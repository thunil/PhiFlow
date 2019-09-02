from __future__ import division

from .domain import *
from phi.math import *
from operator import itemgetter
import itertools
import tensorflow as tf
from phi.math.sampled import *
# Many functions used from gridliquid
from .gridliquid import *


class FlipLiquidPhysics(Physics):

    def __init__(self, pressure_solver=None):
        Physics.__init__(self, {'obstacles': ['obstacle'], 'inflows': 'inflow'})
        self.pressure_solver = pressure_solver

    def step(self, state, dt=1.0, obstacles=(), inflows=(), **dependent_states):
        # Not really needed, but here just to be sure
        assert len(dependent_states) == 0
        domaincache = domain(state, obstacles)
        
        # Inflow and forces
        points, velocity = self.add_inflow(state, inflows, dt)

        # Update the active mask based on the new fluid-filled grid cells (for pressure solve)
        active_mask = self.update_active_mask(domaincache, points)

        # Create velocity field from particle velocities and make it divergence free. Then interpolate back the change to the particle velocities.
        velocity_field = grid(domaincache.grid, points, velocity, staggered=True)
        velocity_field = domaincache.with_hard_boundary_conditions(velocity_field)

        velocity_field_with_forces = self.apply_field_forces(state, velocity_field, dt)
        div_free_velocity_field = divergence_free(velocity_field_with_forces, domaincache, self.pressure_solver, state=state)
        
        velocity += self.particle_velocity_change(domaincache, points, (div_free_velocity_field - velocity_field))

        # Advect the points and remove the particles that went out of the simulation boundaries.
        points = self.advect_points(domaincache, points, div_free_velocity_field, dt)
        points, velocity = self.remove_out_of_bounds(state, points, velocity)

        # Update new active mask after advection
        active_mask = self.update_active_mask(domaincache, points)
        
        return state.copied_with(points=points, velocity=velocity, active_mask=active_mask, age=state.age + dt)



    def advect_points(self, domaincache, points, velocity, dt):
        # For now the distance is fixed, we could make it dependent on velocity later on.
        #max_vel = math.max(math.abs(velocity.staggered))
        _, ext_velocity = extrapolate(velocity, domaincache.active(), dx=1.0, distance=30)
        ext_velocity = domaincache.with_hard_boundary_conditions(ext_velocity)

        # Runge Kutta 3rd order advection scheme
        velocity_RK1 = grid_to_particles(domaincache.grid, points, ext_velocity, staggered=True)
        velocity_RK2 = grid_to_particles(domaincache.grid, (points + 0.5 * dt * velocity_RK1), ext_velocity, staggered=True)
        velocity_RK3 = grid_to_particles(domaincache.grid, (points + 0.75 * dt * velocity_RK2), ext_velocity, staggered=True)

        new_points = points + 1/9 * dt * (2 * velocity_RK1 + 3 * velocity_RK2 + 4 * velocity_RK3)
        return new_points


    def add_inflow(self, state, effects, dt):
        inflow_density = math.zeros_like(state.active_mask)
        for effect in effects:
            inflow_density = effect.apply_grid(inflow_density, state.grid, staggered=False, dt=dt)
        inflow_points = random_grid_to_coords(inflow_density, state.particles_per_cell)
        points = math.concat([state.points, inflow_points], axis=1)
        velocity = math.concat([state.velocity, 0.0 * (inflow_points)], axis=1)

        return points, velocity


    def apply_field_forces(self, state, velocity, dt):
        forces = dt * (state.gravity + state.trained_forces.staggered)
        return velocity + forces

    
    def update_active_mask(self, domaincache, points):
        density = grid(domaincache.grid, points)
        active_mask = create_binary_mask(density, threshold=0.0)
        domaincache._active = active_mask

        return active_mask


    def particle_velocity_change(self, domaincache, points, velocity_field_change):
        solid_paddings, open_paddings = domaincache.domain._get_paddings(lambda material: material.solid)
        active_mask = domaincache._active
        mask = active_mask[(slice(None),) + tuple([slice(1,-1)] * domaincache.rank) + (slice(None),)]
        mask = math.pad(mask, solid_paddings, "constant", 0)
        mask = math.pad(mask, open_paddings, "constant", 1)

        # We redefine the borders, when there is a solid wall we want to extrapolate to these cells. (Even if there is fluid in the solid wall, we overwrite it with the extrapolated value)
        extrapolate_mask = mask * active_mask
        
        # Interpolate the change from the grid and add it to the particle velocity
        _, ext_gradp = extrapolate(velocity_field_change, extrapolate_mask, dx=1.0, distance=2)
        gradp_particles = grid_to_particles(domaincache.grid, points, ext_gradp, staggered=True)

        return gradp_particles

    
    def remove_out_of_bounds(self, state, points, velocity):
        # Remove out of bounds
        indices = math.to_int(math.floor(points))
        shape = [points.shape[0], -1, points.shape[-1]]
        #shape = [world.batch_size, -1, state.domain.rank]
        mask = math.prod((indices >= 0) & (indices <= [d-1 for d in state.grid.dimensions]), axis=-1)

        # Out of bounds particles will be deleted
        points = math.boolean_mask(points, mask)
        points = math.reshape(points, shape)

        velocity = math.boolean_mask(velocity, mask)
        velocity = math.reshape(velocity, shape)

        return points, velocity


FLIPLIQUID = FlipLiquidPhysics()


class FlipLiquid(State):
    __struct__ = State.__struct__.extend(('points', 'velocity', '_active_mask', 'trained_forces'),
                            ('_domain', '_gravity'))

    def __init__(self, state_domain=Open2D,
                 density=0.0, velocity=0.0, gravity=-9.81, batch_size=None, particles_per_cell=1):
        State.__init__(self, tags=('liquid', 'velocityfield'), batch_size=batch_size)
        self._domain = state_domain
        self._density = density
        self.particles_per_cell = particles_per_cell
        self.batch_size = batch_size
        self._active_mask = create_binary_mask(self._density, threshold=0)
        self.domaincache = None
        self.domaincache = domain(self, ())
        self.domaincache._active = self._active_mask

        self._last_pressure = None
        self._last_pressure_iterations = None

        # Density only used to initialize the particle array, afterwards density is never used for calculations again.
        # points has dimensions (batch, particle_number, spatial_rank), when we concatenate we always add to the "particle_number" dimension.
        self.points = random_grid_to_coords(self._density, particles_per_cell)
        self.velocity = zeros_like(self.points) + velocity

        if isinstance(gravity, (tuple, list)):
            assert len(gravity) == state_domain.rank
            self._gravity = np.array(gravity)
        else:
            assert state_domain.rank >= 1
            gravity = [gravity] + ([0] * (state_domain.rank - 1))
            self._gravity = np.array(gravity)

        # When you want to train a force, you need to overwrite this value with a tf.Variable that is trainable. This initialization is only a dummy value.
        self.trained_forces = zeros(self.grid.staggered_shape())

    def default_physics(self):
        return FLIPLIQUID

    @property
    def density_field(self):
        return grid(self.grid, self.points, duplicate_handling='add')

    @property
    def _density(self):
        return self._density_field

    @_density.setter
    def _density(self, value):
        self._density_field = initialize_field(value, self.grid.shape())

    @property
    def velocity_field(self):
        return grid(self.grid, self.points, self.velocity, staggered=True)

    @property
    def _velocity(self):
        return self.velocity_field
    
    @_velocity.setter
    def _velocity(self, value):
        self.velocity = zeros_like(self.points) + value

    @property
    def active_mask(self):
        return self._active_mask

    @property
    def domain(self):
        return self._domain

    @property
    def grid(self):
        return self.domain.grid

    @property
    def rank(self):
        return self.grid.rank

    @property
    def gravity(self):
        return self._gravity

    @property
    def pressure(self):
        return self._last_pressure

    @property
    def last_pressure_iterations(self):
        return self._last_pressure_iterations

    def __repr__(self):
        return "Liquid[points: %s, velocity: %s]" % (self.points, self.velocity)

    def __add__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copied_with(velocity=self.velocity + other)
        else:
            return self.copied_with(density=self.points + other)

    def __sub__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copied_with(velocity=self.velocity - other)
        else:
            return self.copied_with(density=self.points - other)
