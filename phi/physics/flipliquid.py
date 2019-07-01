from .domain import *
from phi.math import *
from operator import itemgetter
import itertools
from phi.math.sampled import *
# Many functions used from gridliquid
from .gridliquid import *


class FlipLiquidPhysics(Physics):

    def __init__(self, pressure_solver=None):
        Physics.__init__(self, {'obstacles': ['obstacle'], 'inflows': 'inflow'})
        self.pressure_solver = pressure_solver

    def step(self, state, dt=1.0, obstacles=(), inflows=(), **dependent_states):
        assert len(dependent_states) == 0
        domaincache = domain(state, obstacles)
        # step
        inflow_density = dt * inflow(inflows, state.grid)
        inflow_points = random_grid_to_coords(inflow_density, state.particles_per_cell)
        points = math.concat([state.points, inflow_points], axis=1)
        velocity = math.concat([state.velocity, math.zeros_like(inflow_points)], axis=1)
        forces = dt * state.gravity
        velocity += forces

        # Update the active mask based on the new fluid-filled grid cells (for pressure solve)
        density = grid(state.grid, points)
        velocity_field = grid(state.grid, points, velocity, staggered=True)
        velocity_field = domaincache.with_hard_boundary_conditions(velocity_field)
        active_mask = create_binary_mask(density, threshold=0.0)
        domaincache._active = active_mask

        _, ext_velocity = extrapolate(velocity_field, domaincache.active(), dx=1.0, distance=2)
        ext_velocity = domaincache.with_hard_boundary_conditions(ext_velocity)

        state.vel_before = velocity_field
        state.div_before = ext_velocity.divergence()

        div_free_velocity_field = divergence_free(velocity_field, domaincache, self.pressure_solver, state=state)

        state.vel_after = div_free_velocity_field

        solid_paddings, open_paddings = domaincache.domain._get_paddings(lambda material: material.solid)
        active_mask = domaincache.active()
        mask = active_mask[(slice(None),) + tuple([slice(1,-1)] * domaincache.rank) + (slice(None),)]
        mask = math.pad(mask, solid_paddings, "constant", 0)
        mask = math.pad(mask, open_paddings, "constant", 1)

        # We redefine the borders, when there is a solid wall we want to extrapolate to these cells.
        extrapolate_mask = mask * active_mask
        
        # Interpolate the change from the grid and add it to the particle velocity
        _, ext_gradp = extrapolate((div_free_velocity_field - velocity_field), extrapolate_mask, dx=1.0, distance=2)
        state.ext_gradp = ext_gradp
        gradp_particles = grid_to_particles(state.grid, points, ext_gradp, staggered=True)
        velocity += gradp_particles

        #max_vel = math.max(math.abs(velocity.staggered))
        _, ext_velocity = extrapolate(div_free_velocity_field, domaincache.active(), dx=1.0, distance=30)
        ext_velocity = domaincache.with_hard_boundary_conditions(ext_velocity)

        grid_advection_velocity = grid_to_particles(state.grid, points, ext_velocity, staggered=True)
        points += dt * grid_advection_velocity

        state.advection_velocity = ext_velocity

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
        
        return state.copied_with(points=points, velocity=velocity, age=state.age + dt)


FLIPLIQUID = FlipLiquidPhysics()


class FlipLiquid(State):
    __struct__ = State.__struct__.extend(('points', 'velocity'),
                            ('_domain', '_gravity'))

    def __init__(self, domain=Open2D,
                 density=0.0, velocity=0.0, gravity=-9.81, batch_size=None, particles_per_cell=1):
        State.__init__(self, tags=('liquid', 'velocityfield'), batch_size=batch_size)
        self._domain = domain
        self._density = density
        self.particles_per_cell = particles_per_cell
        self.batch_size = batch_size
        self.domaincache = None
        self._last_pressure = None
        self._last_pressure_iterations = None
        self.advection_velocity = None
        self.div_before = None
        self.vel_before = None
        self.vel_after = None
        self.ext_gradp = None

        # Density only used to initialize the particle array, afterwards density is never used for calculations again.
        # points has dimensions (batch, particle_number, spatial_rank), when we concatenate we always add to the "particle_number" dimension.
        self.points = random_grid_to_coords(self._density, particles_per_cell)
        self.velocity = zeros_like(self.points) + velocity

        if isinstance(gravity, (tuple, list)):
            assert len(gravity) == domain.rank
            self._gravity = np.array(gravity)
        elif domain.rank == 1:
            self._gravity = np.array([gravity])
        else:
            assert domain.rank >= 2
            gravity = ([0] * (domain.rank - 2)) + [gravity] + [0]
            self._gravity = np.array(gravity)

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
    def last_pressure(self):
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

