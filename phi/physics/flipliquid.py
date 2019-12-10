from __future__ import division

from phi import math, struct
from phi.physics.material import Material
from .gridliquid import liquid_divergence_free
from phi.physics.field.util import extrapolate
from .field.mask import union_mask
from .field.effect import Gravity, gravity_tensor, effect_applied
from .pressuresolver.solver_api import FluidDomain
from .physics import Physics, StateDependency
from .domain import DomainState
from .field.sampled import SampledField, distribute_points


class Flip(Physics):
    """
Physics for Fluid Implicit Particles simulation for liquids.
Supports obstacles, density effects and global gravity.
    """

    def __init__(self, pressure_solver=None):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True)])
        self.pressure_solver = pressure_solver

    def step(self, liquid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=()):
        # We advect as the last part of the step, because we must make sure we have divergence free velocity fields. We cannot advect first assuming the input is divergence free because it never will be due to the velocities being stored on the particles.

        fluiddomain = self.get_particle_domain(liquid, obstacles)
        # Create velocity field from particle velocities and make it divergence free. Then interpolate back the change to the particle velocities.
        velocity_field = liquid.velocity.at(liquid.staggered_grid('staggered', 0))

        velocity_field_with_forces = self.apply_forces(velocity_field, gravity, dt)
        div_free_velocity_field = liquid_divergence_free(liquid, velocity_field_with_forces, fluiddomain, self.pressure_solver)

        velocity = liquid.velocity.data + self.particle_velocity_change(fluiddomain, liquid.points,
                                                                        (div_free_velocity_field - velocity_field))

        # Advect the points
        points = self.advect_points(fluiddomain, liquid.points, div_free_velocity_field, dt)

        # Inflow    
        inflow_density = liquid.domain.centered_grid(0)
        for effect in density_effects:
            inflow_density = effect_applied(effect, inflow_density, dt=dt)
        inflow_points = distribute_points(inflow_density.data, liquid.particles_per_cell)
        points = math.concat([points, inflow_points], axis=1)
        velocity = math.concat([velocity, 0.0 * (inflow_points)], axis=1)

        # Remove the particles that went out of the simulation boundaries
        points, velocity = self.remove_out_of_bounds(liquid, points, velocity)

        return liquid.copied_with(points=points, velocity=velocity, age=liquid.age + dt)

    @staticmethod
    def apply_forces(velocity, gravity, dt=1.0):
        forces = dt * gravity_tensor(gravity, velocity.rank)
        return velocity.with_data(velocity.staggered_tensor() + forces)

    @staticmethod
    def particle_velocity_change(fluiddomain, points, velocity_field_change):
        pad_values = struct.map(lambda solid: int(not solid), Material.solid(fluiddomain.domain.boundaries))
        if isinstance(pad_values, (list, tuple)):
            pad_values = [0] + list(pad_values) + [0]

        active_mask = fluiddomain.active.data
        mask = active_mask[(slice(None),) + tuple([slice(1, -1)] * fluiddomain.rank) + (slice(None),)]
        mask = math.pad(mask, [[0,0]] + [[1, 1]] * fluiddomain.rank + [[0,0]], constant_values=pad_values)

        # We redefine the borders, when there is a solid wall we want to extrapolate to these cells. (Even if there is fluid in the solid wall, we overwrite it with the extrapolated value)
        extrapolate_mask = mask * active_mask

        # Interpolate the change from the grid and add it to the particle velocity
        ext_gradp, _ = extrapolate(velocity_field_change, extrapolate_mask, voxel_distance=2)
        # Sample_at requires physical coordinates
        gradp_particles = ext_gradp.sample_at(points * ext_gradp.dx)

        return gradp_particles

    @staticmethod
    def advect_points(fluiddomain, points, velocity_field, dt):
        ext_velocity, _ = extrapolate(velocity_field, fluiddomain.active_tensor(), voxel_distance=30)
        ext_velocity = fluiddomain.with_hard_boundary_conditions(ext_velocity)

        # Runge-Kutta 3rd order advection scheme
        # Sample_at requires physical coordinates
        velocity_rk1 = ext_velocity.sample_at(points * ext_velocity.dx)
        velocity_rk2 = ext_velocity.sample_at(points * ext_velocity.dx + 0.5 * dt * velocity_rk1)
        velocity_rk3 = ext_velocity.sample_at(points * ext_velocity.dx + 0.5 * dt * velocity_rk2)
        velocity_rk4 = ext_velocity.sample_at(points * ext_velocity.dx + 1 * dt * velocity_rk3)

        new_points = points + 1 / 6 * dt * (1 * velocity_rk1 + 2 * velocity_rk2 + 2 * velocity_rk3 + 1 * velocity_rk4)
        return new_points

    @staticmethod
    def remove_out_of_bounds(liquid, points, velocity):
        # Remove out of bounds
        indices = math.to_int(math.floor(points))
        shape = [points.shape[0], -1, points.shape[-1]]
        mask = math.prod((indices >= 0) & (indices <= [d - 1 for d in liquid.domain.resolution]), axis=-1)

        # Out of bounds particles will be deleted
        points = math.boolean_mask(points, mask)
        points = math.reshape(points, shape)

        velocity = math.boolean_mask(velocity, mask)
        velocity = math.reshape(velocity, shape)

        return points, velocity

    @staticmethod
    def get_particle_domain(liquid, obstacles):
        if obstacles is not None:
            obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
            # Difference with grid-based liquid simulations
            obstacle_grid = obstacle_mask.at(liquid.staggered_grid('center', 0).center_points, collapse_dimensions=False)
            mask = 1 - obstacle_grid
        else:
            mask = liquid.centered_grid('mask', 1)
        # If extrapolation of the accessible mask isn't constant, then no boundary conditions will be correct.
        extrapolation = Material.accessible_extrapolation_mode(liquid.domain.boundaries)
        mask = mask.copied_with(extrapolation=extrapolation)

        active_mask = mask * liquid.active_mask.at(liquid.domain)
        return FluidDomain(liquid.domain, obstacles, active=active_mask.copied_with(extrapolation='constant'), accessible=mask)


FLIP = Flip()


@struct.definition()
class FlipLiquid(DomainState):

    def __init__(self, domain, points, velocity=0.0, particles_per_cell=1, tags=('flipliquid', ), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return FLIP

    @struct.variable()
    def points(self, points):
        return points

    @struct.variable(default=0.0)
    def velocity(self, velocity):
        return SampledField('velocity', self.points, data=velocity, mode='mean')

    @property
    def density(self):
        return SampledField('density', self.points, data=1.0, mode='add')

    @property
    def active_mask(self):
        return SampledField('active_mask', self.points, data=1.0, mode='any')

    @struct.constant(default=1)
    def particles_per_cell(self, particles_per_cell):
        return particles_per_cell

    def __repr__(self):
        return "Liquid[density: %s, velocity: %s]" % (self.density, self.velocity)
