from __future__ import division

from phi import math, struct
from .gridliquid import extrapolate, liquid_divergence_free
from .field.mask import union_mask
from .field.effect import Gravity, gravity_tensor, effect_applied
from .pressuresolver.base import FluidDomain
from .physics import Physics, StateDependency
from .domain import DomainState
from .field.sampled import SampledField, random_grid_to_coords


def get_particle_domain(liquid, obstacles):
    if liquid.domaincache is None or not liquid.domaincache.is_valid(obstacles):
        if obstacles is not None:
            obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
            # Difference with grid-based liquid simulations
            obstacle_grid = obstacle_mask.at(liquid.staggered_grid('center', 0).center_points,
                                             collapse_dimensions=False).data
            mask = 1 - obstacle_grid
        else:
            mask = math.ones(liquid.domain.centered_shape(name='active')).data

        if liquid.domaincache is None:
            active_mask = mask
        else:
            active_mask = mask * liquid.domaincache.active()
        return FluidDomain(liquid.domain, obstacles, active=active_mask, accessible=mask)
    else:
        return liquid.domaincache


class FlipLiquidPhysics(Physics):
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

        fluiddomain = get_particle_domain(liquid, obstacles)
        fluiddomain._active = liquid.active_mask.at(liquid.domain).data

        # Create velocity field from particle velocities and make it divergence free. Then interpolate back the change to the particle velocities.
        velocity_field = liquid.velocity.at(liquid.staggered)

        velocity_field_with_forces = self.apply_field_forces(liquid, velocity_field, gravity, dt)
        div_free_velocity_field = liquid_divergence_free(liquid, velocity_field_with_forces, fluiddomain,
                                                         self.pressure_solver)

        velocity = liquid.velocity.data + self.particle_velocity_change(fluiddomain, liquid.points,
                                                                        (div_free_velocity_field - velocity_field))

        # Advect the points
        points = self.advect_points(fluiddomain, liquid.points, div_free_velocity_field, dt)

        # Inflow    
        inflow_density = liquid.domain.centered_grid(0)
        for effect in density_effects:
            inflow_density = effect_applied(effect, inflow_density, dt=dt)
        inflow_points = random_grid_to_coords(inflow_density.data, liquid.particles_per_cell)
        points = math.concat([points, inflow_points], axis=1)
        velocity = math.concat([velocity, 0.0 * (inflow_points)], axis=1)

        # Remove the particles that went out of the simulation boundaries
        points, velocity = self.remove_out_of_bounds(liquid, points, velocity)

        return liquid.copied_with(points=points, velocity=velocity, domaincache=fluiddomain, age=liquid.age + dt)

    @staticmethod
    def apply_field_forces(liquid, velocity_field, gravity, dt):
        forces = dt * (gravity_tensor(gravity, liquid.rank) + liquid.trained_forces.staggered_tensor())
        forces = liquid.domain.staggered_grid(forces)
        return velocity_field + forces

    @staticmethod
    def particle_velocity_change(fluiddomain, points, velocity_field_change):
        solid_paddings, open_paddings = fluiddomain.domain._get_paddings(lambda material: material.solid)
        active_mask = fluiddomain._active
        mask = active_mask[(slice(None),) + tuple([slice(1, -1)] * fluiddomain.rank) + (slice(None),)]
        mask = math.pad(mask, solid_paddings, "constant", 0)
        mask = math.pad(mask, open_paddings, "constant", 1)

        # We redefine the borders, when there is a solid wall we want to extrapolate to these cells. (Even if there is fluid in the solid wall, we overwrite it with the extrapolated value)
        extrapolate_mask = mask * active_mask

        # Interpolate the change from the grid and add it to the particle velocity
        _, ext_gradp = extrapolate(fluiddomain.domain, velocity_field_change, extrapolate_mask, distance=2)
        gradp_particles = ext_gradp.sample_at(points)

        return gradp_particles

    @staticmethod
    def advect_points(fluiddomain, points, velocity_field, dt):
        _, ext_velocity = extrapolate(fluiddomain.domain, velocity_field, fluiddomain.active(), distance=30)
        ext_velocity = fluiddomain.with_hard_boundary_conditions(ext_velocity)

        # Runge-Kutta 3rd order advection scheme
        velocity_rk1 = ext_velocity.sample_at(points)
        velocity_rk2 = ext_velocity.sample_at(points + 0.5 * dt * velocity_rk1)
        velocity_rk3 = ext_velocity.sample_at(points + 0.5 * dt * velocity_rk2)
        velocity_rk4 = ext_velocity.sample_at(points + 1 * dt * velocity_rk3)

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


FLIP_LIQUID = FlipLiquidPhysics()


@struct.definition()
class FlipLiquid(DomainState):

    def __init__(self, domain, points, velocity=0.0, particles_per_cell=1, tags=('flipliquid', ), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))
        self.centered = self.centered_grid('centered', 0)
        self.staggered = self.staggered_grid('staggered', 0)
        self._domaincache = get_particle_domain(self, ())
        self._domaincache._active = self.active_mask.at(domain).data

    def default_physics(self):
        return FLIP_LIQUID

    @struct.attr()
    def points(self, points):
        return points

    @struct.attr(default=0.0)
    def velocity(self, v):
        return SampledField('velocity', self.points, data=v, mode='mean')

    @property
    def density(self):
        return SampledField('density', self.points, data=1.0, mode='add')

    @property
    def active_mask(self):
        return SampledField('active_mask', self.points, data=1.0, mode='any')

    @struct.attr(default=None)
    def domaincache(self, domaincache):  # Domain cache sort of redundant, can be merged with active_mask
        return domaincache

    @struct.attr(default=0.0)
    def trained_forces(self, f):
        return self.staggered_grid('trained_forces', f)

    @struct.prop(default=1)
    def particles_per_cell(self, p):
        return p

    def __repr__(self):
        return "Liquid[density: %s, velocity: %s]" % (self.density, self.velocity)
