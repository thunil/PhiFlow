# Because division is different in Python 2 and 3
from __future__ import division

import itertools

import numpy as np

from phi import math, struct
from .domain import DomainState
from .field import StaggeredGrid, advect, union_mask, SampledField
from .field.effect import Gravity, gravity_tensor, effect_applied
from .field.sampled import distribute_points
from .field.util import extrapolate, create_surface_mask
from .fluid import solve_pressure, Fluid
from .material import Material
from .physics import StateDependency, Physics
from .pressuresolver.solver_api import FluidDomain


class FreeSurfaceFlow(Physics):
    """
Physics for Grid-based liquid simulation directly advecting the density.
Supports obstacles, density effects and global gravity.
    """

    def __init__(self, pressure_solver=None, extrapolation_distance=30):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True)])
        self.pressure_solver = pressure_solver
        self.extrapolation_distance = extrapolation_distance

    def step(self, liquid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=()):
        if isinstance(liquid, LevelsetLiquid):
            return _LevelsetPhysics().step(liquid, self.pressure_solver, dt, obstacles, gravity, density_effects)
        if isinstance(liquid, FlipLiquid):
            return _FlipPhysics().step(liquid, self.pressure_solver, dt, obstacles, gravity, density_effects)

        fluiddomain = self.get_domain(liquid, obstacles)
        
        ext_velocity, _ = extrapolate(liquid.velocity, fluiddomain.active_tensor(), voxel_distance=self.extrapolation_distance)
        ext_velocity = fluiddomain.with_hard_boundary_conditions(ext_velocity)

        density = advect.semi_lagrangian(liquid.density, ext_velocity, dt=dt)
        velocity = advect.semi_lagrangian(ext_velocity, ext_velocity, dt=dt)

        for effect in density_effects:
            density = effect_applied(effect, density, dt=dt)

        # Update the active mask based on the new fluid-filled grid cells (for pressure solve)
        fluiddomain = fluiddomain.copied_with(active=liquid.domain.centered_grid(liquid_mask(density.data, threshold=0.1), extrapolation='constant'))

        forces = liquid.staggered_grid('forces', 0).staggered_tensor() + dt * gravity_tensor(gravity, liquid.rank)
        velocity = velocity + liquid.domain.staggered_grid(forces)
        velocity = liquid_divergence_free(liquid, velocity, fluiddomain, self.pressure_solver)
        velocity = fluiddomain.with_hard_boundary_conditions(velocity)

        return liquid.copied_with(density=density, velocity=velocity, age=liquid.age + dt)

    @staticmethod
    def get_domain(liquid, obstacles):
        # if liquid.domaincache is None or not liquid.domaincache.is_valid(obstacles):
        if obstacles is not None:
            obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
            obstacle_grid = obstacle_mask.at(liquid.velocity.center_points, collapse_dimensions=False)
            mask = 1 - obstacle_grid
        else:
            mask = liquid.centered_grid('mask', 1)

        extrapolation = Material.accessible_extrapolation_mode(liquid.domain.boundaries)
        mask = mask.copied_with(extrapolation=extrapolation)

        active_mask = mask * liquid_active_mask(liquid)
        return FluidDomain(liquid.domain, obstacles, active=active_mask.copied_with(extrapolation='constant'), accessible=mask)
    # else:
    #     return liquid.domaincache.copied_with(active=liquid_active_mask(liquid))


def liquid_active_mask(fluid):
    if isinstance(fluid, Fluid):
        return fluid.domain.centered_grid(liquid_mask(fluid.density.data, threshold=0.1), extrapolation='constant')
    else:
        return fluid.active_mask


def liquid_divergence_free(liquid, velocity, fluiddomain, pressure_solver=None):
    assert isinstance(velocity, StaggeredGrid)
    ext_velocity, _ = extrapolate(velocity, fluiddomain.active_tensor(), voxel_distance=2)
    ext_velocity = fluiddomain.with_hard_boundary_conditions(ext_velocity)
    divergence_field = ext_velocity.divergence(physical_units=False)
    pressure, iteration = solve_pressure(divergence_field, fluiddomain, pressure_solver=pressure_solver)
    pressure_gradient = StaggeredGrid.gradient(pressure)
    pressure_gradient = pressure_gradient.copied_with(data=[pressure_gradient.data[i] * velocity.dx[i] for i in range(velocity.rank)])
    velocity -= fluiddomain.with_hard_boundary_conditions(pressure_gradient)
    return velocity


def liquid_mask(tensor, threshold=1e-5):
    """
    Builds a binary tensor with the same shape as the input tensor. Wherever tensor is greater than threshold, the binary mask will contain a '1', else the entry will be '0'.
        :param tensor: density tensor (float)
        :param threshold: Optional scalar value. Threshold relative to the maximal value in the tensor, must be between 0 and 1. Default is 1e-5.
        :return: A tensor which is a binary mask of the given input tensor.
    """
    f_max = math.max(math.abs(tensor))
    scaled_tensor = math.divide_no_nan(math.abs(tensor), f_max)
    binary_mask = math.ceil(scaled_tensor - threshold)

    return binary_mask


class _LevelsetPhysics(object):

    def step(self, liquid, pressure_solver, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=()):
        fluiddomain = FreeSurfaceFlow.get_domain(liquid, obstacles)

        # Assume input has a divergence free velocity
        sdf, velocity = self.advect(liquid, fluiddomain, dt)
        # Update active mask after advection
        # We take max of the dx, because currently my implementation only accepts scalar dx, i.e. constant ratio rescaling.
        fluiddomain = fluiddomain.copied_with(active=liquid.domain.centered_grid(self.update_active_mask(sdf, density_effects, dx=max(sdf.dx), dt=dt), extrapolation='constant'))

        sdf = recompute_sdf(sdf, fluiddomain.active_tensor(), velocity, distance=liquid.distance, dt=dt)

        velocity = self.apply_forces(velocity, gravity, dt)
        velocity = liquid_divergence_free(liquid, velocity, fluiddomain, pressure_solver)
        velocity = fluiddomain.with_hard_boundary_conditions(velocity)

        return liquid.copied_with(sdf=sdf, velocity=velocity, active_mask=fluiddomain.active, age=liquid.age + dt)

    @staticmethod
    def advect(liquid, fluiddomain, dt):
        # Advect liquid SDF and velocity using extrapolated velocity
        ext_velocity_free, _ = extrapolate(liquid.velocity, fluiddomain.active_tensor(), voxel_distance=liquid.distance)
        ext_velocity = fluiddomain.with_hard_boundary_conditions(ext_velocity_free)

        # When advecting SDF we don't want to replicate boundary values when the sample coordinates are out of bounds, we want the fluid to move further away from the boundary. We increase the distance when sampling outside of the boundary.
        rank = liquid.rank
        padded_sdf = math.pad(liquid.sdf.data, [[0, 0]] + [[1, 1]] * rank + [[0, 0]], "symmetric")

        zero = math.zeros_like(liquid.sdf.data)
        padded_cells = 0

        updim = True
        if updim:
            # For just upper dimension
            padded = math.pad(zero, [[0, 0]] + [([1, 0] if i == (rank - 2) else [1, 1]) for i in range(rank)] + [[0, 0]], "constant", constant_values=0)
            padded_cells = math.pad(padded, [[0, 0]] + [([0, 1] if i == (rank - 2) else [0, 0]) for i in range(rank)] + [[0, 0]], "constant", constant_values=max(ext_velocity.dx))
        else:
            # Creating a mask for padding in all directions (in case we don't want the special case for upper dimension)
            for d in range(rank):
                padded = math.pad(zero, [[0, 0]] + [([0, 0] if d == i else [1, 1]) for i in range(rank)] + [[0, 0]], "constant", constant_values=0)
                padded = math.pad(padded, [[0, 0]] + [([1, 1] if d == i else [0, 0]) for i in range(rank)] + [[0, 0]], "constant", constant_values=1)

                padded_cells += padded

            padded_cells = max(ext_velocity.dx) * math.sqrt(padded_cells)

        # Increase distance outside of boundaries by dx, this will make sure that during advection we have proper wall separation
        padded_sdf += padded_cells

        padded_sdf = liquid.centered_grid('padded_sdf', padded_sdf)
        padded_ext_v = liquid.staggered_grid('padded_extrapolated_velocity', math.pad(ext_velocity.staggered_tensor(), [[0, 0]] + [[1, 1]] * rank + [[0, 0]], "symmetric"))

        padded_sdf = advect.semi_lagrangian(padded_sdf, padded_ext_v, dt=dt)
        stagger_slice = tuple([slice(1, -1) for i in range(rank)])
        sdf = liquid.centered_grid('sdf', padded_sdf.data[(slice(None),) + stagger_slice + (slice(None),)])

        # Advect the extrapolated velocity that hasn't had BC applied. This will make sure no interpolation occurs with 0 from BC.
        velocity = advect.semi_lagrangian(ext_velocity_free, ext_velocity, dt=dt)
        return sdf, velocity

    @staticmethod
    def update_active_mask(sdf, effects, dx=1.0, dt=1.0):
        # The "sdf" parameter is a Field here, for utility when creating zero Fields
        # Find the active cells from the Signed Distance Field

        ones = math.ones_like(sdf.data)
        active_mask = math.where(sdf.data < 0.5 * dx, ones, 0.0 * ones)
        inflow_grid = sdf.with_data(math.zeros_like(sdf.data))

        for effect in effects:
            inflow_grid = effect_applied(effect, inflow_grid, dt=dt)

        inflow_mask = liquid_mask(inflow_grid.data, threshold=0)
        # Logical OR between the masks
        active_mask = active_mask + inflow_mask - active_mask * inflow_mask
        return active_mask

    @staticmethod
    def apply_forces(velocity, gravity, dt=1.0):
        forces = dt * gravity_tensor(gravity, velocity.rank)
        return velocity.with_data(velocity.staggered_tensor() + forces)


@struct.definition()
class LevelsetLiquid(DomainState):

    def __init__(self, domain, active_mask=0.0, velocity=0.0, distance=30, tags=('sdfliquid', 'velocityfield'), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return FreeSurfaceFlow()

    @struct.variable(default=0.0)
    def velocity(self, velocity):
        return self.staggered_grid('velocity', velocity)

    @struct.variable(default=0.0)
    def active_mask(self, active_mask):
        return self.domain.centered_grid(active_mask, extrapolation='constant')

    @struct.variable(default=None, dependencies=[DomainState.domain, 'velocity', 'active_mask', 'distance'])
    def sdf(self, s):
        if s is None:
            _, s = extrapolate(self.velocity, self.active_mask.data, voxel_distance=self.distance)

        return self.centered_grid('SDF', s)

    @struct.constant(default=10)
    def distance(self, distance):
        """
    Defines the distance in grid cells over which should be extrapolated, i.e. distance over which the SDF value is correct.
        """
        return distance

    def __repr__(self):
        return "Liquid[SDF: %s, velocity: %s]" % (self.sdf, self.velocity)


def recompute_sdf(sdf, active_mask, velocity, distance=10, dt=1.0):
    """
        :param sdf: a CenteredGrid that can be used for calculations.
        :param active_mask: a tensor that is a binary mask to indicate where fluid is present
        :return s_distance: a CenteredGrid containing the signed distance field
    """
    sdf_data = sdf.data
    dx = sdf.dx
    signs = -1 * (2 * active_mask - 1)
    s_distance = 2.0 * (distance + 1) * signs
    surface_mask = create_surface_mask(active_mask)

    # For new active cells via inflow (cells that were outside fluid in old sdf) we want to initialize their signed distance to the default
    # Previously initialized with -0.5*dx, i.e. the cell is completely full (center is 0.5*dx inside the fluid surface). For stability and looks this was changed to 0 * dx, i.e. the cell is only half full. This way small changes to the SDF won't directly change neighbouring empty cells to fluidcells.
    sdf_data = math.where((active_mask >= 1) & (sdf_data >= 0.5 * max(dx)), -0.0 * math.ones_like(sdf_data), sdf_data)
    # Use old Signed Distance values at the surface, then completely recompute the Signed Distance Field
    s_distance = math.where((surface_mask >= 1), sdf_data, s_distance)

    dims = range(sdf.rank)
    directions = np.array(list(itertools.product(
        *np.tile((-1, 0, 1), (len(dims), 1))
    )))

    for _ in range(distance):
        # Create a copy of current distance
        buffered_distance = 1.0 * s_distance
        for d in directions:
            if (d == 0).all():
                continue
            # Shift the field in direction d, compare new distances to old ones.
            d_slice = tuple([(slice(1, None) if d[i] == -1 else slice(0, -1) if d[i] == 1 else slice(None)) for i in dims])
            d_dist = math.pad(s_distance, [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in dims] + [[0, 0]], "symmetric")
            d_dist = d_dist[(slice(None),) + d_slice + (slice(None),)]
            d_dist += np.sqrt((dx * d).dot(dx * d)) * signs
            # Update smaller distances and prevent updating the distance at the surface
            updates = (math.abs(d_dist) < math.abs(buffered_distance)) & (surface_mask <= 0)
            buffered_distance = math.where(updates, d_dist, buffered_distance)

        s_distance = buffered_distance

    distance_limit = -distance * (2 * active_mask - 1)
    s_distance = math.where(math.abs(s_distance) < distance, s_distance, distance_limit)

    return sdf.copied_with(data=s_distance)


class _FlipPhysics(object):
    """
Physics for Fluid Implicit Particles simulation for liquids.
Supports obstacles, density effects and global gravity.
    """

    def step(self, liquid, pressure_solver, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=()):
        # We advect as the last part of the step, because we must make sure we have divergence free velocity fields. We cannot advect first assuming the input is divergence free because it never will be due to the velocities being stored on the particles.
        fluiddomain = self.get_particle_domain(liquid, obstacles)

        # Create velocity field from particle velocities and make it divergence free. Then interpolate back the change to the particle velocities.
        velocity_field = liquid.velocity.at(liquid.staggered_grid('staggered', 0))

        velocity_field_with_forces = self.apply_forces(velocity_field, gravity, dt)
        div_free_velocity_field = liquid_divergence_free(liquid, velocity_field_with_forces, fluiddomain, pressure_solver)

        velocity = liquid.velocity.data + self.particle_velocity_change(fluiddomain, liquid.points.data,
                                                                        (div_free_velocity_field - velocity_field))

        # Advect the points
        points = self.advect_points(fluiddomain, liquid.points.data, div_free_velocity_field, dt)

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


@struct.definition()
class FlipLiquid(DomainState):

    def __init__(self, domain, points, velocity=0.0, particles_per_cell=1, tags=('flipliquid', ), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return FreeSurfaceFlow()

    @struct.variable()
    def points(self, points):
        if isinstance(points, SampledField):
            return points
        return SampledField('points', points, data=points, mode='any')

    @struct.variable(default=0.0)
    def velocity(self, velocity):
        if isinstance(velocity, SampledField):
            return velocity
        if isinstance(velocity, (int, float)):
            velocity = [velocity] * self.rank
        return SampledField('velocity', self.points.data, data=velocity, mode='mean')

    @property
    def density(self):
        return SampledField('density', self.points.data, data=1.0, mode='add')

    @property
    def active_mask(self):
        return SampledField('active_mask', self.points.data, data=1.0, mode='any')

    @struct.constant(default=1)
    def particles_per_cell(self, particles_per_cell):
        return particles_per_cell

    def __repr__(self):
        return "Liquid[density: %s, velocity: %s]" % (self.density, self.velocity)
