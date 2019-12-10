# Because division is different in Python 2 and 3
from __future__ import division

import itertools

import numpy as np

from phi import math, struct
from phi.physics.field import union_mask
from phi.physics.field.util import extrapolate, create_surface_mask
from phi.physics.material import Material
from .domain import DomainState
from .field import StaggeredGrid
from .field import advect
from .field.effect import Gravity, gravity_tensor, effect_applied
from .fluid import solve_pressure, Fluid
from .physics import StateDependency, Physics
from .pressuresolver.solver_api import FluidDomain


class IncompressibleLiquid(Physics):
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
        if isinstance(liquid, SDFLiquid):
            return _SDF_LIQUID.step(liquid, self.pressure_solver, dt, obstacles, gravity, density_effects)

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


INCOMPRESSIBLE_LIQUID = IncompressibleLiquid()


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


class _SDFLiquidPhysics(object):

    def step(self, liquid, pressure_solver, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=()):
        fluiddomain = IncompressibleLiquid.get_domain(liquid, obstacles)

        # Assume input has a divergence free velocity
        sdf, velocity = self.advect(liquid, fluiddomain, dt)
        # Update active mask after advection
        # We take max of the dx, because currently my implementation only accepts scalar dx, i.e. constant ratio rescaling.
        fluiddomain = fluiddomain.copied_with(active=liquid.domain.centered_grid(self.update_active_mask(sdf.data, density_effects, dx=max(sdf.dx), dt=dt), extrapolation='constant'))

        sdf = recompute_sdf(sdf, fluiddomain.active_tensor(), velocity, distance=liquid.distance, dt=dt)

        velocity = self.apply_forces(velocity, gravity, dt)
        velocity = liquid_divergence_free(liquid, velocity, fluiddomain, pressure_solver)

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
        # Find the active cells from the Signed Distance Field

        ones = math.ones_like(sdf)
        active_mask = math.where(sdf < 0.5 * dx, ones, 0.0 * ones)
        inflow_grid = math.zeros_like(active_mask)

        for effect in effects:
            inflow_grid = effect_applied(effect, inflow_grid, dt=dt)

        inflow_mask = liquid_mask(inflow_grid, threshold=0)
        # Logical OR between the masks
        active_mask = active_mask + inflow_mask - active_mask * inflow_mask
        return active_mask

    @staticmethod
    def apply_forces(velocity, gravity, dt=1.0):
        forces = dt * gravity_tensor(gravity, velocity.rank)
        return velocity.with_data(velocity.staggered_tensor() + forces)


_SDF_LIQUID = _SDFLiquidPhysics()


@struct.definition()
class SDFLiquid(DomainState):

    def __init__(self, domain, active_mask=0.0, velocity=0.0, distance=30, tags=('sdfliquid', 'velocityfield'), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return INCOMPRESSIBLE_LIQUID

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
