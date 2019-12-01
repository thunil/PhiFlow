# Because division is different in Python 2 and 3
from __future__ import division

import itertools
import numpy as np

from phi import math, struct
from .physics import StateDependency, Physics
from .pressuresolver.base import FluidDomain
from .field import advect, StaggeredGrid, union_mask
from .field.effect import Gravity, gravity_tensor, effect_applied
from .domain import DomainState
from .fluid import solve_pressure


def get_domain(liquid, obstacles):
    if liquid.domaincache is None or not liquid.domaincache.is_valid(obstacles):
        if obstacles is not None:
            obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
            obstacle_grid = obstacle_mask.at(liquid.velocity.center_points, collapse_dimensions=False).data
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


class GridLiquidPhysics(Physics):
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
        fluiddomain = get_domain(liquid, obstacles)
        fluiddomain._active = create_binary_mask(liquid.density.data, threshold=0.1)
        s_distance, ext_velocity = extrapolate(liquid.domain, liquid.velocity, fluiddomain.active(), distance=self.extrapolation_distance)
        ext_velocity = fluiddomain.with_hard_boundary_conditions(ext_velocity)

        density = advect.semi_lagrangian(liquid.density, ext_velocity, dt=dt)
        velocity = advect.semi_lagrangian(ext_velocity, ext_velocity, dt=dt)

        for effect in density_effects:
            density = effect_applied(effect, density, dt=dt)

        # Update the active mask based on the new fluid-filled grid cells (for pressure solve)
        fluiddomain._active = create_binary_mask(density.data, threshold=0.1)

        forces = liquid.staggered_grid('forces', 0).staggered_tensor() + dt * gravity_tensor(gravity, liquid.rank)
        velocity = velocity + liquid.domain.staggered_grid(forces)
        velocity = liquid_divergence_free(liquid, velocity, fluiddomain, self.pressure_solver)

        return liquid.copied_with(density=density, velocity=velocity, signed_distance=s_distance,
                                  domaincache=fluiddomain, age=liquid.age + dt)


GRID_LIQUID = GridLiquidPhysics()


@struct.definition()
class GridLiquid(DomainState):

    def __init__(self, domain, density=0.0, velocity=0.0, tags=('gridliquid', 'velocityfield'), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return GRID_LIQUID

    @struct.attr(default=0.0)
    def density(self, d):
        return self.centered_grid('density', d)

    @struct.attr(default=0.0)
    def velocity(self, v):
        return self.staggered_grid('velocity', v)

    @struct.attr(default=0.0)
    def signed_distance(self, s):
        return self.centered_grid('SDF', s)

    @struct.attr(default=None)
    def domaincache(self, d):
        return d

    def __repr__(self):
        return "Liquid[density: %s, velocity: %s]" % (self.density, self.velocity)


def liquid_divergence_free(liquid, velocity, fluiddomain, pressure_solver=None):
    assert isinstance(velocity, StaggeredGrid)
    _, ext_velocity = extrapolate(liquid.domain, velocity, fluiddomain.active(), distance=2)
    ext_velocity = fluiddomain.with_hard_boundary_conditions(ext_velocity)
    divergence_field = ext_velocity.divergence(physical_units=False)
    pressure, iteration = solve_pressure(divergence_field, fluiddomain, pressure_solver=pressure_solver)
    pressure_gradient = StaggeredGrid.gradient(pressure)
    pressure_gradient = pressure_gradient.copied_with(data=[pressure_gradient.data[i] * velocity.dx[i] for i in range(velocity.rank)])
    velocity -= fluiddomain.with_hard_boundary_conditions(pressure_gradient)
    return velocity


def create_binary_mask(tensor, threshold=1e-5):
    """
    Builds a binary tensor with the same shape as the input tensor. Wherever tensor is greater than threshold, the binary mask will contain a '1', else the entry will be '0'.
        :param threshold: Optional scalar value. Threshold relative to the maximal value in the tensor, must be between 0 and 1. Default is 1e-5.
        :return: A tensor which is a binary mask of the given input tensor.
    """
    f_max = math.max(math.abs(tensor))
    scaled_tensor = math.divide_no_nan(math.abs(tensor), f_max)
    binary_mask = math.ceil(scaled_tensor - threshold)

    return binary_mask


def create_surface_mask(particle_mask):
    # When we create inner contour, we don't want the fluid-wall boundaries to show up as surface, so we should pad with symmetric edge values.
    mask = math.pad(particle_mask, [[0, 0]] + [[1, 1]] * math.spatial_rank(particle_mask) + [[0, 0]], "constant")
    dims = range(math.spatial_rank(mask))
    bcs = math.zeros_like(particle_mask)

    # Move in every possible direction to assure corners are properly set.
    directions = np.array(list(itertools.product(
        *np.tile((-1, 0, 1), (len(dims), 1))
    )))

    for d in directions:
        d_slice = tuple([(slice(2, None) if d[i] == -1 else slice(0, -2) if d[i] == 1 else slice(1, -1)) for i in dims])
        center_slice = tuple([slice(1, -1) for _ in dims])

        # Create inner contour of particles
        bc_d = math.maximum(mask[(slice(None),) + d_slice + (slice(None),)],
                            mask[(slice(None),) + center_slice + (slice(None),)]) - \
               mask[(slice(None),) + d_slice + (slice(None),)]
        bcs = math.maximum(bcs, bc_d)
    return bcs


def extrapolate(domain, input_field, active_mask, distance=10):
    """
    Create a signed distance field for the grid, where negative signs are fluid cells and positive signs are empty cells. The fluid surface is located at the points where the interpolated value is zero. Then extrapolate the input field into the air cells.
        :param domain: Domain that can create new Fields
        :param input_field: Field to be extrapolated
        :param active_mask: One dimensional binary mask indicating where fluid is present
        :param distance: Optional maximal distance (in number of grid cells, i.e. local coordinates) where signed distance should still be calculated / how far should be extrapolated.
        :return s_distance: tensor containing signed distance field
        :return ext_field: a new Field with extrapolated values
    """
    ext_data = input_field.data
    dx = input_field.dx
    if isinstance(input_field, StaggeredGrid):
        ext_data = input_field.staggered_tensor()
        active_mask = math.pad(active_mask, [[0, 0]] + [[0, 1]] * input_field.rank + [[0, 0]], "constant")

    dims = range(input_field.rank)
    # Larger than distance to be safe. It could start extrapolating velocities from outside distance into the field.
    signs = -1 * (2 * active_mask - 1)
    s_distance = 2.0 * (distance + 1) * signs
    surface_mask = create_surface_mask(active_mask)

    # surface_mask == 1 doesn't output a tensor, just a scalar, but >= works.
    # Initialize the distance with 0 at the surface
    # Previously initialized with -0.5*dx, i.e. the cell is completely full (center is 0.5*dx inside the fluid surface). For stability and looks this was changed to 0 * dx, i.e. the cell is only half full. This way small changes to the SDF won't directly change neighbouring empty cells to fluid cells.
    s_distance = math.where((surface_mask >= 1), -0.0 * math.ones_like(s_distance), s_distance)

    directions = np.array(list(itertools.product(
        *np.tile((-1, 0, 1), (len(dims), 1))
    )))

    # First make a move in every positive direction (StaggeredGrid velocities there are correct, we want to extrapolate these)
    if isinstance(input_field, StaggeredGrid):
        for d in directions:
            if (d <= 0).all():
                continue

            # Shift the field in direction d, compare new distances to old ones.
            d_slice = tuple(
                [(slice(1, None) if d[i] == -1 else slice(0, -1) if d[i] == 1 else slice(None)) for i in dims])

            d_field = math.pad(ext_data,
                               [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in
                                           dims] + [[0, 0]], "symmetric")
            d_field = d_field[(slice(None),) + d_slice + (slice(None),)]

            d_dist = math.pad(s_distance,
                              [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in dims] + [
                                  [0, 0]], "symmetric")
            d_dist = d_dist[(slice(None),) + d_slice + (slice(None),)]
            d_dist += np.sqrt((dx * d).dot(dx * d)) * signs

            if (d.dot(d) == 1) and (d >= 0).all():
                # Pure axis direction (1,0,0), (0,1,0), (0,0,1)
                updates = (math.abs(d_dist) < math.abs(s_distance)) & (surface_mask <= 0)
                updates_velocity = updates & (signs > 0)
                ext_data = math.where(
                    math.concat([(math.zeros_like(updates_velocity) if d[i] == 1 else updates_velocity) for i in dims],
                                axis=-1), d_field, ext_data)
                s_distance = math.where(updates, d_dist, s_distance)
            else:
                # Mixed axis direction (1,1,0), (1,1,-1), etc.
                continue

    for _ in range(distance):
        # Create a copy of current distance
        buffered_distance = 1.0 * s_distance
        for d in directions:
            if (d == 0).all():
                continue

            # Shift the field in direction d, compare new distances to old ones.
            d_slice = tuple(
                [(slice(1, None) if d[i] == -1 else slice(0, -1) if d[i] == 1 else slice(None)) for i in dims])

            d_field = math.pad(ext_data,
                               [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in
                                           dims] + [[0, 0]], "symmetric")
            d_field = d_field[(slice(None),) + d_slice + (slice(None),)]

            d_dist = math.pad(s_distance,
                              [[0, 0]] + [([0, 1] if d[i] == -1 else [1, 0] if d[i] == 1 else [0, 0]) for i in dims] + [
                                  [0, 0]], "symmetric")
            d_dist = d_dist[(slice(None),) + d_slice + (slice(None),)]
            d_dist += np.sqrt((dx * d).dot(dx * d)) * signs

            # We only want to update velocity that is outside of fluid
            updates = (math.abs(d_dist) < math.abs(buffered_distance)) & (surface_mask <= 0)
            updates_velocity = updates & (signs > 0)
            ext_data = math.where(math.concat([updates_velocity] * math.spatial_rank(ext_data), axis=-1), d_field,
                                  ext_data)
            buffered_distance = math.where(updates, d_dist, buffered_distance)

        s_distance = buffered_distance

    # Cut off inaccurate values
    distance_limit = -distance * (2 * active_mask - 1)
    s_distance = math.where(math.abs(s_distance) < distance, s_distance, distance_limit)

    if isinstance(input_field, StaggeredGrid):
        ext_field = domain.staggered_grid(ext_data, extrapolation=domain.boundaries.extrapolation_mode)
        stagger_slice = tuple([slice(0, -1) for i in dims])
        s_distance = s_distance[(slice(None),) + stagger_slice + (slice(None),)]
    else:
        ext_field = input_field.copied_with(data=ext_data)

    return s_distance, ext_field
