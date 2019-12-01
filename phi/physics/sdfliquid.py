from __future__ import division

import itertools
import numpy as np

from phi import math, struct
from .physics import StateDependency, Physics
from .pressuresolver.base import FluidDomain
from .field import advect, StaggeredGrid
from .field.effect import Gravity, gravity_tensor, effect_applied
from .domain import DomainState
from .fluid import solve_pressure

from .gridliquid import extrapolate, get_domain, create_binary_mask, create_surface_mask, liquid_divergence_free


class SDFLiquidPhysics(Physics):

    def __init__(self, pressure_solver=None):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True)])
        self.pressure_solver = pressure_solver

    def step(self, liquid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=()):
        fluiddomain = get_domain(liquid, obstacles)
        # Need correct active mask for advection
        fluiddomain._active = liquid.active_mask

        # Assume input has a divergence free velocity
        sdf, velocity = self.advect(liquid, fluiddomain, dt)
        # Update active mask after advection
        # We take max of the dx, because currently my implementation only accepts scalar dx, i.e. constant ratio rescaling.
        fluiddomain._active = self.update_active_mask(sdf.data, density_effects, dx=max(sdf.dx), dt=dt)

        sdf = recompute_sdf(sdf, fluiddomain.active(), velocity, distance=liquid.distance, dt=dt)

        velocity = self.apply_forces(liquid, velocity, gravity, dt)
        velocity = liquid_divergence_free(liquid, velocity, fluiddomain, self.pressure_solver)

        return liquid.copied_with(sdf=sdf, velocity=velocity, domaincache=fluiddomain, active_mask=fluiddomain.active(), age=liquid.age + dt)

    @staticmethod
    def advect(liquid, fluiddomain, dt):
        # Advect liquid SDF and velocity using extrapolated velocity
        _, ext_velocity_free = extrapolate(liquid.domain, liquid.velocity, fluiddomain.active(), distance=liquid.distance)
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

        inflow_mask = create_binary_mask(inflow_grid)
        # Logical OR between the masks
        active_mask = active_mask + inflow_mask - active_mask * inflow_mask
        return active_mask

    @staticmethod
    def apply_forces(liquid, velocity, gravity, dt=1.0):
        forces = dt * (gravity_tensor(gravity, liquid.rank) + liquid.trained_forces.staggered_tensor())
        forces = liquid.domain.staggered_grid(forces)
        return velocity + forces


SDF_LIQUID = SDFLiquidPhysics()


@struct.definition()
class SDFLiquid(DomainState):

    def __init__(self, domain, density=0.0, velocity=0.0, distance=30, tags=('sdfliquid', 'velocityfield'), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

        self._domaincache = get_domain(self, ())
        self._active_mask = create_binary_mask(self.density.data, threshold=0)
        self._domaincache._active = self._active_mask
        self._sdf_data, _ = extrapolate(self.domain, self.velocity, self._active_mask, distance=distance)
        self._sdf = self.centered_grid('sdf', self._sdf_data)

    def default_physics(self):
        return SDF_LIQUID

    @struct.attr(default=0.0)
    def density(self, d):
        return self.centered_grid('density', d)

    @struct.attr(default=0.0)
    def velocity(self, v):
        return self.staggered_grid('velocity', v)

    @struct.attr(default=0.0)
    def sdf(self, s):
        return self.centered_grid('SDF', s)

    @struct.attr(default=0.0)
    def active_mask(self, a):
        return a

    @struct.attr(default=None)
    def domaincache(self, d):
        return d

    @struct.attr(default=0.0)
    def trained_forces(self, f):
        return self.staggered_grid('trained_forces', f)

    @struct.prop(default=10)
    def distance(self, d):
        return d

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

    # Rough error correction for disappearing SDF
    s_distance -= dt * math.max(math.abs(velocity.staggered_tensor())) * 0.01

    return sdf.copied_with(data=s_distance)
