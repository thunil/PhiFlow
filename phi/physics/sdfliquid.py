from .domain import *
from phi.math import *
from operator import itemgetter
import itertools
# Many functions used from gridliquid
from .gridliquid import *


class SDFLiquidPhysics(Physics):

    def __init__(self, pressure_solver=None):
        Physics.__init__(self, {'obstacles': ['obstacle'], 'inflows': 'inflow'})
        self.pressure_solver = pressure_solver

    def step(self, state, dt=1.0, obstacles=(), inflows=(), **dependent_states):
        assert len(dependent_states) == 0
        domaincache = domain(state, obstacles)

        #state.mask_before = state.velocity
        active_mask = self.update_active_mask(state.sdf, inflows, domaincache)
        #domaincache._active = active_mask

        velocity = self.apply_forces(state, dt)
        velocity = divergence_free(velocity, domaincache, self.pressure_solver, state=state)

        sdf, velocity = self.advect(state, velocity, dt)
        active_mask = self.update_active_mask(sdf, inflows, domaincache)
        #domaincache._active = active_mask

        sdf = recompute_sdf(sdf, active_mask, distance=state._distance)

        # Necessary for obstacle advection (so they don't move using extrapolated velocity in air)
        vel_active = self.staggered_active_mask(domaincache._active)
        velocity = velocity * vel_active
        state.mask_after = velocity
        
        return state.copied_with(sdf=sdf, velocity=velocity, active_mask=active_mask, age=state.age + dt)


    def advect(self, state, velocity, dt):
        dx = 1.0
        #state.mask_before = 1.0 * state.velocity
        #max_vel = math.max(math.abs(state.velocity.staggered))     # Extrapolate based on max velocity
        _, ext_velocity_free = extrapolate(velocity, state.active_mask, dx=dx, distance=state._distance)
        ext_velocity = state.domaincache.with_hard_boundary_conditions(ext_velocity_free)

        state.mask_after = ext_velocity_free


        # When advecting SDF we don't want to replicate boundary values when the sample coordinates are out of bounds, we want the fluid to move further away from the boundary. We increase the distance when sampling outside of the boundary.
        rank = state.rank
        padded_sdf = math.pad(state.sdf, [[0,0]] + [[1,1]]*rank + [[0,0]], "symmetric")
        padded_ext_v = StaggeredGrid(math.pad(ext_velocity.staggered, [[0,0]] + [[1,1]]*rank + [[0,0]], "symmetric"))

        zero = math.zeros_like(state.sdf)
        #padded_cells = math.pad(zero, [[0,0]] + [[1,1]]*rank + [[0,0]], "constant", constant_values=dx)
        padded_cells = 0

        updim = True
        if updim:
            # For just upper dimension
            padded = math.pad(zero, [[0,0]] + [([1,0] if i == (rank-2) else [1,1]) for i in range(rank)] + [[0,0]], "constant", constant_values=0)
            padded_cells = math.pad(padded, [[0,0]] + [([0,1] if i == (rank-2) else [0,0]) for i in range(rank)] + [[0,0]], "constant", constant_values=dx)
        else:
            for d in range(rank):
                padded = math.pad(zero, [[0,0]] + [([0,0] if d == i else [1,1]) for i in range(rank)] + [[0,0]], "constant", constant_values=0)
                padded = math.pad(padded, [[0,0]] + [([1,1] if d == i else [0,0]) for i in range(rank)] + [[0,0]], "constant", constant_values=1)
                
                padded_cells += padded

            padded_cells = dx * math.sqrt(padded_cells)

        # Increase distance outside of boundaries by dx, this will make sure that during advection we have proper wall separation
        padded_sdf += padded_cells

        padded_sdf = padded_ext_v.advect(padded_sdf, dt=dt)
        stagger_slice = tuple([slice(1,-1) for i in range(rank)])
        sdf = padded_sdf[(slice(None),) + stagger_slice + (slice(None),)]

        #sdf = ext_velocity.advect(state.sdf, dt=dt)
        # Advect the extrapolated velocity that hasn't had BC applied. This will make sure no interpolation occurs with 0 from BC.
        velocity = ext_velocity.advect(ext_velocity_free, dt=dt)

        return sdf, velocity

    def update_active_mask(self, sdf, inflows, domaincache):
        # Find the active cells from the Signed Distance Field
        
        dx = 1.0    # In case dx is used later
        ones = math.ones_like(sdf)
        active_mask = math.where(sdf < 0.5*dx, ones, 0.0 * ones)
        inflow_mask = create_binary_mask(inflow(inflows, domaincache.grid), threshold=0)
        # Logical OR between the masks
        active_mask = active_mask + inflow_mask - active_mask * inflow_mask
        # Set the new active mask in domaincache
        domaincache._active = active_mask

        return active_mask

    def staggered_active_mask(self, active_mask):
        rank = spatial_rank(active_mask)
        center = math.pad(active_mask, [[0,0]] + [[0,1]]*rank + [[0,0]], "constant")
        mask = []

        for d in range(rank):
            upper = math.pad(active_mask, [[0,0]] + [([1,0] if d == i else [0,1]) for i in range(rank)] + [[0,0]], "constant")

            mask.append(math.maximum(center, upper))
        
        return math.concat(mask, axis=-1)

    def apply_forces(self, state, dt):
            return state.velocity + dt * (state.gravity + state.trained_forces.staggered)


SDFLIQUID = SDFLiquidPhysics()


class SDFLiquid(State):
    __struct__ = State.__struct__.extend(('_sdf', '_velocity', '_active_mask', '_pressure', 'mask_before', 'mask_after', 'trained_forces'),
                            ('_domain', '_gravity'))

    def __init__(self, state_domain=Open2D,
                 density=0.0, velocity=zeros, gravity=-9.81, batch_size=None, distance=30):
        State.__init__(self, tags=('liquid', 'velocityfield'), batch_size=batch_size)
        self._domain = state_domain
        self._distance = distance
        self._density = density
        self._velocity = velocity
        self._active_mask = create_binary_mask(self._density, threshold=0)
        self._sdf, _ = extrapolate(self.velocity, self._active_mask, distance=distance)
        # Initialize correct shape
        self._last_pressure = math.zeros_like(self._density)
        self._last_pressure_iterations = None

        # Initialize the active mask for the first step. First step advection will ignore obstacles. But there shouldn't be any velocity in the obstacles anyway, so there should be no difference.
        self.domaincache = None
        self.domaincache = domain(self, ())
        self.domaincache._active = self._active_mask

        self.mask_before = math.zeros_like(self._density)
        self.mask_after = math.zeros_like(self._density)

        if isinstance(gravity, (tuple, list)):
            assert len(gravity) == state_domain.rank
            self._gravity = np.array(gravity)
        else:
            assert state_domain.rank >= 1
            gravity = [gravity] + ([0] * (state_domain.rank - 1))
            self._gravity = np.array(gravity)

        self.trained_forces = math.zeros_like(self._velocity.staggered)

    def default_physics(self):
        return SDFLIQUID

    @property
    def signed_distance(self):
        return self._sdf
    
    @property
    def sdf(self):
        return self._sdf

    @property
    def _density(self):
        return self._density_field

    @_density.setter
    def _density(self, value):
        self._density_field = initialize_field(value, self.grid.shape())

    @property
    def velocity(self):
        return self._velocity

    @property
    def _velocity(self):
        return self._velocity_field

    @_velocity.setter
    def _velocity(self, value):
        self._velocity_field = initialize_field(value, self.grid.staggered_shape())

    @property
    def active_mask(self):
        return self._active_mask

    @property
    def pressure(self):
        return self._last_pressure

    @property
    def _pressure(self):
        return self._last_pressure

    @_pressure.setter
    def _pressure(self, value):
        self._last_pressure = value

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
        return "Liquid[SDF: %s, velocity: %s, active mask: %s]" % (self._sdf, self.velocity, self.active_mask)

    def __add__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copied_with(velocity=self.velocity + other)
        else:
            return self.copied_with(sdf=math.min(self._sdf, other))

    def __sub__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copied_with(velocity=self.velocity - other)
        else:
            return self.copied_with(sdf=math.max(self._sdf, -other))


def recompute_sdf(sdf, active_mask, distance=10, dx=1.0):
    signs = -1 * (2*active_mask - 1)
    s_distance = 2.0 * (distance+1) * signs
    surface_mask = create_surface_mask(active_mask)

    # For new active cells via inflow (cells that were outside fluid in old sdf) we want to initialize their signed distance to the default
    # Previously initialized with -0.5*dx, i.e. the cell is completely full (center is 0.5*dx inside the fluid surface). For stability and looks this was changed to 0 * dx, i.e. the cell is only half full. This way small changes to the SDF won't directly change neighbouring empty cells to fluidcells.
    sdf = math.where((active_mask >= 1) & (sdf >= 0.5*dx), -0.0*dx * math.ones_like(sdf), sdf)
    # Use old Signed Distance values at the surface, then completely recompute the Signed Distance Field
    s_distance = math.where((surface_mask >= 1), sdf, s_distance)

    dims = range(spatial_rank(sdf))
    directions = np.array(list(itertools.product(
        *np.tile( (-1,0,1) , (len(dims),1) )
        )))

    for _ in range(distance):
        # Create a copy of current distance
        buffered_distance = 1.0 * s_distance
        for d in directions:
            if (d==0).all():
                continue
                
            # Shift the field in direction d, compare new distances to old ones.
            d_slice = tuple([(slice(1, None) if d[i] == -1 else slice(0,-1) if d[i] == 1 else slice(None)) for i in dims])

            d_dist = math.pad(s_distance, [[0,0]] + [([0,1] if d[i] == -1 else [1,0] if d[i] == 1 else [0,0]) for i in dims] + [[0,0]], "symmetric")
            d_dist = d_dist[(slice(None),) + d_slice + (slice(None),)]
            d_dist += dx * np.sqrt(d.dot(d)) * signs

            # Prevent updating the distance at the surface
            updates = (math.abs(d_dist) < math.abs(buffered_distance)) & (surface_mask <= 0)
            buffered_distance = math.where(updates, d_dist, buffered_distance)

        s_distance = buffered_distance

    distance_limit = -distance * (2*active_mask - 1)
    s_distance = math.where(math.abs(s_distance) < distance, s_distance, distance_limit)

    return s_distance