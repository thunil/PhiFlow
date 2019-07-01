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

        #max_vel = math.max(math.abs(state.velocity.staggered))
        _, ext_velocity = extrapolate(state.velocity, domaincache.active(), dx=1.0, distance=30)
        ext_velocity = domaincache.with_hard_boundary_conditions(ext_velocity)
        sdf = ext_velocity.advect(state.sdf, dt=dt)
        velocity = ext_velocity.advect(ext_velocity, dt=dt)

        # Find the active cells from the Signed Distance Field
        ones = math.ones_like(sdf)
        # In case dx is used later
        dx=1.0
        active_mask = math.where(sdf < 0.5*dx, ones, 0.0 * ones)
        inflow_mask = create_binary_mask(inflow(inflows, state.grid), threshold=0)
        # Logical OR between the masks
        active_mask = active_mask + inflow_mask - active_mask * inflow_mask
        domaincache._active = active_mask

        forces = dt * state.gravity
        velocity = velocity + forces

        velocity = divergence_free(velocity, domaincache, self.pressure_solver, state=state)

        sdf = recompute_sdf(sdf, active_mask, distance=30)
        
        return state.copied_with(sdf=sdf, velocity=velocity, age=state.age + dt)


SDFLIQUID = SDFLiquidPhysics()


class SDFLiquid(State):
    __struct__ = State.__struct__.extend(('_sdf', '_velocity'),
                            ('_domain', '_gravity'))

    def __init__(self, domain=Open2D,
                 density=0.0, velocity=zeros, gravity=-9.81, batch_size=None):
        State.__init__(self, tags=('liquid', 'velocityfield'), batch_size=batch_size)
        self._domain = domain
        self._density = density
        self._velocity = velocity
        particle_mask = create_binary_mask(self._density, threshold=0)
        self._sdf, _ = extrapolate(self.velocity, particle_mask, distance=30)
        self.domaincache = None
        self._last_pressure = None
        self._last_pressure_iterations = None

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
        return "Liquid[SDF: %s, velocity: %s]" % (self._sdf, self.velocity)

    def __add__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copied_with(velocity=self.velocity + other)
        else:
            return self.copied_with(sdf=math.min(self._sdf, other))

    def __sub__(self, other):
        if isinstance(other, StaggeredGrid):
            return self.copied_with(velocity=self.velocity - other)
        else:
            return self.copied_with(sdf=math.min(self._sdf, -other))


def recompute_sdf(sdf, active_mask, dx=1.0, distance=10):
    s_distance = -2.0 * (distance+1) * (2*active_mask - 1)
    signs = -1 * (2*active_mask - 1)
    surface_mask = create_surface_mask(active_mask)

    # For new active cells via inflow (cells that were outside fluid in old sdf) we want to initialize their signed distance to the default
    sdf = math.where((active_mask >= 1) & (sdf >= 0.5*dx), -0.5*dx * math.ones_like(sdf), sdf)
    # Use old Signed Distance values at the surface, then completely recompute the Signed Distance Field
    s_distance = math.where((surface_mask >= 1), sdf, s_distance)

    dims = range(spatial_rank(sdf))
    directions = np.array(list(itertools.product(
        *np.tile( (-1,0,1) , (len(dims),1) )
        )))

    for _ in range(distance):
        for d in directions:
            if (d==0).all():
                continue
                
            # Shift the field in direction d, compare new distances to old ones.
            d_slice = [(slice(1, None) if d[i] == -1 else slice(0,-1) if d[i] == 1 else slice(None)) for i in dims]

            d_dist = math.pad(s_distance, [[0,0]] + [([0,1] if d[i] == -1 else [1,0] if d[i] == 1 else [0,0]) for i in dims] + [[0,0]], "symmetric")
            d_dist = d_dist[[slice(None)] + d_slice + [slice(None)]]
            d_dist += dx * np.sqrt(d.dot(d)) * signs

            updates = math.abs(d_dist) < math.abs(s_distance)
            s_distance = math.where(updates, d_dist, s_distance)

    return s_distance