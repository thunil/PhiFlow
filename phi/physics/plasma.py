"""
Definition of plasma, HasegawaWakatani Model, as well as plasma-related functions.
"""
from numbers import Number

import numpy as np
from numba import jit, stencil, prange
from phi import math, struct

from phi.physics.domain import Domain, DomainState
from phi.physics.field import CenteredGrid, StaggeredGrid, advect, union_mask
from phi.physics.field.effect import Gravity, effect_applied, gravity_tensor
from phi.physics.material import OPEN, Material, PERIODIC
from phi.physics.physics import Physics, StateDependency
from phi.physics.pressuresolver.solver_api import FluidDomain
from phi.physics.pressuresolver.sparse import SparseCG

import time


@struct.definition()
class PlasmaHW(DomainState):
    """
    Following Hasegawa-Wakatani Model of incompressible Plasma
    A PlasmaHW state consists of:
    - Density field (centered grid)
    - Phi field (centered grid)
    - Omega field (centered grid)
    """

    def __init__(self, domain, tags=('plasma'), name='plasmaHW', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))
        self.nu = 1.  # nu_e/(1.96*w_ce)
        #density2d = self.density.copied_with(
        #    data=math.reshape(self.density.data, [-1] + list(self.density.data.shape[2:])),
        #    box=domain.box.without_axis(0))
        #_, density_grad_x = density2d.gradient(difference='central', padding='wrap').unstack()
        #self.kappa = 1 / np.sum(density2d.data) * density_grad_x.data  # density_grad_x.data * (1/np.sum(density2d.data))

    @struct.variable(default=0, dependencies=DomainState.domain)
    def density(self, density):
        """
        The marker density is stored in a CenteredGrid with dimensions matching the domain.
        It describes the number of particles per physical volume.
        """
        return self.centered_grid('density', density)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def phi(self, phi):
        """
        The marker phi is stored in a CenteredGrid with dimensions matching the domain.
        It describes ..... # TODO
        """
        return self.centered_grid('phi', phi)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def omega(self, omega):
        """
        The marker omega is stored in a CenteredGrid with dimensions matching the domain.
        It describes ..... # TODO
        """
        #if np.array_equal(omega, omega.item(0)) and omega.item(0) != 0 and PERIODIC in self.DomainState.domain.boundary:
        #    raise "Cannot set Omega to non zero constant with Periodic boundary condition, as the solution for Phi is then undefined"
        return self.centered_grid('omega', omega)

    # Fields for checking things are working

    @struct.variable(default=0, dependencies=DomainState.domain)
    def laplace_phi(self, laplace_phi):
        return self.centered_grid('laplace_phi', laplace_phi)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def laplace_n(self, laplace_n):
        return self.centered_grid('laplace_n', laplace_n)

    def __repr__(self):
        return "plasma[density: %s, phi: %s, omega %s]" % (self.density, self.phi, self.omega)

    def __mul__(self, other):
        return self.copied_with(density=self.density*other, phi=self.phi*other, omega=self.omega*other)

    __rmul__ = __mul__

    def __add__(self, other):
        return self.copied_with(density=self.density+other.density, phi=self.phi+other.phi, omega=self.omega+other.omega)


class HasegawaWakatani(Physics):
    r"""
    Physics modelling the Hasegawa-Wakatani equations with diffusion factor.
    Solves for phi using:  poisson_solver
    Solves Poisson Bracket using:  Arakawa Scheme
    Diffuses small scales using:  N times laplace on field to diffuse

    Hasegawa-Wakatani Equations:
    $$
        \partial_t \Omega = \frac{1}{\nu} \nabla_{||}^2(n-\phi) - [\phi,\Omega] + nu_\bot \nabla^{2N} \Omega \\
        \partial_t n = \frac{1}{\nu} \nabla^2_{||}(n-\phi) - [\phi,n] - \kappa_n\partial_y\phi + nu_\bot \nabla^{2N} n
    $$

    """

    def __init__(self, poisson_solver=None, dim=2, N=3, c1=1, nu=10**-6, K0=0.15, arakawa_coeff=1, kappa_coeff=1):
        """
        :param poisson_solver: Pressure solver to use for solving for phi
        :param dim: Dimension of HW model
        :type dim: int
        :param N: Apply laplace N times on field for diffusion (nabla^(2*N))
        :type N: int
        :param c1: Adiabatic parameter (0.1: hydrodynamic - 5: adiabatic). T/(n_0 e^2 eta_||) * (k_||^2)/(c_s/L_n)
        :type c1: float
        :param K0: 
        :type K0: float
        :param nu: dissipation parameter (10^-10 - 10^-4)
        :type nu: float
        """
        Physics.__init__(self, [  # No effects currently supported for the fields
            #StateDependency('obstacles', 'obstacle'),
            #StateDependency('gravity', 'gravity', single_state=True),
            #StateDependency('density_effects', 'density_effect', blocking=True),
            #StateDependency('velocity_effects', 'velocity_effect', blocking=True)
        ])
        self.poisson_solver = poisson_solver
        self.dim = dim
        self.N = N
        self.c1 = c1
        self.nu = nu
        self.K0 = K0
        self.arakawa_coeff = arakawa_coeff
        self.kappa_coeff = kappa_coeff
        # Derived Values
        self.L = 2*np.pi/K0
        self.energy = 0
        self.enstrophy = 0
        if self.dim == 2:
            print("Using: Step 2D.")
            self.step = self.rk4_2d
        else:
            print("Using: Step 3D.")
            self.step = self.step3d
        #print("{:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} ".format(
        #       "o", "n", "nu", "dzdz", "[p,o]", "nab2o", "[p,n]", "k*dyp", "nab2n"))
        print(self)

    def __repr__(self):
        formatted_str = "\n".join([
            "Hasegawa-Wakatani {}D".format(self.dim),
            "- Poisson Solver: {}".format(self.poisson_solver),
            "- Dissipation Order: laplace^{}".format(self.N),
            "- Adiabatic Parameter (c1): {}".format(self.c1),
            "- Dissipation Parameter (nu): {}".format(self.nu),
            "- (K0): {}".format(self.K0),
            "- (L): ".format(self.L),
            "- Arakawa Coefficient: {}".format(self.arakawa_coeff),
            "- Kappa Coefficient: {}".format(self.kappa_coeff),
            "----------------------------------------------------------\n"
        ])
        return formatted_str

    def rk4_2d(self, plasma, dt=0.1):
        """Runge-Kutta 4 function
        y_{n+1} = y_n  + (k1 + 2*k2 + 2*k3 + k4)/6
        k1 = dt * f(y_n,        t)
        k2 = dt * f(y_n + k1/2, t + dt/2)
        k3 = dt * f(y_n + k2/2, t + dt/2)
        k4 = dt * f(y_n + k3,   t + dt)
        """
        yn = plasma
        k1 = dt*self.gradient_2d(yn)
        k2 = dt*self.gradient_2d(yn + k1*0.5)
        k3 = dt*self.gradient_2d(yn + k2*0.5)
        k4 = dt*self.gradient_2d(yn + k3)
        res = yn + (k1 + 2*k2 + 2*k3 + k4)*(1/6)
        print("{:<7.2f} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g}".format(
            plasma.age + dt,
            np.max(yn.density.data),
            np.max(k1.density.data),
            np.max(k2.density.data),
            np.max(k3.density.data),
            np.max(k4.density.data)
        ))
        return res.copied_with(res, age=plasma.age+dt)

    def euler_2d(self, plasma, dt=0.1):
        # Recast to 3D: return Z from Batch-axis
        plasma_grad = self.gradient_2d(plasma)
        p = plasma.phi + dt * plasma_grad.phi
        o = plasma.omega + dt * plasma_grad.omega
        n = plasma.density + dt * plasma_grad.density
        return plasma.copied_with(density=n, omega=o, phi=p, age=plasma.age+dt)

    def calculate_properties(self, plasma):
        self.energy = get_total_energy(plasma.density, plasma.phi)
        self.enstrophy = get_generalized_enstrophy(plasma.density, plasma.phi)
        print("{:>8.2g}  {:>8.2g}".format(np.max(self.energy.data), np.max(self.enstrophy.data)))
        return

    def gradient_2d(self, plasma):
        """
        2D Hasegawa-Wakatani Equations:
        time-derivative   = parallel_mix - poiss_bracket - spatial_derivative     + damping/diffusion
        ------------------|--------------|---------------|------------------------|----------------------
        $$
        \partial_t \Omega = c_1 (n-\phi) - [\phi,\Omega]                          + nu \nabla^{2N} \Omega \\
        \partial_t n      = c_1 (n-\phi) - [\phi,n]      - \kappa_n\partial_y\phi + nu \nabla^{2N} n
        $$
        """
        # pylint: disable-msg = arguments-differ
        # Step 1: New Phy (Poisson equation). phi_0 = nabla^-2_bot Omega_0
        # Calculate: Omega -> Phi
        # Pressure Solvers require exact mean of zero. Set mean to zero:
        o2d = plasma.omega - math.mean(plasma.omega.data[0, ..., 0])  # NOTE: Only in 2D
        p, _ = solve_poisson(o2d, FluidDomain(plasma.domain))

        # Compute in numpy arrays through .data
        #dy_o, dx_o = o2d.gradient(difference='central', padding='wrap').unstack()
        dy_p, dx_p = p.gradient(difference='central', padding='wrap').unstack()
        dy_n, dx_n = plasma.density.gradient(difference='central', padding='wrap').unstack()
        #dx_o = dx_o[0, 0, ...]; dy_o = dy_o[0, 0, ...]
        dx_p, dy_p = dx_p[0, 0, ...], dy_p[0, 0, ...]
        dx_n, dy_n = dx_n[0, 0, ...], dy_n[0, 0, ...]

        # Step 2.1: New Omega.
        o = (self.c1 * (plasma.density - p).data[0, ..., 0]
             - self.arakawa_coeff * periodic_arakawa(p.data[0, ..., 0], plasma.omega.data[0, ..., 0])
             + self.nu * diffuse(plasma.omega, self.N))
        # Step 2.2: New Density.
        kappa = dx_n/plasma.density.data[0, ..., 0]
        n = (self.c1 * (plasma.density - p).data[0, ..., 0]
             - self.arakawa_coeff * periodic_arakawa(p.data[0, ..., 0], plasma.density.data[0, ..., 0])
             - self.kappa_coeff * kappa * dy_p
             + self.nu * diffuse(plasma.density, self.N))

        # Debug print
        # print("{:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | ".format(
        #     np.max(o), np.max(n), np.max(p.data[0, ..., 0]),
        #     np.max(self.c1 * (plasma.density - p).data[0, ..., 0]),
        #     np.max(self.arakawa_coeff * periodic_arakawa(p.data[0, ..., 0], o2d.data[0, ..., 0])),
        #     np.max(self.nu * diffuse(o2d, self.N)),
        #     np.max(self.arakawa_coeff * periodic_arakawa(p.data[0, ..., 0], plasma.density.data[0, ..., 0])),
        #     np.max(self.kappa_coeff * kappa * dy_p),
        #     np.max(self.nu * diffuse(plasma.density, self.N))
        # ))

        return plasma.copied_with(
            density=plasma.density.copied_with(data=math.reshape(n, plasma.density.data.shape), box=plasma.domain.box),
            omega=plasma.omega.copied_with(data=math.reshape(o, plasma.omega.data.shape), box=plasma.domain.box),
            phi=p
        )

    def step3d(self, plasma, dt=0.1):
        """
        Computes the next state of a physical system, given the current state.
        Solves the simulation for a time increment self.dt.

        :param plasma: Initial state of the plasma for the Hasegawa-Wakatani Model
        :type plasma: PlasmaHW
        :param dt: time increment, float (can be positive, negative or zero)
        :type dt: float
        :returns: dict from String to List<State>
        :rtype: PlasmaHW
        """
        t = time.time()
        # pylint: disable-msg = arguments-differ
        domain3d = plasma.domain
        p3d = plasma.phi
        o3d = plasma.omega
        n3d = plasma.density
        shape3d = o3d.data.shape
        # Cast to 2D: Move Z-axis to Batch-axis (order: z, y, x)
        domain2d = Domain(domain3d.resolution[1:], box=domain3d.box.without_axis(0))
        o2d = o3d.copied_with(data=math.reshape(o3d.data, [-1] + list(o3d.data.shape[2:])), box=domain2d.box)
        n2d = n3d.copied_with(data=math.reshape(n3d.data, [-1] + list(n3d.data.shape[2:])), box=domain2d.box)
        # Step 1: New Phy (Poisson equation). phi_0 = nabla^-2_bot Omega_0
        # Calculate: Omega -> Phi
        # Pressure Solvers require exact mean of zero. Set mean to zero:
        #o2d -= o2d.mean()  # NOTE: Only in 2D
        p2d, _ = solve_poisson(o2d, FluidDomain(domain2d))
        p3d = p2d.copied_with(data=math.reshape(p2d.data, shape3d), box=domain3d.box)
        # Calculate: grad_z. Laplace Operator (Gradient) on 1 Dimension
        dzdz = (n3d - p3d).laplace(axes=[0]).data[..., 0]

        # Testing laplace
        t_0 = time.time()
        laplace_phi_3d = plasma.laplace_phi
        laplace_phi = math.nd.laplace(p3d.data[..., 0], axes=[0], padding='wrap')
        laplace_phi = laplace_phi_3d.copied_with(data=math.reshape(laplace_phi, shape3d), box=domain3d.box)
        laplace_n_3d = plasma.laplace_n
        laplace_n = math.nd.laplace(n3d.data[..., 0], axes=[0], padding='wrap')
        laplace_n = laplace_n_3d.copied_with(data=math.reshape(laplace_n, shape3d), box=domain3d.box)
        print("Laplace Time:  {}".format(time.time()-t_0))

        # Compute in numpy arrays through .data
        t_0 = time.time()
        dy_o, dx_o = o2d.gradient(difference='central', padding='wrap').unstack()
        dy_p, dx_p = p2d.gradient(difference='central', padding='wrap').unstack()
        dy_n, dx_n = n2d.gradient(difference='central', padding='wrap').unstack()
        print("Gradient Time: {}".format(time.time()-t_0))

        # Diffusion Components
        t_0 = time.time()
        dif_o = diffuse(o3d, self.N)
        dif_n = diffuse(n3d, self.N)
        print("Diffuse Time:  {}".format(time.time()-t_0))

        # Step 2.1: New Omega.
        t_0 = time.time()
        o = (1 / plasma.nu * dzdz 
            - periodic_arakawa_3d(p2d.data[..., 0], o2d.data[..., 0])
            + dif_o)
        print("Omega Time:    {}".format(time.time()-t_0))
        # Step 2.2: New Density.
        t_0 = time.time()
        kappa = dx_n.data * (1 / np.sum(n2d.data))
        n = (1 / plasma.nu * dzdz 
            - periodic_arakawa_3d(p2d.data[..., 0], n2d.data[..., 0])
            - kappa * dy_p.data
            + dif_n)
        print("Density Time:  {}".format(time.time()-t_0))

        # Recast to 3D: return Z from Batch-axis
        p3d = euler(p3d, p3d.copied_with(data=math.reshape(p2d.data, shape3d), box=domain3d.box), dt)
        o3d = euler(o3d, o3d.copied_with(data=math.reshape(o, shape3d), box=domain3d.box), dt)
        n3d = euler(n3d, n3d.copied_with(data=math.reshape(n, shape3d), box=domain3d.box), dt)
        print("Total Time:    {}".format(time.time()-t))
        # p3d = advect.semi_lagrangian(p3d, p2d.copied_with(data=math.reshape(p2d.data, shape3d), box=domain3d.box), dt=dt)
        # o3d = advect.semi_lagrangian(o3d, o3d.copied_with(data=math.reshape(o2d.data, shape3d), box=domain3d.box), dt=dt)
        # n3d = advect.semi_lagrangian(n3d, n3d.copied_with(data=math.reshape(n2d.data, shape3d), box=domain3d.box), dt=dt)
        # n3d = n3d.normalized(n3d)
        # print("density = {}".format(density))

        # Debug print
        print("{:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | ".format(
            np.max(o), np.max(n), plasma.nu,
            np.max(dzdz),
            np.max(periodic_arakawa_3d(p2d.data[..., 0], o2d.data[..., 0])),
            np.max(dif_o),
            np.max(periodic_arakawa_3d(p2d.data[..., 0], n2d.data[..., 0])),
            np.max(kappa * dy_p.data),
            np.max(dif_n)
        ))

        return plasma.copied_with(density=n3d, omega=o3d, phi=p3d,
                                  laplace_phi=laplace_phi, laplace_n=laplace_n,
                                  age=(plasma.age + dt))


HASEGAWAWAKATANI = HasegawaWakatani()


def get_total_energy(n, phi):
    """
    Calculate total energy of HW plasma
    $E = \frac{1}{2} \int  d^2x (\tilde{n}^2  + |\nabla_\bot \tilde{\phi}|^2)$
    """
    return 0.5 * math.sum(n**2 + math.abs(phi.gradient(difference='central', padding='wrap'))**2)


def get_generalized_enstrophy(n, phi):
    """
    Calculate generalized enstrophy of HW plasma
    $U = \frac{1}{2} \int d^2 x (\tilde{n} - \nabla^2_\bot \tilde{\phi})^2 \equiv \frac{1}{2} \int d^2 x (\tilde{n} - \tilde{\Omega})^2$
    """
    return 0.5 * math.sum((n - phi.gradient(difference='central', padding='wrap'))**2)


def diffuse(field, N=1):
    """
    returns nu*nabla_\bot^(2*N)*field :: allows diffusion of accumulation on small scales
    :param field: field to be diffused
    :type field: Field or Array or Tensor
    :param N: order of diffusion (nabla^(2*N))
    :type N: int
    :param nu: perpendicular nu
    :type nu: Field/Array/Tensor or int/float
    :returns: nabla^{2*N}(field)
    """
    # Second Order for Diffusion
    #nabla2_o = math.nd.laplace(o3d.data[0, ..., 0], axes=[1, 2], padding='wrap')
    #nabla2_n = math.nd.laplace(n3d.data[0, ..., 0], axes=[1, 2], padding='wrap')
    #return math.nd.laplace(field.data[0, ..., 0], axes=[1, 2], padding='wrap')
    if N == 0:
    	ret_field = 0
    else:
        # Apply laplace N times in perpendicular ([y, x])
        ret_field = field#.data[0, ..., 0]#
        for _ in range(N):
            ret_field = ret_field.laplace(axes=[1, 2])  # DOES NOT WORK
            #if field.rank == 3:
            #    ret_field = math.nd.laplace(ret_field, axes=axes, padding='wrap')
            #else:
            #    ret_field = math.nd.laplace(ret_field)
    return ret_field.data[0, ..., 0]


def solve_poisson(omega2d, fluiddomain, poisson_solver=SparseCG()):
    """
    Computes the pressure from the given Omega field with z in batch-axis using the specified solver.
    :param omega2d: CenteredGrid
    :param fluiddomain: FluidDomain instance
    :type fluiddomain: FluidDomain
    :param poisson_solver: PressureSolver to use, None for default
    :return: scalar tensor or CenteredGrid, depending on the type of divergence
    """
    assert isinstance(omega2d, CenteredGrid)
    phi2d, iteration = poisson_solver.solve(omega2d.data, fluiddomain, guess=None)
    if isinstance(omega2d, CenteredGrid):
        phi2d = CenteredGrid(phi2d, omega2d.box, name='phi')
    return phi2d, iteration


def euler(x, dx, dt):
    return x + dx * dt


def leapfrog():
    return


def rk4():
    return


@stencil
def arakawa_stencil(zeta, psi):
    return (zeta[1, 0] * (psi[0, 1] - psi[0, -1] + psi[1, 1] - psi[1, -1])
            - zeta[-1, 0] * (psi[0, 1] - psi[0, -1] + psi[-1, 1] - psi[-1, -1])
            - zeta[0, 1] * (psi[1, 0] - psi[-1, 0] + psi[1, 1] - psi[-1, 1])
            + zeta[0, -1] * (psi[1, 0] - psi[-1, 0] + psi[1, -1] - psi[-1, -1])
            + zeta[1, -1] * (psi[1, 0] - psi[0, -1])
            + zeta[1, 1] * (psi[0, 1] - psi[1, 0])
            - zeta[-1, 1] * (psi[0, 1] - psi[-1, 0])
            - zeta[-1, -1] * (psi[-1, 0] - psi[0, -1]))


@jit
def arakawa_vec(zeta, psi, d):
    return (zeta[2:, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[2:, 2:] - psi[2:, 0:-2])
            - zeta[0:-2, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[0:-2, 2:] - psi[0:-2, 0:-2])
            - zeta[1:-1, 2:] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 2:] - psi[0:-2, 2:])
            + zeta[1:-1, 0:-2] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 0:-2] - psi[0:-2, 0:-2])
            + zeta[2:, 0:-2] * (psi[2:, 1:-1] - psi[1:-1, 0:-2])
            + zeta[2:, 2:] * (psi[1:-1, 2:] - psi[2:, 1:-1])
            - zeta[0:-2, 2:] * (psi[1:-1, 2:] - psi[0:-2, 1:-1])
            - zeta[0:-2, 0:-2] * (psi[0:-2, 1:-1] - psi[1:-1, 0:-2])) / (4 * d**2)


def arakawa(z, p, d=1.):
    return arakawa_stencil(z, p) / (12 * (d**2))


def periodic_arakawa(zeta, psi, d=1.):
    ''' 2D periodic padding and apply arakawa stencil to padded matrix '''
    z = periodic_padding(zeta)
    p = periodic_padding(psi)
    #return arakawa_stencil(z, p, d=d)[1:-1, 1:-1]
    return arakawa_vec(z, p, d=d)


@jit
def arakawa_3d(z, p, d=1.):
    res = z.copy()
    for i in prange(z.shape[0]):
        res[i] = arakawa_stencil(z[i], p[i])
    return res / (12 * (d**2))


def periodic_arakawa_3d(zeta, psi, d=1.):
    ''' periodic padding and apply arakawa stencil to padded matrix '''
    z = periodic_padding(zeta)
    p = periodic_padding(psi)
    ret = arakawa_3d(z[1:-1, ...], p[1:-1, ...])[:, 1:-1, 1:-1]
    return ret


def periodic_padding(A):
    return np.pad(A, 1, mode='wrap')


def get_sigma(Te0, me, ve):
    """
    $\bar{sigma} = T_{e0}/(m_e \nu_e)$
    """
    return Te0 / (me * ve)


def get_mu(me, mi, Ti, Te, sigma):
    """
    $\mu = (m_e/m_i)^{1/2} (T_i/T_e)^{5/2} \bar{\sigma}
    """
    return np.sqrt(me / mi) * np.sqrt((Ti / Te)**(5)) * sigma


def get_speed_of_sound(T, M):
    """sound speed c_s^2 = T/M"""
    return np.sqrt(T/M)


def get_driftwave_dispersion_scale(c, M, T, e, B):
    return np.sqrt(c**2 * M * T / (e**2 * B**2))
