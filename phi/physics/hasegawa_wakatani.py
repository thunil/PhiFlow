"""
Definition of plasma, HasegawaWakatani Model, as well as plasma-related functions.
"""
import numpy as np
from numba import jit, f4, f8
from phi import math
from phi.physics.physics import Physics
from phi.physics.field import CenteredGrid
from phi.physics.pressuresolver.solver_api import FluidDomain
from phi.physics.pressuresolver.sparse import SparseCG
from phi.physics.pressuresolver.fourier import FourierSolver
import time
import logging


class HasegawaWakatani2D(Physics):
    r"""
    Physics modelling the Hasegawa-Wakatani equations with diffusion factor
    Used for: Studying Resistive drift- wave turbulence in slab geometry for cold ions with isothermal electrons
    - Equilibrium constant magnetic field, B, in z-direction (batch dimension)
    - Equilibrium initial density, n_0, has gradient (dn_0/dx) < 0 (negative n direction)
    - Equilibrium density scale, L_n = n_0 / |dn_0/dx|
    - Neglect: finite Larmor radius effects, temperature gradients, temperature fluctuations
    - Using dimensionless variables: x=x/rho_s, y=y/rho_s, t=tc_s/L_n

    Solves for phi using:  poisson_solver
    Solves Poisson Bracket using:  Arakawa Scheme
    Diffuses small scales using:  N times laplace on field to diffuse

    Hasegawa-Wakatani Equations:
        \partial_t \Omega = \frac{1}{\nu} \nabla_{||}^2(n-\phi) - [\phi,\Omega]                          + nu_\bot \nabla^{2N} \Omega \\
        \partial_t n      = \frac{1}{\nu} \nabla^2_{||}(n-\phi) - [\phi,n]      - \kappa_n\partial_y\phi + nu_\bot \nabla^{2N} n
    """

    def __init__(self, poisson_solver=SparseCG(), dim=2, N=3, c1=1, nu=10**-6, K0=0.15, arakawa_coeff=1, kappa_coeff=1):
        """
        :param poisson_solver: Pressure solver to use for solving for phi
        :param dim: Dimension of HW model
        :type dim: int
        :param N: Apply laplace N times on field for diffusion (nabla^(2*N))
        :type N: int
        :param c1: Adiabatic parameter (0.1: hydrodynamic - 5: adiabatic). T/(n_0 e^2 eta_||) * (k_||^2)/(c_s/L_n)
        :type c1: float
        :param nu: dissipation parameter (10^-10 - 10^-4)
        :type nu: float
        :param K0:
        :type K0: float
        :param arakawa_coeff: Coefficient (Multiplier) for poisson bracket
        :type arakawa_coeff: int or float
        :param kappa_coeff: Coefficient (Multiplier) for kappa term
        :type kappa_coeff: int of float
        :param initial_density: Initial density to use for kappa term. Shape: (1, N, N, 1)
        :type initial_density: array
        """
        Physics.__init__(self, [
            # No effects currently supported for the fields
        ])
        self.poisson_solver = poisson_solver #SparseCG() # FourierSolver()#
        #self.laplace = lambda x: x.laplace(axes=[1, 2])
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
        self.step = self.rk4_step
        print(self)

    def __repr__(self):
        formatted_str = "\n".join([
            "Hasegawa-Wakatani {}D".format(self.dim),
            "- Poisson Solver: {}".format(self.poisson_solver),
            "- Dissipation Order: laplace^{}".format(self.N),
            "- Adiabatic Parameter (c1): {}".format(self.c1),
            "- Dissipation Parameter (nu): {}".format(self.nu),
            "- (K0): {}".format(self.K0),
            "- (L): {}".format(self.L),
            "- Arakawa Coefficient: {}".format(self.arakawa_coeff),
            "- Kappa Coefficient: {}".format(self.kappa_coeff),
            "----------------------------------------------------------\n"
        ])
        return formatted_str

    def rk4_step(self, plasma, dt=0.1):
        """Runge-Kutta 4 function
        y_{n+1} = y_n  + (k1 + 2*k2 + 2*k3 + k4)/6
        k1 = dt * f(y_n,        t)
        k2 = dt * f(y_n + k1/2, t + dt/2)
        k3 = dt * f(y_n + k2/2, t + dt/2)
        k4 = dt * f(y_n + k3,   t + dt)
        """
        # RK4
        t0 = time.time()
        if plasma.energy == 0:
            plasma = plasma.copied_with(energy=get_total_energy(plasma.density, self.get_phi(plasma)),
                                        enstrophy=get_generalized_enstrophy(plasma.density, plasma.omega))
        yn = plasma
        pn = self.get_phi(yn)  # TODO: only execute for t=0
        k1 = dt*self.gradient_2d(yn, pn, dt=0)
        p1 = self.get_phi(yn + k1*0.5)
        k2 = dt*self.gradient_2d(yn + k1*0.5, p1, dt=dt/2)
        p2 = self.get_phi(yn + k2*0.5)
        k3 = dt*self.gradient_2d(yn + k2*0.5, p2, dt=dt/2)
        p3 = self.get_phi(yn + k3)
        k4 = dt*self.gradient_2d(yn + k3, p3, dt=dt)
        #p4 = self.get_phi(k4)
        y1 = yn + (k1 + 2*k2 + 2*k3 + k4)*(1/6)  # TODO: currently adds two timesteps
        phi = self.get_phi(y1)
        t1 = time.time()
        # Predicted Energy
        pred_E = y1.energy
        pred_U = y1.enstrophy
        # Actual Energy
        E = get_total_energy(y1.density, phi)
        U = get_generalized_enstrophy(y1.density, y1.omega)
        # Percentage Deviation
        perc_E = 1 - pred_E/E
        perc_U = 1 - pred_U/U
        print(" | ".join([
            f"{plasma.age + dt:<7.04g}",
            f"{np.max(np.abs(yn.density.data)):>7.02g}",
            f"{np.max(np.abs(k1.density.data)):>7.02g}",
            f"{np.max(np.abs(k2.density.data)):>7.02g}",
            f"{np.max(np.abs(k3.density.data)):>7.02g}",
            f"{np.max(np.abs(k4.density.data)):>7.02g}",
            f"{t1-t0:>6.02f}s",
            f"{perc_E:>8.02g}",
            f"{perc_U:>8.02g}"
        ]))
        return y1.copied_with(energy=E, enstrophy=U, phi=phi)

    def euler(self, plasma, dt=0.1):
        # Recast to 3D: return Z from Batch-axis
        plasma_grad = self.gradient_2d(plasma)
        p = plasma_grad.phi  # NOT A GRADIENT
        o = plasma.omega + dt * plasma_grad.omega
        n = plasma.density + dt * plasma_grad.density
        return plasma.copied_with(density=n, omega=o, phi=p, age=plasma.age+dt, grad_density=plasma_grad.density)

    def gradient_invariants(self, density, omega, phi):
        """returns dE, dU for timestepping
        Equations:
        dE/dt = G_n + G_c - D^E
        dZ/dt = G_n - D^U

        discretized using np.sum to get scalar of 2D
        
        :param plasma: [description]
        :type plasma: [type]
        """
        # Define parameters
        n = density.data[0, ..., 0]
        p = phi.data[0, ..., 0]
        o = omega.data[0, ..., 0]
        #omega = omega - math.mean(omega)
        # Gamma_n = - \int{d^2 x \tilde{n} \frac{\partial \tilde{\phi}}{\partial y}}
        dy_p, dx_p = phi.gradient(difference='central', padding='circular').unstack()
        gamma_n = -math.sum(n * dy_p[0, 0, ...])
        # Gamma_c = c_1 \int{d^2 x /\tilde{n} - \tilde{\phi})^2}
        gamma_c = self.c1 * math.sum((n - p)**2)
        DE = math.sum(n*diffuse(density, self.N)-p*diffuse(omega, self.N))
        DU = -math.sum((n-o)*(diffuse(density, self.N)-diffuse(omega, self.N)))
        # dE/dt = G_n - G_c - DE
        dE = gamma_n - gamma_c - DE
        # dU/dt = G_n - DU
        dU = gamma_n - DU
        return dE, dU

    def get_phi(self, plasma, guess=None):
        # Calculate Phi from Omega
        o_mean = math.mean(plasma.omega.data[0, ..., 0])
        phi, _ = solve_poisson(plasma.omega - o_mean,
                               FluidDomain(plasma.domain),
                               poisson_solver=self.poisson_solver,
                               guess=guess)
        # Readjust to mean
        phi += o_mean
        return phi

    def gradient_2d(self, plasma, phi, dt=0):
        """
        2D Hasegawa-Wakatani Equations:
        time-derivative   = parallel_mix - poiss_bracket - spatial_derivative     + damping/diffusion
        ------------------|--------------|---------------|------------------------|----------------------
        \partial_t \Omega = c_1 (\phi-n) - [\phi,\Omega]                          + nu \nabla^{2N} \Omega \\
        \partial_t n      = c_1 (\phi-n) - [\phi,n]      - \kappa_n\partial_y\phi + nu \nabla^{2N} n
        """
        # Compute in numpy arrays through .data
        #dy_o, dx_o = o2d.gradient(difference='central', padding='circular').unstack()
        dy_p, dx_p = phi.gradient(difference='central', padding='circular').unstack()
        dy_n, dx_n = plasma.density.gradient(difference='central', padding='circular').unstack()
        #dx_o = dx_o[0, ..., 0]; dy_o = dy_o[0, ..., 0]
        dx_p, dy_p = dx_p[0, ..., 0], dy_p[0, ..., 0]
        dx_n, dy_n = dx_n[0, ..., 0], dy_n[0, ..., 0]

        # Calculate Gradients
        diff = (phi - plasma.density).data[0, ..., 0]
        # Step 2.1: New Omega.
        o = (self.c1 * diff
             - self.arakawa_coeff * periodic_arakawa(phi.data[0, ..., 0], plasma.omega.data[0, ..., 0])
             + self.nu * diffuse(plasma.omega, self.N))

        # Step 2.2: New Density.
        n = (self.c1 * diff
             - self.arakawa_coeff * periodic_arakawa(phi.data[0, ..., 0], plasma.density.data[0, ..., 0])
             - self.kappa_coeff * dy_p
             + self.nu * diffuse(plasma.density, self.N))

        #print("{:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | {:>7.2g} | ".format(
            # Fields
        #    np.max(o), np.max(n), np.max(phi.data[0, ..., 0]),
        #    np.max(self.c1 * (plasma.density - phi).data[0, ..., 0]),
        #    np.max(self.arakawa_coeff * periodic_arakawa(phi.data[0, ..., 0], plasma.omega.data[0, ..., 0])),
        #    np.max(self.nu * diffuse(plasma.omega.data[0, ..., 0], self.N)),
        #    np.max(self.arakawa_coeff * periodic_arakawa(phi.data[0, ..., 0], plasma.density.data[0, ..., 0])),
        #    np.max(self.kappa_coeff * kappa * dy_p),
        #    np.max(self.nu * diffuse(plasma.density, self.N))
        #))

        energy, enstrophy = self.gradient_invariants(plasma.density, plasma.omega, phi)

        return plasma.copied_with(
            density=plasma.density.copied_with(data=math.reshape(n, plasma.density.data.shape)),
            omega=plasma.omega.copied_with(data=math.reshape(o, plasma.omega.data.shape)),
            phi=phi,  # NOT A GRADIENT
            energy=energy,
            enstrophy=enstrophy,
            age=plasma.age+dt
        )


def solve_poisson(omega2d, fluiddomain, poisson_solver=SparseCG(accuracy=1e-5), guess=None):
    """
    Computes the pressure from the given Omega field with z in batch-axis using the specified solver.

    :param omega2d: CenteredGrid
    :param fluiddomain: FluidDomain instance
    :type fluiddomain: FluidDomain
    :param poisson_solver: PressureSolver to use, None for default
    :return: scalar tensor or CenteredGrid, depending on the type of divergence
    """
    assert isinstance(omega2d, CenteredGrid)
    phi2d, iteration = poisson_solver.solve(field=omega2d.data, domain=fluiddomain, guess=guess)
    if isinstance(omega2d, CenteredGrid):
        phi2d = CenteredGrid(phi2d, omega2d.box, name='phi')
    return phi2d, iteration


def get_total_energy(n, phi):
    """
    Calculate total energy of HW plasma
    $E = \frac{1}{2} \int  d^2x (\tilde{n}^2  + |\nabla_\bot \tilde{\phi}|^2)$
    """
    nabla_phi = phi.gradient(difference='central', padding='circular')
    return 0.5 * math.sum(n**2 + math.abs(nabla_phi)**2).data[0, ..., 0]


def get_generalized_enstrophy(n, omega):
    """
    Calculate generalized enstrophy of HW plasma
    $U = \frac{1}{2} \int d^2 x (\tilde{n} - \nabla^2_\bot \tilde{\phi})^2 \equiv \frac{1}{2} \int d^2 x (\tilde{n} - \tilde{\Omega})^2$
    """
    return 0.5 * math.sum(((n - omega)*(n - omega))).data[0, ..., 0]


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
    ret_field = field
    if N == 0:
    	ret_field = CenteredGrid(np.zeros(ret_field.data.shape))
    else:
        # Apply laplace N times in perpendicular ([y, x])
        ret_field = field
        for _ in range(N):
            ret_field = ret_field.laplace(axes=[1, 2])
            #ret_field = math.fourier_laplace(ret_field)
    return ret_field.data[0, ..., 0]


@jit(#f4[:,:](f8[:,:], f8[:,:], f4),
     cache=True, nopython=True)#, nogil=True, parallel=True)
def arakawa_vec(zeta, psi, d):
    """2D periodic first-order Arakawa
    requires 1 cell padded input on each border"""
    return (zeta[2:, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[2:, 2:] - psi[2:, 0:-2])
            - zeta[0:-2, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, 0:-2] + psi[0:-2, 2:] - psi[0:-2, 0:-2])
            - zeta[1:-1, 2:] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 2:] - psi[0:-2, 2:])
            + zeta[1:-1, 0:-2] * (psi[2:, 1:-1] - psi[0:-2, 1:-1] + psi[2:, 0:-2] - psi[0:-2, 0:-2])
            + zeta[2:, 0:-2] * (psi[2:, 1:-1] - psi[1:-1, 0:-2])
            + zeta[2:, 2:] * (psi[1:-1, 2:] - psi[2:, 1:-1])
            - zeta[0:-2, 2:] * (psi[1:-1, 2:] - psi[0:-2, 1:-1])
            - zeta[0:-2, 0:-2] * (psi[0:-2, 1:-1] - psi[1:-1, 0:-2])) / (4 * d**2)


def periodic_arakawa(zeta, psi, d=1.):
    """2D periodic padding and apply Arakawa stencil to padded matrix"""
    z = periodic_padding(zeta)
    p = periodic_padding(psi)
    #return arakawa_stencil(z, p, d=d)[1:-1, 1:-1]
    return arakawa_vec(z, p, d=d)


def periodic_padding(A):
    return math.pad(A, 1, mode='circular')
