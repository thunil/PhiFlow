"""
Definition of plasma, HasegawaWakatani Model, as well as plasma-related functions.
"""
from numbers import Number

import numpy as np
from phi import math, struct

from phi.physics.domain import Domain, DomainState
from phi.physics.field import CenteredGrid, StaggeredGrid, advect, union_mask
from phi.physics.field.effect import Gravity, effect_applied, gravity_tensor
from phi.physics.material import OPEN, Material
from phi.physics.physics import Physics, StateDependency
from phi.physics.pressuresolver.solver_api import FluidDomain
from phi.physics.pressuresolver.sparse import SparseCG


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
        density2d = self.density.copied_with(
            data=math.reshape(self.density.data, [-1] + list(self.density.data.shape[2:])),
            box=domain.box.without_axis(0))
        _, density_grad_x = density2d.gradient().unstack()
        self.kappa = 1 / np.sum(density2d.data) * density_grad_x.data  # density_grad_x.data * (1/np.sum(density2d.data))

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
        return self.centered_grid('omega', omega)

    def __repr__(self):
        return "plasma[density: %s, phi: %s, omega %s]" % (self.density, self.phi, self.omega)


class HasegawaWakatani(Physics):
    r"""
    Physics modelling the Hasegawa-Wakatani equations.
    Supports buoyancy proportional to the marker density.  # TODO: Adjust
    Supports obstacles, density effects, velocity effects, global gravity.  # TODO: Adjust

    Hasegawa-Wakatani Equations:
    $$
        \partial_t \Omega = \frac{1}{\nu} \nabla_{||}^2(n-\phi) - \{\phi,\Omega\} \\
        \partial_t n = \frac{1}{\nu} \nabla^2_{||}(n-\phi) - \{\phi,n\} - \kappa_n\partial_y\phi 
    $$

    """

    def __init__(self, pressure_solver=None, conserve_density=True):
        Physics.__init__(self, [  # No effects currently supported for the fields
            #StateDependency('obstacles', 'obstacle'),
            #StateDependency('gravity', 'gravity', single_state=True),
            #StateDependency('density_effects', 'density_effect', blocking=True),
            #StateDependency('velocity_effects', 'velocity_effect', blocking=True)
        ])
        self.pressure_solver = pressure_solver  # TODO: Adjust
        self.conserve_density = conserve_density  # TODO: Adjust

    def step(self, plasma, dt=1.0):
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
        # pylint: disable-msg = arguments-differ
        domain3d = plasma.domain
        phi3d = plasma.phi
        omega3d = plasma.omega
        density3d = plasma.density
        shape3d = omega3d.data.shape

        # Cast to 2D: Move Z-axis to Batch-axis (order: z, y, x)
        domain2d = Domain(domain3d.resolution[1:], box=domain3d.box.without_axis(0))
        omega2d = omega3d.copied_with(data=math.reshape(omega3d.data, [-1] + list(omega3d.data.shape[2:])), box=domain2d.box)
        density2d = density3d.copied_with(data=math.reshape(density3d.data, [-1] + list(density3d.data.shape[2:])), box=domain2d.box)

        # Step 1: New Phy (Poisson equation). phi_0 = ∇^-2_bot Omega_0
        # mask = plasma.domain.centered_grid(1, extrapolation='replicate')  # Tell solver: no obstacles, etc.
        # Calculate Phi from Omega
        phi2d, _ = solve_pressure(omega2d, FluidDomain(domain2d))
        phi3d = phi2d.copied_with(data=math.reshape(phi2d.data, shape3d), box=domain3d.box)

        # Calculate Poisson Bracket components. Gradient on all axes (x, y)
        dy_omega, dx_omega = omega2d.gradient().unstack()
        dy_phi, dx_phi = phi2d.gradient().unstack()
        dy_density, dx_density = density2d.gradient().unstack()
        # Calculate Z grad. Laplace Operator (Gradient) on 1 Dimension
        dzdz = (density3d - phi3d).laplace(axes=[0])

        # Compute in numpy arrays through .data
        # Step 2.1: New Omega.
        # $\partial_t \Omega = \frac{1}{\nu} (\partial_{z}^2 n - \partial^2_{z}\phi)
        #                      - \partial_x\phi\partial_y\Omega + \partial_y\phi_0\partial_x\Omega$
        omega = 1 / plasma.nu * dzdz.data[..., 0] \
            - dx_phi.data * dy_omega.data + dy_phi.data * dx_omega.data
        print("omega = 1/{} * {} - {} + {}".format(
            plasma.nu,
            np.max(dzdz.data[..., 0]),
            np.max(dx_phi.data * dy_omega.data),
            np.max(dy_phi.data * dx_omega.data)
        ))
        # Step 2.2: New Density.
        # $\partial_t n = \frac{1}{\nu} (\partial^2_{z} n - \partial^2_{z}\phi)
        #                 - \partial_x\phi\partial_y n     + \partial_y\phi\partial_x n
        #                 - \frac{1}{n} \partial_x n \partial_y\phi$
        kappa = dx_density.data * (1 / np.sum(density2d.data))
        density = 1 / plasma.nu * dzdz.data[..., 0] \
            - dx_phi.data * dy_density.data + dy_phi.data * dx_density.data \
            - kappa * dy_phi.data
        print("density = 1/{} * {} - {} + {} - {}".format(
            plasma.nu,
            np.max(dzdz.data[..., 0]),
            np.max(dx_phi.data * dy_density.data),
            np.max(dy_phi.data * dx_density.data),
            np.max(kappa * dy_phi.data)
        ))

        # Recast to 3D: return Z from Batch-axis
        phi3d = phi2d.copied_with(data=math.reshape(phi2d.data, shape3d), box=domain3d.box)
        omega3d = omega3d.copied_with(data=math.reshape(omega, shape3d), box=domain3d.box)
        density3d = density3d.copied_with(data=math.reshape(density, shape3d), box=domain3d.box)

        return plasma.copied_with(density=density3d, omega=omega3d, phi=phi3d, age=(plasma.age + dt))


HASEGAWAWAKATANI = HasegawaWakatani()


def solve_pressure(omega2d, fluiddomain, pressure_solver=None):
    """
    Computes the pressure from the given Omega field with z in batch-axis using the specified solver.
    :param omega2d: CenteredGrid
    :param fluiddomain: FluidDomain instance
    :type fluiddomain: FluidDomain
    :param pressure_solver: PressureSolver to use, None for default
    :return: scalar tensor or CenteredGrid, depending on the type of divergence
    """
    assert isinstance(omega2d, CenteredGrid)
    if pressure_solver is None:
        pressure_solver = SparseCG()
    phi2d, iteration = pressure_solver.solve(omega2d.data, fluiddomain, pressure_guess=None)
    if isinstance(omega2d, CenteredGrid):
        phi2d = CenteredGrid(phi2d, omega2d.box, name='phi')
    return phi2d, iteration


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
