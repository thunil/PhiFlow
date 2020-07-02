from phi import struct
from phi.physics.domain import DomainState
import numpy as np
from phi import math


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

    @struct.variable(default=0, dependencies=DomainState.domain)
    def density(self, density):
        """
        The marker density is stored in a CenteredGrid with dimensions matching the domain.
        It describes the number of particles per physical volume.
        """
        return self.centered_grid('density', density)

    @struct.constant(default=0, dependencies=DomainState.domain)
    def initial_density(self, initial_density):
        """
        The marker density is stored in a CenteredGrid with dimensions matching the domain.
        It describes: the number of particles per physical volume in the grid.
        """
        return self.centered_grid('initial_density', initial_density)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def phi(self, phi):
        """
        The marker phi is stored in a CenteredGrid with dimensions matching the domain.
        It describes: Fluctuations in the Potential
        """
        return self.centered_grid('phi', phi)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def omega(self, omega):
        """
        The marker omega is stored in a CenteredGrid with dimensions matching the domain.
        It describes: Fluctuations in the Vorticity. Defined as the laplace of the Potential Fluctuations.
        NOTE: Cannot set Omega to non-zero constant with Periodic boundary condition, as the solution for Phi is then undefined (Poisson Equation of non-zero constant)
        """
        return self.centered_grid('omega', omega)

    # Other properties

    @struct.variable(default=0, dependencies=DomainState.domain)
    def energy(self, energy):
        """Total energy of HW plasma"""
        return self.energy

    @struct.variable(default=0, dependencies=DomainState.domain)
    def enstrophy(self, enstrophy):
        """Generalized enstrophy"""
        return self.enstrophy

    # Constructed properties
    @property
    def potential_vorticity(self):
        return self.omega - self.phi

    # Fields for checking things are working

    @struct.variable(default=0, dependencies=DomainState.domain)
    def grad_density(self, grad_density):
        return self.centered_grid('grad_density', grad_density)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def laplace_phi(self, laplace_phi):
        return self.centered_grid('laplace_phi', laplace_phi)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def laplace_n(self, laplace_n):
        return self.centered_grid('laplace_n', laplace_n)

    def __repr__(self):
        return "plasma[density: %s, phi: %s, omega %s]" % (self.density, self.phi, self.omega)

    def __mul__(self, other):
        return self.copied_with(density=self.density*other,
                                phi=self.phi*other,
                                omega=self.omega*other,
                                energy=self.energy*other,
                                enstrophy=self.enstrophy*other)

    __rmul__ = __mul__

    def __add__(self, other):
        return self.copied_with(density=self.density+other.density,
                                phi=self.phi+other.phi,
                                omega=self.omega+other.omega,
                                energy=self.energy+other.energy,
                                enstrophy=self.enstrophy+other.enstrophy,
                                age=max(self.age, other.age))
