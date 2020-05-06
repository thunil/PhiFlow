import sys
from phi.flow import *  # Use NumPy
from phi.physics.hasegawa_wakatani import *  # Plasma Physics
from phi.physics.plasma_field import PlasmaHW  # Plasma Field
MODE = "NumPy"

import sys
import os
sys.path.append(os.getcwd())
import pandas as pd

DESCRIPTION = """
Hasegawa-Wakatani Plasma
"""
c1_dict = {"hydrodynamic": 0.1,
           "transition": 1,
           "adiabatic": 5}
grid_dict = {"test": (16, 16),
             "coarse": (128, 128),
             "fine": (1024, 1024)}
K0_dict = {"small": 0.15,  # focus on high-k
           "large": 0.0375}  # focus on low-k
nu_dict = {"coarse-large": 5*10**-10,
           "fine-small": 10**-4}


step_size = 10**-3
initial_state = {
    "grid": [256, 256],#grid_dict['coarse'],      # Grid size in points (resolution)
    "K0":   0.15,#K0_dict['large'],         # Box size defining parameter
    "N":    3,                        # N*2 order of dissipation
    "nu":   10**-8,#nu_dict['coarse-large'],  # Dissipation scaling coefficient
    "c1":   1,#c1_dict['transition'],     # Adiabatic parameter
    "kappa_coeff":   1,#0**3,
    "arakawa_coeff": 1,
}


def get_box_size(k0):
    return 2*np.pi/k0


N = initial_state['grid'][1]
shape = (1, *initial_state['grid'], 1)
del initial_state['grid']


domain = Domain([N, N],
                box=AABox(0, [get_box_size(initial_state['K0'])]*len([N, N])),  # NOTE: Assuming square
                boundaries=(PERIODIC, PERIODIC)  # Each dim: OPEN / CLOSED / PERIODIC
                )
fft_random = CenteredGrid.sample(Noise(), domain)
integral = np.sum(fft_random.data**2)
fft_random /= np.sqrt(integral)
fft_random *= 100


scalars = {"energy": [],
           "enstrophy": []}

plasma_hw = PlasmaHW(domain,
                     #kappa=10,
                     density=fft_random,
                     phi=-0.5*fft_random,
                     omega=-0.5*fft_random,
                     initial_density=fft_random
                     )
class PlasmaSim(App):

    def __init__(self, initial_state, resolution):
        App.__init__(self, 'Plasma Sim', DESCRIPTION, summary='plasma' + 'x'.join([str(d) for d in resolution]), framerate=20)
        self.plasma = world.add(
            plasma_hw,
            physics=HasegawaWakatani2D(**initial_state)#, poisson_solver=FourierSolver())
        )
        # Add Fields
        self.dt = EditableFloat('dt', step_size)
        self.add_field('Density', lambda: self.plasma.density)
        self.add_field('Phi', lambda: self.plasma.phi)
        self.add_field('Omega', lambda: self.plasma.omega)
        self.add_field('Initial Density', lambda: self.plasma.initial_density)
        self.add_field('Gradient Density', lambda: self.plasma.grad_density)
        self.add_field('Potential Vorticity', lambda: self.plasma.potential_vorticity)

    def action_reset(self):
        self.steps = 0
        plasma = PlasmaHW(
            domain,
            density=fft_random,
            omega=-0.5*fft_random,
            initial_density=fft_random
        )
        self.plasma.density = plasma.density
        self.plasma.omega = plasma.omega
        self.plasma.phi = plasma.phi
        self.plasma.initial_density = plasma.initial_density

    def step(self):
        world.step(dt=self.dt)
        scalars["energy"].append(self.plasma.energy)
        scalars["enstrophy"].append(self.plasma.enstrophy)
        if self.steps % 1 == 0:
            self.scene.write(self.plasma.density, names='density', frame=self.steps)
            self.scene.write(self.plasma.omega, names='omega', frame=self.steps)
            self.scene.write(self.plasma.phi, names='phi', frame=self.steps)
            pd.DataFrame(scalars).to_csv('test.csv')

        # Not Working
        #self.scene.write_sim_frame([
        #    self.plasma.density.state,
        #    self.plasma.omega.state,
        #    self.plasma.phi.state], 
        #    fieldnames=['density', 'omega', 'phi'], frame=self.steps)
        #energy_array[i] = world.state.plasma.energy


show(PlasmaSim(initial_state, [N, N]), display=('Density', 'Phi', 'Omega', 'Initial Density', 'Gradient Density'), framerate=1, debug=True)
