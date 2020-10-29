import scipy.ndimage
import os
import sys
from phi.flow import *  # Use NumPy
from phi.physics.plasma import *  # Plasma tools
MODE = "NumPy"

sys.path.append(os.getcwd())

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
nu_dict = {"coarse-large": 5 * 10**-10,
           "fine-small": 10**-4}


step_size = 10**-5
initial_state = {
    "grid": grid_dict['coarse'],      # Grid size in points (resolution)
    "K0": K0_dict['large'],         # Box size defining parameter
    "N": 1,                        # N*2 order of dissipation
    "nu": 0,  # nu_dict['coarse-large'],  # Dissipation scaling coefficient
    "c1": 50,  # c1_dict['adiabatic'],     # Adiabatic parameter
    "kappa_coeff": 0,
    "arakawa_coeff": 0,
}


def get_box_size(k0):
    return 2 * np.pi / k0


N = initial_state['grid'][1]
shape = (1, *initial_state['grid'], 1)
del initial_state['grid']

box_field = np.zeros(shape)
box_field[0, int(N / 4):int(3 * N / 4), int(N / 4):int(3 * N / 4), 0] = np.ones((int(N / 2), int(N / 2)))
box_field += 1

line_field = np.zeros(shape)
line_field[0, (int(N / 2) - 1):(int(N / 2) + 1), :, 0] = np.ones((2, N))
line_field += 1

cross_field = np.zeros(shape)
cross_field[0, (int(N / 2) - 1):(int(N / 2) + 1), :, 0] = np.ones((2, N))
cross_field[0, :, (int(N / 2) - 1):(int(N / 2) + 1), 0] = np.ones((N, 2))
cross_field += 1

slope_field = np.array([np.arange(N) for _ in range(N)]).reshape(shape) / 10 + 1

step_field = np.ones(shape)
step_field[0, :, int(N / 2):, 0] = 2
step_field = step_field[::-1].T

x_field = scipy.ndimage.rotate(cross_field[0, :, :, 0], angle=45, reshape=False).reshape(shape) + 1

random_field = np.random.uniform(0, 0.5, size=shape)

perturbation_field = np.ones(shape) * 10
for i in [np.random.randint(0, shape[1], size=3)]:
    for j in [np.random.randint(0, shape[2], size=3)]:
        perturbation_field[0, i, j, 0] += np.random.rand() * np.random.choice([-1, +1]) * 3


def gaus2d(shape, stds, phys_scales=(5, 5), centers=(0, 0)):
    x = np.linspace(-phys_scales[0], phys_scales[1], shape[0])
    y = np.linspace(-phys_scales[1], phys_scales[1], shape[1])
    x, y = np.meshgrid(x, y)
    mx, my = centers
    sx, sy = stds
    return np.exp(-((x - mx)**2. / (2. * sx ** 2.) + (y - my)**2. / (2. * sy ** 2.))) / np.sqrt(2. * np.pi * sx * sy)


gaussian_field = gaus2d(shape[1:3], (2, 2)).reshape(shape)


random_field -= np.mean(random_field)  # zero center


domain = Domain([N, N],
                box=AABox(0, [get_box_size(initial_state['K0'])] * len([N, N])),  # NOTE: Assuming square
                boundaries=(PERIODIC, PERIODIC)  # Each dim: OPEN / CLOSED / PERIODIC
                )
fft_random = CenteredGrid.sample(Noise(), domain)
integral = np.sum(fft_random.data**2)
fft_random /= np.sqrt(integral)


density_data = []
omega_data = []
phi_data = []


class PlasmaSim(App):

    def __init__(self, initial_state, resolution):
        App.__init__(self, 'Plasma Sim', DESCRIPTION, summary='plasma' + 'x'.join([str(d) for d in resolution]), framerate=20)
        self.plasma = world.add(
            PlasmaHW(
                domain,
                density=fft_random,
                omega=0.5 * fft_random,
                phi=0.5 * fft_random,
                initial_density=fft_random
            ),
            physics=HasegawaWakatani(**initial_state)
        )
        # Add Fields
        self.dt = EditableFloat('dt', step_size)
        self.add_field('Density', lambda: self.plasma.density)
        self.add_field('Phi', lambda: self.plasma.phi)
        self.add_field('Omega', lambda: self.plasma.omega)
        self.add_field('Initial Density', lambda: self.plasma.initial_density)
        self.add_field('Gradient Density', lambda: self.plasma.grad_density)

    def action_reset(self):
        self.steps = 0
        plasma = PlasmaHW(
            domain,
            density=fft_random,
            omega=-0.5 * fft_random,
            phi=-0.5 * fft_random,
            initial_density=fft_random
        )
        self.plasma.density = plasma.density
        self.plasma.omega = plasma.omega
        self.plasma.phi = plasma.phi
        self.plasma.initial_density = plasma.initial_density

    def step(self):
        world.step(dt=self.dt)
        # density_data.append(world.state.plasmaGW.density.data)
        # omega_data.append(world.state.plasmaGW.omega.data)
        # phi_data.append(world.state.plasmaHW.phi.data)
        if self.steps % 100 == 0:
            self.scene.write(self.plasma.density, names='density', frame=self.steps)
            self.scene.write(self.plasma.omega, names='omega', frame=self.steps)
            self.scene.write(self.plasma.phi, names='phi', frame=self.steps)
        # self.scene.write_sim_frame([
        #    self.plasma.density.state,
        #    self.plasma.omega.state,
        #    self.plasma.phi.state],
        #    fieldnames=['density', 'omega', 'phi'], frame=self.steps)
        #energy_array[i] = world.state.plasma.energy


show(PlasmaSim(initial_state, [N, N]), display=('Density', 'Phi', 'Omega', 'Initial Density', 'Gradient Density'), framerate=1, debug=True)
#a = PlasmaSim(initial_state, [N, N])
