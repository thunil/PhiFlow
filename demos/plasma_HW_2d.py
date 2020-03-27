import sys
from phi.flow import *  # Use NumPy
from phi.physics.plasma import *  # Plasma tools
MODE = "NumPy"

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


step_size = 10**-13
initial_state = {
    "grid": grid_dict['coarse'],      # Grid size in points (resolution)
    "K0":   K0_dict['large'],         # Box size defining parameter
    "N":    1,                        # N*2 order of dissipation
    "nu":   nu_dict['coarse-large'],  # Dissipation scaling coefficient
    "c1":   10000000*c1_dict['adiabatic'],     # Adiabatic parameter
    "kappa_coeff":   1,
    "arakawa_coeff": 1,
}


def get_box_size(k0):
    return 2*np.pi/k0


N = initial_state['grid'][1]
shape = (1, *initial_state['grid'], 1)
del initial_state['grid']

box_field = np.zeros(shape)
box_field[0, int(N/4):int(3*N/4), int(N/4):int(3*N/4), 0] = np.ones((int(N/2), int(N/2)))
box_field += 1

line_field = np.zeros(shape)
line_field[0, (int(N/2)-1):(int(N/2)+1), :, 0] = np.ones((2, N))
line_field += 1

cross_field = np.zeros(shape)
cross_field[0, (int(N/2)-1):(int(N/2)+1), :, 0] = np.ones((2, N))
cross_field[0, :, (int(N/2)-1):(int(N/2)+1), 0] = np.ones((N, 2))
cross_field += 1

slope_field = np.array([np.arange(N) for _ in range(N)]).reshape(shape)/10 + 1

step_field = np.ones(shape)
step_field[0, :, int(N/2):, 0] = 2
step_field = step_field[::-1].T

import scipy.ndimage
x_field = scipy.ndimage.rotate(cross_field[0, :, :, 0], angle=45, reshape=False).reshape(shape)+1

random_field = np.random.uniform(0, 0.5, size=shape)

perturbation_field = np.ones(shape)*10
for i in [np.random.randint(0, shape[1], size=3)]:
    for j in [np.random.randint(0, shape[2], size=3)]:
        perturbation_field[0, i, j, 0] += np.random.rand()*np.random.choice([-1, +1])*3

def gaus2d(shape, stds, phys_scales=(5, 5), centers=(0, 0)):
    x = np.linspace(-phys_scales[0], phys_scales[1], shape[0])
    y = np.linspace(-phys_scales[1], phys_scales[1], shape[1])
    x, y = np.meshgrid(x, y)
    mx, my = centers
    sx, sy = stds
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x-mx)**2. / (2. * sx**2.) + (y-my)**2. / (2. * sy**2.)))
gaussian_field = gaus2d(shape[1:3], (2, 2)).reshape(shape)


random_field -= np.mean(random_field)  # zero center


class PlasmaSim(App):

    def __init__(self, initial_state, resolution):
        App.__init__(self, 'Plasma Sim', DESCRIPTION, summary='plasma' + 'x'.join([str(d) for d in resolution]), framerate=20)
        plasma = self.plasma = world.add(
            PlasmaHW(
                Domain(
                    # dx = self.box.size / self.resolution
                    resolution,
                    box=AABox(0, [get_box_size(initial_state['K0'])]*len(resolution)),  # NOTE: Assuming square
                    boundaries=(PERIODIC, PERIODIC)  # Each dim: OPEN / CLOSED / PERIODIC
                ),
                density=2*random_field,
                omega=random_field,
                phi=random_field,
                initial_density=random_field
            ),
            physics=HasegawaWakatani(**initial_state)
        )
        # Add Fields
        self.dt = EditableFloat('dt', step_size)
        self.add_field('Density', lambda: plasma.density)
        self.add_field('Phi', lambda: plasma.phi)
        self.add_field('Omega', lambda: plasma.omega)

    def action_reset(self):
        self.steps = 0
        plasma = PlasmaHW(
            Domain(
                [N, N],
                box=box[0:N, 0:N],
                boundaries=(CLOSED, CLOSED)  # Each dim: OPEN / CLOSED / PERIODIC
            ),
            density=gaussian_field,#np.ones(shape=shape),
            omega=-gaussian_field,#np.random.uniform(low=1, high=10, size=shape),
            phi=-gaussian_field#np.random.uniform(low=1, high=10, size=shape)
        )
        self.plasma.density = plasma.density #np.random.uniform(low=1, high=2, size=shape)
        self.plasma.omega = plasma.omega #np.random.uniform(low=1, high=2, size=shape)
        self.plasma.phi = plasma.phi #np.random.uniform(low=1, high=2, size=shape)

    def step(self):
        world.step(dt=self.dt)


show(PlasmaSim(initial_state, [N, N]), display=('Density', 'Phi', 'Omega'), framerate=1, debug=True)

