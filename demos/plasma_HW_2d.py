import sys
from phi.flow import *  # Use NumPy
from phi.physics.plasma import *  # Plasma tools
MODE = "NumPy"

DESCRIPTION = """
Hasegawa-Wakatani Plasma
"""
N = 64
shape = (1, N, N, 1)

class PlasmaSim(App):

    def __init__(self, resolution):
        App.__init__(self, 'Plasma Sim', DESCRIPTION, summary='plasma' + 'x'.join([str(d) for d in resolution]), framerate=20)
        plasma = self.plasma = world.add(
            PlasmaHW(
                Domain(
                    resolution,
                    box=box[0:N, 0:N],
                    boundaries=(CLOSED, CLOSED)  # Each dim: OPEN / CLOSED / PERIODIC
                ),
                density=np.ones(shape=shape),
                omega=np.random.uniform(low=1, high=10, size=shape),
                phi=np.random.uniform(low=1, high=10, size=shape)
            ),
            physics=HasegawaWakatani(kap=0.1)
        )
        # Add Fields
        self.dt = EditableFloat('dt', 0.01)
        self.add_field('Density', lambda: plasma.density)
        self.add_field('Phi', lambda: plasma.phi)
        self.add_field('Omega', lambda: plasma.omega)

    def action_reset(self):
        self.steps = 0
        self.plasma.density = np.random.uniform(low=0, high=1, size=shape)
        self.plasma.omega = np.random.uniform(low=1, high=2, size=shape)
        self.plasma.phi = np.random.uniform(low=1, high=2, size=shape)

    def step(self):
        world.step(dt=self.dt)


show(PlasmaSim([N, N]), display=('Density', 'Phi', 'Omega'), framerate=1, debug=True)
