import sys
from phi.flow import *  # Use NumPy
from phi.physics.plasma import *  # Plasma tools
MODE = "NumPy"

DESCRIPTION = """
Hasegawa-Wakatani Plasma
"""


class PlasmaSim(App):

    def __init__(self, resolution):
        App.__init__(self, 'Plasma Sim', DESCRIPTION, summary='plasma' + 'x'.join([str(d) for d in resolution]), framerate=20)
        plasma = self.plasma = world.add(
            PlasmaHW(
                Domain(
                    resolution,
                    box=box[0:32, 0:32, 0:32],
                    boundaries=CLOSED
                ),
                density=np.ones(shape=(1,32,32,32,1)), # np.random.uniform(low=0, high=1, size=(1,32,32,32,1)),
                omega=np.random.uniform(low=0, high=1, size=(1,32,32,32,1)),
                phi=np.zeros((1,32,32,32,1))
            ),
            physics=HasegawaWakatani()
        )
        # Add Fields
        self.add_field('Density', lambda: plasma.density)
        self.add_field('Phi', lambda: plasma.phi)
        self.add_field('Omega', lambda: plasma.omega)

    def action_reset(self):
        self.steps = 0
        self.plasma.density = np.ones(shape=(1,32,32,32,1))  # np.random.uniform(low=0, high=1, size=(1,32,32,32,1))
        self.plasma.omega = np.random.uniform(low=0, high=1, size=(1,32,32,32,1))
        self.plasma.phi = np.zeros((1,32,32,32,1))

    def step(self):
        world.step()


show(PlasmaSim([32, 32, 32]), display=('Density', 'Phi', 'Omega'), framerate=1)
