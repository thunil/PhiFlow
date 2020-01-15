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
                density=np.reshape(np.random.uniform(low=0, high=1, size=(32,32,32)), newshape=(1,32,32,32,1))
            ),
            physics=HasegawaWakatani()
        )
        # Add Fields
        self.add_field('Density', lambda: plasma.density)
        self.add_field('Phi', lambda: plasma.phi)
        self.add_field('Omega', lambda: plasma.omega)

    def action_reset(self):
        self.steps = 0
        self.plasma.density = 1.0

    def step(self):
        world.step()

show(PlasmaSim([32, 32, 32]), display=('Density', 'Phi', 'Omega'), framerate=1)
