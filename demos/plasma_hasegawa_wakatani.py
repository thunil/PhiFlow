import sys
from phi.flow import *  # Use NumPy
from phi.physics.plasma import *  # Plasma tools
MODE = "NumPy"

DESCRIPTION = """
Hasegawa-Wakatani Plasma
"""

N = 32

class PlasmaSim(App):

    def __init__(self, resolution):
        App.__init__(self, 'Plasma Sim', DESCRIPTION, summary='plasma' + 'x'.join([str(d) for d in resolution]), framerate=20)
        plasma = self.plasma = world.add(
            PlasmaHW(
                Domain(
                    resolution,
                    box=box[0:N, 0:N, 0:N],
                    boundaries=(PERIODIC,PERIODIC,PERIODIC)#OPEN#CLOSED#PERIODIC
                ),
                density=np.ones(shape=(1,N,N,N,1)), # np.random.uniform(low=0, high=1, size=(1,32,32,32,1)),
                omega=np.ones(shape=(1,N,N,N,1)),#np.random.uniform(low=0, high=1, size=(1,32,32,32,1)),
                phi=np.random.uniform(low=0, high=1, size=(1,N,N,N,1))#,np.ones(shape=(1,32,32,32,1))#np.zeros((1,32,32,32,1))
            ),
            physics=HasegawaWakatani()
        )
        # Add Fields
        self.dt = EditableFloat('dt', 0.01)
        self.add_field('Density', lambda: plasma.density)
        self.add_field('Phi', lambda: plasma.phi)
        self.add_field('Omega', lambda: plasma.omega)
        self.add_field('Laplace Phi', lambda: plasma.laplace_phi)
        self.add_field('Laplace Density', lambda: plasma.laplace_n)

    def action_reset(self):
        self.steps = 0
        self.plasma.density = np.random.uniform(low=0, high=1, size=(1,N,N,N,1))  #np.ones(shape=(1,32,32,32,1))
        self.plasma.omega = np.random.uniform(low=0, high=1, size=(1,N,N,N,1))
        self.plasma.phi = np.random.uniform(low=0, high=1, size=(1,N,N,N,1))#np.zeros((1,32,32,32,1))

    def step(self):
        world.step(dt=self.dt)


show(PlasmaSim([N,N,N]), display=('Density', 'Phi', 'Omega'), framerate=1, debug=True)
