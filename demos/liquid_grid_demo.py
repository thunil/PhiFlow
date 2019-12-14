from phi.flow import *


class GridDemo(App):

    def __init__(self, size=(64, 80)):
        App.__init__(self, 'Grid-based simulation', 'Grid-based liquid simulation.', stride=1)
        domain = Domain(size, SLIPPERY)
        # --- Initial state ---
        self.initial_density = domain.centered_grid(0.0).data
        self.initial_density[:, size[-2] * 6 // 8: size[-2] * 8 // 8 - 1, size[-1] * 2 // 8: size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8: size[-2] * 2 // 8, size[-1] * 0 // 8: size[-1] * 8 // 8, :] = 1
        # --- Create liquid and expose fields to GUI ---
        self.liquid = world.add(Fluid(domain, density=self.initial_density), physics=FreeSurfaceFlow())
        self.add_field('Density', lambda: self.liquid.density)
        self.add_field('Velocity', lambda: self.liquid.velocity)

    def step(self):
        world.step(dt=0.1)

    def action_reset(self):
        self.liquid.density = self.initial_density
        self.liquid.velocity = 0.0
        self.steps = 0


show(framerate=3, display=('Density', 'Velocity'))
