from phi.flow import *


class SDFDemo(App):

    def __init__(self, size=(64, 80)):
        App.__init__(self, 'SDF simulation', 'Signed Distance based liquid simulation.', stride=1)
        domain = Domain(size, SLIPPERY)
        self.distance = max(size)
        # --- Initial state ---
        self.initial_density = domain.centered_grid(0.0).data
        self.initial_density[:, size[-2] * 6 // 8: size[-2] * 8 // 8 - 1, size[-1] * 2 // 8: size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8: size[-2] * 2 // 8, size[-1] * 0 // 8: size[-1] * 8 // 8, :] = 1
        # --- Create liquid and expose fields to GUI ---
        self.liquid = world.add(SDFLiquid(domain, density=self.initial_density, distance=self.distance))
        self.add_field('Fluid', lambda: self.liquid.active_mask)
        self.add_field('Signed Distance Field', lambda: self.liquid.sdf)
        self.add_field('Velocity', lambda: self.liquid.velocity)

    def step(self):
        world.step(dt=0.1)

    def action_reset(self):
        active_mask = create_binary_mask(self.initial_density, threshold=0)
        self.liquid.sdf, _ = extrapolate(self.liquid.domain, self.liquid.staggered_grid('velocity', 0.0), active_mask,
                                         distance=self.distance)
        self.liquid.domaincache._active = active_mask
        self.liquid.velocity = 0.0
        self.steps = 0


show(framerate=3, display=('Fluid', 'Velocity'))
