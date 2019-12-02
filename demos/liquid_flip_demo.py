from phi.flow import *


PARTICLES_PER_CELL = 8


class FlipDemo(App):

    def __init__(self, size=(64, 64)):
        App.__init__(self, 'FLIP simulation', 'Fluid Implicit Particle liquid simulation.', stride=3)
        domain = Domain(size, SLIPPERY, box=AABox(0, [32,32]))
        # --- Initial state ---
        self.initial_density = domain.centered_grid(0.0).data
        self.initial_density[:, size[-2] * 6 // 8: size[-2] * 8 // 8 - 1, size[-1] * 2 // 8: size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8: size[-2] * 2 // 8, size[-1] * 0 // 8: size[-1] * 8 // 8, :] = 1
        self.initial_velocity = [0.0, 0.0]
        self.initial_points = random_grid_to_coords(self.initial_density, PARTICLES_PER_CELL)
        # --- Create liquid and expose fields to GUI ---
        self.liquid = world.add(FlipLiquid(domain, points=self.initial_points, velocity=self.initial_velocity, particles_per_cell=PARTICLES_PER_CELL))
        self.add_field('Fluid', lambda: self.liquid.active_mask.at(domain))
        self.add_field('Density', lambda: self.liquid.density.at(domain))
        self.add_field('Velocity', lambda: self.liquid.velocity.at(self.liquid.staggered_grid('staggered', 0)))

    def step(self):
        world.step(dt=0.3)
        self.info('Particle count: %d' % self.liquid.points.shape[1])

    def action_reset(self):
        self.liquid.points = random_grid_to_coords(self.initial_density, PARTICLES_PER_CELL)
        self.liquid.velocity = self.initial_velocity
        self.steps = 0


FlipDemo().prepare().step()
show(framerate=3, display=('Fluid', 'Velocity'))
