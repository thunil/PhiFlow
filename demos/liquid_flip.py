from phi.flow import *


PARTICLES_PER_CELL = 8
DESCRIPTION = """
The Fluid Implicit Particle (FLIP) is a hybrid particle-grid approach for simulating liquids.
It preserves liquid volume much better than purely grid-based approaches.

For initialization, %d particles are randomly distributed in each occupied cell.
""" % PARTICLES_PER_CELL


class FlipDemo(App):

    def __init__(self, size=(64, 64)):
        App.__init__(self, 'FLIP simulation', DESCRIPTION, stride=10)
        domain = Domain(size, CLOSED, box=AABox(0, [32,32]))
        # --- Initial state ---
        self.initial_density = domain.centered_grid(0.0).data
        self.initial_density[:, size[-2] * 6 // 8: size[-2] * 8 // 8 - 1, size[-1] * 2 // 8: size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8: size[-2] * 2 // 8, size[-1] * 0 // 8: size[-1] * 8 // 8, :] = 1
        self.initial_velocity = [0.0, 0.0]
        self.initial_points = distribute_points(self.initial_density, PARTICLES_PER_CELL)
        # --- Create liquid and expose fields to GUI ---
        self.liquid = world.add(FlipLiquid(domain, points=self.initial_points, velocity=self.initial_velocity, particles_per_cell=PARTICLES_PER_CELL), physics=FreeSurfaceFlow())
        self.add_field('Fluid', lambda: self.liquid.active_mask.at(domain))
        self.add_field('Density', lambda: self.liquid.density.at(domain))
        self.add_field('Velocity', lambda: self.liquid.velocity.at(self.liquid.staggered_grid('staggered', 0)))

    def step(self):
        world.step(dt=0.1)
        self.info('%d: Particle count: %d' % (self.steps, self.liquid.points.data.shape[1]))

    def action_reset(self):
        self.liquid.points = distribute_points(self.initial_density, PARTICLES_PER_CELL)
        self.liquid.velocity = self.initial_velocity
        self.steps = 0


show(display=('Fluid', 'Velocity'))
