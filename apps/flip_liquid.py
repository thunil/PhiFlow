from phi.flow import *
from phi.math.sampled import *

class ParticleBasedLiquid(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Particle-based Liquid", stride=3)

        size = [64, 80]
        domain = Domain(size, SLIPPERY)
        self.particles_per_cell = 8

        self.initial_density = zeros(domain.grid.shape())
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1

        #self.initial_velocity = [1.42, 0]
        self.initial_velocity = 0.0
        
        self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=-0.5, particles_per_cell=self.particles_per_cell)
        #world.Inflow(Sphere((10,32), 5), rate=0.2)

        self.add_field("Fluid", lambda: self.liquid.domaincache.active())
        self.add_field("Density", lambda: self.liquid.density_field)
        self.add_field("Points", lambda: grid(self.liquid.grid, self.liquid.points, self.liquid.points))
        self.add_field("Velocity", lambda: self.liquid.velocity_field.staggered)
        self.add_field("Pressure", lambda: self.liquid.last_pressure)
        self.add_field("Advection Velocity on grid", lambda: self.liquid.advection_velocity.staggered)

        self.add_field("Divergence before", lambda: self.liquid.div_before)
        self.add_field("Velocity before pressure solve", lambda: self.liquid.vel_before.staggered)
        self.add_field("Velocity after pressure solve", lambda: self.liquid.vel_after.staggered)
        self.add_field("Extrapolated Pressure Gradient grid", lambda: self.liquid.ext_gradp.staggered)


    def step(self):
        world.step(dt=0.5)

    def action_reset(self):
        self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
        self.liquid.velocity = zeros_like(self.liquid.points) + self.initial_velocity
        self.time = 0


app = ParticleBasedLiquid().show(production=__name__ != "__main__", framerate=3, display=("Density", "Velocity"))
