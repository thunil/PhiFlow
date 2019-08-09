from phi.tf.flow import *
from phi.math.sampled import *

class ParticleBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Particle-based Liquid TF", stride=3)

        size = [64, 80]
        domain = Domain(size, SLIPPERY)
        self.particles_per_cell = 4

        self.initial_density = zeros(domain.grid.shape())
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1

        #self.initial_velocity = [1.42, 0]
        self.initial_velocity = 0.0
        
        self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=-2.0, particles_per_cell=self.particles_per_cell)
        #world.Inflow(Sphere((10,32), 5), rate=0.2)

        session = Session(Scene.create('liquid'))
        tf_bake_graph(world, session)

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Density", lambda: self.liquid.density_field)
        self.add_field("Points", lambda: grid(self.liquid.grid, self.liquid.points, self.liquid.points))
        self.add_field("Velocity", lambda: self.liquid.velocity_field.staggered)
        self.add_field("Pressure", lambda: self.liquid.last_pressure)


    def step(self):
        world.step(dt=0.1)

    def action_reset(self):
        self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
        self.liquid.velocity = zeros_like(self.liquid.points) + self.initial_velocity
        self.time = 0


app = ParticleBasedLiquid().show(production=__name__ != "__main__", framerate=3, display=("Density", "Velocity"))
