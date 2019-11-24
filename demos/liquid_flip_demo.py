from phi.flow import * 


class FlipDemo(App):

    def __init__(self):
        App.__init__(self, 'FLIP simulation', "Fluid Implicit Particle liquid simulation.", stride=3)

        self.dt = 0.3
        size = [64, 64]
        domain = Domain(size, SLIPPERY)
        self.particles_per_cell = 8

        self.initial_density = domain.centered_grid(0.0).data
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1


        self.initial_velocity = [0.0,0.0]

        self.initial_points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
        
        self.liquid = world.add(FlipLiquid(domain, points=self.initial_points, velocity=self.initial_velocity, gravity=-4.0, particles_per_cell=self.particles_per_cell))

        self.add_field("Fluid", lambda: self.liquid.active_mask.center_sample())
        self.add_field("Density", lambda: self.liquid.density.center_sample())
        self.add_field("Velocity", lambda: self.liquid.velocity.stagger_sample())


    def step(self):
        world.step(dt=self.dt)

        print("Amount of particles:" + str(math.sum(self.liquid.density.center_sample().data)))


    def action_reset(self):
        self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
        self.liquid.velocity = self.initial_velocity
        self.time = 0


show(framerate=3, display=("Fluid", "Velocity"))
