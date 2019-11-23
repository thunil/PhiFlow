from phi.flow import * 


class GridDemo(App):

    def __init__(self):
        App.__init__(self, 'Grid-based simulation', "Grid-based liquid simulation.", stride=1)

        size = [64, 80]
        domain = Domain(size, SLIPPERY)
        self.dt = 0.1

        self.initial_density = domain.centered_grid(0.0).data
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1

        self.liquid = world.add(GridLiquid(domain, density=self.initial_density, velocity=0.0))
        #world.Inflow(Sphere((10,32), 5), rate=0.2)

        self.add_field("Density", lambda: self.liquid.density)
        self.add_field("Velocity", lambda: self.liquid.velocity)


    def step(self):
        world.step(dt=self.dt)

    def action_reset(self):
        self.liquid.density = self.initial_density
        self.liquid.velocity = 0.0
        self.time = 0


show(framerate=3, display=("Density", "Velocity"))
