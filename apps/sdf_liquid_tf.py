from phi.tf.flow import *

class SDFBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Signed Distance based Liquid", stride=3)

        size = [80,64]
        domain = Domain(size, SLIPPERY)

        self.distance = 70

        self.initial_density = zeros(domain.grid.shape())
        self.initial_velocity = zeros(domain.grid.staggered_shape())
        #self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8-2, size[-1] * 3 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 0, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        #self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1
        #self.initial_velocity.staggered[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 3 // 8 : size[-1] * 6 // 8 + 1, :] = [-2.0, -0.0]

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=-5.0, distance=self.distance)
        #world.Inflow(Sphere((70,32), 8), rate=0.2)

        session = Session(Scene.create('test'))
        tf_bake_graph(world, session)

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Mask Before", lambda: self.liquid.mask_before.staggered)
        self.add_field("Mask After", lambda: self.liquid.mask_after.staggered)
        self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
        self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
        self.add_field("Velocity Centered", lambda: self.liquid.velocity)
        self.add_field("Divergence Velocity", lambda: self.liquid.velocity.divergence())
        self.add_field("Pressure", lambda: self.liquid.pressure)


    def step(self):
        world.step(dt=0.1)

    def action_reset(self):
        particle_mask = create_binary_mask(self.initial_density, threshold=0)
        self.liquid._sdf, _ = extrapolate(self.initial_velocity, particle_mask, distance=self.distance)
        self.liquid.domaincache._active = particle_mask
        self.liquid.velocity = self.initial_velocity
        self.time = 0


app = SDFBasedLiquid().show(production=__name__ != "__main__", framerate=2, display=("Signed Distance Field", "Velocity"))
