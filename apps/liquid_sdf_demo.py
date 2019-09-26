import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *  # Use TensorFlow
    mode = 'TensorFlow'
else:
    from phi.flow import *  # Use NumPy
    mode = 'NumPy'


class SDFDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'SDF simulation', "Signed Distance based liquid simulation using %s backend." % mode, stride=1)

        self.dt = 0.1
        size = [64, 80]
        domain = Domain(size, SLIPPERY)

        self.distance = max(size)

        self.initial_density = zeros(domain.grid.shape())
        self.initial_velocity = zeros(domain.grid.staggered_shape())
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1
        self.initial_velocity.staggered[:, size[-2] * 3 // 8 : size[-2] * 6 // 8 + 1, size[-1] * 3 // 8 : size[-1] * 6 // 8 + 1, :] = [0, -0.0]

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=-9.81, distance=self.distance)
        #world.Inflow(Sphere((70,32), 8), rate=0.2)

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
        self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
        self.add_field("Pressure", lambda: self.liquid.last_pressure)


    def step(self):
        world.step(dt=self.dt)

    def action_reset(self):
        active_mask = create_binary_mask(self.initial_density, threshold=0)
        self.liquid._sdf, _ = extrapolate(self.initial_velocity, active_mask, distance=self.distance)
        self.liquid.domaincache._active = active_mask
        self.liquid.velocity = self.initial_velocity
        self.time = 0


app = SDFDemo().show(production=__name__ != "__main__", framerate=2, display=("Signed Distance Field", "Velocity"))
