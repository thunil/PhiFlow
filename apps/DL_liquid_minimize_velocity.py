from phi.tf.flow import *

class SDFBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Signed Distance based Liquid", stride=3, learning_rate=1e-0)

        size = [80,64]
        domain = Domain(size, SLIPPERY)

        self.distance = 80
        self.dt = 0.1

        self.initial_density_data = zeros(domain.grid.shape())
        self.initial_velocity_data = zeros(domain.grid.staggered_shape())
        self.initial_density_data[:, size[-2] * 2 // 8 : size[-2] * 6 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8 - 1, :] = 1
        #self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1
        #self.initial_velocity_data.staggered[:, size[-2] * 2 // 8 : size[-2] * 6 // 8 - 0, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = [-2.0, 0.0]

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density_data, velocity=self.initial_velocity_data, gravity=-5.0, distance=self.distance)
        #world.Inflow(Sphere((70,32), 8), rate=0.2)


        # Forces to be trained are directly added onto velocity, therefore should have same shape.
        with self.model_scope():
            self.forces = tf.Variable(tf.zeros(domain.grid.staggered_shape().staggered), name="TrainedForces", trainable=True)
        self.reset_forces = self.forces.assign(tf.zeros(domain.grid.staggered_shape().staggered))

        self.session = Session(Scene.create('liquid'))

        self.state_in = placeholder_like(self.liquid.state)
        self.state_in.trained_forces = self.forces
        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

        self.loss = l2_loss(self.state_out.velocity.staggered)
        self.add_objective(self.loss, "Unsupervised_Loss")

        # Two thresholds for the world_step
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))


        self.add_field("Trained Forces", lambda: self.session.run(self.forces)) # feed_dict=self._feed_dict(None, False)
        self.add_field("State in SDF", lambda: self.session.run(self.state_in.sdf, self.base_feed_dict))
        self.add_field("State out SDF", lambda: self.session.run(self.state_out.sdf, self.base_feed_dict))
        

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
        self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
        self.add_field("Pressure", lambda: self.liquid.pressure)


    def step(self):
        # Run optimization step
        self.base_feed_dict.update({self.state_in.active_mask: self.liquid.state.active_mask, self.state_in.sdf: self.liquid.state.sdf, self.state_in.velocity.staggered: self.liquid.state.velocity.staggered})
        TFModel.step(self)
        self.current_loss = self.session.run(self.loss, self.base_feed_dict)
        # Use trained forces to do a step when loss is small enough
        if self.current_loss < self.loss_threshold or self.steps > self.step_threshold:
            self.steps = 0
            self.world_steps += 1
            self.liquid.trained_forces = self.session.run(self.forces)
            world.step(dt=self.dt)


    def action_reset(self):
        # particle_mask = create_binary_mask(self.initial_density_data, threshold=0)
        # self.liquid._sdf, _ = extrapolate(self.initial_velocity_data, particle_mask, distance=self.distance)
        # self.liquid.domaincache._active = particle_mask
        # self.liquid.velocity = self.initial_velocity_data
        # self.session.run(self.reset_forces)
        # self.time = 0

        #Temporary: Make this button do a step using the pretrained forces
        self.liquid.trained_forces = self.session.run(self.forces)
        world.step(dt=self.dt)


app = SDFBasedLiquid().show(production=__name__ != "__main__", framerate=2, display=("Signed Distance Field", "Velocity"))
