from phi.tf.flow import *

# TODO: doesn't work yet, trained forces isn't changing, I assume that there is no gradient between the ball and the liquid, need to find out where the gradient backpropagation fails.

def ball_movement(previous_ball, dt, **dependent_states):
    # TODO: Doesn't work with multiple batches yet
    # There should only be one velocity state passed.
    assert len(dependent_states['velocity_state']) == 1
    # We want to move the ball according to the velocity field passed in dependent_states
    velocity_state = dependent_states['velocity_state'][0]
    velocity_field = velocity_state.velocity.staggered

    # Reshape the sample_coords correctly. [batch , z,y,x,... , spatial_rank]
    new_shape = math.ones_like(velocity_state.grid.shape())
    new_shape[-1] = velocity_field.shape[-1]
    sample_coords = math.reshape(previous_ball.center, new_shape)
    # Resample the velocity field at the center of the ball to see what the ball is advected with
    ball_velocity = math.resample(velocity_field, sample_coords)
    ball_velocity = math.reshape(ball_velocity, previous_ball.center.shape)

    center_coords = previous_ball.center + dt * ball_velocity
    grid_dimensions = velocity_state.grid.dimensions
    radius = previous_ball.radius
    # Clamp the coordinates so the ball can't leave the domain. We assume the ball was summoned in a closed battlefield and cannot run from its destiny.
    center_coords = math.minimum(math.maximum(radius, center_coords), grid_dimensions - radius)

    return Sphere(center_coords, radius)


class SDFBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Deep Learning with Liquid movements", stride=3, learning_rate=1e-1)

        size = [40,32]
        domain = Domain(size, SLIPPERY)

        self.distance = 40
        self.dt = 0.1

        self.initial_density_data = zeros(domain.grid.shape())
        self.initial_velocity_data = zeros(domain.grid.staggered_shape())
        self.initial_density_data[:, size[-2] * 2 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        #self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1
        #self.initial_velocity_data.staggered[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 0, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = [-2.0, 0.0]

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density_data, velocity=self.initial_velocity_data, gravity=-1.0, distance=self.distance)
        #world.Inflow(Sphere((70,32), 8), rate=0.2)


        # Forces to be trained are directly added onto velocity, therefore should have same shape.
        with self.model_scope():
            self.forces = tf.Variable(tf.zeros(domain.grid.staggered_shape().staggered), name="TrainedForces", trainable=True)
        self.reset_forces = self.forces.assign(tf.zeros(domain.grid.staggered_shape().staggered))


        self.ball = world.Obstacle(Sphere([20.0, 16.0], 2.0), tags=())
        self.ball.physics = GeometryMovement(ball_movement)
        self.ball.physics.dependencies.update({'velocity_state': 'velocityfield'})

        self.sess = Session(Scene.create('liquid'))

        # Explicitly written out as I needed to set the tf Variable somewhere
        self.state_in = placeholder_like(world.state)
        dt = self.dt
        # Perhaps too hard-coded. Set Liquid trained force to tf Variable
        self.state_in.states[0].trained_forces = self.forces
        self.state_out = world.physics.step(self.state_in, dt=dt)
        world.physics = BakedWorldPhysics(world.physics, self.sess, self.state_in, self.state_out, dt)

        for sysstate in world.state.states:
            sysstate_in = self.state_in[sysstate.trajectorykey]
            sysstate_out = self.state_out[sysstate.trajectorykey]
            baked_physics = BakedPhysics(self.sess, sysstate_in, sysstate_out, dt)
            world.physics.add(sysstate.trajectorykey, baked_physics)


        # Temporary: Liquid states
        #self.state_in = state_in.states[0]
        #self.state_out = state_out.states[0]

        #self.ball_in = state_in.states[1]
        #self.ball_center = state_out.states[1].geometry.center

        # Try to find a force to bring the ball to the target position
        self.target_center = tf.constant([10.0, 16.0])

        self.force_weight = self.editable_float('Force_Weight', 1.0)
        self.loss = l2_loss(self.state_out.states[1].geometry.center - self.target_center) + self.force_weight * l2_loss(self.forces)
        self.add_objective(self.loss, "Unsupervised_Loss")

        # Two thresholds for the world_step
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))

        self.add_field("Trained Forces", lambda: self.sess.run(self.forces)) # feed_dict=self._feed_dict(None, False)
        self.add_field("State in SDF", lambda: self.sess.run(self.state_in.states[0].sdf, self.base_feed_dict))
        self.add_field("State out SDF", lambda: self.sess.run(self.state_out.states[0].sdf, self.base_feed_dict))

        self.add_field("Ball Location", lambda: self.ball.geometry.value_at(indices_tensor(self.initial_density_data)))
        

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
        self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
        self.add_field("Pressure", lambda: self.liquid.pressure)


    def step(self):
        # Run optimization step
        self.base_feed_dict.update({self.state_in.states[0].active_mask: self.liquid.state.active_mask, self.state_in.states[0].sdf: self.liquid.state.sdf, self.state_in.states[0].velocity.staggered: self.liquid.state.velocity.staggered, self.state_in.states[1].geometry.center: self.ball.geometry.center})
        TFModel.step(self)
        self.current_loss = self.sess.run(self.loss, self.base_feed_dict)
        # Use trained forces to do a step when loss is small enough
        if self.current_loss < self.loss_threshold or self.steps > self.step_threshold:
            self.steps = 0
            self.world_steps += 1
            self.liquid.trained_forces = self.sess.run(self.forces)
            world.step(dt=self.dt)


    def action_reset(self):
        # particle_mask = create_binary_mask(self.initial_density_data, threshold=0)
        # self.liquid._sdf, _ = extrapolate(self.initial_velocity_data, particle_mask, distance=self.distance)
        # self.liquid.domaincache._active = particle_mask
        # self.liquid.velocity = self.initial_velocity_data
        # self.sess.run(self.reset_forces)
        # self.time = 0

        #Temporary: Make this button do a step using the pretrained forces
        self.liquid.trained_forces = self.sess.run(self.forces)
        world.step(dt=self.dt)


app = SDFBasedLiquid().show(production=__name__ != "__main__", framerate=2, display=("Signed Distance Field", "Velocity"))
