from phi.tf.flow import *
from phi.tf.util import residual_block
import inspect, os


def forcenet2d_3x_16(field, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("ForceNet"):
        # Field should have the shape of the grid (not staggered grid)
        y = field
        downres_steps = 3
        downres_padding = sum([2 ** i for i in range(downres_steps)])
        y = tf.pad(y, [[0, 0], [0, downres_padding], [0, downres_padding], [0, 0]])
        resolutions = [ y ]
        filter_count = 16
        res_block_count = 2
        for i in range(downres_steps): # 1/2, 1/4
            y = tf.layers.conv2d(resolutions[0], filter_count, 2, strides=2, activation=tf.nn.relu, padding="valid", name="downconv_%d"%i, trainable=trainable, reuse=reuse)
            for j, nb_channels in enumerate([filter_count] * res_block_count):
                y = residual_block(y, nb_channels, name="downrb_%d_%d" % (i,j), training=training, trainable=trainable, reuse=reuse)
            resolutions.insert(0, y)

        for j, nb_channels in enumerate([filter_count] * res_block_count):
            y = residual_block(y, nb_channels, name="centerrb_%d" % j, training=training, trainable=trainable, reuse=reuse)

        for i in range(1, len(resolutions)):
            y = upsample2x(y)
            res_in = resolutions[i][:, 0:y.shape[1], 0:y.shape[2], :]
            y = tf.concat([y, res_in], axis=-1)
            if i < len(resolutions)-1:
                y = tf.pad(y, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="SYMMETRIC")
                y = tf.layers.conv2d(y, filter_count, 2, 1, activation=tf.nn.relu, padding="valid", name="upconv_%d" % i, trainable=trainable, reuse=reuse)
                for j, nb_channels in enumerate([filter_count] * res_block_count):
                    y = residual_block(y, nb_channels, 2, name="uprb_%d_%d" % (i, j), training=training, trainable=trainable, reuse=reuse)
            else:
                # Last iteration
                y = tf.pad(y, [[0,0], [1,1], [1,1], [0,0]], mode="SYMMETRIC")
                y = tf.layers.conv2d(y, 2, 2, 1, activation=None, padding="valid", name="upconv_%d"%i, trainable=trainable, reuse=reuse)
    force = StaggeredGrid(y)
    path = os.path.join(os.path.dirname(inspect.getabsfile(forcenet2d_3x_16)), "forcenet2d_3x_16")
    return force, path



class SDFBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Signed Distance based Liquid", stride=3, learning_rate=1e-3)

        size = [32,40]
        domain = Domain(size, SLIPPERY)

        self.distance = 40
        self.dt = 0.1

        self.initial_density_data = zeros(domain.grid.shape())
        self.initial_velocity_data = zeros(domain.grid.staggered_shape())
        self.initial_density_data[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        #self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1
        #self.initial_velocity_data.staggered[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 0, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = [-2.0, 0.0]

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density_data, velocity=self.initial_velocity_data, gravity=-5.0, distance=self.distance)
        #world.Inflow(Sphere((70,32), 8), rate=0.2)


        self.sess = Session(Scene.create('liquid'))

        # Construct model
        self.state_in = placeholder_like(self.liquid.state) # Forces based on input SDF

        with self.model_scope():
            self.forces, _ = forcenet2d_3x_16(tf.constant(self.liquid._density))

        self.state_in.trained_forces = self.forces
        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)
        

        # Try to find a force to bring it to the target state
        self.target_density_data = zeros(domain.grid.shape())
        self.target_density_data[:, size[-2] * 2 // 8 : size[-2] * 6 // 8 - 0, size[-1] * 6 // 8 : size[-1] * 8 // 8 - 1, :] = 1
        target_sdf = recompute_sdf(self.target_density_data, self.target_density_data, distance=self.distance)
        self.target_state_sdf = tf.constant(target_sdf)


        self.force_weight = self.editable_float('Force_Weight', 1.0)
        self.loss = l2_loss(self.state_out.sdf - self.target_state_sdf) + self.force_weight * l2_loss(self.forces)
        #self.loss = l2_loss(self.state_out.velocity.staggered)
        self.add_objective(self.loss, "Unsupervised_Loss")

        # Two thresholds for the world_step
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))

        self.add_field("Trained Forces", lambda: self.sess.run(self.forces, feed_dict={self.state_in.sdf: self.liquid.state.sdf, self.state_in.velocity.staggered: self.liquid.state.velocity.staggered}))
        self.add_field("State in SDF", lambda: self.sess.run(self.state_in.sdf, self.base_feed_dict))
        self.add_field("State out SDF", lambda: self.sess.run(self.state_out.sdf, self.base_feed_dict))
        

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
        self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
        self.add_field("Pressure", lambda: self.liquid.pressure)


    def step(self):
        # Run optimization step
        self.base_feed_dict.update({
            self.state_in.active_mask: self.liquid.state.active_mask, self.state_in.sdf: self.liquid.state.sdf, self.state_in.velocity.staggered: self.liquid.state.velocity.staggered})
        TFModel.step(self)
        self.current_loss = self.sess.run(self.loss, self.base_feed_dict)
        # Use trained forces to do a step when loss is small enough
        if self.current_loss < self.loss_threshold or self.steps > self.step_threshold:
            self.steps = 0
            self.world_steps += 1
            self.liquid.trained_forces = self.sess.run(self.forces, feed_dict={self.state_in.sdf: self.liquid.state.sdf, self.state_in.velocity.staggered: self.liquid.state.velocity.staggered})
            world.step(dt=self.dt)


    def action_reset(self):
        particle_mask = create_binary_mask(self.initial_density_data, threshold=0)
        self.liquid._sdf, _ = extrapolate(self.initial_velocity_data, particle_mask, distance=self.distance)
        self.liquid.domaincache._active = particle_mask
        self.liquid.velocity = self.initial_velocity_data
        self.time = 0

        #Temporary: Make this button do a step using the pretrained forces
        # self.liquid.trained_forces = self.sess.run(self.forces, feed_dict={self.state_in.sdf: self.liquid.state.sdf, self.state_in.velocity.staggered: self.liquid.state.velocity.staggered})
        # world.step(dt=self.dt)


app = SDFBasedLiquid().show(production=__name__ != "__main__", framerate=2, display=("Signed Distance Field", "Velocity"))
