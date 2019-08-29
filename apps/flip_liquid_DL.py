from phi.tf.flow import *
from phi.math.sampled import *
from phi.tf.util import residual_block
import inspect, os



def forcenet2d_3x_16(field, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("ForceNet"):
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
    return force



class ParticleBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Particle-based Liquid DL", stride=3, learning_rate=1e-3)

        size = [32, 40]
        domain = Domain(size, SLIPPERY)
        self.particles_per_cell = 4
        self.dt = 0.1

        self.initial_density = zeros(domain.grid.shape())
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        #self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1

        #self.initial_velocity = [1.42, 0]
        self.initial_velocity = 0.0
        
        self.liquid = world.FlipLiquid(state_domain=domain, density=0.0, velocity=self.initial_velocity, gravity=-5.0, particles_per_cell=self.particles_per_cell)
        #world.Inflow(Sphere((10,32), 5), rate=0.2)

        self.sess = Session(Scene.create('liquid'))
        self.state_in = placeholder_like(self.liquid.state, particles=True)

        with self.model_scope():
            self.forces = forcenet2d_3x_16(tf.constant(self.liquid.density_field))

        self.state_in.trained_forces = self.forces
        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

        # Try to find a force to bring it to the target state
        target_density_data = zeros(domain.grid.shape())
        target_density_data[:, size[-2] * 2 // 8 : size[-2] * 6 // 8 - 0, size[-1] * 6 // 8 : size[-1] * 8 // 8 - 1, :] = self.particles_per_cell
        #target_points_data = active_centers(target_density_data, self.particles_per_cell)

        self.force_weight = self.editable_float('Force_Weight', 1.0, (1e-5, 1e3))
        self.loss = l2_loss(self.state_out.density_field - target_density_data) + self.force_weight * l2_loss(self.forces)
        #self.loss = l2_loss(self.state_out.velocity)
        self.add_objective(self.loss, "Unsupervised_Loss")

        # Two thresholds for the world_step
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))

        self.add_field("Trained Forces", lambda: self.sess.run(self.forces))

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Density", lambda: self.liquid.density_field)
        self.add_field("Points", lambda: grid(self.liquid.grid, self.liquid.points, self.liquid.points))
        self.add_field("Velocity", lambda: self.liquid.velocity_field.staggered)
        self.add_field("Pressure", lambda: self.liquid.last_pressure)


    def step(self):
        print("Amount of particles:" + str(math.sum(self.liquid.density_field)))
        # Run optimization step
        self.base_feed_dict.update({
            self.state_in.active_mask: self.liquid.state.active_mask, self.state_in.points: self.liquid.state.points, self.state_in.velocity: self.liquid.state.velocity
            })
        TFModel.step(self)
        self.current_loss = self.sess.run(self.loss, self.base_feed_dict)
        # Use trained forces to do a step when loss is small enough
        if self.current_loss < self.loss_threshold or self.steps > self.step_threshold:
            self.steps = 0
            self.world_steps += 1
            self.liquid.trained_forces = self.sess.run(self.forces)
            world.step(dt=self.dt)


    def action_reset(self):
        self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
        self.liquid.velocity = zeros_like(self.liquid.points) + self.initial_velocity
        self.time = 0

        #Temporary: Make this button do a step using the pretrained forces
        # self.liquid.trained_forces = self.sess.run(self.forces)
        # world.step(dt=self.dt)


app = ParticleBasedLiquid().show(production=__name__ != "__main__", framerate=3, display=("Density", "Trained Forces"))
