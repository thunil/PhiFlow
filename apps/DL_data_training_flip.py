from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *


def forcenet2d_3x_16(initial_density, initial_velocity, target_density, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    y = tf.concat([initial_density, initial_velocity.staggered[:, :-1, :-1, :], target_density], axis=-1)
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



class LiquidNetworkTraining(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Network Training for pre-generated FLIP Liquid simulation data", stride=1, learning_rate=1e-3, validation_batch_size=1)

        self.size = np.array([32, 40])
        domain = Domain(self.size, SLIPPERY)
        self.particles_per_cell = 4
        self.dt = 0.1
        self.gravity = -0.0

        self.initial_density = placeholder(np.concatenate(([None], self.size, [1])))
        self.initial_velocity = StaggeredGrid(placeholder(np.concatenate(([None], self.size+1, [len(self.size)]))))

        particle_points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
        particle_velocity = grid_to_particles(domain.grid, particle_points, self.initial_velocity, staggered=True)

        # Initialization doesn't matter, training data is fed later
        # Question: Do we want gravity at all.
        self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=particle_velocity, gravity=self.gravity, particles_per_cell=self.particles_per_cell)

        # Train Neural Network to find forces
        self.target_density = placeholder(domain.grid.shape())

        with self.model_scope():
            self.forces = forcenet2d_3x_16(self.initial_density, self.initial_velocity, self.target_density)
        self.liquid.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.liquid.state, dt=self.dt)

        # Two thresholds for the world_step and editable float force_weight
        self.force_weight = self.editable_float('Force_Weight', 1.0, (1e-5, 1e3))
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))


        self.loss = l2_loss(self.state_out.density_field - self.target_density) + self.force_weight * l2_loss(self.forces)

        self.add_objective(self.loss, "Unsupervised_Loss")

        self.add_field("Trained Forces", self.forces)
        self.add_field("Target", self.target_density)

        self.add_field("Fluid", self.liquid.active_mask)
        self.add_field("Density", self.liquid.density_field)
        # self.add_field("Points", grid(self.liquid.grid, self.liquid.points, self.liquid.points))
        # self.add_field("Velocity", self.liquid.velocity_field.staggered)

        self.set_data(
            train = Dataset.load('~/phi/model/flip-datagen', range(10,20)), 
            val = Dataset.load('~/phi/model/flip-datagen', range(10)), 
            placeholders = (self.initial_density, self.initial_velocity.staggered, self.target_density),
            channels = ('initial_density', 'initial_velocity_staggered', 'target_density')
            )


app = LiquidNetworkTraining().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Target"))
