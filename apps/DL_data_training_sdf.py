from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *


def forcenet2d_3x_16(initial_density, initial_velocity, target_density, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("ForceNet"):
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
        TFModel.__init__(self, "Network Training for pre-generated SDF Liquid simulation data", stride=1, learning_rate=1e-3, validation_batch_size=1)

        self.size = np.array([32, 40])
        domain = Domain(self.size, SLIPPERY)
        self.dt = 0.1
        self.gravity = -0.0

        #self.initial_density = placeholder(domain.grid.shape())
        #self.initial_velocity = placeholder(domain.grid.staggered_shape())

        self.initial_density = placeholder(np.concatenate(([None], self.size, [1])))
        self.initial_velocity = StaggeredGrid(placeholder(np.concatenate(([None], self.size+1, [len(self.size)]))))

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity)

        # Train Neural Network to find forces
        self.target_sdf = placeholder_like(self.liquid.sdf)

        with self.model_scope():
            self.forces = forcenet2d_3x_16(self.liquid.sdf, self.liquid.velocity, self.target_sdf)
        self.liquid.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.liquid.state, dt=self.dt)

        # Two thresholds for the world_step and editable float force_weight
        self.force_weight = self.editable_float('Force_Weight', 1.0, (1e-5, 1e3))
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))


        self.loss = l2_loss(self.state_out.sdf - self.target_sdf) + self.force_weight * l2_loss(self.forces)

        self.add_objective(self.loss, "Unsupervised_Loss")

        self.add_field("Trained Forces", self.forces)
        self.add_field("Target", self.target_sdf)

        self.add_field("Initial", self.initial_density)
        self.add_field("Signed Distance Field", self.liquid.sdf)
        self.add_field("Velocity",self.liquid.velocity.staggered)

        self.set_data(
            train = Dataset.load('~/phi/model/sdf-datagen', range(10,20)), 
            val = Dataset.load('~/phi/model/sdf-datagen', range(10)), 
            placeholders = (self.initial_density, self.initial_velocity.staggered, self.target_sdf),
            channels = ('initial_sdf', 'initial_velocity_staggered', 'target_sdf')
            )



app = LiquidNetworkTraining().show(production=__name__ != "__main__", framerate=3, display=("Fluid", "Velocity"))
