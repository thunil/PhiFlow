from phi.tf.flow import *


# Training Parameters
num_steps = 200
batch_size = 1
display_step = 10

# Network Parameters
size = [32,40]
dropout = 0.75 # Dropout, probability to keep units
keep_prob = tf.constant(dropout) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    #x = tf.reshape(x, shape=[-1, size[0]+1, size[1]+1, 2])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, rate=1-dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



class SDFBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Signed Distance based Liquid", stride=3, learning_rate=1e-1)

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
            # Store layers weight & bias
            weights = {
                # 5x5 conv, 1 input, 32 outputs
                'wc1': tf.Variable(tf.random_normal([5, 5, 2, 32])),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd1': tf.Variable(tf.random_normal([9*11*1*64, 1024])),
                # 1024 inputs, 10 outputs (class prediction)
                'out': tf.Variable(tf.random_normal([1024, 2]))
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'out': tf.Variable(tf.random_normal([2]))
            }

            #self.forces = StaggeredGrid(conv_net(self.state_in.velocity.staggered, weights, biases, keep_prob))

            #self.forces = StaggeredGrid(tf.Variable(tf.random_normal(domain.grid.staggered_shape().staggered), trainable=True))

            # weight = tf.Variable(tf.random_normal(domain.grid.staggered_shape().staggered), trainable=True)
            # self.forces = weight * self.state_in.velocity

            # kernel = tf.Variable(tf.random_normal([5,5,2,2]))
            # self.forces = tf.nn.conv2d(self.state_in.velocity.staggered, kernel, strides=[1,1,1,1], padding='SAME')
            # self.forces = StaggeredGrid(self.forces)

            kernel1 = tf.Variable(tf.random_normal([5,5,2,32]))
            kernel2 = tf.Variable(tf.random_normal([5,5,32,64]))
            kernel3 = tf.Variable(tf.random_normal([33*41*64, 33*41*2]))
            bias1 = tf.Variable(tf.random_normal([32]))
            bias2 = tf.Variable(tf.random_normal([64]))
            bias3 = tf.Variable(tf.random_normal([33*41*2]))

            # Convolution Layer
            conv1 = conv2d(self.state_in.velocity.staggered, kernel1, bias1)
            conv2 = conv2d(conv1, kernel2, bias2)

            # conv2 shape: [1,33,41,64]

            fc1 = tf.reshape(conv2, [-1, kernel3.get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, kernel3), bias3)
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.reshape(fc1, [-1, 33, 41, 2])

            self.forces = StaggeredGrid(fc1)

            
        self.state_in.trained_forces = self.forces
        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)
        

        # Try to find a force to bring it to the target state
        self.target_density_data = zeros(domain.grid.shape())
        self.target_density_data[:, size[-2] * 2 // 8 : size[-2] * 6 // 8 - 0, size[-1] * 6 // 8 : size[-1] * 8 // 8 - 1, :] = 1
        target_sdf = recompute_sdf(self.target_density_data, self.target_density_data, distance=self.distance)
        self.target_state_sdf = tf.constant(target_sdf)


        self.force_weight = self.editable_float('Force_Weight', 1.0)
        #self.loss = l2_loss(self.state_out.sdf - self.target_state_sdf) + self.force_weight * l2_loss(self.forces)
        self.loss = l2_loss(self.state_out.velocity.staggered)
        self.add_objective(self.loss, "Unsupervised_Loss")

        # Two thresholds for the world_step
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))

        self.add_field("Trained Forces", lambda: self.sess.run(self.forces.staggered, feed_dict={self.state_in.sdf: self.liquid.state.sdf, self.state_in.velocity.staggered: self.liquid.state.velocity.staggered}))
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
        # particle_mask = create_binary_mask(self.initial_density_data, threshold=0)
        # self.liquid._sdf, _ = extrapolate(self.initial_velocity_data, particle_mask, distance=self.distance)
        # self.liquid.domaincache._active = particle_mask
        # self.liquid.velocity = self.initial_velocity_data
        # self.sess.run(self.reset_forces)
        # self.time = 0

        #Temporary: Make this button do a step using the pretrained forces
        self.liquid.trained_forces = self.sess.run(self.forces, feed_dict={self.state_in.sdf: self.liquid.state.sdf, self.state_in.velocity.staggered: self.liquid.state.velocity.staggered})
        world.step(dt=self.dt)


app = SDFBasedLiquid().show(production=__name__ != "__main__", framerate=2, display=("Signed Distance Field", "Velocity"))
