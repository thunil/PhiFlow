from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *


def insert_circles(field, centers, radii, values=None):
    """
Field should be a density/active mask/velocity field with shape [batch, coordinate_dimensions, components].
Centers should be given in index format (highest dimension first) and values should be integers that index into the field. Can be a list of coordinates.
Radii can be a single value if it is the same for all centers, otherwise specify a radius for every center value in the list of centers.
Values should specify the vector that goes into the entry of the corresponding circle (list of vectors if there are multiple centers).
    """

    indices = indices_tensor(field).astype(int)
    indices = math.reshape(indices, [indices.shape[0], -1, indices.shape[-1]])[0]

    # Both index and centers need to be np arrays (or TF tensors?) in order for the subtraction to work properly
    centers = np.array(centers)

    # Loop through entire field and mark the cells that are in the circle
    for index in indices:
        where_circle = (math.sum((index - centers)**2, axis=-1) <= radii**2)

        if (where_circle).any():
            field_index = (slice(None),) + tuple(math.unstack(index)) + (slice(None),)

            if values is None:
                # Insert scalar density/fluid mask
                field[field_index] = 1
            else:
                # Insert vector field
                values_index = math.where(where_circle)[0]     # Always take first possible circle
                field[field_index] = values[values_index]

    return field


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
    return force


class RandomLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Random Liquid simulation for Deep Learning", stride=3, learning_rate=1e-3)

        self.size = [32, 40]
        domain = Domain(self.size, SLIPPERY)
        self.dt = 0.1
        self.gravity = -4.0

        self.initial_density = zeros(domain.grid.shape())
        # Initial velocity different for FLIP, so set it separately over there
        self.initial_velocity = zeros(domain.grid.staggered_shape())

        number_of_circles = np.random.randint(1, min(self.size)/2)
        centers = np.array([np.random.randint(i, size=number_of_circles) for i in self.size]).reshape([-1, len(self.size)])
        radii = np.random.uniform(0, min(self.size)/number_of_circles, size=number_of_circles)
        velocities = np.array([np.random.uniform(-min(self.size)/4, min(self.size)/4, size=number_of_circles) for _ in self.size]).reshape([-1, len(self.size)])

        self.initial_density = insert_circles(self.initial_density, centers, radii)
        self.initial_velocity = StaggeredGrid(insert_circles(self.initial_velocity.staggered, centers, radii, velocities))

        # Choose whether you want a particle-based FLIP simulation or a grid-based SDF simulation
        self.flip = False
        if self.flip:
            # FLIP simulation
            self.particles_per_cell = 4
            self.initial_velocity = np.random.uniform(-min(self.size)/4, min(self.size)/4, size=len(self.size))
            
            self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=0.0, gravity=0.0, particles_per_cell=self.particles_per_cell)

            self.add_field("Fluid", lambda: self.liquid.active_mask)
            self.add_field("Density", lambda: self.liquid.density_field)
            self.add_field("Points", lambda: grid(self.liquid.grid, self.liquid.points, self.liquid.points))
            self.add_field("Velocity", lambda: self.liquid.velocity_field.staggered)
            self.add_field("Pressure", lambda: self.liquid.pressure)

        else:
            # SDF simulation
            self.distance = max(self.size)

            self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=zeros(domain.grid.staggered_shape()), gravity=0.0, distance=self.distance)

            self.add_field("Fluid", lambda: self.liquid.active_mask)
            self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
            self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
            self.add_field("Pressure", lambda: self.liquid.pressure)


        # Train Neural Network to find forces
        self.sess = Session(Scene.create('liquid'))
        self.state_in = placeholder_like(self.liquid.state, particles=self.flip)

        self.forces = forcenet2d_3x_16(tf.constant(self.liquid._density))
        self.state_in.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

        # Two thresholds for the world_step and editable float force_weight
        self.force_weight = self.editable_float('Force_Weight', 1.0, (1e-5, 1e3))
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))


        # Target state being random number of steps in the future of initial
        state = self.liquid.state

        # Set the gravity and velocities during the steps
        state._gravity = self.gravity
        state._velocity = self.initial_velocity

        self.target_steps = np.random.randint(1, 20)
        print("Target steps:" + str(self.target_steps)) 
        for _ in range(self.target_steps):
            state = self.liquid.default_physics().step(state, dt=self.dt)

        # Define what target is used for loss calculation
        if self.flip:
            # FLIP simulation
            self.target_data = state.density_field
            self.target = placeholder_like(self.target_data)

            self.loss = l2_loss(self.state_out.density_field - self.target) + self.force_weight * l2_loss(self.forces)

        else:
            # SDF simulation
            self.target_active = state.active_mask
            self.add_field("Target Active Mask", lambda: self.target_active)

            self.target_data = state.sdf
            self.target = placeholder_like(self.target_data)

            self.loss = l2_loss(self.state_out.sdf - self.target) + self.force_weight * l2_loss(self.forces)

        self.add_objective(self.loss, "Unsupervised_Loss")

        self.add_field("Trained Forces", lambda: self.sess.run(self.forces))
        self.add_field("Target", lambda: self.target_data)


    def step(self):
        if self.flip:
            print("Amount of particles:" + str(math.sum(self.liquid.density_field)))

            self.base_feed_dict.update({self.state_in.points: self.liquid.points})

        else:
            self.base_feed_dict.update({self.state_in.sdf: self.liquid.sdf})

        # Run optimization step
        self.base_feed_dict.update({
            self.state_in.active_mask: self.liquid.active_mask,self.state_in.velocity: self.liquid.velocity,
            self.target: self.target_data
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
        self.initial_density = zeros(self.liquid.grid.shape())
        self.initial_velocity = zeros(self.liquid.grid.staggered_shape())

        number_of_circles = np.random.randint(1, min(self.size)/2)
        centers = np.array([np.random.randint(i, size=number_of_circles) for i in self.size]).reshape([-1, len(self.size)])
        radii = np.random.uniform(0, min(self.size)/number_of_circles, size=number_of_circles)
        velocities = np.array([np.random.uniform(-min(self.size)/4, min(self.size)/4, size=number_of_circles) for _ in self.size]).reshape([-1, len(self.size)])

        self.initial_density = insert_circles(self.initial_density, centers, radii)
        self.initial_velocity = StaggeredGrid(insert_circles(self.initial_velocity.staggered, centers, radii, velocities))
        

        if self.flip:
            self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)

            self.initial_velocity = np.random.uniform(-min(self.size)/4, min(self.size)/4, size=len(self.size))
            self.liquid.velocity = zeros_like(self.liquid.points) + self.initial_velocity

        else:
            particle_mask = create_binary_mask(self.initial_density, threshold=0)
            self.liquid._sdf, _ = extrapolate(self.initial_velocity, particle_mask, distance=self.distance)
            self.liquid._active_mask = particle_mask
            self.liquid.velocity = self.initial_velocity


        # Target state being random number of steps in the future of initial
        state = self.liquid.state

        # Set the gravity and velocities during the steps
        state._gravity = self.gravity
        state._velocity = self.initial_velocity

        self.target_steps = np.random.randint(1, 20)
        print("Target steps:" + str(self.target_steps)) 
        for i in range(self.target_steps):
            state = self.liquid.default_physics().step(state, dt=self.dt)
        
        if self.flip:
            self.target_data = state.density_field
        else:
            self.target_data = state.sdf
            self.target_active = state.active_mask

        self.time = 0



app = RandomLiquid().show(production=__name__ != "__main__", framerate=3, display=("Fluid", "Velocity"))
