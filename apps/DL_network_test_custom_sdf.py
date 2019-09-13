from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *
from phi.physics.forcenet import *


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
            field_index = [slice(None)] + math.unstack(index) + [slice(None)]

            if values is None:
                # Insert scalar density/fluid mask
                field[field_index] = 1
            else:
                # Insert vector field
                values_index = math.where(where_circle)[0]     # Always take first possible circle
                field[field_index] = values[values_index]

    return field



class LiquidNetworkTesting(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Network Testing for pre-generated SDF Liquid simulation data", stride=1, learning_rate=1e-3, validation_batch_size=1)

        # Load the model data from the training app, so we can test that network on testing simulation data.

        self.size = np.array([32, 40])
        domain = Domain(self.size, SLIPPERY)
        self.dt = 0.01
        self.gravity = -0.0

        self.liquid = world.SDFLiquid(state_domain=domain, density=0.0, velocity=0.0, gravity=self.gravity)

        # Train Neural Network to find forces
        self.target_sdf = placeholder(domain.grid.shape())
        self.state_in = placeholder_like(self.liquid.state)

        with self.model_scope():
            self.forces = forcenet2d_3x_16(self.state_in.sdf, self.state_in.velocity, self.target_sdf)
        self.state_in.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

        self.session.initialize_variables()


        self.initial_density = zeros(domain.grid.shape())
        # Initial velocity different for FLIP, so set it separately over there
        self.initial_velocity = zeros(domain.grid.staggered_shape())
        self.target_density = zeros(domain.grid.shape())


        #-------- INITIAL --------#

        ### CIRCLES ###
        # number_of_circles = 1
        # centers = np.array([16, 30])
        # radii = np.array([8])
        # velocities = np.array([0.0, 0.0])

        # self.initial_density = insert_circles(self.initial_density, centers, radii)
        # self.initial_velocity = StaggeredGrid(insert_circles(self.initial_velocity.staggered, centers, radii, velocities))

        ### OTHER SHAPES ###
        self.initial_density[:, self.size[-2] * 3 // 8 : self.size[-2] * 7 // 8, self.size[-1] * 1 // 8 : self.size[-1] * 3 // 8 - 1, :] = 1
        self.initial_density[:, self.size[-2] * 1 // 8 : self.size[-2] * 5 // 8, self.size[-1] * 5 // 8 : self.size[-1] * 7 // 8 - 1, :] = 1
        #self.initial_velocity.staggered[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 3 // 8 : size[-1] * 6 // 8 + 1, :] = [-2.0, -0.0]


        #-------- TARGET --------#

        ### CIRCLES ###
        # number_of_circles = 1
        # centers = np.array([16, 10])
        # radii = np.array([8])
        # velocities = np.array([0.0, 0.0])

        # self.target_density = insert_circles(self.target_density, centers, radii)

        ### OTHER SHAPES ###
        self.target_density[:, self.size[-2] * 3 // 8 : self.size[-2] * 5 // 8 - 1, self.size[-1] * 2 // 8 : self.size[-1] * 6 // 8, :] = 1

        self.distance = max(self.size)

        initial_mask = create_binary_mask(self.initial_density, threshold=0)
        self.initial_sdf_data, _ = extrapolate(self.initial_velocity, initial_mask, distance=self.distance)

        target_mask = create_binary_mask(self.target_density, threshold=0)
        self.target_sdf_data, _ = extrapolate(self.initial_velocity, target_mask, distance=self.distance)


        self.active_mask = create_binary_mask(self.initial_density, threshold=0)

        self.feed = {
            self.state_in.active_mask: self.active_mask,
            self.state_in.sdf: self.initial_sdf_data,
            self.state_in.velocity.staggered: self.initial_velocity.staggered, 
            self.target_sdf: self.target_sdf_data
            }

        self.loss = l2_loss(self.state_in.sdf - self.target_sdf)

        self.add_field("Trained Forces", lambda: self.session.run(self.forces, feed_dict=self.feed))
        self.add_field("Target SDF", lambda: self.session.run(self.target_sdf, feed_dict=self.feed))

        ones = math.ones_like(self.target_sdf)
        self.target_mask = math.where(self.target_sdf < 0.5, ones, 0.0 * ones)

        self.add_field("Target Fluid", lambda: self.session.run(self.target_mask, feed_dict=self.feed))

        self.add_field("Fluid", lambda: self.session.run(self.state_in.active_mask, feed_dict=self.feed))
        self.add_field("SDF", lambda: self.session.run(self.state_in.sdf, feed_dict=self.feed))
        self.add_field("Velocity", lambda: self.session.run(self.state_in.velocity.staggered, feed_dict=self.feed))


    def step(self):
        [active_mask, sdf, velocity_staggered] = self.session.run([self.state_out.active_mask, self.state_out.sdf, self.state_out.velocity.staggered], feed_dict=self.feed)

        self.feed.update({
            self.state_in.active_mask: active_mask,
            self.state_in.sdf: sdf,
            self.state_in.velocity.staggered: velocity_staggered
            })

        self.current_loss = self.session.run(self.loss, feed_dict=self.feed)


    def action_reset(self):
        self.feed = {
            self.state_in.active_mask: self.active_mask,
            self.state_in.sdf: self.initial_sdf_data,
            self.state_in.velocity.staggered: self.initial_velocity.staggered, 
            self.target_sdf: self.target_sdf_data
            }



app = LiquidNetworkTesting().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Fluid"))
