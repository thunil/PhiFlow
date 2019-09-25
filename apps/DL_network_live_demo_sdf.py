from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *
from phi.physics.forcenet import *


def insert_circle(field, center, radius):
    indices = indices_tensor(field).astype(int)
    
    where_circle = math.expand_dims(math.sum((indices - center)**2, axis=-1) <= radius**2, axis=-1)
    field = math.where(where_circle, math.ones_like(field), field)

    return field



class LiquidNetworkSDFdemo(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Live Demo for SDF trained network", stride=1, learning_rate=1e-3, validation_batch_size=1)

        # Load the model data from the training app, so we can test that network on testing simulation data.

        self.size = np.array([32, 40])
        domain = Domain(self.size, SLIPPERY)
        self.dt = 0.01
        self.gravity = -0.0

        self.liquid = world.SDFLiquid(state_domain=domain, density=0.0, velocity=0.0, gravity=self.gravity)


        self.initial_density = zeros(domain.grid.shape())
        # Initial velocity different for FLIP, so set it separately over there
        self.initial_velocity = zeros(domain.grid.staggered_shape())
        self.target_density = zeros(domain.grid.shape())


        #-------- INITIAL --------#

        ### CIRCLES ###
        center = [16, 30]
        radius = 8

        self.initial_density = insert_circle(self.initial_density, center, radius)


        #-------- TARGET --------#

        self.target_x = self.editable_int('Target Circle X coordinate', int(self.size[1]//2), (0, self.size[1]))

        self.target_y = self.editable_int('Target Circle Y coordinate', int(self.size[0]//2), (0, self.size[0]))
        
        ### CIRCLES ###
        center = math.stack([self.target_y, self.target_x])
        radius = 8

        self.target_density = insert_circle(self.target_density, center, radius)

        self.distance = max(self.size)

        initial_mask = create_binary_mask(self.initial_density, threshold=0)
        self.initial_sdf_data, _ = extrapolate(self.initial_velocity, initial_mask, distance=self.distance)

        target_mask = create_binary_mask(self.target_density, threshold=0)
        self.target_sdf, _ = extrapolate(self.initial_velocity, target_mask, distance=self.distance)

        self.active_mask = create_binary_mask(self.initial_density, threshold=0)


        # Train Neural Network to find forces
        self.state_in = placeholder_like(self.liquid.state)

        with self.model_scope():
            self.forces = forcenet2d_3x_16(self.state_in.sdf, self.state_in.velocity, self.target_sdf)
        self.state_in.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

        #self.session.initialize_variables()

        self.feed = {
            self.state_in.active_mask: self.active_mask,
            self.state_in.sdf: self.initial_sdf_data,
            self.state_in.velocity.staggered: self.initial_velocity.staggered
            }

        self.feed.update(self.editable_values_dict())

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
        self.feed.update(self.editable_values_dict())

        self.current_loss = self.session.run(self.loss, feed_dict=self.feed)


    def action_reset(self):
        self.feed.update({
            self.state_in.active_mask: self.active_mask,
            self.state_in.sdf: self.initial_sdf_data,
            self.state_in.velocity.staggered: self.initial_velocity.staggered, 
            self.target_sdf: self.target_sdf_data
            })
        self.feed.update(self.editable_values_dict())



app = LiquidNetworkSDFdemo().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Fluid"))
