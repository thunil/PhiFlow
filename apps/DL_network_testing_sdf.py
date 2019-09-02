from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *
from phi.physics.forcenet import *


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

        # Set the first batch data
        channels = ('initial_sdf', 'initial_velocity_staggered', 'target_sdf')

        self._testing_set = Dataset.load('~/phi/model/sdf-datagen', range(480,520))
        self._test_reader = BatchReader(self._testing_set, channels)
        self._test_iterator = self._test_reader.all_batches(batch_size=1, loop=True)

        [initial_sdf_data, initial_velocity_staggered_data, target_sdf_data] = next(self._test_iterator)

        ones = math.ones_like(initial_sdf_data)
        active_mask = math.where(initial_sdf_data < 0.5, ones, 0.0 * ones)

        self.feed = {
            self.state_in.active_mask: active_mask,
            self.state_in.sdf: initial_sdf_data,
            self.state_in.velocity.staggered: initial_velocity_staggered_data, 
            self.target_sdf: target_sdf_data
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
        [initial_sdf_data, initial_velocity_staggered_data, target_sdf_data] = next(self._test_iterator)

        ones = math.ones_like(initial_sdf_data)
        active_mask = math.where(initial_sdf_data < 0.5, ones, 0.0 * ones)

        self.feed = {
            self.state_in.active_mask: active_mask,
            self.state_in.sdf: initial_sdf_data,
            self.state_in.velocity.staggered: initial_velocity_staggered_data, 
            self.target_sdf: target_sdf_data
            }



app = LiquidNetworkTesting().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Target Fluid"))
