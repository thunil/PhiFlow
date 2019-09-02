from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *
from phi.physics.forcenet import *


class LiquidNetworkTesting(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Network Testing for pre-generated FLIP Liquid simulation data", stride=1, learning_rate=1e-3, validation_batch_size=1)

        # Load the model data from the training app, so we can test that network on testing simulation data.

        self.size = np.array([32, 40])
        domain = Domain(self.size, SLIPPERY)
        self.particles_per_cell = 4
        self.dt = 0.1
        self.gravity = -0.0

        self.liquid = world.FlipLiquid(state_domain=domain, density=0.0, velocity=0.0, gravity=self.gravity, particles_per_cell=self.particles_per_cell)

        # Train Neural Network to find forces
        self.target_density = placeholder(domain.grid.shape())
        self.state_in = placeholder_like(self.liquid.state, particles=True)

        with self.model_scope():
            self.forces = forcenet2d_3x_16(self.state_in.density_field, self.state_in.velocity_field, self.target_density)
        self.state_in.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

        self.session.initialize_variables()

        # Set the first batch data
        channels = ('initial_density', 'initial_velocity_staggered', 'target_density')

        self._testing_set = Dataset.load('~/phi/model/flip-datagen', range(20,30))
        self._test_reader = BatchReader(self._testing_set, channels)
        self._test_iterator = self._test_reader.all_batches(batch_size=1, loop=True)

        [initial_density_data, initial_velocity_staggered_data, target_density_data] = next(self._test_iterator)

        particle_points = random_grid_to_coords(initial_density_data, self.particles_per_cell)
        particle_velocity = grid_to_particles(domain.grid, particle_points, StaggeredGrid(initial_velocity_staggered_data), staggered=True)

        active_mask = create_binary_mask(initial_density_data, threshold=0)

        self.feed = {
            self.state_in.active_mask: active_mask,
            self.state_in.points: particle_points,
            self.state_in.velocity: particle_velocity, 
            self.target_density: target_density_data
            }

        self.add_field("Trained Forces", lambda: self.session.run(self.forces, feed_dict=self.feed))
        self.add_field("Target", lambda: self.session.run(self.target_density, feed_dict=self.feed))

        self.add_field("Fluid", lambda: self.session.run(self.state_in.active_mask, feed_dict=self.feed))
        self.add_field("Density", lambda: self.session.run(self.state_in.density_field, feed_dict=self.feed))
        self.add_field("Velocity", lambda: self.session.run(self.state_in.velocity_field.staggered, feed_dict=self.feed))



    def step(self):
        [active_mask, particle_points, particle_velocity] = self.session.run([self.state_out.active_mask, self.state_out.points, self.state_out.velocity], feed_dict=self.feed)

        self.feed.update({
            self.state_in.active_mask: active_mask,
            self.state_in.points: particle_points,
            self.state_in.velocity: particle_velocity
            })

        #self.world_steps += 1


    def action_reset(self):
        [initial_density_data, initial_velocity_staggered_data, target_density_data] = next(self._test_iterator)

        particle_points = random_grid_to_coords(initial_density_data, self.particles_per_cell)
        particle_velocity = grid_to_particles(self.liquid.grid, particle_points, StaggeredGrid(initial_velocity_staggered_data), staggered=True)

        active_mask = create_binary_mask(initial_density_data, threshold=0)

        self.feed.update({
            self.state_in.active_mask: active_mask,
            self.state_in.points: particle_points,
            self.state_in.velocity: particle_velocity, 
            self.target_density: target_density_data
            })



app = LiquidNetworkTesting().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Fluid"))
