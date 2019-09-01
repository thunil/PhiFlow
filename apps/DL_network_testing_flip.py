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

        #self.sess = Session(Scene.create('liquid'))
        self.session.initialize_variables()

        channels = ('initial_density', 'initial_velocity_staggered', 'target_density')

        # Set the first batch data
        self._testing_set = Dataset.load('~/phi/model/flip-datagen', range(20,30))
        self._test_reader = BatchReader(self._testing_set, channels)
        self._test_iterator = self._test_reader.all_batches(batch_size=1, loop=True)

        [self.initial_density_data, self.initial_velocity_staggered_data, self.target_density_data] = next(self._test_iterator)

        self.feed = {
            self.initial_density: self.initial_density_data,
            self.initial_velocity.staggered: self.initial_velocity_staggered_data, 
            self.target_density: self.target_density_data
            }


        self.add_field("Trained Forces", self.session.run(self.forces, self.feed))
        self.add_field("Target", self.session.run(self.target_density, self.feed))

        self.add_field("Fluid", self.session.run(self.liquid.active_mask, self.feed))
        self.add_field("Density", self.session.run(self.liquid.density_field, self.feed))
        # self.add_field("Points", grid(self.liquid.grid, self.liquid.points, self.liquid.points))
        # self.add_field("Velocity", self.liquid.velocity_field.staggered)



    def step(self):
        [self.initial_density_data, self.initial_velocity_staggered_data] = self.session.run([self.state_out.density_field, self.state_out.velocity_field.staggered], feed_dict=self.feed)

        self.feed = {
            self.initial_density: self.initial_density_data,
            self.initial_velocity.staggered: self.initial_velocity_staggered_data, 
            self.target_density: self.target_density_data
            }

        self.world_steps += 1


    def action_reset(self):
        # Update data to a new simulation of testing set
        [self.initial_density_data, self.initial_velocity_staggered_data, self.target_density_data] = next(self._test_iterator)

        self.feed = {
            self.initial_density: self.initial_density_data,
            self.initial_velocity.staggered: self.initial_velocity_staggered_data, 
            self.target_density: self.target_density_data
            }



app = LiquidNetworkTesting().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Fluid"))
