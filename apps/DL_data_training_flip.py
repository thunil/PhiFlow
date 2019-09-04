from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *
from phi.physics.forcenet import *


class LiquidNetworkTraining(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Network Training for pre-generated FLIP Liquid simulation data", stride=1, learning_rate=1e-3, validation_batch_size=1)

        self.size = np.array([32, 40])
        domain = Domain(self.size, SLIPPERY)
        self.particles_per_cell = 4
        # Don't think timestep plays a role during training, but it's still needed for the computation graph.
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

        # Do multiple steps so the network learns how the liquid changes shape
        for _ in range(5):
            self.state_out = self.liquid.default_physics().step(self.state_out, dt=self.dt)

        # Two thresholds for the world_step and editable float force_weight
        self.force_weight = self.editable_float('Force_Weight', 1e-2, (1e-5, 1e3))

        # For larger initial velocities we need a large force to work against it.
        self.loss = l2_loss(self.state_out.density_field - self.target_density) + self.force_weight * math.divide_no_nan(l2_loss(self.forces), math.max(self.initial_velocity.staggered))

        self.add_objective(self.loss, "Unsupervised_Loss")

        self.add_field("Trained Forces", self.forces)
        self.add_field("Target", self.target_density)

        self.add_field("Fluid", self.liquid.active_mask)
        self.add_field("Density", self.liquid.density_field)
        # self.add_field("Points", grid(self.liquid.grid, self.liquid.points, self.liquid.points))
        self.add_field("Velocity", self.liquid.velocity_field.staggered)

        self.set_data(
            train = Dataset.load('~/phi/model/flip-datagen', range(1700)), 
            #val = Dataset.load('~/phi/model/flip-datagen', range(100)), 
            placeholders = (self.initial_density, self.initial_velocity.staggered, self.target_density),
            channels = ('initial_density', 'initial_velocity_staggered', 'target_density')
            )


app = LiquidNetworkTraining().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Target"))
