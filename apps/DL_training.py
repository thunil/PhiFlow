from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *
from phi.physics.forcenet import *

# Set the simulation type: either FLIP or SDF
mode = 'SDF'

class LiquidNetworkTraining(TFModel):
    def __init__(self):
        TFModel.__init__(self, 'Liquid DL Training', "Network training for pre-generated %s liquid simulation data" % mode, stride=1, learning_rate=1e-3, validation_batch_size=1)

        self.size = np.array([64, 96])
        domain = Domain(self.size, SLIPPERY)
        self.dataset_size = 500
        
        # Don't think timestep plays a role during training, but it's still needed for the computation graph.
        self.dt = 0.01
        self.gravity = -0.0
    

        if mode == 'FLIP':
            self.particles_per_cell = 4

            self.initial_density = placeholder(np.concatenate(([None], self.size, [1])))
            self.initial_velocity = StaggeredGrid(placeholder(np.concatenate(([None], self.size+1, [len(self.size)]))))

            particle_points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
            particle_velocity = grid_to_particles(domain.grid, particle_points, self.initial_velocity, staggered=True)

            # Initialization doesn't matter, training data is fed later
            self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=particle_velocity, gravity=self.gravity, particles_per_cell=self.particles_per_cell)

            # Train Neural Network to find forces
            self.target_density = placeholder(domain.grid.shape())

            with self.model_scope():
                self.forces = forcenet2d_3x_16(self.initial_density, self.initial_velocity, self.target_density)

            self.force_weight = self.editable_float('Force_Weight', 1e-2, (1e-5, 1e3))

        else:
            # Initial density just needs to be a tensor so the simulation can run on tensorflow backend, but the value of initial density doesn't matter, will be overwritten later with sdf.
            self.initial_density = tf.zeros(domain.grid.shape())       
            self.initial_velocity = StaggeredGrid(placeholder(np.concatenate(([None], self.size+1, [len(self.size)]))))

            self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity)

            self.liquid.sdf = placeholder(np.concatenate(([None], self.size, [1])))

            # Train Neural Network to find forces
            self.target_sdf = placeholder_like(self.liquid.sdf)

            with self.model_scope():
                self.forces = forcenet2d_3x_16(self.liquid.sdf, self.liquid.velocity, self.target_sdf)

            self.force_weight = self.editable_float('Force_Weight', 1e-1, (1e-5, 1e3))


        self.liquid.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.liquid.state, dt=self.dt)

        # Do multiple steps so the network learns how the liquid changes shape
        for _ in range(4):
            self.state_out = self.liquid.default_physics().step(self.state_out, dt=self.dt)


        if mode == 'FLIP':
            ### USING DENSITY FIELD FOR LOSS
            #self.loss = l2_loss(self.state_out.density_field - self.target_density) + self.force_weight * math.divide_no_nan(l2_loss(self.forces), math.max(self.initial_velocity.staggered))


            ### USING TILING TO CREATE TARGET POINTS
            self.target_points = active_centers(self.target_density, 1)
            number_particles = math.shape(self.state_out.points)[1]

            self.target_points = tf.tile(self.target_points, [1, self.particles_per_cell+2, 1])
            self.target_points = tf.slice(self.target_points, (0,0,0), (1, number_particles, 2))


            ### CREATING MORE THAN NECESSARY, THEN SHRINK TO CREATE TARGET POINTS
            # self.target_points = active_centers(self.target_density, self.particles_per_cell+2)
            # self.target_points = tf.slice(self.target_points, (0,0,0), (1, math.shape(self.state_out.points)[1], 2))

            self.loss = l2_loss(self.state_out.points - self.target_points) + self.force_weight * math.divide_no_nan(l2_loss(self.forces), math.max(self.initial_velocity.staggered))


            self.add_objective(self.loss, "Unsupervised_Loss")

            self.add_field("Trained Forces", self.forces)
            self.add_field("Target", self.target_density)

            self.add_field("Fluid", self.liquid.active_mask)
            self.add_field("Initial", self.liquid.density_field)
            self.add_field("Velocity", self.liquid.velocity_field.staggered)

            self.set_data(
                train = Dataset.load('~/phi/model/flip-datagen-3', range(self.dataset_size)), 
                placeholders = (self.initial_density, self.initial_velocity.staggered, self.target_density),
                channels = ('initial_density', 'initial_velocity_staggered', 'target_density')
                )

        else:
            self.loss = l2_loss(self.state_out.sdf - self.target_sdf) + self.force_weight * math.divide_no_nan(l2_loss(self.forces), math.max(self.initial_velocity.staggered))

            self.add_objective(self.loss, "Unsupervised_Loss")

            self.add_field("Trained Forces", self.forces)
            self.add_field("Target", self.target_sdf)

            self.add_field("Initial", self.liquid.sdf)
            self.add_field("Velocity",self.liquid.velocity.staggered)

            self.set_data(
                train = Dataset.load('~/phi/model/sdf-datagen', range(self.dataset_size)),  
                placeholders = (self.liquid.sdf, self.initial_velocity.staggered, self.target_sdf),
                channels = ('initial_sdf', 'initial_velocity_staggered', 'target_sdf')
                )


app = LiquidNetworkTraining().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Velocity"))
