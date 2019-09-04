from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *
from phi.physics.forcenet import *


class LiquidNetworkTraining(TFModel):
    def __init__(self):
        TFModel.__init__(self, "Network Training for pre-generated SDF Liquid simulation data", stride=1, learning_rate=1e-3, validation_batch_size=1)

        self.size = np.array([32, 40])
        domain = Domain(self.size, SLIPPERY)
        # Don't think timestep plays a role during training, but it's still needed for the computation graph.
        self.dt = 0.1
        self.gravity = -0.0

        self.initial_density = tf.zeros(domain.grid.shape())       # Initial density just needs to be a tensor so the simulation can run on tensorflow backend, but the value of initial density doesn't matter, will be overwritten later with sdf.
        self.initial_velocity = StaggeredGrid(placeholder(np.concatenate(([None], self.size+1, [len(self.size)]))))

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity)

        self.liquid.sdf = placeholder(np.concatenate(([None], self.size, [1])))

        # Train Neural Network to find forces
        self.target_sdf = placeholder_like(self.liquid.sdf)

        with self.model_scope():
            self.forces = forcenet2d_3x_16(self.liquid.sdf, self.liquid.velocity, self.target_sdf)
        self.liquid.trained_forces = self.forces

        self.state_out = self.liquid.default_physics().step(self.liquid.state, dt=self.dt)

        # Do multiple steps so the network learns how the liquid changes shape
        for _ in range(5):
            self.state_out = self.liquid.default_physics().step(self.state_out, dt=self.dt)

        # Two thresholds for the world_step and editable float force_weight
        self.force_weight = self.editable_float('Force_Weight', 1e-3, (1e-5, 1e3))


        self.loss = l2_loss(self.state_out.sdf - self.target_sdf) + self.force_weight * l2_loss(self.forces)

        self.add_objective(self.loss, "Unsupervised_Loss")

        self.add_field("Trained Forces", self.forces)
        self.add_field("Target", self.target_sdf)

        self.add_field("Initial", self.liquid.sdf)
        self.add_field("Velocity",self.liquid.velocity.staggered)

        self.set_data(
            train = Dataset.load('~/phi/model/sdf-datagen', range(450)), 
            val = Dataset.load('~/phi/model/sdf-datagen', range(450, 500)), 
            placeholders = (self.liquid.sdf, self.initial_velocity.staggered, self.target_sdf),
            channels = ('initial_sdf', 'initial_velocity_staggered', 'target_sdf')
            )



app = LiquidNetworkTraining().show(production=__name__ != "__main__", framerate=3, display=("Trained Forces", "Velocity"))
