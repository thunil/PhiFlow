from phi.tf.flow import *
from phi.math.sampled import *

class ParticleBasedLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Particle-based Liquid DL", stride=3, learning_rate=1e-1)

        size = [32, 40]
        domain = Domain(size, SLIPPERY)
        self.particles_per_cell = 4
        self.dt = 0.1

        self.initial_density = zeros(domain.grid.shape())
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1

        #self.initial_velocity = [1.42, 0]
        self.initial_velocity = 0.0
        
        self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=-5.0, particles_per_cell=self.particles_per_cell)
        #world.Inflow(Sphere((10,32), 5), rate=0.2)

        # Forces to be trained are directly added onto velocity, therefore should have same shape.
        with self.model_scope():
            self.forces = tf.Variable(tf.zeros(self.liquid.points.shape), name="TrainedForces", trainable=True)
        self.reset_forces = self.forces.assign(tf.zeros(self.liquid.points.shape))

        # Set up the Tensorflow state and step
        # We do this manually because we need to add the trained forces
        self.sess = Session(Scene.create('liquid'))

        self.state_in = placeholder_like(self.liquid.state, particles=True)
        self.state_in.trained_forces = self.forces
        self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

        # Try to find a force to bring it to the target state
        target_density_data = zeros(domain.grid.shape())
        target_density_data[:, size[-2] * 2 // 8 : size[-2] * 6 // 8 - 0, size[-1] * 6 // 8 : size[-1] * 8 // 8 - 1, :] = 1
        #target_points_data = active_centers(target_density_data, self.particles_per_cell)
        self.target_state_density = tf.constant(target_density_data)

        self.force_weight = self.editable_float('Force_Weight', 1.0, (1e-5, 1e3))
        #self.loss = l2_loss(self.state_out.density_field - self.target_state_density) + self.force_weight * l2_loss(self.forces)
        self.loss = l2_loss(self.state_out.velocity)
        self.add_objective(self.loss, "Unsupervised_Loss")

        # Two thresholds for the world_step
        self.loss_threshold = EditableFloat('Loss_Threshold', 1e-1, (1e-5, 10))
        self.step_threshold = EditableFloat('Step_Threshold', 100, (1, 1e4))

        self.add_field("Trained Forces", lambda: grid(self.liquid.grid, self.liquid.points, self.sess.run(tf.slice(self.forces, [0,0,0], self.liquid.points.shape)), staggered=True))

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Density", lambda: self.liquid.density_field)
        self.add_field("Points", lambda: grid(self.liquid.grid, self.liquid.points, self.liquid.points))
        self.add_field("Velocity", lambda: self.liquid.velocity_field.staggered)
        self.add_field("Pressure", lambda: self.liquid.last_pressure)


    def step(self):
        print("Amount of particles:" + str(math.sum(self.liquid.density_field)))
        # Run optimization step
        self.base_feed_dict.update({
            self.state_in.active_mask: self.liquid.state.active_mask, self.state_in.points: self.liquid.state.points, self.state_in.velocity: self.liquid.state.velocity
            })
        TFModel.step(self)
        self.current_loss = self.sess.run(self.loss, self.base_feed_dict)
        # Use trained forces to do a step when loss is small enough
        if self.current_loss < self.loss_threshold or self.steps > self.step_threshold:
            self.steps = 0
            self.world_steps += 1
            self.liquid.trained_forces = self.sess.run(tf.slice(self.forces, [0,0,0], self.liquid.points.shape))
            world.step(dt=self.dt)


    def action_reset(self):
        # self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
        # self.liquid.velocity = zeros_like(self.liquid.points) + self.initial_velocity
        # self.sess.run(self.reset_forces)
        # self.time = 0

        #Temporary: Make this button do a step using the pretrained forces
        self.liquid.trained_forces = self.sess.run(self.forces)
        world.step(dt=self.dt)


app = ParticleBasedLiquid().show(production=__name__ != "__main__", framerate=3, display=("Density", "Trained Forces"))
