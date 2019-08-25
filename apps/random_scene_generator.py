from phi.tf.flow import *
from phi.math.sampled import *


def insert_circle(field, center, radius):
    """
Field should be a density/active mask field with shape [batch, coordinate_dimensions, 1]
Center should be given in index format (highest dimension first) and values should be integers that index into the field. 
    """
    assert field.shape[-1] == 1
    # TODO: Add support for center and radius to be arrays, so multiple circles can be drawn in one iteration.
    indices = indices_tensor(field).astype(int)
    indices = math.reshape(indices, [indices.shape[0], -1, indices.shape[-1]])[0]

    # Loop through entire field and mark the cells that are in the circle
    for index in indices:
        if math.sum((index - center)**2) <= radius**2:
            field_index = [slice(None)] + math.unstack(index) + [0]
            field[field_index] = 1

    return field



class RandomLiquid(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Random Liquid simulation generator", stride=3, learning_rate=1e-3)

        size = [32, 40]
        domain = Domain(size, SLIPPERY)
        self.dt = 0.1
        self.gravity = -4.0

        self.initial_density = zeros(domain.grid.shape())
        # self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        # self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1

        self.initial_density = insert_circle(self.initial_density, [20, 20], 4.0)


        self.sess = Session(Scene.create('liquid'))

        # Choose whether you want a particle-based FLIP simulation or a grid-based SDF simulation
        self.flip = False
        if self.flip:
            # FLIP simulation
            self.particles_per_cell = 4
            self.initial_velocity = 0.0
            
            self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity, particles_per_cell=self.particles_per_cell)

            self.state_in = placeholder_like(self.liquid.state, particles=True)
            self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

            self.add_field("Fluid", lambda: self.liquid.active_mask)
            self.add_field("Density", lambda: self.liquid.density_field)
            self.add_field("Points", lambda: grid(self.liquid.grid, self.liquid.points, self.liquid.points))
            self.add_field("Velocity", lambda: self.liquid.velocity_field.staggered)
            self.add_field("Pressure", lambda: self.liquid.pressure)

        else:
            # SDF simulation
            self.distance = max(size)
            self.initial_velocity = zeros(domain.grid.staggered_shape())

            self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity, distance=self.distance)

            self.state_in = placeholder_like(self.liquid.state)
            self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)


            self.add_field("Fluid", lambda: self.liquid.active_mask)
            self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
            self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
            self.add_field("Pressure", lambda: self.liquid.pressure)


    def step(self):
        if self.flip:
            print("Amount of particles:" + str(math.sum(self.liquid.density_field)))
        world.step(dt=self.dt)


    def action_reset(self):
        if self.flip:
            self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)
            self.liquid.velocity = zeros_like(self.liquid.points) + self.initial_velocity

        else:
            particle_mask = create_binary_mask(self.initial_density, threshold=0)
            self.liquid._sdf, _ = extrapolate(self.initial_velocity, particle_mask, distance=self.distance)
            self.liquid.domaincache._active = particle_mask
            self.liquid.velocity = self.initial_velocity

        self.time = 0



app = RandomLiquid().show(production=__name__ != "__main__", framerate=3, display=("Fluid", "Velocity"))
