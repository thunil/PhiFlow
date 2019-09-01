from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *


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


#world.batch_size = 4


class RandomLiquid(TFModel):

    def __init__(self):
        # Choose whether you want a particle-based FLIP simulation or a grid-based SDF simulation
        self.flip = True
        
        if self.flip:
            TFModel.__init__(self, "FLIP datagen", stride=1, learning_rate=1e-3)
        else:
            TFModel.__init__(self, "SDF datagen", stride=1, learning_rate=1e-3)

        self.size = [32, 40]
        domain = Domain(self.size, SLIPPERY)
        self.dt = 0.1
        self.gravity = -4.0

        self.record_steps = 12

        self.initial_density = zeros(domain.grid.shape())
        # Initial velocity different for FLIP, so set it separately over there
        self.initial_velocity = zeros(domain.grid.staggered_shape())

        number_of_circles = np.random.randint(1, min(self.size)/2)
        centers = np.array([np.random.randint(i, size=number_of_circles) for i in self.size]).reshape([-1, len(self.size)])
        radii = np.random.uniform(1, min(self.size)/number_of_circles, size=number_of_circles)
        velocities = np.array([np.random.uniform(-min(self.size)/4, min(self.size)/4, size=number_of_circles) for _ in self.size]).reshape([-1, len(self.size)])

        self.initial_density = insert_circles(self.initial_density, centers, radii)
        self.initial_velocity = StaggeredGrid(insert_circles(self.initial_velocity.staggered, centers, radii, velocities))


        self.sess = Session(Scene.create('liquid'))
        if self.flip:
            # FLIP simulation
            self.particles_per_cell = 4
            self.initial_velocity = np.random.uniform(-min(self.size)/4, min(self.size)/4, size=len(self.size))
            
            self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity, particles_per_cell=self.particles_per_cell)

            self.state_in = placeholder_like(self.liquid.state, particles=True)
            self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

            self.scene.write_sim_frame([self.liquid.density_field], ['target_density'], frame=1)

            self.add_field("Fluid", lambda: self.liquid.active_mask)
            self.add_field("Density", lambda: self.liquid.density_field)
            self.add_field("Points", lambda: grid(self.liquid.grid, self.liquid.points, self.liquid.points))
            self.add_field("Velocity", lambda: self.liquid.velocity_field.staggered)
            self.add_field("Pressure", lambda: self.liquid.pressure)

        else:
            # SDF simulation
            self.distance = max(self.size)

            self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity, distance=self.distance)

            self.state_in = placeholder_like(self.liquid.state)
            self.state_out = self.liquid.default_physics().step(self.state_in, dt=self.dt)

            self.scene.write_sim_frame([self.liquid.sdf], ['target_sdf'], frame=1)

            self.add_field("Fluid", lambda: self.liquid.active_mask)
            self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
            self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
            self.add_field("Pressure", lambda: self.liquid.pressure)


    def step(self):
        # TODO: record_steps doesn't quite work with "Run Sequence" button of Dash GUI
        if self.flip:
            print("Amount of particles:" + str(math.sum(self.liquid.density_field)))

            if self.steps >= self.record_steps:
                self.scene.write_sim_frame([self.liquid.density_field, self.liquid.velocity_field.staggered], ['initial_density', 'initial_velocity_staggered'], frame=1)

                self.new_scene()
                self.steps = 0
                self.action_reset()
                self.info('Starting data generation in scene %s' % self.scene)
                self.record_steps = np.random.randint(2, 20)

                self.scene.write_sim_frame([self.liquid.density_field], ['target_density'], frame=1)
            else:
                world.step(dt=self.dt)

        else:
            if self.steps >= self.record_steps:
                self.scene.write_sim_frame([self.liquid.sdf, self.liquid.velocity.staggered], ['initial_sdf', 'initial_velocity_staggered'], frame=1)

                self.new_scene()
                self.steps = 0
                self.action_reset()
                self.info('Starting data generation in scene %s' % self.scene)
                self.record_steps = np.random.randint(2, 20)

                self.scene.write_sim_frame([self.liquid.sdf], ['target_sdf'], frame=1)
            else:
                world.step(dt=self.dt)


    def action_reset(self):
        self.initial_density = zeros(self.liquid.grid.shape())
        self.initial_velocity = zeros(self.liquid.grid.staggered_shape())

        number_of_circles = np.random.randint(1, min(self.size)/2)
        centers = np.array([np.random.randint(i, size=number_of_circles) for i in self.size]).reshape([-1, len(self.size)])
        radii = np.random.uniform(0, min(self.size)/number_of_circles, size=number_of_circles)
        velocities = np.array([np.random.uniform(-min(self.size)/4, min(self.size)/4, size=number_of_circles) for _ in self.size]).reshape([-1, len(self.size)])

        self.initial_density = insert_circles(self.initial_density, centers, radii)
        self.initial_velocity = StaggeredGrid(insert_circles(self.initial_velocity.staggered, centers, radii, velocities))
        

        if self.flip:
            self.liquid.points = random_grid_to_coords(self.initial_density, self.particles_per_cell)

            self.initial_velocity = np.random.uniform(-min(self.size)/4, min(self.size)/4, size=len(self.size))
            self.liquid.velocity = zeros_like(self.liquid.points) + self.initial_velocity

        else:
            particle_mask = create_binary_mask(self.initial_density, threshold=0)
            self.liquid._sdf, _ = extrapolate(self.initial_velocity, particle_mask, distance=self.distance)
            self.liquid._active_mask = particle_mask
            self.liquid.velocity = self.initial_velocity

        self.time = 0



app = RandomLiquid().show(production=__name__ != "__main__", framerate=3, display=("Fluid", "Velocity"))
