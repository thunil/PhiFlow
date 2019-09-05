from __future__ import division

from phi.tf.flow import *
from phi.math.sampled import *


"""
This app will generate the necessary initial and target npz files to use for testing a network. Simply adjust the initial density/velocity and target density to the scenario you wish to test, and run the file.
"""


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


class CustomLiquid(TFModel):

    def __init__(self):
        # Choose whether you want a particle-based FLIP simulation or a grid-based SDF simulation
        self.flip = False
        
        if self.flip:
            TFModel.__init__(self, "FLIP custom", stride=1, learning_rate=1e-3)
        else:
            TFModel.__init__(self, "SDF custom", stride=1, learning_rate=1e-3)

        self.size = [32, 40]
        domain = Domain(self.size, SLIPPERY)
        self.dt = 0.1
        self.gravity = -9.81

        self.initial_density = zeros(domain.grid.shape())
        # Initial velocity different for FLIP, so set it separately over there
        self.initial_velocity = zeros(domain.grid.staggered_shape())
        self.target_density = zeros(domain.grid.shape())


        #-------- INITIAL --------#

        ### CIRCLES ###
        number_of_circles = 1
        centers = np.array([16, 10])
        radii = np.array([10])
        velocities = np.array([0.0, 0.0])

        self.initial_density = insert_circles(self.initial_density, centers, radii)
        self.initial_velocity = StaggeredGrid(insert_circles(self.initial_velocity.staggered, centers, radii, velocities))

        ### OTHER SHAPES ###
        #self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        #self.initial_velocity.staggered[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 3 // 8 : size[-1] * 6 // 8 + 1, :] = [-2.0, -0.0]


        #-------- TARGET --------#

        ### CIRCLES ###
        number_of_circles = 1
        centers = np.array([16, 30])
        radii = np.array([10])
        velocities = np.array([0.0, 0.0])

        self.target_density = insert_circles(self.target_density, centers, radii)

        ### OTHER SHAPES ###
        #self.target_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1

        self.add_field("Initial Density", lambda: self.initial_density)
        self.add_field("Target Density", lambda: self.target_density)
        self.add_field("Initial Velocity", lambda: self.initial_velocity.staggered)
        

        if self.flip:
            # FLIP simulation
            self.particles_per_cell = 4
            self.initial_velocity = [0.0, 0.0]
            
            self.liquid = world.FlipLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity, particles_per_cell=self.particles_per_cell)

            self.scene.write_sim_frame(
                [self.initial_density * self.particles_per_cell, 
                self.liquid.velocity_field.staggered, 
                self.target_density * self.particles_per_cell], 
                ['initial_density', 
                'initial_velocity_staggered', 
                'target_density'], 
                frame=1)

        else:
            # SDF simulation
            self.distance = max(self.size)

            self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=self.gravity, distance=self.distance)

            target_mask = create_binary_mask(self.target_density, threshold=0)
            target_sdf, _ = extrapolate(self.initial_velocity, target_mask, distance=self.distance)

            self.scene.write_sim_frame(
                [self.liquid.sdf, 
                self.initial_velocity.staggered, 
                target_sdf], 
                ['initial_sdf', 
                'initial_velocity_staggered', 
                'target_sdf'], 
                frame=1)

CustomLiquid()
