from phi import math
from phi.math.nd import *
from phi.math.geom import *
import numpy as np


def particles_to_grid(griddef, points, values=None, staggered=False, weight_factor=1.0):
    """
Projects particles onto the grid (along with values that the particles carry).

    :param weight_factor: defines how far values can influence a grid value, i.e. how blurry the grid should look.
    """
    def distance(x, y):
        """
    Euclidian distance (can be changed) between two tensors without square root to save computation resources.
    Input shapes to Output shape: (b, m, n, c) and (b, p, c) -> (b, m, n, p, 1)
        """
        # Expand dims makes sure that the resulting shape has a last dimension of 1 (scalar field)
        distance = math.sqrt(math.expand_dims(math.sum((x - y)**2, axis=-1), axis=-1))
        return distance


    if values is None:
        valid_indices = math.to_int(math.floor(points))
        valid_indices = math.minimum(math.maximum(0, valid_indices), griddef.resolution-1)
        # Correct format for math.scatter
        valid_indices = batch_indices(valid_indices)
        
        # Duplicate Handling always add except for active mask, but we can construct that from the density mask.
        ones = math.expand_dims(math.prod(math.ones_like(points), axis=-1), axis=-1)

        return math.scatter(points, valid_indices, ones, griddef.shape(1), duplicates_handling='add')

    else:
        # Shape multiplications: (b, m, n, p, 1) * (b, p, c) = (b, m, n, p, c)
        # We sum up along the "-2" axis (axis with size p) and have a resulting grid (b, m, n, c)
        # Abbreviations: b=batchsize, [m,n]=2D-grid-dimensions, p=number-of-particles, c=components-of-vector

        # Epsilon to prevent divide by 0 and small values
        epsilon = 1e-6

        # The following method could be used for density field when weight_factor is chosen small (less extrapolation, most cells too far away and receive value 0).

        if staggered:
            dims = range(len(griddef.resolution))
            # Staggered grids only for vector fields
            assert values.shape[-1] == len(dims)

            result = []
            for d in dims: 
                cell_staggered = griddef.staggered_points(d)
                # Correct format for distance calculation
                cell_staggered = math.expand_dims(cell_staggered, axis=-2)
                dist = distance(cell_staggered, points)       
                
                scaling_dist = math.exp(-dist/weight_factor)
                normalizing_factor = 1/(math.sum(scaling_dist, axis=-2) + epsilon)
                staggered_values = math.expand_dims(values[...,d], axis=-1)
                grid_values = math.sum(scaling_dist * staggered_values, axis=-2) * normalizing_factor

                grid_values = math.where(math.abs(grid_values) < epsilon, 0.0*grid_values, grid_values)

                result.append(grid_values)
            
            return StaggeredGrid(math.concat(result, axis=-1))

        else:
            cell_centers = griddef.center_points()
            # Correct format for distance calculation
            cell_centers = math.expand_dims(cell_centers, axis=-2)
            dist = distance(cell_centers, points)
            
            scaling_dist = math.exp(-dist/weight_factor)
            normalizing_factor = 1/(math.sum(scaling_dist, axis=-2) + epsilon)
            grid_values = math.sum(scaling_dist * values, axis=-2) * normalizing_factor

            grid_values = math.where(math.abs(grid_values) < epsilon, 0.0*grid_values, grid_values)

            return grid_values



def active_centers(array, particles_per_cell=1):
    index_array = []
    batch_size = math.staticshape(array)[0] if math.staticshape(array)[0] is not None else 1

    for batch in range(batch_size):
        indices = math.where(array[batch,...,0] > 0)
        indices = math.to_float(indices)

        # For Deep Learning simulations where the target state needs to have same particle count as initial state. For all other purposes this method should be called with particles_per_cell set to the default 1.
        temp = []
        for _ in range(particles_per_cell):
            # Uniform distribution over cell
            temp.append(indices)
        index_array.append(math.concat(temp, axis=0))
    try:
        index_array = math.stack(index_array)
    except ValueError:
        raise ValueError("all arrays in the batch must have the same number of active cells.")
    return index_array + 0.5


def random_grid_to_coords(array, particles_per_cell=1):
    index_array = []
    batch_size = math.staticshape(array)[0] if math.staticshape(array)[0] is not None else 1
    
    for batch in range(batch_size):
        indices = math.where(array[batch,...,0] > 0)
        indices = math.to_float(indices)

        temp = []
        for _ in range(particles_per_cell):
            # Uniform distribution over cell
            temp.append(indices + math.random_like(indices))
        index_array.append(math.concat(temp, axis=0))
    try:
        index_array = math.stack(index_array)
    except ValueError:
        raise ValueError("all arrays in the batch must have the same number of active cells.")
    
    return index_array


def grid_to_particles(griddef, points, values, staggered=False):
    if staggered:
        values = values.staggered
        dims = range(len(griddef.resolution))
        # Staggered grids only for vector fields
        assert values.shape[-1] == len(dims)

        result = []
        oneD_ones = math.unstack(math.ones_like(points), axis=-1)[0]
        for d in dims:
            staggered_offset = math.stack([(0.0 * oneD_ones if i == d else -0.5 * oneD_ones) for i in dims], axis=-1)

            indices = (points + staggered_offset)
            values_d = math.expand_dims(math.unstack(values, axis=-1)[d], axis=-1)

            result.append(math.resample(values_d, indices, boundary="REPLICATE"))

        return math.concat(result, axis=-1)

    else:
        return math.resample(values, points-0.5, boundary="REPLICATE")


def batch_indices(indices):
    """
Reshapes the indices, such that aside from indices they also contain batch number. For example the entry (32, 40) as coordinates of batch 2 will become (2, 32, 40).
Transform shape (b, p, d) to (b, p, d+1) where batch size is b, number of particles is p and number of dimensions is d. 
    """
    batch_size = indices.shape[0]
    out_spatial_rank = len(indices.shape) - 2
    out_spatial_size = math.shape(indices)[1:-1]

    batch_ids = math.reshape(math.range_like(indices, batch_size), [batch_size] + [1] * out_spatial_rank)
    tile_shape = math.pad(out_spatial_size, [[1,0]], constant_values=1)
    batch_ids = math.expand_dims(math.tile(batch_ids, tile_shape), axis=-1)

    return math.concat((batch_ids, indices), axis=-1)