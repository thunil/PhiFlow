from phi import math
from phi.math.nd import *
from phi.math.geom import *
from phi.physics.gridliquid import *
import numpy as np


def particles_to_grid(griddef, points, values=None, duplicate_handling='mean', staggered=False):
    valid_indices = math.to_int(math.floor(points))
    valid_indices = math.minimum(math.maximum(0, valid_indices), griddef.resolution-1)
    # Correct format for math.scatter
    valid_indices = batch_indices(valid_indices)

    # Duplicate Handling always add except for active mask, but we can construct that from the density mask.
    ones = math.expand_dims(math.prod(math.ones_like(points), axis=-1), axis=-1)

    density =  math.scatter(points, valid_indices, ones, griddef.shape(1), duplicates_handling='add')

    if values is None:
        return density

    else:
        if staggered:
            dims = range(len(griddef.resolution))
            # Staggered grids only for vector fields
            assert values.shape[-1] == len(dims)

            active_mask = create_binary_mask(density, threshold=0)
            mask = math.pad(active_mask, [[0, 0]] + [[1, 1]] * spatial_rank(active_mask) + [[0, 0]], "constant")
            
            result = []
            oneD_ones = math.unstack(math.ones_like(values), axis=-1)[0]
            staggered_shape = [i+1 for i in griddef.resolution]

            for d in dims: 
                staggered_offset = math.stack([(0.5 * oneD_ones if i == d else 0.0 * oneD_ones) for i in dims], axis=-1)

                indices = math.to_int(math.floor(points + staggered_offset))
                
                valid_indices = math.maximum(0, math.minimum(indices, griddef.resolution))
                valid_indices = batch_indices(valid_indices)

                values_d = math.expand_dims(math.unstack(values, axis=-1)[d], axis=-1)
                result.append(math.scatter(points, valid_indices, values_d, [indices.shape[0]] + staggered_shape + [1], duplicates_handling='mean'))

                d_slice = tuple([(slice(0, -2) if i == d else slice(1,-1)) for i in dims])
                active_mask = math.minimum(mask[(slice(None),) + d_slice + (slice(None),)], active_mask)
            
            grid_values = StaggeredGrid(math.concat(result, axis=-1))
            # Fix values at lower boundary of liquids (using StaggeredGrid these might not receive a value, so we replace it with a value inside the liquid)
            _, grid_values = extrapolate(grid_values, active_mask, distance=2)

            return grid_values

        else:
            return math.scatter(points, valid_indices, values, griddef.shape(values.shape[-1]), duplicates_handling=duplicate_handling)


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