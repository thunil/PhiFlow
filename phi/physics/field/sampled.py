from phi.physics.gridliquid import extrapolate
from phi.physics.domain import Domain
from phi.physics.material import SLIPPERY
from phi import struct, math
import numpy as np
from .field import Field
from .grid import CenteredGrid
from .staggered_grid import StaggeredGrid, unstack_staggered_tensor
from .flag import SAMPLE_POINTS


@struct.definition()
class SampledField(Field):

    def __init__(self, name, sample_points, data=1, mode='add', point_count=None, **kwargs):
        Field.__init__(self, **struct.kwargs(locals(), ignore=['point_count']))
        self._point_count = point_count

    def sample_at(self, points, collapse_dimensions=True):
        raise NotImplementedError()

    def at(self, other_field, collapse_dimensions=True, force_optimization=False, return_self_if_compatible=False):
        if isinstance(other_field, SampledField) and other_field.sample_points is self.sample_points:
            return self
        elif isinstance(other_field, CenteredGrid):
            return self.center_sample(other_field).at(other_field, collapse_dimensions, force_optimization, return_self_if_compatible)

        elif isinstance(other_field, StaggeredGrid):
            return self.stagger_sample(other_field).at(other_field, collapse_dimensions, force_optimization, return_self_if_compatible)

        else:
            return self

    def center_sample(self, field):
        """
    Samples the SampledField at the centers of a CenteredGrid. The value of the field parameter does not influence the result, only the resolution and box of the field.
        :param field: Field that contains valid resolution and box
        :return : sampled CenteredGrid
        """
        valid_indices = math.to_int(math.floor(self.sample_points))
        valid_indices = math.minimum(math.maximum(0, valid_indices), field.resolution-1)
        # Correct format for math.scatter
        valid_indices = batch_indices(valid_indices)

        scattered = math.scatter(self.sample_points, valid_indices, self.data, math.concat([[valid_indices.shape[0]], field.resolution, [1]], axis=-1), duplicates_handling=self.mode)

        return field.copied_with(data=scattered)


    def stagger_sample(self, field):
        """
    Samples the SampledField on a StaggeredGrid (see where the values of StaggeredGrids are defined in the documentation). The value of the field parameter does not influence the result, only the resolution and box of the field.
        :param field: Field that contains valid resolution and box
        :return : sampled StaggeredGrid
        """
        valid_indices = math.to_int(math.floor(self.sample_points))
        valid_indices = math.minimum(math.maximum(0, valid_indices), field.resolution-1)
        # Correct format for math.scatter
        valid_indices = batch_indices(valid_indices)

        active_mask = math.scatter(self.sample_points, valid_indices, 1, math.concat([[valid_indices.shape[0]], field.resolution, [1]], axis=-1), duplicates_handling='any')

        mask = math.pad(active_mask, [[0, 0]] + [[1, 1]] * self.rank + [[0, 0]], "constant")

        if isinstance(self.data, (int, float, np.ndarray)):
            values = math.zeros_like(self.sample_points) + self.data
        else:
            values = self.data
        
        result = []
        oneD_ones = math.unstack(math.ones_like(values), axis=-1)[0]
        staggered_shape = [i+1 for i in field.resolution]

        dims = range(field.rank)
        for d in dims: 
            staggered_offset = math.stack([(0.5 * oneD_ones if i == d else 0.0 * oneD_ones) for i in dims], axis=-1)

            indices = math.to_int(math.floor(self.sample_points + staggered_offset))
            
            valid_indices = math.maximum(0, math.minimum(indices, field.resolution))
            valid_indices = batch_indices(valid_indices)

            values_d = math.expand_dims(math.unstack(values, axis=-1)[d], axis=-1)
            result.append(math.scatter(self.sample_points, valid_indices, values_d, [indices.shape[0]] + staggered_shape + [1], duplicates_handling=self.mode))

            d_slice = tuple([(slice(0, -2) if i == d else slice(1,-1)) for i in dims])
            u_slice = tuple([(slice(2, None) if i == d else slice(1,-1)) for i in dims])
            active_mask = math.minimum(mask[(slice(None),) + d_slice + (slice(None),)], active_mask)
            active_mask = math.minimum(mask[(slice(None),) + u_slice + (slice(None),)], active_mask)
        
        staggered_tensor_prep = unstack_staggered_tensor(math.concat(result, axis=-1))
        grid_values = StaggeredGrid.from_tensors('staggered', staggered_tensor_prep)
        # Fix values at boundary of liquids (using StaggeredGrid these might not receive a value, so we replace it with a value inside the liquid)
        _, grid_values = extrapolate(Domain(field.resolution, SLIPPERY, field.box), grid_values, active_mask, distance=2)

        return grid_values


    @struct.attr()
    def data(self, data):
        if isinstance(data, (tuple, list, np.ndarray)):
            data = math.zeros_like(self.sample_points) + data
        return data

    @struct.prop(default='add')
    def mode(self, mode):
        assert mode in ('add', 'mean', 'any')
        return mode

    @struct.attr()
    def sample_points(self, sample_points):
        assert math.ndims(sample_points) == 3, sample_points.shape
        return sample_points

    @property
    def shape(self):
        with struct.anytype():
            if math.ndims(self.data) > 0:
                data_shape = (self._batch_size, self._point_count, self.component_count)
            else:
                data_shape = ()
            return self.copied_with(data=data_shape,
                                    sample_points=(self._batch_size, self._point_count, self.rank))

    @property
    def rank(self):
        return math.staticshape(self.sample_points)[-1]

    @property
    def component_count(self):
        if math.ndims(self.data) == 0:
            return 1
        return math.shape(self.data)[-1]

    def unstack(self):
        raise NotImplementedError()

    @property
    def points(self):
        if SAMPLE_POINTS in self.flags or self.sample_points is self.data:
            return self
        return SampledField(self.name+'.points', self.sample_points, self.sample_points, flags=[SAMPLE_POINTS])

    def compatible(self, other_field):
        if not other_field.has_points:
            return True
        if isinstance(other_field, SampledField) and other_field.sample_points is self.sample_points:
            return True
        return False

    def __repr__(self):
        return '%s[%sx(%d), %dD]' % (self.__class__.__name__, self._point_count if self._point_count is not None else '?', self.component_count, self.rank)


def batch_indices(indices):
    """
Reshapes the indices, such that aside from indices they also contain batch number. For example the entry (32, 40) as coordinates of batch 2 will become (2, 32, 40).
Transform shape (b, p, d) to (b, p, d+1) where batch size is b, number of particles is p and number of dimensions is d. 
    """
    batch_size = indices.shape[0]
    out_spatial_rank = len(indices.shape) - 2
    out_spatial_size = math.shape(indices)[1:-1]

    batch_range = math.backend.choose_backend(indices).range(batch_size)
    batch_ids = math.reshape(batch_range, [batch_size] + [1] * out_spatial_rank)
    tile_shape = math.pad(out_spatial_size, [[1,0]], constant_values=1)
    batch_ids = math.expand_dims(math.tile(batch_ids, tile_shape), axis=-1)

    return math.concat((batch_ids, indices), axis=-1)


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
