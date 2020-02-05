from unittest import TestCase

import numpy as np
from phi.geom import AABox
#from phi.tf import tf
# pylint: disable-msg = redefined-builtin, redefined-outer-name, unused-wildcard-import, wildcard-import
from phi.math import *
from phi.physics.plasma import *

#if tf.__version__[0] == '2':
#    print('Adjusting for tensorflow 2.0')
#    tf = tf.compat.v1
#    tf.disable_eager_execution()

INPUT_CHANGED_MSG = "Side Effect detected: solver causes input array to change"
OUTPUT_NOT_UNIFORM = "Uniform input does not yield uniform output. Recommendation: visualize array: import matplotlib.pyplot as plt; plt.imshow(array_out); plt.colorbar()"


class TestMath(TestCase):

    def test_arakawa(self):
        """ Test the Arakawa Implementation using two functions """
        # Test Sizes
        size = 64
        shape = (size, size)
        max_range = np.pi/2
        step_size = max_range/(shape[0]-2) # comparison is on full scale
        # Test Functions
        f = lambda x, y: x*x+y
        g = lambda x, y: np.sin(x)*y
        # True value: dx(x*x+y)*dy(sin(x)*y) - dy(x*x+y)*dx(sin(x)*y)
        h = lambda x, y: (2*x)*(np.sin(x)) - (1)*(np.cos(x)*y)
        # Build Arrays
        f_arr = np.empty(shape)
        g_arr = np.empty(shape)
        h_arr = np.empty(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                f_arr[i, j] = f(i*step_size, j*step_size)
                g_arr[i, j] = g(i*step_size, j*step_size)
                h_arr[i, j] = h(i*step_size, j*step_size)
        # Compute Arakawa
        res = arakawa(f_arr, g_arr, d=step_size)
        # Calculate Percentage Difference
        mask = (h_arr[1:-1, 1:-1] != 0)
        dev = np.abs(np.divide((h_arr - res)[1:-1, 1:-1][mask], h_arr[1:-1, 1:-1][mask]))
        # Check it is in an acceptable boundary
        self.assertLess(np.max(dev), 0.7)
    
    def test_laplace(self):
        """ test laplace for cylindrical behavior """
        # Setup uniform 3D Array
        N = 16
        array_in = np.ones((N, N, N))
        # Compute laplace for each dimension for circular
        laplace_axis_z = laplace(array_in, padding='circular', axes=[0])
        laplace_axis_y = laplace(array_in, padding='circular', axes=[1])
        laplace_axis_x = laplace(array_in, padding='circular', axes=[2])
        self.assertEqual(np.max(np.abs(laplace_axis_z-laplace_axis_z.mean())), 0, OUTPUT_NOT_UNIFORM)
        self.assertEqual(np.max(np.abs(laplace_axis_y-laplace_axis_y.mean())), 0, OUTPUT_NOT_UNIFORM)
        self.assertEqual(np.max(np.abs(laplace_axis_x-laplace_axis_x.mean())), 0, OUTPUT_NOT_UNIFORM)
        # Replicate
        laplace_axis_z = laplace(array_in, padding='replicate', axes=[0])
        laplace_axis_y = laplace(array_in, padding='replicate', axes=[1])
        laplace_axis_x = laplace(array_in, padding='replicate', axes=[2])
        self.assertEqual(np.max(np.abs(laplace_axis_z-laplace_axis_z.mean())), 0, OUTPUT_NOT_UNIFORM)
        self.assertEqual(np.max(np.abs(laplace_axis_y-laplace_axis_y.mean())), 0, OUTPUT_NOT_UNIFORM)
        self.assertEqual(np.max(np.abs(laplace_axis_x-laplace_axis_x.mean())), 0, OUTPUT_NOT_UNIFORM)
        # Wrap
        laplace_axis_z = laplace(array_in, padding='wrap', axes=[0])
        laplace_axis_y = laplace(array_in, padding='wrap', axes=[1])
        laplace_axis_x = laplace(array_in, padding='wrap', axes=[2])
        self.assertEqual(np.max(np.abs(laplace_axis_z-laplace_axis_z.mean())), 0, OUTPUT_NOT_UNIFORM)
        self.assertEqual(np.max(np.abs(laplace_axis_y-laplace_axis_y.mean())), 0, OUTPUT_NOT_UNIFORM)
        self.assertEqual(np.max(np.abs(laplace_axis_x-laplace_axis_x.mean())), 0, OUTPUT_NOT_UNIFORM)
