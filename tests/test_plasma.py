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

    def test_periodic_poisson_solver_SparseCG(self):
        """ check that periodic pressure solve of uniform array is uniform """
        N = 16
        # Define Domain
        from phi.physics.domain import Domain
        from phi.physics.material import PERIODIC
        domain = Domain((N, N), boundaries=(PERIODIC, PERIODIC))
        from phi.physics.pressuresolver.solver_api import FluidDomain
        fluid_domain = FluidDomain(domain)
        # Define uniform array of ones
        array_in_2d = np.zeros((1, N, N, 1))
        # Compute SparseCG
        from phi.physics.pressuresolver.sparse import SparseCG
        poisson_solver = SparseCG()
        array_out, iteration = poisson_solver.solve(array_in_2d, fluid_domain, guess=None)
        array_out = array_out.reshape((N, N))
        # Ensure Input has not been changed
        self.assertTrue(np.array_equal(np.zeros((1, N, N, 1)), array_in_2d), INPUT_CHANGED_MSG)
        # Compare
        self.assertEqual(np.max(np.abs(array_out - array_out.mean())), 0, OUTPUT_NOT_UNIFORM)

    def test_periodic_poisson_solver_SparseSciPy(self):
        """ check that periodic pressure solve of uniform array is uniform """
        N = 16
        # Define Domain
        from phi.physics.domain import Domain
        from phi.physics.material import PERIODIC
        domain = Domain((N, N), boundaries=(PERIODIC, PERIODIC))
        from phi.physics.pressuresolver.solver_api import FluidDomain
        fluid_domain = FluidDomain(domain)
        # Define uniform array of ones
        array_in_2d = np.zeros((1, N, N, 1))
        # Compute SparseSciPy
        from phi.physics.pressuresolver.sparse import SparseSciPy
        poisson_solver = SparseSciPy()
        array_out, iteration = poisson_solver.solve(array_in_2d, fluid_domain, guess=None)
        array_out = array_out.reshape((N, N))
        # Ensure Input has not been changed
        self.assertTrue(np.array_equal(np.zeros((1, N, N, 1)), array_in_2d), INPUT_CHANGED_MSG)
        # Compare
        self.assertEqual(np.max(np.abs(array_out - array_out.mean())), 0, OUTPUT_NOT_UNIFORM)

    def test_periodic_poisson_solver_GeometricCG(self):
        """ check that periodic pressure solve of uniform array is uniform """
        N = 16
        # Define Domain
        from phi.physics.domain import Domain
        from phi.physics.material import PERIODIC
        domain = Domain((N, N), boundaries=(PERIODIC, PERIODIC))
        from phi.physics.pressuresolver.solver_api import FluidDomain
        fluid_domain = FluidDomain(domain)
        # Define uniform array of ones
        array_in_2d = np.zeros((1, N, N, 1))
        # Compute GeometricCG
        from phi.physics.pressuresolver.geom import GeometricCG
        poisson_solver = GeometricCG()
        array_out, iteration = poisson_solver.solve(array_in_2d, fluid_domain, guess=None)
        array_out = array_out.reshape((N, N))
        # Ensure Input has not been changed
        self.assertTrue(np.array_equal(np.zeros((1, N, N, 1)), array_in_2d), INPUT_CHANGED_MSG)
        # Compare
        self.assertEqual(np.max(np.abs(array_out - array_out.mean())), 0, OUTPUT_NOT_UNIFORM)

    def higher_order_central_diff(self):
        """test for higher order central difference working in 2D"""
        from phi.math.nd import finite_diff
        n = 3
        tensor = np.arange(0, n**2).reshape(1, n, n, 1)
        padding = 'circular'
        axes = [1, 0]
        order = 2
        accuracy = 2
        results = finite_diff(tensor, order, axes, accuracy=accuracy, padding=padding, dx=1)
        self.assertEqual(results[0, ..., 0], np.array([[9, 9, 9], [0, 0, 0], [-9, -9, -9]]))
        self.assertEqual(results[0, ..., 1], np.array([[3, 0, -3], [3, 0, -3], [3, 0, -3]]))

    def higher_order_central_diff_sum(self):
        """test for higher order central difference working in 2D"""
        from phi.math.nd import sum_finite_diff
        n = 3
        tensor = np.arange(0, n**2).reshape(1, n, n, 1)
        padding = 'circular'
        axes = [1, 0]
        order = 2
        accuracy = 2
        results = sum_finite_diff(tensor, order, axes, accuracy=accuracy, padding=padding, dx=1)
        self.assertEqual(results[0, ..., 0], np.array([[12, 9, 6], [3, 0, -3], [-6, -9, -12]]))


class TestPhysics(TestCase):

    def test_hasegawa_wakatani_2d_adiabatic(self):
        """test HW 2D model in the adiabatic limit for square data"""
        from phi.physics.world import World
        from phi.physics.field import Noise
        from phi.physics.material import PERIODIC
        from phi.physics.hasegawa_wakatani import HasegawaWakatani2D  # Plasma Physics
        from phi.physics.plasma_field import PlasmaHW  # Plasma Field
        # Define Setup
        MODE = "NumPy"
        step_size = 10**-2
        initial_state = {
            "grid": [64, 64],    # Grid size in points (resolution)
            "K0":   0.0375/2,    # Box size defining parameter
            "N":    1,           # N*2 order of dissipation
            "nu":   0,           # Dissipation scaling coefficient
            "c1":   5,           # Adiabatic parameter
            "kappa_coeff":   0,
            "arakawa_coeff": 1,
        }
        # Process Setup
        N = initial_state['grid'][1]
        shape = (1, *initial_state['grid'], 1)
        del initial_state['grid']
        def get_box_size(k0):
            return 2*np.pi/k0
        # Setup Experiment
        domain = Domain([N, N],
                        box=AABox(0, [get_box_size(initial_state['K0'])]*len([N, N])),  # NOTE: Assuming square
                        boundaries=(PERIODIC, PERIODIC)  # Each dim: OPEN / CLOSED / PERIODIC
                        )
        fft_random = CenteredGrid.sample(Noise(), domain)
        integral = np.sum(fft_random.data**2)
        fft_random /= np.sqrt(integral)
        # Instantiate Physics
        world = World()
        world.add(
            PlasmaHW(
                domain,
                density=fft_random,
                omega=0.5*fft_random,
                phi=0.5*fft_random,
                initial_density=fft_random
            ),
            physics=HasegawaWakatani2D(**initial_state)
        )
        # Test
        def get_adiabatic_deviation(plasma):
            diff = (plasma.density - plasma.phi).data[0, ..., 0]
            return np.max(np.abs(diff))

        for i in range(200):
            world.step(dt=step_size)
            diff = get_adiabatic_deviation(world.state.plasma)  # TODO: Fix (KeyError)
            if diff < 1e-5:
                break
        self.assertLess(diff, 1e-5)
