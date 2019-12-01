from unittest import TestCase
from phi.flow import *


class TestFlipLiquid(TestCase):
    def test_direct_liquid(self):
        liquid = FlipLiquid(Domain([16, 16]), points=np.zeros([1,0,2]))
        assert liquid.default_physics() == FLIP_LIQUID
        liquid2 = FLIP_LIQUID.step(liquid)
        assert(liquid2.age == 1.0)
        assert(liquid.age == 0.0)
        assert(liquid2.name == liquid.name)

    def test_flipliquid(self):
        world = World()
        world.batch_size = 2
        liquid = world.add(FlipLiquid(Domain([16, 16]), points=np.zeros([1,0,2])))
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(liquid)
        self.assertAlmostEqual(liquid.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)
        #self.assertEqual(liquid._batch_size, 2)

    def test_flipliquid_initializers(self):
        def typetest(liquid):
            self.assertIsInstance(liquid, FlipLiquid)
            self.assertIsInstance(liquid.velocity.at(liquid.staggered_grid('test', 0)), StaggeredGrid)
            np.testing.assert_equal(liquid.density.at(liquid.centered_grid('test', 0)).data.shape, [1,4,4,1])
            np.testing.assert_equal(liquid.velocity.at(liquid.staggered_grid('test', 0)).staggered_tensor().shape, [1,5,5,2])
        typetest(FlipLiquid(Domain([4, 4]), points=np.zeros([1,0,2]), velocity=0.0))
        typetest(FlipLiquid(Domain([4, 4]), points=np.zeros([1,0,2])))