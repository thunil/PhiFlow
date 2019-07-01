from unittest import TestCase
from phi.flow import *


class TestFlipLiquid(TestCase):
    def test_direct_liquid(self):
        liquid = FlipLiquid(Domain([16, 16]))
        assert liquid.default_physics() == FLIPLIQUID
        liquid2 = FLIPLIQUID.step(liquid)
        assert(liquid2.age == 1.0)
        assert(liquid.age == 0.0)
        assert(liquid2.trajectorykey == liquid.trajectorykey)

    def test_flipliquid(self):
        world = World()
        world.batch_size = 2
        liquid = world.FlipLiquid(Domain([16, 16]))
        inflow = world.Inflow(Sphere((8, 8), radius=4))
        world.step()
        world.step(liquid)
        self.assertAlmostEqual(world.state.age, 2.0)
        self.assertAlmostEqual(liquid.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)
        self.assertEqual(liquid._batch_size, 2)

    def test_flipliquid_initializers(self):
        def typetest(liquid):
            self.assertIsInstance(liquid, FlipLiquid)
            self.assertIsInstance(liquid.velocity_field, StaggeredGrid)
            np.testing.assert_equal(liquid.density_field.shape, [1,4,4,1])
            np.testing.assert_equal(liquid.velocity_field.shape, [1,5,5,2])
        typetest(FlipLiquid(Domain([4, 4]), density=0.0, velocity=0.0))
        typetest(FlipLiquid(Domain([4, 4]), density=1.0, velocity=1.0))
        typetest(FlipLiquid(Domain([4, 4])))