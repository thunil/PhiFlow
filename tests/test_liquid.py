from unittest import TestCase

import numpy as np

from phi.geom import Sphere
from phi.physics.domain import Domain
from phi.physics.field import StaggeredGrid
from phi.physics.field.effect import Inflow
from phi.physics.fluid import Fluid
from phi.physics.liquid import FlipLiquid, FreeSurfaceFlow, LevelsetLiquid
from phi.physics.world import World
from phi.tf.session import Session
from phi.tf.world import tf_bake_graph


class TestFlipLiquid(TestCase):
    def test_direct_liquid(self):
        liquid = FlipLiquid(Domain([16, 16]), points=np.zeros([1,0,2]))
        liquid2 = FreeSurfaceFlow().step(liquid)
        assert(liquid2.age == 1.0)
        assert(liquid.age == 0.0)
        assert(liquid2.name == liquid.name)

    def test_flip(self):
        world = World()
        world.batch_size = 2
        liquid = world.add(FlipLiquid(Domain([16, 16]), points=np.zeros([1,0,2])))
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(liquid)
        self.assertAlmostEqual(liquid.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)

    def test_flip_tf(self):
        world = World()
        world.batch_size = 2
        liquid = world.add(FlipLiquid(Domain([16, 16]), points=np.zeros([1,0,2])))
        world.add(Inflow(Sphere((8, 8), radius=4)))
        tf_bake_graph(world, Session(None))
        world.step()
        world.step()
        self.assertIsInstance(liquid.velocity.data, np.ndarray)

    def test_flip_initializers(self):
        def typetest(liquid):
            self.assertIsInstance(liquid, FlipLiquid)
            self.assertIsInstance(liquid.velocity.at(liquid.staggered_grid('test', 0)), StaggeredGrid)
            np.testing.assert_equal(liquid.density.at(liquid.centered_grid('test', 0)).data.shape, [1,4,4,1])
            np.testing.assert_equal(liquid.velocity.at(liquid.staggered_grid('test', 0)).staggered_tensor().shape, [1,5,5,2])
        typetest(FlipLiquid(Domain([4, 4]), points=np.zeros([1,0,2]), velocity=0.0))
        typetest(FlipLiquid(Domain([4, 4]), points=np.zeros([1,0,2])))


class TestDensityBasedLiquid(TestCase):

    def test_direct_density_liquid(self):
        liquid = Fluid(Domain([16, 16]))
        liquid2 = FreeSurfaceFlow().step(liquid)
        assert(liquid2.age == 1.0)
        assert(liquid.age == 0.0)
        assert(liquid2.name == liquid.name)

    def test_density_liquid(self):
        world = World()
        world.batch_size = 2
        liquid = world.add(Fluid(Domain([16, 16])), physics=FreeSurfaceFlow())
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(liquid)
        self.assertAlmostEqual(liquid.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)

    def test_density_liquid_tf(self):
        world = World()
        world.batch_size = 2
        liquid = world.add(Fluid(Domain([16, 16])), physics=FreeSurfaceFlow())
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        tf_bake_graph(world, Session(None))
        world.step()
        world.step()
        self.assertIsInstance(liquid.velocity.unstack()[0].data, np.ndarray)


class TestLevelsetLiquid(TestCase):

    def test_levelset_liquid_tf(self):
        world = World()
        world.batch_size = 2
        liquid = world.add(LevelsetLiquid(Domain([16, 16])), physics=FreeSurfaceFlow())
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        tf_bake_graph(world, Session(None))
        world.step()
        world.step()
        self.assertIsInstance(liquid.velocity.unstack()[0].data, np.ndarray)
