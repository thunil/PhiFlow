from unittest import TestCase
from phi.flow import *


class TestGridLiquid(TestCase):

    def test_direct_liquid(self):
        liquid = Fluid(Domain([16, 16]))
        liquid2 = INCOMPRESSIBLE_LIQUID.step(liquid)
        assert(liquid2.age == 1.0)
        assert(liquid.age == 0.0)
        assert(liquid2.name == liquid.name)

    def test_gridliquid(self):
        world = World()
        world.batch_size = 2
        liquid = world.add(Fluid(Domain([16, 16])), physics=INCOMPRESSIBLE_LIQUID)
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(liquid)
        self.assertAlmostEqual(liquid.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)
