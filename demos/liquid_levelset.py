from phi.flow import *


DESCRIPTION = """
Grid-based liquid simulation using level sets to define the liquid.
A signed distance field (SDF) is used to track the liquid where negative values correspond to filled cells.
"""


class LevelsetLiquidDemo(App):

    def __init__(self, size=(64, 80)):
        App.__init__(self, 'SDF simulation', DESCRIPTION, stride=10)
        domain = Domain(size, SLIPPERY)
        self.distance = max(size)
        # --- Initial state ---
        initial_fluid_density = union_mask([box[:15, :], box[48:63, 20:60]])
        # --- Create liquid and expose fields to GUI ---
        self.liquid = world.add(LevelsetLiquid(domain, active_mask=initial_fluid_density, distance=self.distance), physics=FreeSurfaceFlow())
        self.add_field('Fluid', lambda: self.liquid.active_mask)
        self.add_field('Signed Distance Field', lambda: self.liquid.sdf)
        self.add_field('Velocity', lambda: self.liquid.velocity)

    def step(self):
        world.step(dt=0.1)


show(display=('Fluid', 'Velocity'))
