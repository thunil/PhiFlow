from phi.flow import *


class SDFDemo(App):

    def __init__(self, size=(64, 80)):
        App.__init__(self, 'SDF simulation', 'Signed Distance based liquid simulation.', stride=1)
        domain = Domain(size, SLIPPERY)
        self.distance = max(size)
        # --- Initial state ---
        initial_fluid_density = union_mask([box[:15, :], box[48:63, 20:60]])
        # --- Create liquid and expose fields to GUI ---
        self.liquid = world.add(SDFLiquid(domain, active_mask=initial_fluid_density, distance=self.distance))
        self.add_field('Fluid', lambda: self.liquid.active_mask)
        self.add_field('Signed Distance Field', lambda: self.liquid.sdf)
        self.add_field('Velocity', lambda: self.liquid.velocity)

    def step(self):
        world.step(dt=0.3)


show(framerate=3, display=('Fluid', 'Velocity'))
