from phi.tf.flow import *


def ball_movement(previous_ball, dt, **dependent_states):
    # TODO: Doesn't work with multiple batches yet
    # There should only be one velocity state passed.
    assert len(dependent_states['velocity_state']) == 1
    # We want to move the ball according to the velocity field passed in dependent_states
    velocity_state = dependent_states['velocity_state'][0]
    velocity_field = velocity_state.velocity.staggered

    # Reshape the sample_coords correctly. [batch , z,y,x,... , spatial_rank]
    new_shape = math.ones_like(velocity_state.grid.shape())
    new_shape[-1] = velocity_field.shape[-1]
    sample_coords = math.reshape(previous_ball.center, new_shape)
    # Resample the velocity field at the center of the ball to see what the ball is advected with
    ball_velocity = math.resample(velocity_field, sample_coords)
    ball_velocity = math.reshape(ball_velocity, previous_ball.center.shape)

    center_coords = previous_ball.center + dt * ball_velocity
    grid_dimensions = velocity_state.grid.dimensions
    radius = previous_ball.radius
    # Clamp the coordinates so the ball can't leave the domain. We assume the ball was summoned in a closed battlefield and cannot run from its destiny.
    center_coords = math.minimum(math.maximum(radius, center_coords), grid_dimensions - radius)

    return Sphere(center_coords, radius)


class SDFBasedLiquid(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Signed Distance based Liquid", stride=3)

        size = [80,64]
        domain = Domain(size, SLIPPERY)

        self.distance = 60

        self.initial_density = zeros(domain.grid.shape())
        self.initial_velocity = zeros(domain.grid.staggered_shape())
        #self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8-2, size[-1] * 3 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 2 // 8 : size[-1] * 6 // 8, :] = 1
        self.initial_density[:, size[-2] * 0 // 8 : size[-2] * 2 // 8, size[-1] * 0 // 8 : size[-1] * 8 // 8, :] = 1
        #self.initial_velocity.staggered[:, size[-2] * 6 // 8 : size[-2] * 8 // 8 - 1, size[-1] * 3 // 8 : size[-1] * 6 // 8 + 1, :] = [-2.0, -0.0]

        self.liquid = world.SDFLiquid(state_domain=domain, density=self.initial_density, velocity=self.initial_velocity, gravity=-4.0, distance=self.distance)
        #world.Inflow(Sphere((70,32), 8), rate=0.2)

        # We don't want the tag obstacle as this will influence the velocity field, we just want an object that moves with the flow and doesn't change the flow.
        self.ball = world.Obstacle(Sphere([40.0, 32.0], 3.0), tags=())
        self.ball.physics = GeometryMovement(ball_movement)
        self.ball.physics.dependencies.update({'velocity_state': 'velocityfield'})

        #session = Session(Scene.create('test'))
        #tf_bake_graph(world, session)

        self.add_field("Ball Location", lambda: self.ball.geometry.value_at(indices_tensor(self.initial_density)))

        self.add_field("Fluid", lambda: self.liquid.active_mask)
        self.add_field("Mask Before", lambda: self.liquid.mask_before.staggered)
        self.add_field("Mask After", lambda: self.liquid.mask_after.staggered)
        self.add_field("Signed Distance Field", lambda: self.liquid.sdf)
        self.add_field("Velocity", lambda: self.liquid.velocity.staggered)
        self.add_field("Velocity Centered", lambda: self.liquid.velocity)
        self.add_field("Divergence Velocity", lambda: self.liquid.velocity.divergence())
        self.add_field("Pressure", lambda: self.liquid.pressure)


    def step(self):
        world.step(dt=0.1)

    def action_reset(self):
        particle_mask = create_binary_mask(self.initial_density, threshold=0)
        self.liquid._sdf, _ = extrapolate(self.initial_velocity, particle_mask, distance=self.distance)
        self.liquid.domaincache._active = particle_mask
        self.liquid.velocity = self.initial_velocity
        self.time = 0


app = SDFBasedLiquid().show(production=__name__ != "__main__", framerate=2, display=("Signed Distance Field", "Velocity"))
