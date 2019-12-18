# Φ<sub>*Flow*</sub> Liquids

*Note*: The contents of this branch are experimental and may be changed in the future.
See the [master branch](https://github.com/tum-pbs/PhiFlow) for a general description of Φ<sub>*Flow*</sub>.

This branch adds experimental support for free surface liquid simulations, supporting the following types:
- Density-based
- Levelset
- Fluid implicit particle (FLIP)

Liquid simulations have been tested in 2D, both with NumPy and TensorFlow (requires TensorFlow>=1.13).

The liquid demos [liquid_density.py](demos/liquid_density.py), [liquid_flip.py](demos/liquid_flip.py) and [liquid_levelset.py](demos/liquid_levelset.py) run a simple simulation using the three methods.


## Usage

OPEN and CLOSED boundaries are supported and can be mixed.
Inflows are supported by all methods.
Liquids are affected by global gravity.

### Density-based
```python
liquid = world.add(Fluid(domain, density=initial_density), physics=FreeSurfaceFlow())
```

### Levelset
```python
liquid = world.add(LevelsetLiquid(domain, active_mask=initial_density), physics=FreeSurfaceFlow())
```

### FLIP
```python
initial_points = distribute_points(initial_density, particles_per_cell=4)
liquid = world.add(FlipLiquid(domain, points=initial_points, particles_per_cell=4), physics=FreeSurfaceFlow())
```


## Known Issues

- Liquid simulations have a spatial bias, i.e. not all operations are symmetric.
- No support for velocity effects.
- Obstacles do not work with Levelset and FLIP.