# Φ<sub>*Flow*</sub> Liquids

*Note*: The contents of this branch are experimental and may be changed in the future.
See the [master branch](https://github.com/tum-pbs/PhiFlow) for a general description of Φ<sub>*Flow*</sub>.

This branch adds experimental support for liquids, supporting the following simulations:
- Density-based
- Levelset
- FLIP

Liquid simulations have been tested in 2D, both with NumPy and TensorFlow (requires TensorFlow>=1.13).

The liquid demos [liquid_density.py](demos/liquid_density.py), [liquid_flip.py](demos/liquid_flip.py) and [liquid_levelset.py](demos/liquid_levelset.py) run a simple simulation using the three methods.

## Usage

Density-based
```python
liquid = world.add(Fluid(domain, density=initial_density), physics=FreeSurfaceFlow())
```

Levelset
```python
liquid = world.add(LevelsetLiquid(domain, active_mask=initial_density), physics=FreeSurfaceFlow())
```

FLIP
```python
initial_points = distribute_points(initial_density, particles_per_cell=4)
liquid = world.add(FlipLiquid(domain, points=initial_points, particles_per_cell=4), physics=FreeSurfaceFlow())
```


## Known Issues

Liquid simulations have a spatial bias, i.e. now all operations are symmetric.
