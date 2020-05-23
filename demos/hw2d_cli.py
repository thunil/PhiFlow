import sys
from phi.tf.flow import *
from phi.tf.tf_cuda_pressuresolver import CUDASolver
from phi.physics.hasegawa_wakatani import *  # Plasma Physics
from phi.physics.plasma_field import PlasmaHW  # Plasma Field
import pandas as pd
import click

@click.command()
@click.option("--mode", default="NUMPY", type=click.STRING, show_default=True,
              help="NUMPY vs. TENSORFLOW vs. PYTORCH")
@click.option("--step_size", default=10**-3, type=click.FloatRange(0, 1), show_default=True,
              help="Step size in time")
@click.option("--steps", default=10**4, type=click.IntRange(0, None), show_default=True,
              help="Number of steps to take in the simulation")
@click.option("--grid_size", default=128, type=click.IntRange(8, None), show_default=True,
              help="Integer. coarse: 128, fine: 1024")
@click.option("--k0", default=0.15, type=click.FloatRange(0, None), show_default=True,
              help="small: 0.15 (focus on high-k), large: 0.0375 (focus on low-k)")
@click.option("--N", "N", default=3, type=click.IntRange(0, None), show_default=True,
              help="2*N is dissipation exponent")
@click.option("--nu", default=10**-6, type=click.FloatRange(0, 1), show_default=True,
              help="coarse-large: 5*10**-10, fine-small: 10**-4")
@click.option("--c1", default=1, type=click.FloatRange(0, None), show_default=True,
              help="Scale between: hydrodynamic: 0.1, transition: 1, adiabatic: 5")
@click.option("--kappa", default=1, type=click.INT, show_default=True,
              help="Kappa coefficient.")
@click.option("--arakawa_coeff", default=1, type=click.INT, show_default=True,
              help="Poisson Bracket coefficient.")
@click.option("--out", "-o", "output_path", required=True, type=click.STRING, show_default=True,
              help="Output path for writing data.")
def main(mode, step_size, steps, grid_size, k0, N, nu, c1, kappa, arakawa_coeff, output_path):
    MODE=mode
    step_size = float(step_size)
    steps = int(steps)

    DESCRIPTION = """
    Hasegawa-Wakatani Plasma
    """

    initial_state = {
        "grid": (int(grid_size), int(grid_size)),      # Grid size in points (resolution)
        "K0":   k0,         # Box size defining parameter
        "N":    int(N),                        # N*2 order of dissipation
        "nu":   int(nu),#nu_dict['coarse-large'],  # Dissipation scaling coefficient
        "c1":   c1,     # Adiabatic parameter
        "kappa_coeff":   kappa,
        "arakawa_coeff": arakawa_coeff,
    }


    def get_box_size(k0):
        return 2*np.pi/k0


    N = initial_state['grid'][1]
    shape = (1, *initial_state['grid'], 1)
    del initial_state['grid']


    domain = Domain([N, N],
                    box=AABox(0, [get_box_size(initial_state['K0'])]*len([N, N])),  # NOTE: Assuming square
                    boundaries=(PERIODIC, PERIODIC)  # Each dim: OPEN / CLOSED / PERIODIC
                    )
    fft_random = CenteredGrid.sample(Noise(), domain)
    integral = np.sum(fft_random.data**2)
    fft_random /= np.sqrt(integral)


    plasma_hw = PlasmaHW(domain,
                         density=fft_random,
                         phi=-0.5*fft_random,
                         omega=-0.5*fft_random,
                         initial_density=fft_random
                         )
    plasma = world.add(plasma_hw,
                       physics=HasegawaWakatani2D(**initial_state, poisson_solver=CUDASolver())
                       )

    scene = Scene.create(output_path)

    for step in range(steps):
        world.step(dt=step_size)
        scene.write(plasma.density, names='density', frame=step)
        scene.write(plasma.omega, names='omega', frame=step)
        scene.write(plasma.phi, names='phi', frame=step)

    return

if __name__ == "__main__":
    main()