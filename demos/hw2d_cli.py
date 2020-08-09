import sys
import click
import os
import pandas as pd
import phi.flow as flow
import time
from pprint import pprint
from phi.physics.hasegawa_wakatani import *  # Plasma Physics
from phi.physics.plasma_field import PlasmaHW  # Plasma Field
import cv2


global flow
math.set_precision(64)
INTERPOLATION_FUNC = cv2.INTER_CUBIC

translation_dic = {'o': 'output_path',
                   'out': 'output_path',
                   'in': 'in_path',
                   'i': 'in_path'}

@click.command()
@click.option("--mode", "mode", default="NUMPY", type=click.STRING, show_default=True,
              help="NUMPY vs. TENSORFLOW vs. PYTORCH")
@click.option("--step_size", "step_size", default=10**-3, type=click.FloatRange(0, 1), show_default=True,
              help="Step size in time")
@click.option("--steps", "steps", default=10**4, type=click.IntRange(0, None), show_default=True,
              help="Number of steps to take in the simulation")
@click.option("--tmax", "end_time", default=None, type=click.IntRange(0, None), show_default=True,
              help="Time until when the simulation should run. If provided, steps is ignored.")
@click.option("--grid_size", "grid_size", default=128, type=click.IntRange(8, None), show_default=True,
              help="Integer. coarse: 128, fine: 1024")
@click.option("--k0", default=0.15, type=click.FloatRange(0, None), show_default=True,
              help="small: 0.15 (focus on high-k), large: 0.0375 (focus on low-k)")
@click.option("--N", "N", default=3, type=click.IntRange(0, None), show_default=True,
              help="2*N is dissipation exponent")
@click.option("--nu", default=10**-6, type=click.FloatRange(0, 1), show_default=True,
              help="coarse-large: 5*10**-10, fine-small: 10**-4")
@click.option("--c1", default=1, type=click.FloatRange(0, None), show_default=True,
              help="Scale between: hydrodynamic: 0.1, transition: 1, adiabatic: 5")
@click.option("--kappa", default=1,  type=click.INT, show_default=True,
              help="Kappa coefficient.")
@click.option("--arakawa_coeff", default=1, type=click.INT, show_default=True,
              help="Poisson Bracket coefficient.")
@click.option("--out", "-o", "output_path", required=True, type=click.STRING, show_default=True,
              help="Output path for writing data.")
@click.option("--in", "-i", "in_path", default="", type=click.STRING, show_default=True,
              help="Path to previous simulation to continue")
@click.option("--snaps", "snaps", default=1000, type=click.INT, show_default=True,
              help="Intervals in which snapshots are saved")
@click.option("--snaps_time", "snaps_time", default=None, type=click.FloatRange(0, None), show_default=True,
              help="Intervals in which snapshots are saved in time.")
@click.option("--seed", "seed", default=None, type=click.INT, show_default=False,
              help="Index of initial seed. Default is last timestep")
def main(mode, step_size, steps, end_time, grid_size, k0, N, nu, c1, kappa, arakawa_coeff, 
         output_path, in_path, snaps, snaps_time, seed):
    time.sleep(np.random.rand()*10)  # Ensure no conflict with jobs that ran at the same time
    MODE=mode
    DESCRIPTION = """
    Hasegawa-Wakatani Plasma
    """
    Solver = flow.FourierSolver()#SparseCG()

    parameters = {}
    # Load previous
    if (in_path) and (seed is None):
        print("\rLoading parameters of previous run...", end="", flush=True)
        if in_path[-1] == "/":
            in_path = in_path[:-1]
        import json
        if not os.path.exists(f"{in_path}/src/context.json"):
            print("\r[-] NO parameters loaded from previous run.")
        else:
            context = json.load(open(f"{in_path}/src/context.json", "r"))
            argv = context["argv"]
            indices = list(range(len(argv)))
            partial = False
            for i in indices:
                if argv[i][:2] == "--":
                    if "=" in argv[i]:
                        key, val = argv[i][2:].split("=")
                        try:
                            val = float(val)
                        except:
                            pass
                    else:
                        key = argv[i][2:]
                        partial = True
                elif argv[i][0] == "-":
                    key = argv[i][1:]
                    partial = True
                elif partial:
                    val = argv[i]
                    partial = False
                else:
                    continue
                if key in translation_dic:
                    key = translation_dic[key]
                if not partial:
                    parameters[key] = val
        if 'steps' in parameters:
            del parameters['steps']  # Run defined step
        locals().update(parameters)  # Bad practice. Quick and dirty fix.  TODO: Broken?
        print("\r[x] Loaded all parameters from previous run.")
        #pprint(parameters
    if 'grid_size' in parameters:
        grid_size = parameters['grid_size']
    if 'step_size' in parameters:
        step_size = parameters['step_size']
    if 'nu' in parameters:
        nu = parameters['nu']
    if 'N' in parameters:
        N = parameters['N']
    if 'k0' in parameters:
        k0 = parameters['k0']
    if 'c1' in parameters:
        c1 = parameters['c1']

    step_size = float(step_size)
    steps = int(steps)

    # Time Adjustment
    if end_time is not None:
        steps = int(end_time/step_size)
    if snaps_time is not None:
        snaps = int(snaps_time/step_size)

    pprint(locals())
    initial_state = {
        "grid": (int(grid_size), int(grid_size)),      # Grid size in points (resolution)
        "K0":   k0,         # Box size defining parameter
        "N":    int(N),                        # N*2 order of dissipation
        "nu":   nu,#nu_dict['coarse-large'],  # Dissipation scaling coefficient
        "c1":   c1,     # Adiabatic parameter
        "kappa_coeff":   kappa,
        "arakawa_coeff": arakawa_coeff,
    }

    def get_box_size(k0):
        return 2*np.pi/k0

    N = initial_state['grid'][1]
    shape = (1, *initial_state['grid'], 1)
    del initial_state['grid']

    # NOTE: Assuming square
    box = flow.AABox(0, [get_box_size(initial_state['K0'])]*len([N, N]))
    domain = flow.Domain([N, N],
                         box=box,
                         boundaries=(flow.PERIODIC, flow.PERIODIC)  # Each dim: OPEN / CLOSED / PERIODIC
                         )
    fft_random = flow.CenteredGrid.sample(flow.Noise(1), domain)
    integral = np.sum(fft_random.data**2)
    fft_random /= np.sqrt(integral)

    # Load last fields
    if in_path:
        print("\rLoading field values of previous run...", end="", flush=True)
        # Find last item
        files = os.listdir(in_path)
        step_list = [int(f.split("_")[1].split(".")[0]) for f in files
                      if "phi" in f]
        # No seed given: Continue previous simulation
        if seed is None:
            sim_index = int(in_path.split('_')[-1].split('.')[0])
            init_step = max(step_list)
            scene = flow.Scene(dir=output_path, category="", index=sim_index)
            #print(f"-> Continuing last simulation from step={init_step}", flush=True)
        else:
            init_step = seed
            scene = flow.Scene.create(output_path)

        # Load last fields
        init_density, init_phi, init_omega = flow.read_sim_frames(in_path, fieldnames=["density", "phi", "omega"], frames=init_step)
        initial_density = init_density #TODO: Read true init density when restarting##flow.read_sim_frames(in_path, fieldnames="density", frames=0)
        def get_stats(arr):
            return dict(mean=np.mean(arr), var=np.var(arr), sum=np.sum(arr), min=np.min(arr), max=np.max(arr), std=np.std(arr))
        print_stats = lambda init_density: print("  ".join([f"{key}={val:>9.2e}" for key, val in get_stats(init_density).items()]))
        prev_properties = {'init_density': get_stats(init_density),
                           'init_phi': get_stats(init_phi),
                           'init_omega': get_stats(init_omega),
                           'initial_density': get_stats(initial_density)}
        print(f"\r[x] Loaded all field values from previous run. (step={init_step:,})")
        is_valid = lambda arr, prev: np.allclose(list(get_stats(arr).values())[:3], list(prev.values())[:3])
        # Resize sizes
        if (np.array(init_density.shape) > N).any():
            init_density = cv2.resize(init_density[0, ..., 0], dsize=(N, N), interpolation=INTERPOLATION_FUNC).reshape(1, N, N, 1)
            init_phi = cv2.resize(init_phi[0, ..., 0], dsize=(N, N), interpolation=INTERPOLATION_FUNC).reshape(1, N, N, 1)
            init_omega = cv2.resize(init_omega[0, ..., 0], dsize=(N, N), interpolation=INTERPOLATION_FUNC).reshape(1, N, N, 1)
            initial_density = cv2.resize(initial_density[0, ..., 0], dsize=(N, N), interpolation=INTERPOLATION_FUNC).reshape(1, N, N, 1)
            # Normalize Functions
            center = lambda arr, prev: arr-np.mean(arr)+prev['mean']
            eq_var = lambda arr, prev: (arr/np.std(arr))*prev['std']
            content = lambda arr, prev: (arr/np.sum(arr))*prev['sum']
            normalize = lambda arr, prev: center(eq_var(content(arr, prev), prev), prev)
            # Normalize and assert properties
            init_density = normalize(init_density, prev_properties['init_density'])
            assert is_valid(init_density, prev_properties['init_density'])
            init_phi = normalize(init_phi, prev_properties['init_phi'])
            assert is_valid(init_phi, prev_properties['init_phi'])
            init_omega = normalize(init_omega, prev_properties['init_omega'])
            assert is_valid(init_omega, prev_properties['init_omega'])
            initial_density = normalize(initial_density, prev_properties['initial_density'])
        assert init_density.shape == init_phi.shape == init_omega.shape == initial_density.shape, f"\nShape mismatch in loading: density={density.shape}, phi={phi.shape}, omega={omega.shape}, init_density={initial_density.shape}"
        if seed is not None:
            # Write seed data
            scene.write(init_density, names='density', frame=seed)
            scene.write(init_phi, names='phi', frame=seed)
            scene.write(init_omega, names='omega', frame=seed)

    # Initialize
    else:
        init_density = fft_random
        init_phi = 0.5*fft_random
        init_omega = 0.5*fft_random
        initial_density = fft_random
        scene = flow.Scene.create(output_path)
        init_step = 0
        # Write initial data
        scene.write(init_density, names='density', frame=0)
        scene.write(init_phi, names='phi', frame=0)
        scene.write(init_omega, names='omega', frame=0)
    
    plasma_hw = PlasmaHW(domain,
                         density=init_density,
                         phi=init_phi,
                         omega=init_omega,
                         initial_density=initial_density,
                         age=init_step*step_size
                         )
    plasma = flow.world.add(plasma_hw,
                            physics=HasegawaWakatani2D(**initial_state, poisson_solver=Solver)
                            )

    for step in range(steps):
        step_ix = step + init_step + 1
        flow.world.step(dt=step_size)
        if step_ix % snaps == 0:
            scene.write(plasma.density, names='density', frame=step_ix)
            scene.write(plasma.omega, names='omega', frame=step_ix)
            scene.write(plasma.phi, names='phi', frame=step_ix)
        if np.isnan(math.sum(plasma.density).data):
            break

    return

if __name__ == "__main__":
    main()
