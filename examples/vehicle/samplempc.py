import fire
from plot import *
from dynamics.f import f
from pathlib import Path
import sys
import os
import subprocess
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
from casadi import SX, vertcat, norm_2, sqrt
import numpy as np
import scipy.linalg
from scipy.integrate import odeint
import math

np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 200

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from soeampc.datasetutils import import_dataset, merge_parallel_jobs, get_date_string, merge_single_parallel_job, print_dataset_statistics
from soeampc.mpcproblem import MPCQuadraticCostLxLu
from soeampc.samplempc import sample_dataset_from_mpc, computetime_test_fwd_sim
from soeampc.sampler import RandomSampler

fp = Path(os.path.dirname(__file__))
os.chdir(fp)


def export_vehicle_ode_model():
    """
    Build the ACADOS implicit ODE model for the vehicle example.
      - decision variable is v (nu)
      - real input is u = Kdelta @ x + v
        Kdelta is a fixed linear feedback gain read from mpc_parameters/Kdelta.txt
      - extra state s with sdot = -rho*s + w_bar
        auxiliary “safety/robustness bookkeeping” state used to “shrink” the terminal set
    """
    rho = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'rho_c.txt'), delimiter=','))
    w_bar = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'wbar.txt'), delimiter=','))

    nx = 4
    nu = 2

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Kdelta.txt'), delimiter=','),
        (nx, nu)
    ).T

    model_name = 'vehicle'

    # states & controls
    x = SX.sym('x', nx, 1)
    xdot = SX.sym('xdot', nx, 1)

    s = SX.sym('s')
    sdot = SX.sym('sdot')

    v = SX.sym('v', nu, 1)      # optimization variable
    u = Kdelta @ x + v          # applied input

    fx = f(x, u)

    # implicit dynamics: f(x,u) - xdot = 0, and -rho*s + w_bar - sdot = 0
    f_impl = vertcat(vertcat(*fx) - xdot, -rho * s + w_bar - sdot)

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = vertcat(x, s)
    model.xdot = vertcat(xdot, sdot)
    model.u = v
    model.p = []
    model.name = model_name
    return model


def export_vehicle_sim_model():
    """
    Simulation model without s-state.
    This is only used for forward sim timing test.
    """
    nx = 4
    nu = 2

    model_name = 'vehicle'

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Kdelta.txt'), delimiter=','),
        (nx, nu)
    ).T

    x = SX.sym('x', nx, 1)
    xdot = SX.sym('xdot', nx, 1)

    v = SX.sym('v', nu, 1)
    u = Kdelta @ x + v

    fx = f(x, u)
    f_impl = vertcat(*fx) - xdot

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = v
    model.p = []
    model.name = model_name
    return model


def sample_mpc(
        showplot=True,
        experimentname="",
        numberofsamples=int(5000),
        randomseed=42,
        verbose=False,
        generate=True,
        nlpiter=200
):
    """
    Generate a dataset by repeatedly solving the ACADOS NMPC problem for random initial states.

    High-level workflow (same as the quadcopter example)
    ----------------------------------------------------
    1) Read all MPC design parameters from `mpc_parameters/*.txt`
       (Q, R, P, gains K/Kinit/Kdelta, constraint matrices Lx/Lu/Ls, alpha, etc.)
    2) Build an ACADOS OCP (AcadosOcp):
    3) Create a RandomSampler that draws x0 uniformly from [xmin, xmax].
    4) For each sampled x0:
       - create/reset an AcadosOcpSolver instance (to avoid false warm-start coupling across samples)
       - set the initial state constraint x(0)=x0 and s(0)=0
       - build an initialization guess (Xinit, Uinit) using Kinit and forward simulation
       - load the guess into the solver and solve OCP
       - extract optimal state and input trajectories (X, U) and solve statistics
    5) Use `sample_dataset_from_mpc(...)` to store dataset.

    Parameters
    ----------
    showplot:        True, plot a quick feasibility scatter after dataset generation.
    experimentname:  Prefix appended to the dataset name for easier identification.
    numberofsamples: Number of random initial states x0 to solve the MPC for.
    randomseed:     Seed for reproducible sampling.
    verbose:        If True, print additional debugging information (initial guesses, feasibility check).
    generate:       If True, generate/build the ACADOS solver code (first run).
                    For repeated runs or parallel sampling, set this to False.
    nlpiter:        Maximum SQP iterations in ACADOS (higher -> more robust, slower).
    """
    print("\n\n===============================================")
    print("Setting up ACADOS OCP problem (vehicle)")
    print("===============================================\n")

    # Auxiliary-state dynamics parameters for: sdot = -rho*s + w_bar
    rho = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'rho_c.txt'), delimiter=','))
    w_bar = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'wbar.txt'), delimiter=','))

    # Auxiliary-state dynamics parameters for: sdot = -rho*s + w_bar
    nx = 4
    nu = 2

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Kdelta.txt'), delimiter=','),
        (nx, nu)
    ).T
    print("Kdelta=\n", Kdelta, "\n")

    ocp = AcadosOcp()

    model = export_vehicle_ode_model()
    ocp.model = model

    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'Tf.txt'), delimiter=','))
    N = 25
    ocp.dims.N = N

    # model.x = [x; s]  -> nx_ includes s
    nx_ = model.x.size()[0]          # 5
    nx = nx_ - 1                     # 4
    nu = model.u.size()[0]           # 2

    ny = nx_ + nu
    ny_e = nx_

    # initialize s along its dynamics (same as quadcopter)
    Sinit = odeint(lambda y, t: -rho * y + w_bar, 0, np.linspace(0, Tf, N + 1))
    print("Sinit =\n", Sinit, "\n")

    Q = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'Q.txt'), delimiter=','), (nx, nx))
    P = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'P.txt'), delimiter=','), (nx, nx))
    R = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'R.txt'), delimiter=','), (nu, nu))

    Q_ = scipy.linalg.block_diag(Q, 1.0)
    P_ = scipy.linalg.block_diag(P, 1.0)

    K = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'K.txt'), delimiter=','), (nx, nu)).T
    Kinit = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'Kinit.txt'), delimiter=','), (nx, nu)).T

    alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'alpha.txt'), delimiter=','))

    print("Q=\n", Q, "\n")
    print("R=\n", R, "\n")
    print("P=\n", P, "\n")

    # cost
    ocp.cost.W_e = P_
    ocp.cost.W = scipy.linalg.block_diag(Q_, R)

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    ocp.cost.Vx = np.zeros((ny, nx_))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx_:, :nu] = np.eye(nu)

    ocp.cost.Vx_e = np.zeros((ny_e, nx_))
    ocp.cost.Vx_e[:nx, :nx] = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # constraints in the same one-sided form: C*[x;s] + D*v <= 1
    # IMPORTANT: these sizes must match your txt files.
    #
    # Recommended for a first vehicle example:
    #   - only input bounds -> nuconstr = 2*nu = 4
    #   - no state constraints -> nxconstr = 0
    nxconstr = 0
    nuconstr = 2 * nu
    nconstr = nxconstr + nuconstr

    Lx = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Lx.txt'), delimiter=','),
        (nx, nconstr)
    ).T
    Lu = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Lu.txt'), delimiter=','),
        (nu, nconstr)
    ).T
    Ls = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Ls.txt'), delimiter=','),
        (1, nconstr)
    ).T

    # enforce bounds on u = Kdelta x + v
    # same patch as quadcopter:
    # Lu*u <= 1 -> Lu*(Kdelta x + v) <= 1 -> (Lu*Kdelta) x + Lu*v <= 1
    # split rows for +u and -u:
    Lx[nxconstr:nxconstr + nu, :] = Lu[nxconstr:nxconstr + nu] @ Kdelta
    Lx[nxconstr + nu:nxconstr + 2 * nu, :] = Lu[nxconstr + nu:nxconstr + 2 * nu] @ Kdelta

    print("Lx=\n", Lx, "\n")
    print("Lu=\n", Lu, "\n")
    print("Ls=\n", Ls, "\n")

    ocp.constraints.C = np.hstack((Lx, Ls))
    ocp.constraints.D = Lu
    ocp.constraints.lg = -1e5 * np.ones(nconstr)
    ocp.constraints.ug = np.ones(nconstr)

    # soft constraints
    ocp.constraints.Jsg = np.eye(nconstr)
    L2_pen = 1e6
    L1_pen = 1e4
    ocp.cost.Zl = L2_pen * np.ones((nconstr,))
    ocp.cost.Zu = L2_pen * np.ones((nconstr,))
    ocp.cost.zl = L1_pen * np.ones((nconstr,))
    ocp.cost.zu = L1_pen * np.ones((nconstr,))

    # terminal set: x' P x <= (alpha_f - alpha_s*s_T)^2
    alpha_s = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'alpha_s.txt'), delimiter=','))
    ocp.constraints.lh_e = np.array([-1e5])

    ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]
    alpha = alpha_f - alpha_s * (1 - math.exp(-rho * Tf)) / rho * w_bar
    if alpha < 0:
        raise Exception("Terminal set size alpha_f - alpha_s*s_T is negative:", alpha)
    ocp.constraints.uh_e = np.array([alpha ** 2])

    ocp.constraints.x0 = np.zeros(nx_)

    # build MPC object
    mpc = MPCQuadraticCostLxLu(
        f, nx, nu, N, Tf, Q, R, P, alpha_f,
        K, Lx, Lu, Kdelta, alpha_reduced=alpha, S=Sinit, Ls=Ls
    )
    mpc.name = model.name

    # solver options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hpipm_mode = 'ROBUST'

    ocp.solver_options.qp_tol = 1e-8
    ocp.solver_options.levenberg_marquardt = 20.0
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.globalization_use_SOC = 1
    ocp.solver_options.line_search_use_sufficient_descent = 1
    ocp.solver_options.alpha_reduction = 0.1
    ocp.solver_options.alpha_min = 0.0001
    ocp.solver_options.regularize_method = 'MIRROR'

    ocp.solver_options.tf = Tf
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_max_iter = nlpiter

    if generate:
        _ = AcadosOcpSolver(ocp, json_file='acados_ocp_' + model.name + '.json')

    # Sampling bounds for x0 (edit to your scenario):
    # px, py in [-5, 5], psi in [-pi, pi], v in [0, 15]
    xmin = np.array([-5.0, -5.0, -math.pi, 0.0])
    xmax = np.array([+5.0, +5.0, +math.pi, 15.0])

    # derive u bounds from Lu
    umax = np.array([1.0 / Lu[nxconstr + i, i] for i in range(nu)])
    umin = np.array([1.0 / Lu[nxconstr + nu + i, i] for i in range(nu)])
    print("\numin=\n", umin)
    print("\numax=\n", umax)

    sampler = RandomSampler(numberofsamples, mpc.nx, randomseed, xmin, xmax)

    def run(x0, verbose=False):
        acados_ocp_solver = AcadosOcpSolver(
            ocp, json_file='acados_ocp_' + model.name + '.json',
            build=False, generate=False
        )

        # fix initial state and s=0
        acados_ocp_solver.set(0, "lbx", np.append(x0, 0.0))
        acados_ocp_solver.set(0, "ubx", np.append(x0, 0.0))

        # init trajectories
        Xinit = np.linspace(x0, np.zeros(nx), N + 1)
        Uinit = np.zeros((N, nu))

        for i in range(N):
            Uinit[i] = Kinit @ Xinit[i]
            # apply bounds to u = Kdelta x + v  -> bounds for v are (umin - Kdelta x, umax - Kdelta x)
            Uinit[i] = np.clip(Uinit[i], umin - Kdelta @ Xinit[i], umax - Kdelta @ Xinit[i])
            Xinit[i + 1] = mpc.forward_simulate_single_step(Xinit[i], Uinit[i])

        if verbose:
            print("\nx0=\n", x0)
            print("\nXinit=\n", Xinit)
            print("\nUinit=\n", Uinit)
            print("\nfeasible=", mpc.feasible(Xinit, Uinit, verbose=True))

        for i in range(N):
            acados_ocp_solver.set(i, "x", np.append(Xinit[i], Sinit[i]))
            acados_ocp_solver.set(i, "u", Uinit[i])

        status = acados_ocp_solver.solve()

        X = np.ndarray((N + 1, nx))
        S = np.ndarray(N + 1)
        U = np.ndarray((N, nu))

        for i in range(N):
            X[i, :] = acados_ocp_solver.get(i, "x")[:-1]
            S[i] = acados_ocp_solver.get(i, "x")[-1]
            U[i, :] = acados_ocp_solver.get(i, "u")

        X[N, :] = acados_ocp_solver.get(N, "x")[:-1]
        S[N] = acados_ocp_solver.get(N, "x")[-1]

        computetime = float(acados_ocp_solver.get_stats('time_tot'))
        number_iterations = float(acados_ocp_solver.get_stats('sqp_iter'))
        return X, U, status, computetime, number_iterations

    _, _, _, _, outfile = sample_dataset_from_mpc(mpc, run, sampler, experimentname, verbose=verbose)
    print("Outfile", outfile)

    if showplot:
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)

        # 1) quick feasibility scatter: sampled initial positions (px0 vs py0)
        plot_feas(x0dataset[:, 0], x0dataset[:, 1])

        # 2) trajectory sanity plots (only works if dataset stores full trajectories)
        if Xdataset.ndim == 3 and Udataset.ndim == 3:
            # Randomly select K trajectories from the dataset
            K_show = 1  # increase/decrease as needed
            rng = np.random.default_rng(0)  # fixed seed for reproducible plotting
            n_avail = Xdataset.shape[0]
            idxs = rng.choice(n_avail, size=min(K_show, n_avail), replace=False)

            V_list = [Udataset[i] for i in idxs]  # NOTE: stored "U" is actually v in this setup
            X_list = [Xdataset[i] for i in idxs]
            labels = [f"dataset sample {i}" for i in idxs]

            # (a) plot the optimizer variable v over the horizon
            #plot_vehicle_ol_V(mpc, V_list, labels=labels, plt_show=True)
            plot_vehicle_ol_grid_3x2(
                mpc,
                V_list,
                X_list,
                labels=labels,
                plt_show=True,
                limits={"umin": umin.tolist(), "umax": umax.tolist()},
                input_is_v=True,  # because V_list is "v" in your setup
            )

            # (b) plot predicted horizon trajectories and reconstructed applied input u = Kdelta x + v
            plot_vehicle_ol(
                mpc,
                V_list,
                X_list,
                labels=labels,
                plt_show=True,
                limits={"umin": umin.tolist(), "umax": umax.tolist()}
            )

            # (c) closed-loop style time-series plot (here: still one horizon each, but useful)
            feas_list = [np.ones(V.shape[0], dtype=int) for V in V_list]
            plot_vehicle_cl(
                mpc,
                Utraj=V_list,
                Xtraj=X_list,
                feasible=feas_list,
                labels=labels,
                plt_show=True,
                limits={"umin": umin.tolist(), "umax": umax.tolist()}
            )

        else:
            print("Dataset not stored as trajectories. Shapes:", Xdataset.shape, Udataset.shape)

    return outfile


def parallel_sample_mpc(instances=16, samplesperinstance=int(1e5), prefix="Cluster"):
    now = get_date_string()
    fp = Path(os.path.abspath(os.path.dirname(__file__)))
    print("\n\n===============================================")
    print("Running", instances, "processes to produce", samplesperinstance, "datapoints each")
    print("===============================================\n")

    os.chdir(fp)
    processes = []
    parallel_experiments_common_name = prefix + "_" + str(now) + "_"

    for i in range(instances):
        experimentname = parallel_experiments_common_name + "_" + str(i) + "_"
        command = [
            "python3",
            "samplempc.py",
            "sample_mpc",
            "--showplot=False",
            "--randomseed=None",
            "--experimentname=" + experimentname,
            "--numberofsamples=" + str(samplesperinstance),
            "--generate=False"
        ]
        with open(fp.joinpath('logs', experimentname + ".log"), "wb") as out:
            p = subprocess.Popen(command, stdout=out, stderr=out)
            processes.append(p)

    for p in processes:
        p.wait()

    merge_parallel_jobs([parallel_experiments_common_name], new_dataset_name=parallel_experiments_common_name[:-1])


def computetime_test_fwd_sim_vehicle(dataset="latest"):
    name = 'vehicle'
    model = export_vehicle_sim_model()

    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'Tf.txt'), delimiter=','))
    N = 10

    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = Tf / N
    sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.newton_iter = 3

    acados_integrator = AcadosSimSolver(sim, 'acados_ocp_' + name + '_sim.json')

    def run(x0, V):
        X = np.zeros((V.shape[0] + 1, x0.shape[0]))
        X[0] = np.copy(x0)
        for i in range(len(V)):
            X[i + 1] = acados_integrator.simulate(x=X[i], u=V[i])
        return X

    computetime_test_fwd_sim(run, dataset)


if __name__ == "__main__":
    # This block runs only when you execute this file directly, e.g.:
    #   python3 samplempc.py sample_mpc --numberofsamples=2000
    #
    # `fire.Fire(...)` turns Python functions into a command-line interface (CLI).
    fire.Fire({
        'sample_mpc': sample_mpc, # Solve the MPC for many random initial states and export a dataset
        'parallel_sample_mpc': parallel_sample_mpc,
        'merge_single_parallel_job': merge_single_parallel_job,
        'print_dataset_statistics': print_dataset_statistics,
        'computetime_test_fwd_sim_vehicle': computetime_test_fwd_sim_vehicle,
    })
