"""
samplempc.py  (Vehicle example for SOEAMPC / ACADOS)
====================================================

This version is adapted to the NEW single-track vehicle model with 7 states:

  x = [px, py, psi, v, r, beta, a]
      px    [m]     global position x
      py    [m]     global position y
      psi   [rad]   yaw angle
      v     [m/s]   speed magnitude
      r     [rad/s] yaw rate
      beta  [rad]   slip angle at CoM
      a     [m/s^2] filtered longitudinal acceleration (PT1 state)

Input (applied input to dynamics.f):
  u = [delta, a_cmd]
      delta [rad]   steering angle
      a_cmd [m/s^2] commanded longitudinal acceleration

This sampling script generates datasets by:
  - building an ACADOS OCP,
  - sampling random initial states x0,
  - solving NMPC for each x0,
  - storing (x0, optimal trajectories) into dataset files.

IMPORTANT ABOUT NAMING:
  This framework historically uses the symbol "v" for the OCP decision variable,
  and then applies u = Kdelta*x + v. Here:
    - decision variable is "v" (shape nu=2),
    - applied input is u = [delta, a_cmd] = Kdelta*x + v.
  So the dataset "U" often stores the solver variable "v", not the physical input "u".

FILES EXPECTED IN mpc_parameters/*.txt
-------------------------------------
1) Tf.txt        scalar   (horizon duration)
2) Q.txt         7x7
3) R.txt         2x2
4) P.txt         7x7
5) K.txt         7x2   (will be transposed to 2x7 in code)
6) Kinit.txt     7x2   (same)
7) Kdelta.txt    7x2   (same)
8) Lx.txt        7x(2*nu)  i.e. 7x4  (initially zeros recommended)
9) Lu.txt        2x(2*nu)  i.e. 2x4  (encodes input bounds)
10) Ls.txt       1x(2*nu)  i.e. 1x4  (initially zeros recommended)
11) alpha.txt    scalar
12) alpha_s.txt  scalar
13) rho_c.txt    scalar
14) wbar.txt     scalar

Suggested starter values for the txt files are included at the bottom of this file.
"""

import fire

from plot import *  # note: your plot.py might assume nx=4; you may need to update it
from dynamics.f import f

from pathlib import Path
import sys
import os
import subprocess

import numpy as np
import scipy.linalg
from scipy.integrate import odeint
import math

from casadi import SX, vertcat
from acados_template import (
    AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim
)

# Project imports (SOEAMPC utilities)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from soeampc.datasetutils import (
    import_dataset, merge_parallel_jobs, get_date_string,
    merge_single_parallel_job, print_dataset_statistics
)
from soeampc.mpcproblem import MPCQuadraticCostLxLu
from soeampc.samplempc import sample_dataset_from_mpc, computetime_test_fwd_sim
from soeampc.sampler import RandomSampler


np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 200

# file path of this example folder
fp = Path(os.path.dirname(__file__))
os.chdir(fp)


# ============================================================
# ACADOS models
# ============================================================

def export_vehicle_ode_model():
    """
    Build the implicit ODE model for ACADOS with augmented state [x; s].

    Plant state x (nx=7):
        [px, py, psi, v, r, beta, a]

    Auxiliary "robustness bookkeeping" state s (1):
        sdot = -rho*s + w_bar

    OCP decision variable is v (nu=2), but applied input is:
        u = Kdelta*x + v
    """

    # auxiliary state parameters for sdot = -rho*s + w_bar
    rho = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'rho_c.txt'), delimiter=','))
    w_bar = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'wbar.txt'), delimiter=','))

    # NEW model dimensions
    nx = 7
    nu = 2

    # Kdelta is stored as (nx,nu) in file, we reshape and transpose to (nu,nx)
    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Kdelta.txt'), delimiter=','),
        (nx, nu)
    ).T

    model_name = 'vehicle_dyn'

    # Symbolic variables for implicit ODE: xdot is an algebraic variable for ACADOS implicit form
    x = SX.sym('x', nx, 1)
    xdot = SX.sym('xdot', nx, 1)

    # auxiliary state s
    s = SX.sym('s')
    sdot = SX.sym('sdot')

    # OCP control variable (named v in this framework)
    v = SX.sym('v', nu, 1)

    # applied input to the physical model
    u = Kdelta @ x + v

    # physical dynamics (CasADi function in dynamics/f.py)
    fx = f(x, u)  # list of length 7

    # implicit dynamics constraint: f(x,u) - xdot = 0
    # plus auxiliary dynamics: -rho*s + w_bar - sdot = 0
    f_impl = vertcat(vertcat(*fx) - xdot,
                     -rho * s + w_bar - sdot)

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = vertcat(x, s)           # augmented state: dimension 8
    model.xdot = vertcat(xdot, sdot)  # augmented xdot
    model.u = v                       # ACADOS control = solver variable
    model.p = []
    model.name = model_name
    return model


def export_vehicle_sim_model():
    """
    Simulation model WITHOUT auxiliary state s.
    Used only for forward-simulation timing benchmarks.
    """
    nx = 7
    nu = 2
    model_name = 'vehicle_dyn'

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


# ============================================================
# Main sampling routine
# ============================================================

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
    Generate a dataset by repeatedly solving NMPC for random initial states.

    Steps:
      1) Read parameters from mpc_parameters/*.txt
      2) Build OCP (AcadosOcp)
      3) Create a RandomSampler for initial states x0
      4) For each x0:
         - create a new solver instance (avoid warm-start coupling across samples)
         - set x(0)=x0, s(0)=0
         - build an init guess (Xinit, Uinit) by forward sim using Kinit
         - solve OCP
         - store optimal trajectories
      5) Write dataset file via sample_dataset_from_mpc(...)
    """

    print("\n\n===============================================")
    print("Setting up ACADOS OCP problem (vehicle NEW MODEL)")
    print("===============================================\n")

    # --- auxiliary state dynamics parameters ---
    rho = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'rho_c.txt'), delimiter=','))
    w_bar = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'wbar.txt'), delimiter=','))

    # --- base dimensions ---
    nx = 7
    nu = 2

    # --- read Kdelta (nu x nx) ---
    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath('mpc_parameters', 'Kdelta.txt'), delimiter=','),
        (nx, nu)
    ).T
    print("Kdelta (nu x nx)=\n", Kdelta, "\n")

    # --- create OCP object and attach model ---
    ocp = AcadosOcp()
    model = export_vehicle_ode_model()
    ocp.model = model

    # horizon
    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'Tf.txt'), delimiter=','))
    N = 30  # keep N fixed here; you can also read it from file if desired
    ocp.dims.N = N

    # augmented state dimension includes s
    nx_ = model.x.size()[0]      # 8
    nx = nx_ - 1                 # 7
    nu = model.u.size()[0]       # 2

    # output dimensions for linear least squares
    # y = [x_aug; v]  -> size ny = nx_ + nu
    ny = nx_ + nu
    ny_e = nx_

    # --- init guess for s trajectory (consistent with its own ODE) ---
    # sdot = -rho*s + w_bar, s(0)=0
    Sinit = odeint(lambda y, t: -rho * y + w_bar, 0, np.linspace(0, Tf, N + 1))
    print("Sinit =\n", Sinit, "\n")

    # --- read cost matrices ---
    Q = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'Q.txt'), delimiter=','), (nx, nx))
    P = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'P.txt'), delimiter=','), (nx, nx))
    R = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'R.txt'), delimiter=','), (nu, nu))

    # include cost on s as 1.0 by default
    Q_ = scipy.linalg.block_diag(Q, 1.0)
    P_ = scipy.linalg.block_diag(P, 1.0)

    # --- feedback gains (stored as 7x2; we transpose to 2x7) ---
    K = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'K.txt'), delimiter=','), (nx, nu)).T
    Kinit = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'Kinit.txt'), delimiter=','), (nx, nu)).T

    # terminal set scaling parameter
    alpha_f = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'alpha.txt'), delimiter=','))

    print("Q=\n", Q, "\n")
    print("R=\n", R, "\n")
    print("P=\n", P, "\n")
    print("K (nu x nx)=\n", K, "\n")
    print("Kinit (nu x nx)=\n", Kinit, "\n")

    # ============================================================
    # Cost definition (Linear least squares)
    # ============================================================
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # stage cost: y = Vx*x_aug + Vu*v, minimize ||y - yref||_W
    ocp.cost.W = scipy.linalg.block_diag(Q_, R)
    ocp.cost.W_e = P_

    # mapping from x_aug into y
    ocp.cost.Vx = np.zeros((ny, nx_))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)  # penalize x states
    # (the last state of x_aug is s; its weight is in Q_)

    # mapping from solver variable v into y
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx_:, :nu] = np.eye(nu)  # penalize solver variable v

    # terminal cost mapping
    ocp.cost.Vx_e = np.zeros((ny_e, nx_))
    ocp.cost.Vx_e[:nx, :nx] = np.eye(nx)

    # references are zero by default -> this is a regulator-to-zero dataset
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # ============================================================
    # Constraints
    # We use one-sided linear constraints:
    #     C*[x;s] + D*v <= 1
    # and -1e5 <= ... (lower bound), so effectively only upper bounds.
    #
    # Simplest safe start: ONLY INPUT BOUNDS (delta and a_cmd).
    # ============================================================
    nxconstr = 0
    nuconstr = 2 * nu           # +u and -u for each input => 4
    nconstr = nxconstr + nuconstr

    # read Lx, Lu, Ls from txt
    # expected file shapes: Lx.txt (7x4), Lu.txt (2x4), Ls.txt (1x4)
    Lx = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'Lx.txt'), delimiter=','), (nx, nconstr)).T
    Lu = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'Lu.txt'), delimiter=','), (nu, nconstr)).T
    Ls = np.reshape(np.genfromtxt(fp.joinpath('mpc_parameters', 'Ls.txt'), delimiter=','), (1, nconstr)).T

    # We bound APPLIED input u = Kdelta*x + v, but the solver variable is v.
    # Starting from: Lu*u <= 1
    # Substitute u = Kdelta*x + v:
    #   Lu*(Kdelta*x + v) <= 1  ->  (Lu*Kdelta) x + Lu*v <= 1
    #
    # So we patch Lx rows for the input constraints:
    Lx[nxconstr:nxconstr + nu, :] = Lu[nxconstr:nxconstr + nu] @ Kdelta
    Lx[nxconstr + nu:nxconstr + 2 * nu, :] = Lu[nxconstr + nu:nxconstr + 2 * nu] @ Kdelta

    print("Lx (nconstr x nx)=\n", Lx, "\n")
    print("Lu (nconstr x nu)=\n", Lu, "\n")
    print("Ls (nconstr x 1)=\n", Ls, "\n")

    ocp.constraints.C = np.hstack((Lx, Ls))  # shape (nconstr, nx+1)
    ocp.constraints.D = Lu                    # shape (nconstr, nu)
    ocp.constraints.lg = -1e5 * np.ones(nconstr)
    ocp.constraints.ug = np.ones(nconstr)

    # Soft constraints (slacks on all constraints)
    ocp.constraints.Jsg = np.eye(nconstr)
    L2_pen = 1e6
    L1_pen = 1e4
    ocp.cost.Zl = L2_pen * np.ones((nconstr,))
    ocp.cost.Zu = L2_pen * np.ones((nconstr,))
    ocp.cost.zl = L1_pen * np.ones((nconstr,))
    ocp.cost.zu = L1_pen * np.ones((nconstr,))

    # ============================================================
    # Terminal set constraint:
    #     x^T P x <= (alpha_f - alpha_s * s_T)^2
    # The code uses a closed-form bound for s(T) under sdot=-rho*s+w_bar.
    # ============================================================
    alpha_s = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'alpha_s.txt'), delimiter=','))

    ocp.constraints.lh_e = np.array([-1e5])
    ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]

    # s(T) for sdot=-rho*s+w_bar, s(0)=0:
    # s(T) = (1 - exp(-rho*T))/rho * w_bar
    alpha = alpha_f - alpha_s * (1 - math.exp(-rho * Tf)) / rho * w_bar
    if alpha < 0:
        raise Exception("Terminal set size alpha_f - alpha_s*s_T is negative:", alpha)

    ocp.constraints.uh_e = np.array([alpha ** 2])

    # initial augmented state constraint target (will be overwritten per sample)
    ocp.constraints.x0 = np.zeros(nx_)

    # ============================================================
    # Build MPC wrapper object (used by dataset generation code)
    # ============================================================
    mpc = MPCQuadraticCostLxLu(
        f, nx, nu, N, Tf, Q, R, P, alpha_f,
        K, Lx, Lu, Kdelta, alpha_reduced=alpha, S=Sinit, Ls=Ls
    )
    mpc.name = model.name

    # ============================================================
    # Solver settings (reasonable defaults)
    # ============================================================
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

    # ============================================================
    # Random sampling bounds for initial state x0
    # x0 = [px, py, psi, v, r, beta, a]
    # ============================================================
    xmin = np.array([
        -10.0,      # px [m]
        -10.0,      # py [m]
        -math.pi,   # psi [rad]
        0.0,        # v [m/s]
        -1.0,       # r [rad/s]
        -0.35,      # beta [rad]  (approx +/-20 deg)
        -3.0        # a [m/s^2]
    ])
    xmax = np.array([
        +10.0,
        +10.0,
        +math.pi,
        25.0,
        +1.0,
        +0.35,
        +3.0
    ])

    # ============================================================
    # Derive bounds for SOLVER VARIABLE v from Lu (at x=0):
    # - You are encoding bounds in Lu such that Lu*u <= 1 corresponds to:
    #     +delta <= delta_max, -delta <= delta_max
    #     +a_cmd <= a_max,     -a_cmd <= a_max
    #
    # In this framework, bounds are applied to u = Kdelta*x + v, so:
    #   v_min(x) = umin - Kdelta*x
    #   v_max(x) = umax - Kdelta*x
    # ============================================================
    umax = np.array([1.0 / Lu[nxconstr + i, i] for i in range(nu)])
    umin = np.array([1.0 / Lu[nxconstr + nu + i, i] for i in range(nu)])
    print("\nDerived base bounds for solver variable v (at x=0):")
    print("umin =", umin)
    print("umax =", umax)

    # sampler object
    sampler = RandomSampler(numberofsamples, mpc.nx, randomseed, xmin, xmax)

    # ============================================================
    # Solve MPC once for a given x0 (used by dataset generator)
    # ============================================================
    def run(x0, verbose=False):
        # new solver instance each sample -> avoids cross-sample warm start coupling
        acados_ocp_solver = AcadosOcpSolver(
            ocp, json_file='acados_ocp_' + model.name + '.json',
            build=False, generate=False
        )

        # fix initial augmented state: x(0)=x0, s(0)=0
        acados_ocp_solver.set(0, "lbx", np.append(x0, 0.0))
        acados_ocp_solver.set(0, "ubx", np.append(x0, 0.0))

        # initial guess trajectories (simple linear interpolation to zero)
        Xinit = np.linspace(x0, np.zeros(nx), N + 1)
        Uinit = np.zeros((N, nu))

        # forward-sim init guess using Kinit feedback
        for i in range(N):
            Uinit[i] = Kinit @ Xinit[i]

            # clip solver variable v such that APPLIED input u stays within bounds:
            # u = Kdelta*x + v -> v bounds shift by -Kdelta*x
            Uinit[i] = np.clip(
                Uinit[i],
                umin - Kdelta @ Xinit[i],
                umax - Kdelta @ Xinit[i]
            )

            # one-step forward simulation using wrapper dynamics
            Xinit[i + 1] = mpc.forward_simulate_single_step(Xinit[i], Uinit[i])

        if verbose:
            print("\nx0=\n", x0)
            print("\nXinit=\n", Xinit)
            print("\nUinit=\n", Uinit)
            print("\nfeasible=", mpc.feasible(Xinit, Uinit, verbose=True))

        # load init guess into solver
        for i in range(N):
            acados_ocp_solver.set(i, "x", np.append(Xinit[i], Sinit[i]))
            acados_ocp_solver.set(i, "u", Uinit[i])

        # solve OCP
        status = acados_ocp_solver.solve()

        # extract solution
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

    # ============================================================
    # dataset generation (calls run(x0) many times)
    # ============================================================
    _, _, _, _, outfile = sample_dataset_from_mpc(
        mpc, run, sampler, experimentname, verbose=verbose
    )
    print("Outfile", outfile)

    # ============================================================
    # Optional plotting / quick sanity check
    # ============================================================
    if showplot:
        x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)

        # (1) feasibility scatter: px0 vs py0
        plot_feas(x0dataset[:, 0], x0dataset[:, 1],title=r"feasibility plot")
        # (2) feasibility scatter: px0 vs py0

        feas_pts = np.column_stack((x0dataset[:, 0], x0dataset[:, 1]))
        plot_feas_notfeas(feas_pts, None)

        # (2) Trajectory plots (only if stored as trajectories)
        # NOTE: your plot functions may assume nx=4.
        if Xdataset.ndim == 3 and Udataset.ndim == 3:
            K_show = 1
            rng = np.random.default_rng(0)
            n_avail = Xdataset.shape[0]
            idxs = rng.choice(n_avail, size=min(K_show, n_avail), replace=False)

            V_list = [Udataset[i] for i in idxs]  # stored solver variable v (nu=2)
            X_list = [Xdataset[i] for i in idxs]
            labels = [f"dataset sample {i}" for i in idxs]

            # v over horizon
            #plot_vehicle_ol_V(mpc, V_list, labels=labels, plt_show=True)

            # predicted trajectories and bounds
            #plot_vehicle_ol(
            #    mpc, V_list, X_list,
            #    labels=labels,
            #    plt_show=True,
            #    limits={"umin": umin.tolist(), "umax": umax.tolist()}
            #)

            plot_vehicle_ol_grid_3x3_7state(
                mpc,
                V_list,  # (N,2) per traj (solver variable v)
                X_list,  # (N+1,7) per traj
                labels=labels,
                plt_show=True,
                limits={"umin": umin.tolist(), "umax": umax.tolist()},
                input_is_v=True,
            )

            V_list = [Udataset[i] for i in idxs+1]  # NOTE: stored "U" is actually v in this setup
            X_list = [Xdataset[i] for i in idxs+1]
            labels = [f"dataset sample {i}" for i in idxs+1]

            plot_vehicle_ol_grid_3x3_7state(
                mpc,
                V_list,  # (N,2) per traj (solver variable v)
                X_list,  # (N+1,7) per traj
                labels=labels,
                plt_show=True,
                limits={"umin": umin.tolist(), "umax": umax.tolist()},
                input_is_v=True,
            )

            feas_list = [np.ones(V.shape[0], dtype=int) for V in V_list]
           # plot_vehicle_cl(
           #     mpc,
           #     Utraj=V_list,
           #     Xtraj=X_list,
           #     feasible=feas_list,
           #     labels=labels,
           #     plt_show=True,
           #     limits={"umin": umin.tolist(), "umax": umax.tolist()}
           # )

            plot_vehicle_cl_grid_3x3(
                mpc,
                Utraj=V_list,
                Xtraj=X_list,
                feasible=feas_list,
                labels=labels,
                plt_show=True,
                limits={"umin": umin.tolist(), "umax": umax.tolist()},
                input_is_v=True,
            )
        else:
            print("Dataset not stored as trajectories. Shapes:", Xdataset.shape, Udataset.shape)

    return outfile


# ============================================================
# Parallel sampling helper
# ============================================================

def parallel_sample_mpc(instances=16, samplesperinstance=int(1e5), prefix="Cluster"):
    """
    Spawn multiple processes, each writing its own dataset chunk,
    then merge all chunks into a single dataset.
    """
    now = get_date_string()
    fp_local = Path(os.path.abspath(os.path.dirname(__file__)))

    print("\n\n===============================================")
    print("Running", instances, "processes to produce", samplesperinstance, "datapoints each")
    print("===============================================\n")

    os.chdir(fp_local)
    processes = []
    common_name = prefix + "_" + str(now) + "_"

    for i in range(instances):
        experimentname = common_name + "_" + str(i) + "_"
        command = [
            "python3", "samplempc.py",
            "sample_mpc",
            "--showplot=False",
            "--randomseed=None",
            "--experimentname=" + experimentname,
            "--numberofsamples=" + str(samplesperinstance),
            "--generate=False"
        ]
        with open(fp_local.joinpath('logs', experimentname + ".log"), "wb") as out:
            p = subprocess.Popen(command, stdout=out, stderr=out)
            processes.append(p)

    for p in processes:
        p.wait()

    merge_parallel_jobs([common_name], new_dataset_name=common_name[:-1])


# ============================================================
# Forward sim compute-time benchmark
# ============================================================

def computetime_test_fwd_sim_vehicle(dataset="latest"):
    """
    Benchmark simulation time using AcadosSimSolver with the SIM model (no s-state).
    """
    name = 'vehicle_dyn'
    model = export_vehicle_sim_model()

    Tf = float(np.genfromtxt(fp.joinpath('mpc_parameters', 'Tf.txt'), delimiter=','))
    N = 30

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


# ============================================================
# CLI entrypoint
# ============================================================

if __name__ == "__main__":
    fire.Fire({
        'sample_mpc': sample_mpc,
        'parallel_sample_mpc': parallel_sample_mpc,
        'merge_single_parallel_job': merge_single_parallel_job,
        'print_dataset_statistics': print_dataset_statistics,
        'computetime_test_fwd_sim_vehicle': computetime_test_fwd_sim_vehicle,
    })
