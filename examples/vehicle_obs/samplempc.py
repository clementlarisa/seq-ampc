# =========================
# file: samplempc.py
# path: examples/vehicle_obs/samplempc.py
# =========================
"""
samplempc.py (vehicle_obs) — NMPC dataset generation WITH HARD obstacle avoidance constraints.

Model
-----
Kinematic bicycle (continuous time), defined in dynamics/f.py:

    x = [px, py, psi, v]
    u = [delta, a]

ACADOS decision variable is v_dec (nu=2), applied input is:
    u = Kdelta @ x + v_dec

Obstacle avoidance (HARD constraints)
------------------------------------
We add nonlinear path constraints with parameters:
    p = [o1x, o1y, o2x, o2y, r_safe]

Constraints (must hold at every stage k):
    c1(x,p) = (px-o1x)^2 + (py-o1y)^2 - r_safe^2 >= 0
    c2(x,p) = (px-o2x)^2 + (py-o2y)^2 - r_safe^2 >= 0

Inactive obstacles
------------------
We keep nh=2 always, but "disable" obstacles by setting their centers to FAR=100:
    o = (FAR, FAR)  => distance is huge => constraint always satisfied.

NN training distribution (20/40/40)
-----------------------------------
Deterministic per x0:
    20%: 0 obstacles active
    40%: 1 obstacle active
    40%: 2 obstacles active

Obstacles are generated "in the way" w.r.t. goal=(0,0) with lateral noise,
so they actually matter for avoidance most of the time.

Dataset export alignment
------------------------
sample_dataset_from_mpc(...) may skip infeasible solves.
Therefore after export we reload x0dataset and recompute obstacles for exactly
the saved x0 samples and save:

    datasets/<outfile>/P_obstacles.npy   shape (Nsaved, 5)
    datasets/<outfile>/N_active.npy      shape (Nsaved,)

Validation
----------
A deterministic example forces exactly ONE obstacle on the straight line from x0->goal,
so you can clearly see avoidance in the plot.

Plots
-----
Uses plot.py:
  - plot_vehicle_ol_grid_2x3(...)  (with XY + obstacle circles + clearance)
  - plot_feas(...)
"""

import fire
import os
import sys
import subprocess
import math
import hashlib
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from casadi import SX, vertcat
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel, AcadosSim

from dynamics.f import f

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from seqampc.datasetutils import (
    import_dataset,
    merge_parallel_jobs,
    get_date_string,
    merge_single_parallel_job,
    print_dataset_statistics,
)
from seqampc.mpcproblem import MPCQuadraticCostLxLu
from seqampc.samplempc import sample_dataset_from_mpc, computetime_test_fwd_sim
from seqampc.sampler import RandomSampler

from plot import plot_feas, plot_vehicle_ol_grid_2x3, plot_vehicle_cl_grid_2x3  # (plot_vehicle_cl optional)

np.set_printoptions(edgeitems=3)
np._core.arrayprint._line_width = 200  # avoids numpy.core deprecation warning

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

# =============================================================================
# Obstacle generation (deterministic per x0)
# =============================================================================

FAR = 100  # sentinel for "inactive obstacle"


def _stable_hash_seed_from_x0(x0: np.ndarray, base_seed: int) -> int:
    """
    Deterministic 32-bit RNG seed from x0 + base_seed.

    - Round x0 to fixed decimals (stability)
    - Hash with sha256
    - Use first 32 bits as seed
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    x0r = np.round(x0, decimals=6)
    payload = (str(base_seed) + "|" + ",".join(f"{v:+.6f}" for v in x0r)).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:8], 16)

def _stable_hash_seed_from_pos(x0: np.ndarray, base_seed: int) -> int:
    """Deterministic seed from (px, py) + base_seed."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    payload = (str(base_seed) + "|" + f"{x0[0]:+.6f},{x0[1]:+.6f}").encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:8], 16)

def _sample_two_obstacles_staggered(
    x0,
    rng,
    goal_xy=(0.0, 0.0),
    arena_xy=(20.0, 20.0),
    min_dist_to_x0=2.0,
    lateral_sigma=0.5,
    t1_interval=(0.3, 0.45),
    t2_interval=(0.55, 0.8),
):
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    start = np.array([x0[0], x0[1]], dtype=float)
    goal = np.array(goal_xy, dtype=float)

    d = goal - start
    dn = np.linalg.norm(d)
    if dn < 1e-6:
        return (FAR, FAR), (FAR, FAR)

    dir_u = d / dn
    perp = np.array([-dir_u[1], dir_u[0]])

    # first obstacle (earlier along path)
    t1 = rng.uniform(*t1_interval)
    p1 = start + t1 * d
    p1 = p1 + rng.normal(0, lateral_sigma) * perp

    # second obstacle (later along path)
    t2 = rng.uniform(*t2_interval)
    p2 = start + t2 * d
    p2 = p2 + rng.normal(0, lateral_sigma) * perp

    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))

def _sample_obstacle_in_way(
    x0: np.ndarray,
    rng: np.random.Generator,
    goal_xy=(0.0, 0.0),
    arena_xy=(20.0, 20.0),
    min_dist_to_x0=2.0,
    t_interval=(0.3, 0.75),
    lateral_sigma=1.0,
):
    """
    Sample ONE obstacle likely "in the way" along the line start->goal.

    Steps:
      1) take direction start->goal
      2) pick point start + t*(goal-start)
      3) add perpendicular offset ~ N(0, lateral_sigma)
      4) clamp to arena bounds
      5) ensure not too close to start
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    px0, py0 = float(x0[0]), float(x0[1])

    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    ax, ay = float(arena_xy[0]), float(arena_xy[1])

    start = np.array([px0, py0], dtype=float)
    goal = np.array([gx, gy], dtype=float)

    d = goal - start
    dn = np.linalg.norm(d)
    if dn < 1e-6:
        # start is already at goal: random obstacle in arena
        return float(rng.uniform(-ax, ax)), float(rng.uniform(-ay, ay))

    dir_u = d / dn
    perp = np.array([-dir_u[1], dir_u[0]], dtype=float)

    t = rng.uniform(t_interval[0], t_interval[1])
    p_line = start + t * d

    lat = rng.normal(loc=0.0, scale=lateral_sigma)
    p = p_line + lat * perp

    p[0] = float(np.clip(p[0], -ax, ax))
    p[1] = float(np.clip(p[1], -ay, ay))

    if np.hypot(p[0] - px0, p[1] - py0) < float(min_dist_to_x0):
        # push it a bit away from start in direction of goal
        p = start + float(min_dist_to_x0) * (dir_u)
        p[0] = float(np.clip(p[0], -ax, ax))
        p[1] = float(np.clip(p[1], -ay, ay))

    return float(p[0]), float(p[1])


def obstacle_params_from_x0(
    x0: np.ndarray,
    base_seed: int,
    probs_0_1_2=(0.2, 0.4, 0.4),  # 20/40/40
    r_safe=0.8,
    arena_xy=(6.0, 6.0),
    min_dist_to_x0=1.5,
):
    """
    Deterministically generate obstacle parameter vector p and number of active obstacles.

    Returns:
      p = [o1x,o1y,o2x,o2y,r_safe]
      n_active in {0,1,2}

    Inactive obstacle convention:
      o=(FAR,FAR) disables the constraint effectively.
    """
    seed = _stable_hash_seed_from_x0(x0, base_seed)
    rng = np.random.default_rng(seed)

    p0, p1, p2 = probs_0_1_2
    #r = rng.random()
    r = (seed % 10_000_000) / 10_000_000.0

    if r < p0:
        n_active = 0
    elif r < p0 + p1:
        n_active = 1
    else:
        n_active = 2

    if n_active == 0:
        o1 = (FAR, FAR)
        o2 = (FAR, FAR)
    elif n_active == 1:
        o1 = _sample_obstacle_in_way(
            x0, rng, arena_xy=arena_xy, min_dist_to_x0=min_dist_to_x0
        )
        o2 = (FAR, FAR)
    else:
        o1, o2 = _sample_two_obstacles_staggered(
            x0,
            rng,
            arena_xy=arena_xy,
            min_dist_to_x0=min_dist_to_x0,
            lateral_sigma=0.4,
            t1_interval=(0.3, 0.45),
            t2_interval=(0.6, 0.85),
        )

    p = np.array([o1[0], o1[1], o2[0], o2[1], float(r_safe)+float(0.05)], dtype=float)
    return p, int(n_active)


def obstacles_list_from_p(p: np.ndarray):
    """
    Convert p=[o1x,o1y,o2x,o2y,r_safe] to a list of active obstacle centers.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    o1 = (float(p[0]), float(p[1]))
    o2 = (float(p[2]), float(p[3]))

    obs = []
    if abs(o1[0]) < FAR * 0.1 and abs(o1[1]) < FAR * 0.1:
        obs.append(o1)
    if abs(o2[0]) < FAR * 0.1 and abs(o2[1]) < FAR * 0.1:
        obs.append(o2)
    return obs


# =============================================================================
# ACADOS model export: implicit DAE + obstacle constraints
# =============================================================================

def export_vehicle_ode_model():
    """
    Implicit model for ACADOS OCP with:
      - augmented state: [px, py, psi, v, s]
      - decision input:  v_dec (2)
      - applied input:   u = Kdelta @ x + v_dec
      - auxiliary state: sdot = -rho*s + w_bar
      - nonlinear constraints (path): c1>=0, c2>=0 using obstacle params p
    """
    rho = float(np.genfromtxt(fp.joinpath("mpc_parameters", "rho_c.txt"), delimiter=","))
    w_bar = float(np.genfromtxt(fp.joinpath("mpc_parameters", "wbar.txt"), delimiter=","))

    nx = 4
    nu = 2

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Kdelta.txt"), delimiter=","),
        (nx, nu),
    ).T

    model_name = "vehicle_obs"

    # states
    x = SX.sym("x", nx, 1)         # [px,py,psi,v]
    xdot = SX.sym("xdot", nx, 1)

    # auxiliary s
    s = SX.sym("s")
    sdot = SX.sym("sdot")

    # decision variable
    v_dec = SX.sym("v", nu, 1)
    u = Kdelta @ x + v_dec

    # physical dynamics
    fx = f(x, u)  # list [pxdot,pydot,psidot,vdot]
    f_impl = vertcat(vertcat(*fx) - xdot, -rho * s + w_bar - sdot)

    # parameters: p=[o1x,o1y,o2x,o2y,r_safe]
    o1x = SX.sym("o1x")
    o1y = SX.sym("o1y")
    o2x = SX.sym("o2x")
    o2y = SX.sym("o2y")
    r_safe = SX.sym("r_safe")
    p = vertcat(o1x, o1y, o2x, o2y, r_safe)

    px = x[0]
    py = x[1]

    # obstacle constraints (HARD)
    c1 = (px - o1x) ** 2 + (py - o1y) ** 2 - r_safe ** 2
    c2 = (px - o2x) ** 2 + (py - o2y) ** 2 - r_safe ** 2
    con_h_expr = vertcat(c1, c2)

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = vertcat(x, s)
    model.xdot = vertcat(xdot, sdot)
    model.u = v_dec
    model.p = p
    model.con_h_expr = con_h_expr
    model.name = model_name
    return model


def export_vehicle_sim_model():
    """
    Simulation model (for forward sim timing test), without s and without obstacles.
    """
    nx = 4
    nu = 2
    model_name = "vehicle"

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Kdelta.txt"), delimiter=","),
        (nx, nu),
    ).T

    x = SX.sym("x", nx, 1)
    xdot = SX.sym("xdot", nx, 1)

    v_dec = SX.sym("v", nu, 1)
    u = Kdelta @ x + v_dec

    fx = f(x, u)
    f_impl = vertcat(*fx) - xdot

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = v_dec
    model.p = []
    model.name = model_name
    return model


# =============================================================================
# Main sampling function
# =============================================================================

def sample_mpc(
    showplot=True,
    experimentname="",
    numberofsamples=int(5000),
    randomseed=42,
    verbose=False,
    generate=True,
    nlpiter=300,
    # obstacle config
    r_safe_value=0.5,
    obstacle_arena_xy=(25.0, 25.0),
    obstacle_min_dist_to_x0=3.0,
    probs_0_1_2=(0.2, 0.4, 0.4),  # 20/40/40
    # validation
    validate_one_example=False,
):
    """
    Dataset sampling:
      - solve NMPC for many random x0
      - for each x0 set obstacle params p deterministically from x0 (20/40/40)
      - export dataset using existing sample_dataset_from_mpc(...)
      - AFTER export: reconstruct and save obstacle arrays aligned with saved samples:
            P_obstacles.npy  shape (Nsaved, 5)
            N_active.npy     shape (Nsaved,)
    """
    print("\n\n===============================================")
    print("Setting up ACADOS OCP problem (vehicle + obstacle avoidance, mixed 20/40/40)")
    print("===============================================\n")

    rho = float(np.genfromtxt(fp.joinpath("mpc_parameters", "rho_c.txt"), delimiter=","))
    w_bar = float(np.genfromtxt(fp.joinpath("mpc_parameters", "wbar.txt"), delimiter=","))

    nx = 4
    nu = 2

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Kdelta.txt"), delimiter=","),
        (nx, nu),
    ).T
    print("Kdelta=\n", Kdelta, "\n")

    ocp = AcadosOcp()
    model = export_vehicle_ode_model()
    ocp.model = model

    # IMPORTANT: parameter_values must match np (=5)
    ocp.parameter_values = np.array([FAR, FAR, FAR, FAR, float(r_safe_value)], dtype=float)

    Tf = float(np.genfromtxt(fp.joinpath("mpc_parameters", "Tf.txt"), delimiter=","))
    N = 25
    ocp.dims.N = N

    nx_ = model.x.size()[0]  # 5 (4 + s)
    nx = nx_ - 1             # 4
    nu = model.u.size()[0]   # 2

    # -----------------------------
    # NO REVERSE: enforce v >= v_min at all stages (hard bound)
    # x_aug = [px, py, psi, v, s]  => v index = 3
    v_min = 0.0     # or 0.0 to allow standstill
    v_max = 15.0    # should match your intended max speed

    ocp.constraints.idxbx = np.array([3], dtype=int)
    ocp.constraints.lbx  = np.array([v_min], dtype=float)
    ocp.constraints.ubx  = np.array([v_max], dtype=float)

    ocp.constraints.idxbx_e = np.array([3], dtype=int)
    ocp.constraints.lbx_e   = np.array([v_min], dtype=float)
    ocp.constraints.ubx_e   = np.array([v_max], dtype=float)
    # -----------------------------


    # activate parameters for obstacles
    ocp.dims.np = 5

    # cost dims
    ny = nx_ + nu
    ny_e = nx_

    # warm-start s trajectory
    Sinit = odeint(lambda y, t: -rho * y + w_bar, 0, np.linspace(0, Tf, N + 1))

    # load cost weights
    Q = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "Q.txt"), delimiter=","), (nx, nx))
    P = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "P.txt"), delimiter=","), (nx, nx))
    R = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "R.txt"), delimiter=","), (nu, nu))

    Q_ = scipy.linalg.block_diag(Q, 1.0)
    P_ = scipy.linalg.block_diag(P, 1.0)

    K = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "K.txt"), delimiter=","), (nx, nu)).T
    Kinit = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "Kinit.txt"), delimiter=","), (nx, nu)).T

    alpha_f = float(np.genfromtxt(fp.joinpath("mpc_parameters", "alpha.txt"), delimiter=","))

    # LINEAR_LS cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q_, R)
    ocp.cost.W_e = P_

    ocp.cost.Vx = np.zeros((ny, nx_))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx_:, :nu] = np.eye(nu)

    ocp.cost.Vx_e = np.zeros((ny_e, nx_))
    ocp.cost.Vx_e[:nx, :nx] = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # linear constraints: C*[x;s] + D*v <= 1
    nxconstr = 0
    nuconstr = 2 * nu
    nconstr = nxconstr + nuconstr

    Lx = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Lx.txt"), delimiter=","),
        (nx, nconstr),
    ).T
    Lu = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Lu.txt"), delimiter=","),
        (nu, nconstr),
    ).T
    Ls = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Ls.txt"), delimiter=","),
        (1, nconstr),
    ).T

    # absorb Kdelta into Lx so bounds apply to u = Kdelta*x + v
    Lx[nxconstr:nxconstr + nu, :] = Lu[nxconstr:nxconstr + nu] @ Kdelta
    Lx[nxconstr + nu:nxconstr + 2 * nu, :] = Lu[nxconstr + nu:nxconstr + 2 * nu] @ Kdelta

    ocp.constraints.C = np.hstack((Lx, Ls))
    ocp.constraints.D = Lu
    ocp.constraints.lg = -1e5 * np.ones(nconstr)
    ocp.constraints.ug = np.ones(nconstr)

    # soft linear constraints (ONLY for g)
    L2_pen = 1e6
    L1_pen = 1e6
    ocp.constraints.Jsg = np.eye(nconstr)
    ocp.cost.Zl = L2_pen * np.ones((nconstr,))
    ocp.cost.Zu = L2_pen * np.ones((nconstr,))
    ocp.cost.zl = L1_pen * np.ones((nconstr,))
    ocp.cost.zu = L1_pen * np.ones((nconstr,))

    # nonlinear obstacle constraints: c1>=0, c2>=0  (HARD: no Jsh!)
    ocp.dims.nh = 2
    ocp.constraints.lh = np.array([0.0, 0.0], dtype=float)
    ocp.constraints.uh = np.array([1e12, 1e12], dtype=float)

    # terminal set: x' P x <= alpha^2 (shrunk by s)
    alpha_s = float(np.genfromtxt(fp.joinpath("mpc_parameters", "alpha_s.txt"), delimiter=","))
    ocp.constraints.lh_e = np.array([-1e5])
    ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]

    alpha = alpha_f - alpha_s * (1 - math.exp(-rho * Tf)) / rho * w_bar
    if alpha < 0:
        raise Exception("Terminal radius became negative:", alpha)
    ocp.constraints.uh_e = np.array([alpha ** 2])

    ocp.constraints.x0 = np.zeros(nx_)

    # python-side MPC object (only for init rollout)
    mpc = MPCQuadraticCostLxLu(
        f, nx, nu, N, Tf, Q, R, P, alpha_f,
        K, Lx, Lu, Kdelta, alpha_reduced=alpha, S=Sinit, Ls=Ls
    )
    mpc.name = model.name

    # solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hpipm_mode = "ROBUST"

    ocp.solver_options.qp_tol = 1e-8
    ocp.solver_options.levenberg_marquardt = 10.0
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.globalization_use_SOC = 1
    ocp.solver_options.line_search_use_sufficient_descent = 1
    ocp.solver_options.alpha_reduction = 0.1
    ocp.solver_options.alpha_min = 1e-4
    ocp.solver_options.regularize_method = "MIRROR"

    ocp.solver_options.tf = Tf
    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_max_iter = nlpiter

    if generate:
        _ = AcadosOcpSolver(ocp, json_file="acados_ocp_" + model.name + ".json")

    # sampler bounds
    xmin = np.array([-20.0, -20.0, -math.pi, 0.0])
    xmax = np.array([+20.0, +20.0, math.pi, 12.0])

    # input bounds derived from Lu (same approach as base code)
    umax = np.array([1.0 / Lu[nxconstr + i, i] for i in range(nu)])
    umin = np.array([1.0 / Lu[nxconstr + nu + i, i] for i in range(nu)])
    print("\numin=\n", umin)
    print("\numax=\n", umax)

    sampler = RandomSampler(numberofsamples, mpc.nx, randomseed, xmin, xmax)

    # =========================================================================
    # Solve one OCP instance (called by dataset generator)
    # =========================================================================
    def run_for_dataset(x0, verbose=False):
        """
        Must return exactly:
          X, U, status, computetime, number_iterations

        Obstacles are computed deterministically from x0 and set as parameters at all stages.
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)

        #base_seed = randomseed if randomseed is not None else 0
        #dx = 0.0 - x0[0]
        #dy = 0.0 - x0[1]
        #psi_nom = math.atan2(dy, dx)
        #seed = _stable_hash_seed_from_pos(x0, base_seed) ^ 0xA5A5A5A5
        #rng = np.random.default_rng(seed)
        #x0[2] = (psi_nom + rng.normal(0.0, 20.0 * math.pi / 180.0) + math.pi) % (2 * math.pi) - math.pi
        #print("psi_nom [rad] =", psi_nom, "  [deg] =", psi_nom * 180 / math.pi)

        solver = AcadosOcpSolver(
            ocp,
            json_file="acados_ocp_" + model.name + ".json",
            build=False,
            generate=False,
        )

        p_val, n_active = obstacle_params_from_x0(
            x0=x0,
            base_seed=randomseed if randomseed is not None else 0,
            probs_0_1_2=probs_0_1_2,
            r_safe=r_safe_value,
            arena_xy=obstacle_arena_xy,
            min_dist_to_x0=obstacle_min_dist_to_x0,
        )

        for k in range(N):
            solver.set(k, "p", p_val)
        solver.set(N, "p", p_val)

        solver.set(0, "lbx", np.append(x0, 0.0))
        solver.set(0, "ubx", np.append(x0, 0.0))

        # init guess
        Xinit = np.linspace(x0, np.zeros(nx), N + 1)
        Uinit = np.zeros((N, nu))
        for k in range(N):
            Uinit[k] = Kinit @ Xinit[k]
            Uinit[k] = np.clip(Uinit[k], umin - Kdelta @ Xinit[k], umax - Kdelta @ Xinit[k])
            Xinit[k + 1] = mpc.forward_simulate_single_step(Xinit[k], Uinit[k])

        for k in range(N):
            solver.set(k, "x", np.append(Xinit[k], Sinit[k]))
            solver.set(k, "u", Uinit[k])

        status = solver.solve()

        X = np.zeros((N + 1, nx))
        U = np.zeros((N, nu))

        for k in range(N):
            xk_aug = solver.get(k, "x")
            X[k, :] = xk_aug[:-1]
            U[k, :] = solver.get(k, "u")
        xN_aug = solver.get(N, "x")
        X[N, :] = xN_aug[:-1]

        computetime = float(solver.get_stats("time_tot"))
        number_iterations = float(solver.get_stats("sqp_iter"))

        if verbose:
            obs_list = obstacles_list_from_p(p_val)
            print("status:", status, "time:", computetime, "iters:", number_iterations)
            print("n_active:", n_active, "obstacles:", obs_list, "r_safe:", float(p_val[4]))

        return X, U, status, computetime, number_iterations

    # =========================================================================
    # Generate dataset
    # =========================================================================
    _, _, _, _, outfile = sample_dataset_from_mpc(mpc, run_for_dataset, sampler, experimentname, verbose=verbose)
    print("Outfile", outfile)

    # =========================================================================
    # Reconstruct obstacle arrays aligned with SAVED dataset samples
    # =========================================================================
    x0dataset, Udataset, Xdataset, computetimes = import_dataset(mpc, outfile)


    P_obs = np.zeros((x0dataset.shape[0], 5), dtype=float)
    N_active = np.zeros((x0dataset.shape[0],), dtype=int)

    for i in range(x0dataset.shape[0]):
        p_i, n_i = obstacle_params_from_x0(
            x0=x0dataset[i],
            base_seed=randomseed if randomseed is not None else 0,
            probs_0_1_2=probs_0_1_2,
            r_safe=r_safe_value,
            arena_xy=obstacle_arena_xy,
            min_dist_to_x0=obstacle_min_dist_to_x0,
        )
        P_obs[i, :] = p_i
        N_active[i] = n_i

    from seqampc.config import DATASETS_DIR
    dataset_root = DATASETS_DIR
    dataset_dir = dataset_root / outfile
    dataset_dir.mkdir(parents=True, exist_ok=True)
    #np.save(dataset_dir.joinpath("P_obstacles.npy"), P_obs)
    #np.save(dataset_dir.joinpath("N_active.npy"), N_active)
    # P_obstacles: (Nsaved, 5) = [o1x, o1y, o2x, o2y, r_safe]
    np.savetxt(
        dataset_dir.joinpath("P_obstacles.txt"),
        P_obs,
        fmt="%.10g",
        delimiter=" ",
        header="o1x o1y o2x o2y r_safe",
        comments="",
    )

    # N_active: (Nsaved,)
    np.savetxt(
        dataset_dir.joinpath("N_active.txt"),
        N_active.astype(int),
        fmt="%d",
        delimiter=" ",
        header="n_active",
        comments="",
    )

    uniq, cnt = np.unique(N_active, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
    print("Saved obstacle parameters to: P_obstacles.txt")
    print("Saved obstacle counts     to: N_active.txt")
    print("Obstacle count distribution:", dist)

    # =========================================================================
    # Validation: deterministic obstacle IN THE WAY (forced 1 obstacle)
    # =========================================================================
    if validate_one_example:
        x0_ex = np.array([4.0, 1.0, math.pi, 2.0], dtype=float)  # start right, heading left

        goal = np.array([0.0, 0.0], dtype=float)
        o1 = 0.5 * (x0_ex[:2] + goal)          # exact midpoint on straight line
        o2 = np.array([FAR, FAR], dtype=float) # inactive
        r_safe_test = r_safe_value  # oder kleiner testweise

        p_forced = np.array([o1[0], o1[1], o2[0], o2[1], float(r_safe_value)], dtype=float)

        solver = AcadosOcpSolver(
            ocp,
            json_file="acados_ocp_" + model.name + ".json",
            build=False,
            generate=False,
        )

        for k in range(N):
            solver.set(k, "p", p_forced)
        solver.set(N, "p", p_forced)

        solver.set(0, "lbx", np.append(x0_ex, 0.0))
        solver.set(0, "ubx", np.append(x0_ex, 0.0))

        Xinit = np.linspace(x0_ex, np.zeros(nx), N + 1)
        Uinit = np.zeros((N, nu))
        for k in range(N):
            Uinit[k] = Kinit @ Xinit[k]
            Uinit[k] = np.clip(Uinit[k], umin - Kdelta @ Xinit[k], umax - Kdelta @ Xinit[k])
            Xinit[k + 1] = mpc.forward_simulate_single_step(Xinit[k], Uinit[k])

        for k in range(N):
            solver.set(k, "x", np.append(Xinit[k], Sinit[k]))
            solver.set(k, "u", Uinit[k])

        status_ex = solver.solve()

        Xex = np.zeros((N + 1, nx))
        Uex = np.zeros((N, nu))
        for k in range(N):
            xk_aug = solver.get(k, "x")
            Xex[k, :] = xk_aug[:-1]
            Uex[k, :] = solver.get(k, "u")
        xN_aug = solver.get(N, "x")
        Xex[N, :] = xN_aug[:-1]

        Uap = np.zeros_like(Uex)
        for k in range(N):
            Uap[k, :] = (Kdelta @ Xex[k, :]).reshape(-1) + Uex[k, :]

        ct_ex = float(solver.get_stats("time_tot"))
        it_ex = float(solver.get_stats("sqp_iter"))

        obstacles = obstacles_list_from_p(p_forced)
        r_safe_ex = float(p_forced[4])

        print("\n--- Deterministic example solve (forced 1 obstacle IN THE WAY) ---")
        print("status:", status_ex, "time:", ct_ex, "iters:", it_ex)
        print("obstacle:", obstacles, "r_safe:", r_safe_ex)

        o = np.array(obstacles[0], dtype=float)
        d = np.sqrt((Xex[:, 0] - o[0]) ** 2 + (Xex[:, 1] - o[1]) ** 2)
        clr = d - r_safe_ex
        kmin = int(np.argmin(clr))
        print("min clearance =", float(clr[kmin]), "at k =", kmin, "pos =", (float(Xex[kmin, 0]), float(Xex[kmin, 1])))

        plot_vehicle_ol_grid_2x3(
            mpc,
            Vtraj=[Uap],  # <- applied input!
            Xtraj=[Xex],
            labels=[f"forced-1-obstacle applied-u (status={status_ex})"],
            plt_show=True,
            limits={"umin": umin.tolist(), "umax": umax.tolist()},
            input_is_v=True,  # <- wichtig: jetzt ist es u
            obstacles=[obstacles],
            r_safe=[r_safe_ex],
            show_xy=True,
            show_clearance=True,
            print_clearance=True,
        )

        plot_vehicle_cl_grid_2x3(
            mpc,
            Utraj=[Uex],  # oder [Vex]
            Xtraj=[Xex],
            feasible=None,
            labels=["cl test"],
            limits={"umin": umin.tolist(), "umax": umax.tolist()},
            input_is_v=True,  # wenn Uex = v_dec
            obstacles=[obstacles],
            r_safe=[r_safe_ex],
        )

    # =========================================================================
    # showplot: scatter + pick 3 dataset samples (2 with obstacles, 1 without)
    # =========================================================================
    if showplot:
        plot_feas(x0dataset[:, 0], x0dataset[:, 1])

        if not (Xdataset.ndim == 3 and Udataset.ndim == 3 and Xdataset.shape[0] > 0):
            print("[INFO] Dataset is not stored as full trajectories. Shapes:", Xdataset.shape, Udataset.shape)
            return outfile

        idx0 = np.where(N_active == 0)[0]
        idx1 = np.where(N_active == 1)[0]
        idx2 = np.where(N_active == 2)[0]

        print("Available counts:",
              f"n0={idx0.size}, n1={idx1.size}, n2={idx2.size}")

        rng_sel = np.random.default_rng(123)
        picks = []

        # want: [0, >=1, >=1]  (prefer 0,1,2)
        def pick_one(idxs, forbid):
            if idxs.size == 0:
                return None
            for _ in range(20):
                cand = int(rng_sel.choice(idxs, size=1, replace=False)[0])
                if cand not in forbid:
                    return cand
            # fallback (allow duplicate if unavoidable)
            return int(rng_sel.choice(idxs, size=1, replace=False)[0])

        # 1) try n_active=0
        p = pick_one(idx0, picks)
        if p is None:
            # if no 0s exist, at least pick the smallest available class
            p = pick_one(idx1 if idx1.size else idx2, picks)
        picks.append(p)

        # 2) try n_active=1 (or >=1)
        p = pick_one(idx1, picks)
        if p is None:
            p = pick_one(idx2, picks)
        if p is None:
            # ultimate fallback: anything
            p = int(rng_sel.integers(0, Xdataset.shape[0]))
        picks.append(p)

        # 3) try n_active=2 (or remaining >=1)
        p = pick_one(idx2, picks)
        if p is None:
            p = pick_one(idx1, picks)
        if p is None:
            p = int(rng_sel.integers(0, Xdataset.shape[0]))
        picks.append(p)

        # ensure unique, keep order, max 3
        uniq = []
        for p in picks:
            if p is not None and p not in uniq:
                uniq.append(p)
        picks = uniq[:3]

        print("\n--- Plotting 3 random dataset samples (1 no-obs, 2 with-obs) ---")
        for k, idx in enumerate(picks):
            Xh = Xdataset[idx]
            Uh = Udataset[idx]
            p_h = P_obs[idx]
            obs_h = obstacles_list_from_p(p_h)
            r_h = float(p_h[4])
            na = int(N_active[idx])

            print(f"  sample[{k}] idx={idx}: n_active={na}, obstacles={obs_h}, r_safe={r_h}")

            plot_vehicle_ol_grid_2x3(
                mpc,
                Vtraj=[Uh],
                Xtraj=[Xh],
                labels=[f"dataset OL idx={idx} (n_active={na})"],
                plt_show=True,
                limits={"umin": umin.tolist(), "umax": umax.tolist()},
                input_is_v=True,
                obstacles=[obs_h],
                r_safe=[r_h],
                show_xy=True,
                show_clearance=True,
                print_clearance=True,
                title=f"dataset| OL idx={idx} (n_active={na})"

            )

    return outfile


# =============================================================================
# Parallel sampling / merge / timing test
# =============================================================================

def parallel_sample_mpc(instances=16, samplesperinstance=int(1e5), prefix="Cluster"):
    now = get_date_string()
    fp_local = Path(os.path.abspath(os.path.dirname(__file__)))

    print("\n\n===============================================")
    print("Running", instances, "processes to produce", samplesperinstance, "datapoints each")
    print("===============================================\n")

    os.chdir(fp_local)
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
            "--generate=False",
        ]
        with open(fp_local.joinpath("logs", experimentname + ".log"), "wb") as out:
            p = subprocess.Popen(command, stdout=out, stderr=out)
            processes.append(p)

    for p in processes:
        p.wait()

    merge_parallel_jobs([parallel_experiments_common_name], new_dataset_name=parallel_experiments_common_name[:-1])


def computetime_test_fwd_sim_vehicle(dataset="latest"):
    name = "vehicle"
    model = export_vehicle_sim_model()

    Tf = float(np.genfromtxt(fp.joinpath("mpc_parameters", "Tf.txt"), delimiter=","))
    N = 10

    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = Tf / N
    sim.solver_options.integrator_type = "IRK"
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 1
    sim.solver_options.newton_iter = 3

    acados_integrator = AcadosSimSolver(sim, "acados_ocp_" + name + "_sim.json")

    def run(x0, V):
        X = np.zeros((V.shape[0] + 1, x0.shape[0]))
        X[0] = np.copy(x0)
        for i in range(len(V)):
            X[i + 1] = acados_integrator.simulate(x=X[i], u=V[i])
        return X

    computetime_test_fwd_sim(run, dataset)


if __name__ == "__main__":
    fire.Fire(
        {
            "sample_mpc": sample_mpc,
            "parallel_sample_mpc": parallel_sample_mpc,
            "merge_single_parallel_job": merge_single_parallel_job,
            "print_dataset_statistics": print_dataset_statistics,
            "computetime_test_fwd_sim_vehicle": computetime_test_fwd_sim_vehicle,
        }
    )