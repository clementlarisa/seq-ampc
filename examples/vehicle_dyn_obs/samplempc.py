# =========================
# file: samplempc_obs.py
# path: examples/vehicle_8state_obs/samplempc_obs.py
# =========================
"""
samplempc_obs.py — NMPC dataset generation WITH HARD obstacle avoidance constraints
for the 8-state single-track vehicle model with steering angle as state.

Plant model (continuous time), dynamics/f.py:
    x = [px, py, psi, v, r, beta, a, delta]
    u = [delta_dot, a_cmd]

ACADOS decision variable is v_dec (nu=2), applied input is:
    u = Kdelta @ x + v_dec

Obstacle avoidance (HARD constraints)
------------------------------------
Parameters:
    p = [o1x, o1y, o2x, o2y, r_safe]

Constraints (every stage):
    c1 = (px-o1x)^2 + (py-o1y)^2 - r_safe^2 >= 0
    c2 = (px-o2x)^2 + (py-o2y)^2 - r_safe^2 >= 0

Inactive obstacles:
    set center to FAR=100 -> always satisfied (given your arena).

After dataset export we recompute obstacles for saved x0 and store:
    datasets/<outfile>/P_obstacles.txt   shape (Nsaved, 5)
    datasets/<outfile>/N_active.txt      shape (Nsaved,)
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

from casadi import SX, vertcat
from acados_template import (
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSimSolver,
    AcadosModel,
    AcadosSim,
)

from dynamics.f import f

# Project imports (SOEAMPC utilities)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from soeampc.datasetutils import (
    import_dataset,
    merge_parallel_jobs,
    get_date_string,
    merge_single_parallel_job,
    print_dataset_statistics,
)
from soeampc.mpcproblem import MPCQuadraticCostLxLu
from soeampc.samplempc import sample_dataset_from_mpc, computetime_test_fwd_sim
from soeampc.sampler import RandomSampler

# plots
from plot import (
    plot_feas,
    plot_vehicle_ol_grid_3x3,
    plot_ctdistro,
)

np.set_printoptions(edgeitems=3)
np._core.arrayprint._line_width = 200

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

# =============================================================================
# Obstacle generation (deterministic per x0)
# =============================================================================

FAR = 100.0  # sentinel for "inactive obstacle"


def _stable_hash_seed_from_x0(x0: np.ndarray, base_seed: int) -> int:
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    x0r = np.round(x0, decimals=6)
    payload = (str(base_seed) + "|" + ",".join(f"{v:+.6f}" for v in x0r)).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:8], 16)

def _stable_hash_seed_from_pos(x0: np.ndarray, base_seed: int) -> int:
    """Deterministischer Seed nur aus (px,py) + base_seed."""
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

    t1 = rng.uniform(*t1_interval)
    p1 = start + t1 * d + rng.normal(0, lateral_sigma) * perp

    t2 = rng.uniform(*t2_interval)
    p2 = start + t2 * d + rng.normal(0, lateral_sigma) * perp

    ax, ay = float(arena_xy[0]), float(arena_xy[1])
    p1[0] = float(np.clip(p1[0], -ax, ax))
    p1[1] = float(np.clip(p1[1], -ay, ay))
    p2[0] = float(np.clip(p2[0], -ax, ax))
    p2[1] = float(np.clip(p2[1], -ay, ay))

    if np.linalg.norm(p1 - start) < min_dist_to_x0:
        p1 = start + min_dist_to_x0 * dir_u
    if np.linalg.norm(p2 - start) < min_dist_to_x0:
        p2 = start + (min_dist_to_x0 + 0.5) * dir_u

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
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    px0, py0 = float(x0[0]), float(x0[1])

    start = np.array([px0, py0], dtype=float)
    goal = np.array([float(goal_xy[0]), float(goal_xy[1])], dtype=float)
    ax, ay = float(arena_xy[0]), float(arena_xy[1])

    d = goal - start
    dn = np.linalg.norm(d)
    if dn < 1e-6:
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
        p = start + float(min_dist_to_x0) * dir_u
        p[0] = float(np.clip(p[0], -ax, ax))
        p[1] = float(np.clip(p[1], -ay, ay))

    return float(p[0]), float(p[1])


def obstacle_params_from_x0(
    x0: np.ndarray,
    base_seed: int,
    probs_0_1_2=(0.2, 0.4, 0.4),
    r_safe=1.2,
    arena_xy=(25.0, 25.0),
    min_dist_to_x0=3.0,
):
    """
    Returns:
      p = [o1x,o1y,o2x,o2y,r_safe]
      n_active in {0,1,2}
    """
    seed = _stable_hash_seed_from_x0(x0, base_seed)
    rng = np.random.default_rng(seed)

    p0, p1, p2 = probs_0_1_2
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
            x0, rng, arena_xy=arena_xy, min_dist_to_x0=min_dist_to_x0, lateral_sigma=0.8
        )
        o2 = (FAR, FAR)
    else:
        o1, o2 = _sample_two_obstacles_staggered(
            x0,
            rng,
            arena_xy=arena_xy,
            min_dist_to_x0=min_dist_to_x0,
            lateral_sigma=0.5,
            t1_interval=(0.3, 0.45),
            t2_interval=(0.6, 0.85),
        )

    p = np.array([o1[0], o1[1], o2[0], o2[1], float(r_safe)], dtype=float)
    return p, int(n_active)


def obstacles_list_from_p(p: np.ndarray):
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

def export_vehicle_ode_model_obs():
    """
    Implicit model for ACADOS OCP with:
      - plant state x (nx=8)
      - auxiliary s state (1)
      - decision input v_dec (nu=2), applied u=Kdelta@x + v_dec
      - parameters p=[o1x,o1y,o2x,o2y,r_safe]
      - nonlinear path constraints con_h_expr=[c1,c2]
    """
    rho = float(np.genfromtxt(fp.joinpath("mpc_parameters", "rho_c.txt"), delimiter=","))
    w_bar = float(np.genfromtxt(fp.joinpath("mpc_parameters", "wbar.txt"), delimiter=","))

    nx = 8
    nu = 2

    # Kdelta stored as 8x2 in txt -> reshape (nx,nu) then transpose -> (nu,nx)
    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Kdelta.txt"), delimiter=","),
        (nx, nu),
    ).T

    model_name = "vehicle_8state_obs"

    x = SX.sym("x", nx, 1)
    xdot = SX.sym("xdot", nx, 1)

    s = SX.sym("s")
    sdot = SX.sym("sdot")

    v_dec = SX.sym("v", nu, 1)
    u = Kdelta @ x + v_dec  # u = [delta_dot, a_cmd]

    fx = f(x, u)  # MUST be length 8
    if not isinstance(fx, (list, tuple)) or len(fx) != nx:
        raise RuntimeError(
            f"dynamics.f must return list of length {nx}, got {type(fx)} len={len(fx) if hasattr(fx,'__len__') else '??'}"
        )

    f_impl = vertcat(vertcat(*fx) - xdot, -rho * s + w_bar - sdot)

    # parameters p=[o1x,o1y,o2x,o2y,r_safe]
    o1x = SX.sym("o1x")
    o1y = SX.sym("o1y")
    o2x = SX.sym("o2x")
    o2y = SX.sym("o2y")
    r_safe = SX.sym("r_safe")
    p = vertcat(o1x, o1y, o2x, o2y, r_safe)

    px = x[0]
    py = x[1]
    c1 = (px - o1x) ** 2 + (py - o1y) ** 2 - r_safe ** 2
    c2 = (px - o2x) ** 2 + (py - o2y) ** 2 - r_safe ** 2
    con_h_expr = vertcat(c1, c2)

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = vertcat(x, s)           # augmented state: 9
    model.xdot = vertcat(xdot, sdot)
    model.u = v_dec
    model.p = p
    model.con_h_expr = con_h_expr
    model.name = model_name
    return model


def export_vehicle_sim_model():
    """
    Simulation model (no s, no obstacles) for forward sim benchmarks.
    """
    nx = 8
    nu = 2
    model_name = "vehicle_8state_sim"

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Kdelta.txt"), delimiter=","),
        (nx, nu),
    ).T

    x = SX.sym("x", nx, 1)
    xdot = SX.sym("xdot", nx, 1)

    v = SX.sym("v", nu, 1)
    u = Kdelta @ x + v

    fx = f(x, u)
    if not isinstance(fx, (list, tuple)) or len(fx) != nx:
        raise RuntimeError(f"dynamics.f must return list of length {nx} for sim model")

    f_impl = vertcat(*fx) - xdot

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = v
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
    nlpiter=400,
    # obstacle config
    r_safe_value=1.2,
    obstacle_arena_xy=(25.0, 25.0),
    obstacle_min_dist_to_x0=3.0,
    probs_0_1_2=(0.2, 0.4, 0.4),
):
    print("\n\n==============================================================")
    print("Setting up ACADOS OCP (8-state vehicle + HARD obstacle avoidance)")
    print("==============================================================\n")

    rho = float(np.genfromtxt(fp.joinpath("mpc_parameters", "rho_c.txt"), delimiter=","))
    w_bar = float(np.genfromtxt(fp.joinpath("mpc_parameters", "wbar.txt"), delimiter=","))

    nx = 8
    nu = 2

    Kdelta = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Kdelta.txt"), delimiter=","),
        (nx, nu),
    ).T
    print("Kdelta (nu x nx)=\n", Kdelta, "\n")

    ocp = AcadosOcp()
    model = export_vehicle_ode_model_obs()
    ocp.model = model

    ocp.parameter_values = np.array([FAR, FAR, FAR, FAR, float(r_safe_value)], dtype=float)
    ocp.dims.np = 5

    Tf = float(np.genfromtxt(fp.joinpath("mpc_parameters", "Tf.txt"), delimiter=","))
    N = 40
    ocp.dims.N = N

    # augmented dims
    nx_ = model.x.size()[0]    # 9 (8 plant + s)
    nx = nx_ - 1               # 8 (plant)
    nu = model.u.size()[0]     # 2

    ny = nx_ + nu
    ny_e = nx_

    # warm-start s trajectory
    Sinit = odeint(lambda y, t: -rho * y + w_bar, 0, np.linspace(0, Tf, N + 1))

    # cost
    Q = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "Q.txt"), delimiter=","), (nx, nx))
    P = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "P.txt"), delimiter=","), (nx, nx))
    R = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "R.txt"), delimiter=","), (nu, nu))

    Q_ = scipy.linalg.block_diag(Q, 1.0)
    P_ = scipy.linalg.block_diag(P, 1.0)

    K = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "K.txt"), delimiter=","), (nx, nu)).T
    Kinit = np.reshape(np.genfromtxt(fp.joinpath("mpc_parameters", "Kinit.txt"), delimiter=","), (nx, nu)).T

    alpha_f = float(np.genfromtxt(fp.joinpath("mpc_parameters", "alpha.txt"), delimiter=","))

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
    nconstr = nxconstr + nuconstr  # 4

    Lx = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Lx.txt"), delimiter=","),
        (nx, nconstr),
    ).T                           # (4,8)
    Lu = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Lu.txt"), delimiter=","),
        (nu, nconstr),
    ).T                           # (4,2)
    Ls = np.reshape(
        np.genfromtxt(fp.joinpath("mpc_parameters", "Ls.txt"), delimiter=","),
        (1, nconstr),
    ).T                           # (4,1)

    # absorb Kdelta into Lx so bounds apply to u = Kdelta*x + v
    Lx[nxconstr:nxconstr + nu, :] = Lu[nxconstr:nxconstr + nu] @ Kdelta
    Lx[nxconstr + nu:nxconstr + 2 * nu, :] = Lu[nxconstr + nu:nxconstr + 2 * nu] @ Kdelta

    ocp.constraints.C = np.hstack((Lx, Ls))
    ocp.constraints.D = Lu
    ocp.constraints.lg = -1e5 * np.ones(nconstr)
    ocp.constraints.ug = np.ones(nconstr)

    # soft linear constraints (ONLY for g)
    ocp.constraints.Jsg = np.eye(nconstr)
    L2_pen = 1e4
    L1_pen = 1e4
    ocp.cost.Zl = L2_pen * np.ones((nconstr,))
    ocp.cost.Zu = L2_pen * np.ones((nconstr,))
    ocp.cost.zl = L1_pen * np.ones((nconstr,))
    ocp.cost.zu = L1_pen * np.ones((nconstr,))

    # nonlinear obstacle constraints: c1>=0, c2>=0 (HARD)
    ocp.dims.nh = 2
    ocp.constraints.lh = np.array([0.0, 0.0], dtype=float)
    ocp.constraints.uh = np.array([1e12, 1e12], dtype=float)

    # hard state bounds: v (idx 3) and delta state (idx 7) in AUGMENTED x=[plant;s]
    delta_max = 25.0 * math.pi / 180.0
    v_min = 0.0
    v_max = 20.0

    ocp.constraints.idxbx = np.array([3, 7], dtype=int)
    ocp.constraints.lbx = np.array([v_min, -delta_max], dtype=float)
    ocp.constraints.ubx = np.array([v_max, +delta_max], dtype=float)

    ocp.constraints.idxbx_e = np.array([3, 7], dtype=int)
    ocp.constraints.lbx_e = np.array([v_min, -delta_max], dtype=float)
    ocp.constraints.ubx_e = np.array([v_max, +delta_max], dtype=float)

    # terminal set: x' P x <= alpha^2 (shrunk by s)
    alpha_s = float(np.genfromtxt(fp.joinpath("mpc_parameters", "alpha_s.txt"), delimiter=","))
    ocp.constraints.lh_e = np.array([-1e5])
    ocp.model.con_h_expr_e = ocp.model.x[:nx].T @ P @ ocp.model.x[:nx]
    alpha = alpha_f - alpha_s * (1 - math.exp(-rho * Tf)) / rho * w_bar
    if alpha < 0:
        raise Exception("Terminal radius became negative:", alpha)
    ocp.constraints.uh_e = np.array([alpha ** 2])

    ocp.constraints.x0 = np.zeros(nx_)

    # python-side MPC wrapper
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

    ocp.solver_options.qp_tol = 1e-6
    ocp.solver_options.levenberg_marquardt = 40.0
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

    # sampler bounds for x0 = [px, py, psi, v, r, beta, a, delta] -20.0, -20.0
    xmin = np.array([-20.0, -20.0, -math.pi , 0.2,  -1.0, -0.35, -3.0, -delta_max])
    xmax = np.array([+20.0, +20.0, +math.pi , 12.0, +1.0, +0.35, +3.0, +delta_max])

    # base input bounds derived from Lu at x=0 (requires Lu diagonal nonzero!)
    # u = [delta_dot, a_cmd]
    diag0 = Lu[nxconstr + 0, 0]
    diag1 = Lu[nxconstr + 1, 1]
    diag0b = Lu[nxconstr + nu + 0, 0]
    diag1b = Lu[nxconstr + nu + 1, 1]
    if abs(diag0) < 1e-12 or abs(diag1) < 1e-12 or abs(diag0b) < 1e-12 or abs(diag1b) < 1e-12:
        raise RuntimeError(
            "Lu diagonal entries are zero -> cannot compute umin/umax via 1/Lu.\n"
            f"Lu block values: {diag0=}, {diag1=}, {diag0b=}, {diag1b=}\n"
            "Fix Lu.txt so it represents proper box constraints for inputs."
        )

    umax = np.array([1.0 / Lu[nxconstr + i, i] for i in range(nu)], dtype=float)
    umin = np.array([1.0 / Lu[nxconstr + nu + i, i] for i in range(nu)], dtype=float)

    delta_dot_min, a_cmd_min = float(umin[0]), float(umin[1])
    delta_dot_max, a_cmd_max = float(umax[0]), float(umax[1])

    print("\n================= INPUT BOUNDS (APPLIED u) =================")
    print("u = [delta_dot, a_cmd]")
    print(f"delta_dot_min = {delta_dot_min:+.6f} rad/s")
    print(f"delta_dot_max = {delta_dot_max:+.6f} rad/s")
    print(f"a_cmd_min     = {a_cmd_min:+.6f} m/s^2")
    print(f"a_cmd_max     = {a_cmd_max:+.6f} m/s^2")
    print("umin=", umin)
    print("umax=", umax)
    print("============================================================\n")

    sampler = RandomSampler(numberofsamples, mpc.nx, randomseed, xmin, xmax)

    # DEBUG: check sampler distribution (before feasibility filtering)
    # Xtest = np.array([sampler.sample() for _ in range(2000)])
    # print("Sampler px stats:", Xtest[:, 0].min(), Xtest[:, 0].max(), Xtest[:, 0].mean())
    # print("Sampler py stats:", Xtest[:, 1].min(), Xtest[:, 1].max(), Xtest[:, 1].mean())
    #plot_feas(Xtest[:, 0], Xtest[:, 1], title="x0 sampled (before solve)")

    # ---- counters for prints (attempt / success / fail by obstacles) ----
    attempt_cnt = {0: 0, 1: 0, 2: 0}
    ok_cnt = {0: 0, 1: 0, 2: 0}
    fail_cnt = {0: 0, 1: 0, 2: 0}
    status_cnt = {}

    # -------------------------------------------------------------------------
    # Solve one OCP instance (called by dataset generator)
    # -------------------------------------------------------------------------
    def run_for_dataset(x0, verbose=False):
        x0 = np.asarray(x0, dtype=float).reshape(-1)

        # ---- reduce x0 bias: align heading roughly towards goal ----
        base_seed = randomseed if randomseed is not None else 0
        dx = 0.0 - x0[0]
        dy = 0.0 - x0[1]
        psi_nom = math.atan2(dy, dx)
        seed = _stable_hash_seed_from_pos(x0, base_seed) ^ 0xA5A5A5A5
        rng = np.random.default_rng(seed)
        x0[2] = (psi_nom + rng.normal(0.0, 20.0 * math.pi / 180.0) + math.pi) % (2 * math.pi) - math.pi
        #print("psi_nom [rad] =", psi_nom, "  [deg] =", psi_nom * 180 / math.pi)
        solver = AcadosOcpSolver(
            ocp,
            json_file="acados_ocp_" + model.name + ".json",
            build=False,
            generate=False,
        )

        # set obstacle parameters deterministically from x0
        p_val, n_active = obstacle_params_from_x0(
            x0=x0,
            base_seed=randomseed if randomseed is not None else 0,
            probs_0_1_2=probs_0_1_2,
            r_safe=r_safe_value,
            arena_xy=obstacle_arena_xy,
            min_dist_to_x0=obstacle_min_dist_to_x0,
        )
        attempt_cnt[n_active] = attempt_cnt.get(n_active, 0) + 1

        for k in range(N):
            solver.set(k, "p", p_val)
        solver.set(N, "p", p_val)

        # fix initial augmented state: x(0)=x0, s(0)=0
        solver.set(0, "lbx", np.append(x0, 0.0))
        solver.set(0, "ubx", np.append(x0, 0.0))

        # init guess
        Xinit = np.linspace(x0, np.zeros(nx), N + 1)    # (N+1,8)
        Uinit = np.zeros((N, nu))                       # (N,2)  decision v

        for k in range(N):
            # pre-stabilization warm start
            Uinit[k] = Kinit @ Xinit[k]
            # clip decision so applied u stays in bounds:
            # applied u = Kdelta*x + v  -> v in [umin - Kdelta*x, umax - Kdelta*x]
            Uinit[k] = np.clip(Uinit[k], umin - Kdelta @ Xinit[k], umax - Kdelta @ Xinit[k])
            # forward simulate one step with decision input (mpc expects decision input)
            Xinit[k + 1] = mpc.forward_simulate_single_step(Xinit[k], Uinit[k])

        for k in range(N):
            solver.set(k, "x", np.append(Xinit[k], Sinit[k]))
            solver.set(k, "u", Uinit[k])

        status = int(solver.solve())

        status_cnt[status] = status_cnt.get(status, 0) + 1
        if status == 0:
            ok_cnt[n_active] = ok_cnt.get(n_active, 0) + 1
        else:
            fail_cnt[n_active] = fail_cnt.get(n_active, 0) + 1

        # extract solution (plant only)
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
            print("status:", status, "time:", computetime, "iters:", number_iterations)
            print("n_active:", n_active, "obstacles:", obstacles_list_from_p(p_val), "r_safe:", float(p_val[4]))

        return X, U, status, computetime, number_iterations

    # generate dataset (may skip infeasible)
    _, _, _, _, outfile = sample_dataset_from_mpc(mpc, run_for_dataset, sampler, experimentname, verbose=verbose)
    print("Outfile:", outfile)

    # reconstruct obstacle arrays aligned with saved dataset samples
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

    dataset_root = Path("/share/mihaela-larisa.clement/soeampc-data/archive")
    dataset_dir = dataset_root / outfile
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        dataset_dir.joinpath("P_obstacles.txt"),
        P_obs,
        fmt="%.10g",
        delimiter=" ",
        header="o1x o1y o2x o2y r_safe",
        comments="",
    )
    np.savetxt(
        dataset_dir.joinpath("N_active.txt"),
        N_active.astype(int),
        fmt="%d",
        delimiter=" ",
        header="n_active",
        comments="",
    )

    uniq, cnt = np.unique(N_active, return_counts=True)
    dist_saved = {int(u): int(c) for u, c in zip(uniq, cnt)}

    print("\n================= OBSTACLE / FEASIBILITY STATS =================")
    print("attempted n_active counts:", attempt_cnt)
    print("successful (status==0) by n_active:", ok_cnt)
    print("failed by n_active:", fail_cnt)
    print("solver status histogram:", status_cnt)
    print("saved dataset n_active distribution:", dist_saved)
    print("saved dataset size:", int(x0dataset.shape[0]))
    print("===============================================================\n")

    print("Saved obstacle parameters to:", str(dataset_dir.joinpath("P_obstacles.txt")))
    print("Saved obstacle counts     to:", str(dataset_dir.joinpath("N_active.txt")))

    # =============================
    # Optional plots
    # =============================
    if showplot:
        # scatter of saved x0
        plot_feas(x0dataset[:, 0], x0dataset[:, 1], title="x0 saved (px/py)")

        # compute time distribution (if present)
        try:
            plot_ctdistro(computetimes, plt_show=True)
        except Exception:
            pass

        # ensure full trajectories exist
        if not (Xdataset.ndim == 3 and Udataset.ndim == 3 and Xdataset.shape[0] > 0):
            print("[INFO] Dataset does not contain full trajectories. Shapes:", Xdataset.shape, Udataset.shape)
            return outfile

        # pick up to 3 samples, preferring 0/1/2 obstacles
        idx0 = np.where(N_active == 0)[0]
        idx1 = np.where(N_active == 1)[0]
        idx2 = np.where(N_active == 2)[0]

        rng_sel = np.random.default_rng(123)
        picks = []

        def pick_one(idxs, forbid):
            if idxs.size == 0:
                return None
            for _ in range(20):
                cand = int(rng_sel.choice(idxs, size=1, replace=False)[0])
                if cand not in forbid:
                    return cand
            return int(rng_sel.choice(idxs, size=1, replace=False)[0])

        for group in [idx0, idx1, idx2]:
            p = pick_one(group, picks)
            if p is not None:
                picks.append(p)
        picks = list(dict.fromkeys(picks))[:3]

        print("\n--- Plotting selected dataset samples with obstacles ---")
        for k, idx in enumerate(picks):
            Xh = Xdataset[idx]     # (N+1,nx)
            Uh = Udataset[idx]     # (N,2) solver variable v_dec
            p_h = P_obs[idx]       # [o1x,o1y,o2x,o2y,r_safe]
            r_h = float(p_h[4])
            obs_h = obstacles_list_from_p(p_h)
            na = int(N_active[idx])

            print(f"  sample[{k}] idx={idx}: n_active={na}, obstacles={obs_h}, r_safe={r_h}")
            xmin_plot = np.full((nx,), None, dtype=object)
            xmax_plot = np.full((nx,), None, dtype=object)

            xmin_plot[3] = v_min
            xmax_plot[3] = v_max
            xmin_plot[7] = -delta_max
            xmax_plot[7] = +delta_max

            limits = {
                "umin": umin.tolist(),
                "umax": umax.tolist(),
                "xmin": xmin_plot.tolist(),
                "xmax": xmax_plot.tolist(),
            }

            plot_vehicle_ol_grid_3x3(
                mpc,
                Vtraj=[Uh],
                Xtraj=[Xh],
                labels=[f"idx={idx} (n_active={na})"],
                plt_show=True,
                limits=limits,
                input_is_v=True,
                obstacles=[obs_h],
                r_safe=[r_h],
                show_xy=True,
                show_clearance=True,
                print_clearance=True,
                title=f"8-state OL grid | idx={idx} | n_active={na}",
            )

    return outfile


# =============================================================================
# Parallel sampling / merge / timing test (optional)
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
    name = "vehicle_8state_sim"
    model = export_vehicle_sim_model()

    Tf = float(np.genfromtxt(fp.joinpath("mpc_parameters", "Tf.txt"), delimiter=","))
    N = 40

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
