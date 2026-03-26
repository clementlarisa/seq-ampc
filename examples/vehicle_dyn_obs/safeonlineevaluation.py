# safeonlineevaluation.py — vehicle_dyn_obs
# - Plots only 'good' samples (preferring terminal), without f_cont/RK4 (uses controller(xk)->x_next).
# - Saves plots under a unique name (timestamp + counter + sample idx + controller).
# - Optional: interpolation (upsampling) of trajectories for smoother plots (linear).

import os
import sys
from pathlib import Path
import datetime as _dt

import numpy as np
from tqdm import tqdm
import fire

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

FP = Path(__file__).resolve().parent
from seqampc.config import DATASETS_DIR, MODELS_DIR
DATASETS_ROOT = str(DATASETS_DIR)

os.chdir(FP)
sys.path.append(str(FP.parent.parent))

# -----------------------------------------------------------------------------
# seqampc imports
# -----------------------------------------------------------------------------

from seqampc.safeonline import (
    AMPC,
    SafeOnlineEvaluationAMPC,
    SafeOnlineEvaluationAMPCGroundTruthInit,
)
from seqampc.datasetutils import mpc_dataset_import
from seqampc.trainampc import import_model


# =============================================================================
# Plot helpers
# =============================================================================

def _get_u_limits_from_mpc(mpc):
    """
    Best-effort: tries to extract umin/umax from the MPC.
    Add your attributes here if your MPC uses different names.
    """
    if hasattr(mpc, "umin") and hasattr(mpc, "umax"):
        return np.asarray(mpc.umin).reshape(-1), np.asarray(mpc.umax).reshape(-1)
    if hasattr(mpc, "u_min") and hasattr(mpc, "u_max"):
        return np.asarray(mpc.u_min).reshape(-1), np.asarray(mpc.u_max).reshape(-1)
    return None, None


def _upsample_time_series(Y: np.ndarray, up: int) -> np.ndarray:
    """
    Linear upsampling along the time axis.
    - Y: (T, d)
    - up: 1 => unchanged; 5 => 5x denser
    """
    if up is None or int(up) <= 1:
        return Y
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, d = Y.shape
    if T <= 1:
        return Y

    x_old = np.arange(T, dtype=float)
    x_new = np.linspace(0.0, float(T - 1), int((T - 1) * up + 1))

    Y_new = np.zeros((x_new.size, d), dtype=float)
    for j in range(d):
        Y_new[:, j] = np.interp(x_new, x_old, Y[:, j])
    return Y_new


def plot_vehicle_obs_cl(
    results,
    labels,
    save_path=None,
    mpc=None,
    xy_lim=20.0,
    interp_factor: int = 1,
):
    """
    results: list of dicts (one per controller) with keys:
        X, U, status, p_obs, ...
    """
    nC = len(results)
    assert nC == len(labels)

    p_obs = results[0]["p_obs"].reshape(-1)
    o1 = (float(p_obs[0]), float(p_obs[1]))
    o2 = (float(p_obs[2]), float(p_obs[3]))
    r_safe = float(p_obs[4])

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2)

    ax_xy = fig.add_subplot(gs[:, 0])
    ax_v  = fig.add_subplot(gs[0, 1])
    ax_u0 = fig.add_subplot(gs[1, 1])
    ax_u1 = fig.add_subplot(gs[2, 1])

    # XY
    for k in range(nC):
        X = _upsample_time_series(results[k]["X"], interp_factor)
        ax_xy.plot(X[:, 0], X[:, 1], label=f"{labels[k]} ({results[k]['status']})")

    for (ox, oy) in [o1, o2]:
        ax_xy.add_patch(Circle((ox, oy), r_safe, fill=False))
        ax_xy.plot([ox], [oy], marker="x")

    ax_xy.set_xlabel("px [m]")
    ax_xy.set_ylabel("py [m]")
    ax_xy.set_xlim(-xy_lim, xy_lim)
    ax_xy.set_ylim(-xy_lim, xy_lim)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    ax_xy.legend(loc="best", fontsize=8)

    # v (state index 3)
    for k in range(nC):
        X = _upsample_time_series(results[k]["X"], interp_factor)
        ax_v.plot(X[:, 3], label=labels[k])
    ax_v.set_ylabel("v [m/s]")
    ax_v.grid(True)

    # u
    for k in range(nC):
        U = results[k]["U"]
        if U is None or np.size(U) == 0:
            continue
        U = _upsample_time_series(U, interp_factor)
        ax_u0.plot(U[:, 0], label=labels[k])
        ax_u1.plot(U[:, 1], label=labels[k])

    ax_u0.set_ylabel("delta_dot [rad/s]")
    ax_u0.grid(True)

    ax_u1.set_ylabel("a_cmd [m/s^2]")
    ax_u1.set_xlabel("k")
    ax_u1.grid(True)

    # constraints (if mpc is provided)
    if mpc is not None:
        umin, umax = _get_u_limits_from_mpc(mpc)
        if umin is not None and umax is not None and len(umin) >= 2 and len(umax) >= 2:
            ax_u0.axhline(umax[0], linestyle="--", linewidth=1)
            ax_u0.axhline(umin[0], linestyle="--", linewidth=1)
            ax_u1.axhline(umax[1], linestyle="--", linewidth=1)
            ax_u1.axhline(umin[1], linestyle="--", linewidth=1)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# Utilities
# =============================================================================

def _summarize(all_results, names):
    Ns = len(all_results)
    print("\n===== SUMMARY (extended) =====")
    for j, name in enumerate(names):
        statuses = np.array([all_results[i][j]["status"] for i in range(Ns)])

        feasible_rate = np.mean(statuses != "infeasible_cl")
        terminal_rate = np.mean(statuses == "terminal_set_reached")
        timeout_rate  = np.mean(statuses == "timeout")

        min_clear = np.array([all_results[i][j]["min_clearance"] for i in range(Ns)])
        viol_rate = np.mean(min_clear < 0.0)

        unique, counts = np.unique(statuses, return_counts=True)
        counts_dict = {u: int(c) for u, c in zip(unique, counts)}

        print(f"\n{name}")
        print(f"  counts: {counts_dict}")
        print(f"  feasible_cl:        {feasible_rate:.3f}")
        print(f"  terminal_reached:   {terminal_rate:.3f}")
        print(f"  timeout:            {timeout_rate:.3f}")
        print(f"  obstacle_viol_rate: {viol_rate:.3f}")
        print(f"  min_clear: mean={min_clear.mean():+.3e}, "
              f"median={np.median(min_clear):+.3e}, "
              f"min={min_clear.min():+.3e}, "
              f"p05={np.quantile(min_clear,0.05):+.3e}, "
              f"p95={np.quantile(min_clear,0.95):+.3e}")


def _resolve_dataset_dir(dataset_dir: str) -> Path:
    p = Path(dataset_dir)
    if not p.is_absolute():
        p = DATASETS_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    return p


def _load_obstacle_params(dataset_dir: Path):
    P_path = dataset_dir / "P_obstacles.txt"
    if not P_path.exists():
        raise FileNotFoundError(f"Missing {P_path}")

    P_obs = np.genfromtxt(P_path, dtype=float, comments="#")
    if P_obs.ndim == 0:
        raise ValueError(f"{P_path} seems empty or invalid.")
    if np.isnan(P_obs).any():
        P_obs = np.genfromtxt(P_path, dtype=float, skip_header=1)
    if P_obs.ndim == 1:
        P_obs = P_obs[None, :]

    N_path = dataset_dir / "N_active.txt"
    N_active = None
    if N_path.exists():
        N_active = np.genfromtxt(N_path, dtype=int, comments="#")
        if np.ndim(N_active) == 0:
            N_active = np.array([int(N_active)])

    return P_obs, N_active


def _obstacle_violation_xy(X: np.ndarray, p_obs: np.ndarray):
    o1x, o1y, o2x, o2y, r_safe = p_obs.reshape(-1)

    px = X[:, 0]
    py = X[:, 1]

    c1 = (px - o1x) ** 2 + (py - o1y) ** 2 - r_safe ** 2
    c2 = (px - o2x) ** 2 + (py - o2y) ** 2 - r_safe ** 2

    return float(np.min(np.minimum(c1, c2)))


def _timestamp_tag():
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _unique_plot_path(out_dir: Path, stem: str, ext: str = ".png") -> Path:
    """
    Finds a free filename: stem_000.png, stem_001.png, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(10_000):
        p = out_dir / f"{stem}_{k:03d}{ext}"
        if not p.exists():
            return p
    # fallback
    return out_dir / f"{stem}_{_timestamp_tag()}{ext}"


# =============================================================================
# Closed Loop
# =============================================================================

def closed_loop_experiment_vehicle_obs(x0, p_obs, controller, Nsim=500):
    nx = controller.mpc.nx
    nu = controller.mpc.nu

    X_cl = np.zeros((Nsim, nx))
    U_cl = np.zeros((Nsim - 1, nu))
    V_cl = np.zeros((Nsim - 1, nu))
    feasible_ampc = np.zeros((Nsim - 1))

    X_cl[0] = x0
    status = "running"
    debug_list = []
    for k in range(Nsim - 1):
        xk = X_cl[k]

        # Uses the controller's internal propagation
        v_dec, x_next = controller(xk)
        if hasattr(controller, "feasible_debug"):
            debug_list.append(controller.feasible_debug.copy())
        u_applied = controller.mpc.stabilizing_feedback_controller_clipped_inputs(xk, v_dec)

        V_cl[k] = v_dec
        U_cl[k] = u_applied
        X_cl[k + 1] = x_next
        feasible_ampc[k] = controller.feasible

        feasible_si = controller.mpc.in_state_and_input_constraints(
            X_cl[:k + 2], V_cl[:k + 1], robust=False
        )

        min_clear = _obstacle_violation_xy(X_cl[:k + 2], p_obs)
        feasible_obs = (min_clear >= 0.0)

        if not (feasible_si and feasible_obs):
            status = "infeasible_cl"
            return status, X_cl[:k + 2], U_cl[:k + 1], V_cl[:k + 1], feasible_ampc[:k + 1], min_clear, debug_list

        if controller.mpc.in_terminal_constraints(x_next, robust=False):
            status = "terminal_set_reached"
            return status, X_cl[:k + 2], U_cl[:k + 1], V_cl[:k + 1], feasible_ampc[:k + 1], min_clear, debug_list

    min_clear = _obstacle_violation_xy(X_cl, p_obs)
    return "timeout", X_cl, U_cl, V_cl, feasible_ampc, min_clear, debug_list

# =============================================================================
# Main Evaluation
# =============================================================================

def print_reason_metrics_from_feasible_debug(results):
    """
    results: list of dicts like controller.feasible_debug with keys:
      - "in_state_and_input_constraints"
      - "in_terminal_constraint"
      - "cost_decrease"
    Prints the same metrics as closed_loop_test_reason.
    """
    if len(results) == 0:
        print("No feasible_debug entries collected; cannot compute reason metrics.")
        return None

    flags_ok = np.array([
        (r["in_state_and_input_constraints"] and r["in_terminal_constraint"] and r["cost_decrease"])
        for r in results
    ], dtype=bool)

    rejection_rate = np.mean(~flags_ok)

    rejections = [r for r, ok in zip(results, flags_ok) if not ok]
    if len(rejections) == 0:
        rejection_from_state_and_input_constraint = 0.0
        rejection_from_terminal_constraint = 0.0
        rejection_from_cost_decrease = 0.0
    else:
        rejection_from_state_and_input_constraint = np.mean([not r["in_state_and_input_constraints"] for r in rejections])
        rejection_from_terminal_constraint = np.mean([not r["in_terminal_constraint"] for r in rejections])
        rejection_from_cost_decrease = np.mean([not r["cost_decrease"] for r in rejections])

    print(f"{rejection_rate=}")
    print(f"{rejection_from_state_and_input_constraint=}")
    print(f"{rejection_from_terminal_constraint=}")
    print(f"{rejection_from_cost_decrease=}")

    return {
        "rejection_rate": float(rejection_rate),
        "rejection_from_state_and_input_constraint": float(rejection_from_state_and_input_constraint),
        "rejection_from_terminal_constraint": float(rejection_from_terminal_constraint),
        "rejection_from_cost_decrease": float(rejection_from_cost_decrease),
    }

def closed_loop_test_on_dataset_vehicle_obs(
    dataset_dir: str,
    model_name: str,
    N_samples: int = 100,
    N_sim: int = 100,
    do_plot: bool = False,
    out_dir: str = "figures/safeonline_vehicle_obs",
    n_good_plots: int = 3,
    plot_controller: str = "safe",     # "naive"|"safe"|"safe init"
    prefer_terminal: bool = True,
    xy_lim: float = 15.0,
    interp_factor: int = 1,            # e.g. 5 for smoother curves
    unique_names: bool = True,         # always saves under a new name
    include_safe_init: bool = False,
):
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cwd0 = os.getcwd()
    try:
        os.chdir(FP)
        mpc, X0, V_init, _, _ = mpc_dataset_import(dataset_dir.name)
    finally:
        os.chdir(cwd0)

    P_obs, _ = _load_obstacle_params(dataset_dir)

    from sklearn.model_selection import train_test_split

    # Reproduce sklearn test split (same as training/statistical_test)
    N = X0.shape[0]
    idx = np.arange(N)

    _, idx_test = train_test_split(
        idx,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # Reproduce statistical_test subsampling (same seed and sampling rule)
    p = min(int(N_samples), idx_test.shape[0])

    rng = np.random.default_rng(seed=42)
    sub = rng.choice(idx_test.shape[0], size=p, replace=False)

    idx_eval = idx_test[sub]

    # Apply to all aligned arrays
    X0 = X0[idx_eval]
    V_init = V_init[idx_eval]
    P_obs = P_obs[idx_eval]

    Ns = X0.shape[0]  # now Ns == p
    V_init = V_init[:Ns]
    P_obs = P_obs[:Ns]

    model = import_model(modelname=model_name)

    naive = AMPC(mpc, model)
    safe = SafeOnlineEvaluationAMPC(mpc, model)
    safe_init = SafeOnlineEvaluationAMPCGroundTruthInit(mpc, model)

    if include_safe_init:
        controllers = [naive, safe, safe_init]
        names = ["naive", "safe", "safe init"]
    else:
        controllers = [naive, safe]
        names = ["naive", "safe"]

    if plot_controller not in names:
        raise ValueError(f"plot_controller must be one of {names}")
    plot_j = names.index(plot_controller)

    all_results = []
    good_terminal = []  # list of (i, score)
    good_other = []     # list of (i, score)

    print(f"\nTesting {len(controllers)} controllers on {Ns} initial conditions\n")

    for i in tqdm(range(Ns)):
        x0 = X0[i]
        v0 = V_init[i]
        p_obs = P_obs[i]

        for c in controllers:
            c.set_context(p_obs[:4])
            c.initialize(x0, v0)

        res_i = []
        for c in controllers:
            status, X, U, V, feas, min_clear, dbg_list = closed_loop_experiment_vehicle_obs(
                x0, p_obs, c, N_sim
            )
            res_i.append(
                {
                    "status": status,
                    "X": X,
                    "U": U,
                    "V": V,
                    "feasible": feas,
                    "feasible_init": bool(c.feasible),
                    "p_obs": np.copy(p_obs),
                    "min_clearance": float(min_clear),
                    "feasible_debug_list": dbg_list,
                }
            )
        all_results.append(res_i)

        # bookkeeping: "good" samples for the selected controller
        st = res_i[plot_j]["status"]
        mc = res_i[plot_j]["min_clearance"]
        if st == "terminal_set_reached":
            good_terminal.append((i, mc))
        elif st != "infeasible_cl":
            good_other.append((i, mc))

    # ---------------------------------------------------------
    # Plot: only successful samples
    # ---------------------------------------------------------
    if do_plot and n_good_plots > 0:
        good_terminal.sort(key=lambda t: t[1], reverse=True)
        good_other.sort(key=lambda t: t[1], reverse=True)

        chosen = []
        seen = set()

        if prefer_terminal:
            for i, _ in good_terminal:
                if i not in seen:
                    chosen.append(i)
                    seen.add(i)
                if len(chosen) >= n_good_plots:
                    break

        if len(chosen) < n_good_plots:
            for i, _ in good_other:
                if i not in seen:
                    chosen.append(i)
                    seen.add(i)
                if len(chosen) >= n_good_plots:
                    break

        print(f"\nPlotting {len(chosen)} good samples for controller='{plot_controller}': {chosen}\n")

        tag = _timestamp_tag()
        for idx in chosen:
            stem = f"cl_good_{plot_controller}_idx={idx:04d}_Ns={Ns}_Nsim={N_sim}_{tag}"
            if unique_names:
                save_path = _unique_plot_path(out_dir, stem, ext=".png")
            else:
                save_path = out_dir / f"{stem}.png"

            plot_vehicle_obs_cl(
                results=all_results[idx],
                labels=names,
                save_path=save_path,
                mpc=mpc,
                xy_lim=xy_lim,
                interp_factor=interp_factor,
            )

    # Summary
    print("\n===== SUMMARY =====")
    for j, name in enumerate(names):
        status = np.array([all_results[i][j]["status"] for i in range(Ns)])
        feasible_cl = np.mean(status != "infeasible_cl")
        terminal = np.mean(status == "terminal_set_reached")
        min_clear = np.array([all_results[i][j]["min_clearance"] for i in range(Ns)])
        print(
            f"{name}: feasible_cl={feasible_cl:.3f}, terminal_set_reached={terminal:.3f}, "
            f"min_clear_mean={min_clear.mean():+.3e}, min_clear_min={min_clear.min():+.3e}"
        )
        # collect all feasible_debug entries for this controller over all episodes
        dbg = []
        for i in range(Ns):
            dbg.extend(all_results[i][j]["feasible_debug_list"])

        if len(dbg) == 0:
            print(f"{name}: no feasible_debug available (controller has no safe-eval reasons).")
        else:
            print_reason_metrics_from_feasible_debug(dbg)

    _summarize(all_results, names)
    
    return None  # prevent fire from printing large return values


# =============================================================================

if __name__ == "__main__":
    fire.Fire({
        "closed_loop_test_on_dataset_vehicle_obs": closed_loop_test_on_dataset_vehicle_obs,
    })