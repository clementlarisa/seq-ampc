# plot.py — vehicle_8state_obs (steering angle as STATE)
# ============================================================
# Model:
#   State  x = [px, py, psi, v, r, beta, a, delta]     (nx = 8)
#   Input  u = [dot_delta, a_cmd]                      (nu = 2)
#
# Notes:
# - In dataset generation, the ACADOS solver variable is called "v" (nu=2),
#   and the applied input is u = Kdelta @ x + v.
# - Many routines below accept either:
#     * decision variable v (default input_is_v=True), or
#     * already applied u (input_is_v=False).
# - Obstacle plotting is optional (pass obstacles / r_safe).
#
# IMPORTANT BUGFIX (your screenshot):
# - DO NOT use sharex="col" for the 3x3 grid, because column 3 contains XY (px)
#   and then clearance (time) -> they would share x-axis and you get time
#   starting at negative px values (e.g. -7s).
#   -> This file uses NO sharex across columns. Each time-series axis uses its own time vector.

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def _legend_if_any(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(**kwargs)


def _as_list_per_traj(value, n_traj, default=None):
    """Normalize scalar/list/None to list length n_traj."""
    if value is None:
        return [default for _ in range(n_traj)]
    if isinstance(value, (float, int, np.floating, np.integer)):
        return [float(value) for _ in range(n_traj)]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == n_traj:
            return list(value)
        if n_traj == 1:
            return [value]
    return [default for _ in range(n_traj)]


def _normalize_obstacles(obstacles, n_traj):
    """
    Normalize obstacles into list-per-traj:
      obstacles_per_traj = [ [(ox,oy), ...], [(ox,oy), ...], ... ]
    Accepts:
      - None
      - single list [(ox,oy), ...] when n_traj==1
      - list-per-traj [ [(ox,oy)...], [(ox,oy)...], ... ]
    """
    if obstacles is None:
        return [[] for _ in range(n_traj)]

    # single trajectory: obstacles as [(ox,oy), ...]
    if n_traj == 1 and isinstance(obstacles, (list, tuple)) and len(obstacles) > 0:
        o0 = obstacles[0]
        if isinstance(o0, (tuple, list, np.ndarray)) and len(o0) == 2 and not isinstance(o0[0], (tuple, list, np.ndarray)):
            return [list(obstacles)]

    # list per trajectory
    if isinstance(obstacles, (list, tuple)) and len(obstacles) == n_traj:
        out = []
        for obs_i in obstacles:
            out.append([] if obs_i is None else list(obs_i))
        return out

    return [[] for _ in range(n_traj)]


def _draw_obstacle_circles(ax, obstacles, r_safe, label_prefix="obs"):
    """Draw obstacles as circles; if r_safe is None -> draw centers only."""
    if obstacles is None or len(obstacles) == 0:
        return

    if r_safe is None:
        for i, (ox, oy) in enumerate(obstacles):
            ax.scatter([ox], [oy], s=60)
            ax.text(ox, oy, f" {label_prefix}{i+1}")
        return

    th = np.linspace(0.0, 2.0 * np.pi, 200)
    rs = float(r_safe)
    for i, (ox, oy) in enumerate(obstacles):
        cx = ox + rs * np.cos(th)
        cy = oy + rs * np.sin(th)
        ax.plot(cx, cy, linewidth=2)
        ax.scatter([ox], [oy], s=60)
        ax.text(ox, oy, f" {label_prefix}{i+1}")


def _min_clearance_over_traj(X, obstacles, r_safe):
    """
    clearance[k] = min_i ( ||(px,py)-(ox_i,oy_i)|| - r_safe )
    Returns (clearance, min_clearance).
    """
    if obstacles is None or len(obstacles) == 0 or r_safe is None:
        return None, None

    px = X[:, 0]
    py = X[:, 1]
    clearance = np.full(px.shape, np.inf, dtype=float)
    rs = float(r_safe)

    for (ox, oy) in obstacles:
        d = np.sqrt((px - ox) ** 2 + (py - oy) ** 2) - rs
        clearance = np.minimum(clearance, d)

    return clearance, float(np.min(clearance))


def _reconstruct_u_from_v(mpc, X, V):
    """
    Reconstruct applied input u = Kdelta*x + v.

    X : (N+1,nx) or (N,nx)
    V : (N,nu)
    Returns U : (N,nu)
    """
    if hasattr(mpc, "Kdelta"):
        Kdelta = np.asarray(mpc.Kdelta)  # (nu,nx)
        U = np.zeros_like(V, dtype=float)
        for k in range(V.shape[0]):
            U[k, :] = (Kdelta @ X[k, :]) + V[k, :]
        return U

    if hasattr(mpc, "stabilizing_feedback_controller"):
        U = np.zeros_like(V, dtype=float)
        for k in range(V.shape[0]):
            u_fb = np.asarray(mpc.stabilizing_feedback_controller(X[k, :])).reshape(-1)
            U[k, :] = u_fb + V[k, :]
        return U

    return np.array(V, copy=True)


def _labels_vehicle_8state():
    xlabels = [
        r"$p_x$ [m]",
        r"$p_y$ [m]",
        r"$\psi$ [rad]",
        r"$v$ [m/s]",
        r"$r$ [rad/s]",
        r"$\beta$ [rad]",
        r"$a$ [m/s$^2$]",
        r"$\delta$ [rad]",
    ]
    ulabels = [
        r"$\dot{\delta}$ [rad/s]",
        r"$a_{\mathrm{cmd}}$ [m/s$^2$]",
    ]
    vlabels = [
        r"$v_{\dot{\delta}}$ [rad/s]",
        r"$v_a$ [m/s$^2$]",
    ]
    return xlabels, ulabels, vlabels


# ============================================================
# 3x3 Open-loop grid (8-state) with obstacles
# ============================================================

def plot_vehicle_ol_grid_3x3(
    mpc,
    Vtraj,                 # list of (N,2) decision v OR applied u
    Xtraj,                 # list of (N+1,8)
    labels=None,
    plt_show=True,
    limits=None,           # {"umin":[2], "umax":[2], "xmin":[8], "xmax":[8]}
    input_is_v=True,
    obstacles=None,        # None OR list-per-traj of obstacle lists
    r_safe=None,           # None OR scalar OR list-per-traj
    show_xy=True,
    show_clearance=True,
    print_clearance=True,
    title=None,
):
    """
    3x3 open-loop grid.

    Layout:
      Row 1: dot_delta(t), a_cmd(t), XY (px vs py + obstacles)
      Row 2: psi(t),       v(t),     clearance(t)
      Row 3: r(t),         beta(t),  delta(t)   [state constraint lines drawn here]
    """
    if limits is None:
        limits = {}

    n_traj = len(Vtraj)
    if labels is None:
        labels = [f"traj {i}" for i in range(n_traj)]

    obstacles_per_traj = _normalize_obstacles(obstacles, n_traj)
    r_safe_per_traj = _as_list_per_traj(r_safe, n_traj, default=None)

    # horizon time grid (N+1)
    t = np.linspace(0.0, float(mpc.Tf), int(mpc.N) + 1)

    xlab, ulab, _ = _labels_vehicle_8state()

    # BUGFIX: do NOT share x across columns (column 3 has XY, not time)
    fig, axs = plt.subplots(3, 3, figsize=(14, 9))
    axs = np.asarray(axs)
    if title is not None:
        fig.suptitle(title)

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    ax_ddel = axs[0, 0]
    ax_acmd = axs[0, 1]
    ax_xy   = axs[0, 2]

    ax_psi  = axs[1, 0]
    ax_v    = axs[1, 1]
    ax_clr  = axs[1, 2]

    ax_r    = axs[2, 0]
    ax_beta = axs[2, 1]
    ax_del  = axs[2, 2]

    # grid + labels
    for ax in [ax_ddel, ax_acmd, ax_psi, ax_v, ax_r, ax_beta, ax_del]:
        ax.grid(True)

    ax_ddel.set_ylabel(ulab[0])
    ax_acmd.set_ylabel(ulab[1])

    ax_psi.set_ylabel(xlab[2])
    ax_v.set_ylabel(xlab[3])

    ax_r.set_ylabel(xlab[4])
    ax_beta.set_ylabel(xlab[5])
    ax_del.set_ylabel(xlab[7])

    # time xlabels for time-series axes only
    for ax in [ax_ddel, ax_acmd, ax_psi, ax_v, ax_r, ax_beta, ax_del, ax_clr]:
        ax.set_xlabel("time [s]")

    # XY
    if show_xy:
        ax_xy.set_aspect("equal")
        ax_xy.grid(True)
        ax_xy.set_xlabel(xlab[0])
        ax_xy.set_ylabel(xlab[1])

    # clearance
    if show_clearance:
        ax_clr.grid(True)
        ax_clr.set_xlabel("time [s]")
        ax_clr.set_ylabel("min clearance [m]")
        ax_clr.axhline(0.0, linestyle="--", linewidth=1.5)

    # plot each trajectory
    for i, (V, X) in enumerate(zip(Vtraj, Xtraj)):
        ls = linestyles[i % len(linestyles)]
        name = labels[i]

        X = np.asarray(X, dtype=float)
        V = np.asarray(V, dtype=float)

        # reconstruct applied u if needed
        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X, V)  # (N,2)
        else:
            U = np.array(V, copy=True)

        # inputs (step) -> plot on N+1 time points
        ax_ddel.step(t, np.append([U[0, 0]], U[:, 0]), linestyle=ls, label=name)
        ax_acmd.step(t, np.append([U[0, 1]], U[:, 1]), linestyle=ls, label=name)

        # states
        ax_psi.plot(t[:X.shape[0]],  X[:, 2], linestyle=ls, label=name)
        ax_v.plot(t[:X.shape[0]],    X[:, 3], linestyle=ls, label=name)
        ax_r.plot(t[:X.shape[0]],    X[:, 4], linestyle=ls, label=name)
        ax_beta.plot(t[:X.shape[0]], X[:, 5], linestyle=ls, label=name)
        ax_del.plot(t[:X.shape[0]],  X[:, 7], linestyle=ls, label=name)

        # XY + obstacles
        obs_i = obstacles_per_traj[i]
        r_i = r_safe_per_traj[i]
        if show_xy:
            traj_label  = "trajectory" if i == 0 else None
            start_label = "start"      if i == 0 else None
            goal_label  = "goal"       if i == 0 else None

            ax_xy.plot(X[:, 0], X[:, 1], linestyle=ls, linewidth=2, label=traj_label)
            ax_xy.plot(X[0, 0],  X[0, 1], marker="o", ms=10, mfc="none", mew=2, linestyle="None", label=start_label)
            ax_xy.plot(X[-1, 0], X[-1, 1], marker="x", ms=10, mew=2, linestyle="None", label=goal_label)
            _draw_obstacle_circles(ax_xy, obs_i, r_i)

        # clearance over horizon
        if show_clearance:
            clr, min_clr = _min_clearance_over_traj(X, obs_i, r_i)
            if clr is not None:
                ax_clr.plot(t[:X.shape[0]], clr, linestyle=ls, label=name)
                if print_clearance:
                    print(f"[plot_vehicle_ol_grid_3x3] {name} min clearance = {min_clr:+.3f} m")

    # input bounds (applied u)
    if "umin" in limits and "umax" in limits and limits["umin"] is not None and limits["umax"] is not None:
        umin = limits["umin"]
        umax = limits["umax"]
        # dot_delta
        if umin[0] is not None:
            ax_ddel.plot([t[0], t[-1]], [umin[0], umin[0]], linestyle="dashed", alpha=0.7)
        if umax[0] is not None:
            ax_ddel.plot([t[0], t[-1]], [umax[0], umax[0]], linestyle="dashed", alpha=0.7)
        # a_cmd
        if umin[1] is not None:
            ax_acmd.plot([t[0], t[-1]], [umin[1], umin[1]], linestyle="dashed", alpha=0.7)
        if umax[1] is not None:
            ax_acmd.plot([t[0], t[-1]], [umax[1], umax[1]], linestyle="dashed", alpha=0.7)

    # state bounds for delta (idx 7)
    if "xmin" in limits and "xmax" in limits and limits["xmin"] is not None and limits["xmax"] is not None:
        xmin = limits["xmin"]
        xmax = limits["xmax"]
        if len(xmin) >= 8 and len(xmax) >= 8:
            if xmin[7] is not None:
                ax_del.plot([t[0], t[-1]], [xmin[7], xmin[7]], linestyle="dashed", alpha=0.7)
            if xmax[7] is not None:
                ax_del.plot([t[0], t[-1]], [xmax[7], xmax[7]], linestyle="dashed", alpha=0.7)

    # legends
    _legend_if_any(ax_ddel, loc="best")
    _legend_if_any(ax_acmd, loc="best")
    _legend_if_any(ax_psi,  loc="best")
    _legend_if_any(ax_v,    loc="best")
    _legend_if_any(ax_r,    loc="best")
    _legend_if_any(ax_beta, loc="best")
    _legend_if_any(ax_del,  loc="best")
    if show_xy:
        _legend_if_any(ax_xy, loc="best")
    if show_clearance:
        _legend_if_any(ax_clr, loc="best")

    fig.tight_layout()
    if plt_show:
        plt.show()
    return fig, axs


# ============================================================
# Closed-loop grid (8-state) with obstacles
# ============================================================

def plot_vehicle_cl_grid_3x3(
    mpc,
    Utraj,         # list of (T,2) decision v OR applied u
    Xtraj,         # list of (T+1,8)
    feasible=None, # list of (T,) mask OR None
    labels=None,
    plt_show=True,
    limits=None,   # {"umin":[2], "umax":[2], "xmin":[8], "xmax":[8]}
    input_is_v=True,
    obstacles=None,
    r_safe=None,
    show_xy=True,
    show_clearance=True,
    mark_infeasible=True,
    title=None,
    path=None,
    filename=None,
):
    if limits is None:
        limits = {}

    n_traj = len(Utraj)
    if labels is None:
        labels = [f"traj {i}" for i in range(n_traj)]

    obstacles_per_traj = _normalize_obstacles(obstacles, n_traj)
    r_safe_per_traj = _as_list_per_traj(r_safe, n_traj, default=None)

    Ts = float(mpc.Tf) / float(mpc.N)

    xlab, ulab, _ = _labels_vehicle_8state()

    # BUGFIX: no sharex across columns
    fig, axs = plt.subplots(3, 3, figsize=(14, 9))
    axs = np.asarray(axs)
    if title is not None:
        fig.suptitle(title)

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    ax_ddel = axs[0, 0]
    ax_acmd = axs[0, 1]
    ax_xy   = axs[0, 2]

    ax_psi  = axs[1, 0]
    ax_v    = axs[1, 1]
    ax_clr  = axs[1, 2]

    ax_r    = axs[2, 0]
    ax_beta = axs[2, 1]
    ax_del  = axs[2, 2]

    for ax in [ax_ddel, ax_acmd, ax_psi, ax_v, ax_r, ax_beta, ax_del]:
        ax.grid(True)

    ax_ddel.set_ylabel(ulab[0])
    ax_acmd.set_ylabel(ulab[1])

    ax_psi.set_ylabel(xlab[2])
    ax_v.set_ylabel(xlab[3])

    ax_r.set_ylabel(xlab[4])
    ax_beta.set_ylabel(xlab[5])
    ax_del.set_ylabel(xlab[7])

    for ax in [ax_ddel, ax_acmd, ax_psi, ax_v, ax_r, ax_beta, ax_del, ax_clr]:
        ax.set_xlabel("time [s]")

    if show_xy:
        ax_xy.set_aspect("equal")
        ax_xy.grid(True)
        ax_xy.set_xlabel(xlab[0])
        ax_xy.set_ylabel(xlab[1])

    if show_clearance:
        ax_clr.grid(True)
        ax_clr.set_xlabel("time [s]")
        ax_clr.set_ylabel("min clearance [m]")
        ax_clr.axhline(0.0, linestyle="--", linewidth=1.5)

    # bounds (inputs)
    if "umin" in limits and "umax" in limits and limits["umin"] is not None and limits["umax"] is not None:
        umin = limits["umin"]
        umax = limits["umax"]
        if umin[0] is not None:
            ax_ddel.plot([0, 1], [umin[0], umin[0]], linestyle="dashed", alpha=0.7)  # will be autoscaled
        if umax[0] is not None:
            ax_ddel.plot([0, 1], [umax[0], umax[0]], linestyle="dashed", alpha=0.7)
        if umin[1] is not None:
            ax_acmd.plot([0, 1], [umin[1], umin[1]], linestyle="dashed", alpha=0.7)
        if umax[1] is not None:
            ax_acmd.plot([0, 1], [umax[1], umax[1]], linestyle="dashed", alpha=0.7)

    # bounds (delta state)
    delta_min = delta_max = None
    if "xmin" in limits and "xmax" in limits and limits["xmin"] is not None and limits["xmax"] is not None:
        xmin = limits["xmin"]
        xmax = limits["xmax"]
        if len(xmin) >= 8 and len(xmax) >= 8:
            delta_min = xmin[7]
            delta_max = xmax[7]

    for i in range(n_traj):
        ls = linestyles[i % len(linestyles)]
        name = labels[i]

        X = np.asarray(Xtraj[i], dtype=float)     # (T+1,8)
        V_or_U = np.asarray(Utraj[i], dtype=float) # (T,2)
        Ti = X.shape[0]
        t_i = np.linspace(0.0, (Ti - 1) * Ts, Ti)

        # feasibility mask
        if feasible is None:
            fmask = np.ones((Ti - 1,), dtype=int)
        else:
            fmask = np.asarray(feasible[i]).reshape(-1)
            if fmask.shape[0] != (Ti - 1):
                fmask = fmask[: (Ti - 1)]
                if fmask.shape[0] < (Ti - 1):
                    fmask = np.pad(fmask, (0, (Ti - 1) - fmask.shape[0]), constant_values=1)

        # reconstruct applied u if needed
        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X[:-1, :], V_or_U)  # (T,2)
        else:
            U = np.array(V_or_U, copy=True)

        # inputs (step)
        ax_ddel.step(t_i, np.append([U[0, 0]], U[:, 0]), linestyle=ls, label=name)
        ax_acmd.step(t_i, np.append([U[0, 1]], U[:, 1]), linestyle=ls, label=name)

        if mark_infeasible and np.any(fmask == 0):
            idx = np.where(fmask == 0)[0]
            ax_ddel.plot(t_i[1:Ti][idx], U[idx, 0], marker="x", linestyle="None", markersize=6)
            ax_acmd.plot(t_i[1:Ti][idx], U[idx, 1], marker="x", linestyle="None", markersize=6)

        # states
        ax_psi.plot(t_i,  X[:, 2], linestyle=ls, label=name)
        ax_v.plot(t_i,    X[:, 3], linestyle=ls, label=name)
        ax_r.plot(t_i,    X[:, 4], linestyle=ls, label=name)
        ax_beta.plot(t_i, X[:, 5], linestyle=ls, label=name)
        ax_del.plot(t_i,  X[:, 7], linestyle=ls, label=name)

        # delta bounds (draw once per plot, but harmless if repeated)
        if delta_min is not None:
            ax_del.plot([t_i[0], t_i[-1]], [delta_min, delta_min], linestyle="dashed", alpha=0.7)
        if delta_max is not None:
            ax_del.plot([t_i[0], t_i[-1]], [delta_max, delta_max], linestyle="dashed", alpha=0.7)

        # XY + obstacles
        obs_i = obstacles_per_traj[i]
        r_i = r_safe_per_traj[i]
        if show_xy:
            ax_xy.plot(X[:, 0], X[:, 1], linestyle=ls, linewidth=2, label=name)
            _draw_obstacle_circles(ax_xy, obs_i, r_i)

            if mark_infeasible and np.any(fmask == 0):
                Xk = X[:-1, :]
                idx = np.where(fmask == 0)[0]
                ax_xy.plot(Xk[idx, 0], Xk[idx, 1], marker=".", linestyle="None", markersize=8)

        # clearance
        if show_clearance:
            clr, _ = _min_clearance_over_traj(X, obs_i, r_i)
            if clr is not None:
                ax_clr.plot(t_i, clr, linestyle=ls, label=name)

    _legend_if_any(ax_ddel, loc="best")
    _legend_if_any(ax_acmd, loc="best")
    _legend_if_any(ax_psi,  loc="best")
    _legend_if_any(ax_v,    loc="best")
    _legend_if_any(ax_r,    loc="best")
    _legend_if_any(ax_beta, loc="best")
    _legend_if_any(ax_del,  loc="best")
    if show_xy:
        _legend_if_any(ax_xy, loc="best")
    if show_clearance:
        _legend_if_any(ax_clr, loc="best")

    fig.tight_layout()

    if (path is not None) and (filename is not None):
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig(os.path.join(path, f"{filename}_cl_grid_3x3.png"), dpi=300)

    if plt_show:
        plt.show()

    return fig, axs


# ============================================================
# Simple open-loop plots (kept for compatibility)
# ============================================================

def plot_vehicle_ol_V(mpc, Vtraj, labels=None, plt_show=True):
    """Open-loop plot of decision variables v over horizon (nu=2)."""
    if labels is None:
        labels = [f"traj {i}" for i in range(len(Vtraj))]

    plt.clf()
    t = np.linspace(0, float(mpc.Tf), int(mpc.N) + 1)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    _, _, vlabels = _labels_vehicle_8state()

    for i in range(len(Vtraj)):
        V = np.asarray(Vtraj[i], dtype=float)
        for j in range(2):
            plt.step(
                t,
                np.append([V[0, j]], V[:, j]),
                label=f"{labels[i]} {vlabels[j]}",
                color=colors[j],
                linestyle=linestyles[i % len(linestyles)]
            )

    plt.grid(True)
    plt.ylabel("decision variables v")
    _legend_if_any(plt.gca(), loc="best")

    if plt_show:
        plt.show()


def plot_vehicle_ol(mpc, Vtraj, Xtraj, labels=None, plt_show=True, limits=None, input_is_v=True):
    """
    Open-loop plot of applied inputs and selected states (quick view).
    """
    if limits is None:
        limits = {}
    if labels is None:
        labels = [f"traj {i}" for i in range(len(Vtraj))]

    t = np.linspace(0, float(mpc.Tf), int(mpc.N) + 1)
    xlab, ulab, _ = _labels_vehicle_8state()

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(12, 8)

    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True)

    for i, (V, X) in enumerate(zip(Vtraj, Xtraj)):
        X = np.asarray(X, dtype=float)
        V = np.asarray(V, dtype=float)

        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X, V)
        else:
            U = np.array(V, copy=True)

        ax1.step(t, np.append([U[0, 0]], U[:, 0]), label=f"{labels[i]} {ulab[0]}")
        ax1.step(t, np.append([U[0, 1]], U[:, 1]), label=f"{labels[i]} {ulab[1]}")
        ax2.plot(t[:X.shape[0]], X[:, 3], label=f"{labels[i]} {xlab[3]}")
        ax2.plot(t[:X.shape[0]], X[:, 7], label=f"{labels[i]} {xlab[7]}")
        ax3.plot(X[:, 0], X[:, 1], label=f"{labels[i]} XY")

    ax1.set_xlabel("time [s]")
    ax2.set_xlabel("time [s]")
    ax3.set_xlabel(xlab[0])
    ax3.set_ylabel(xlab[1])
    ax3.set_aspect("equal")

    _legend_if_any(ax1, loc="best")
    _legend_if_any(ax2, loc="best")
    _legend_if_any(ax3, loc="best")

    if plt_show:
        plt.show()
    return fig


# ============================================================
# Feasibility plots + compute time histogram
# ============================================================

def plot_feas(xfeas, yfeas, xlim=None, ylim=None,
              xlabel=r"$p_x$ [m]", ylabel=r"$p_y$ [m]",
              title=None, plt_show=True):
    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    ax.set_aspect('equal')

    if title is not None:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.scatter(xfeas, yfeas, marker=',', color='tab:blue')

    if plt_show:
        plt.show()


def plot_feas_notfeas(feas, notfeas, xlim=None, ylim=None,
                      xlabel=r"$p_x$ [m]", ylabel=r"$p_y$ [m]",
                      plt_show=True, save_tex_path=None):
    def _as_points(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 1:
            raise ValueError("plot_feas_notfeas expects points of shape (N,2).")
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
        raise ValueError(f"Invalid array shape {arr.shape}; expected (N,2).")

    feas_pts = _as_points(feas)
    notfeas_pts = _as_points(notfeas)

    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    ax.set_aspect('equal')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if feas_pts is not None and len(feas_pts) > 0:
        plt.scatter(feas_pts[:, 0], feas_pts[:, 1], marker='.', color='tab:blue')
    if notfeas_pts is not None and len(notfeas_pts) > 0:
        plt.scatter(notfeas_pts[:, 0], notfeas_pts[:, 1], marker='.', color='tab:red')

    if save_tex_path is not None:
        try:
            import tikzplotlib
            tikzplotlib.save(save_tex_path, axis_height='2.4in', axis_width='2.4in')
        except Exception:
            pass

    if plt_show:
        plt.show()


def plot_ctdistro(ct, plt_show=True):
    """Histogram of compute times in ms using log-spaced bins (no legend warning)."""
    plt.clf()
    computetimes = np.asarray(ct, dtype=float) * 1000.0
    if computetimes.size == 0:
        print("plot_ctdistro: empty input.")
        return

    eps = 1e-12
    cmin = max(float(computetimes.min()), eps)
    cmax = max(float(computetimes.max()), cmin * 1.001)
    logbins = np.geomspace(cmin, cmax, 20)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=2.1)

    plt.hist(computetimes, density=False, bins=logbins)
    locs, _ = plt.yticks()
    plt.yticks(locs, np.round(locs / len(computetimes) * 100, 1))
    plt.ylabel('fraction [%]')
    plt.grid(True)
    plt.xlabel('compute time [ms]')
    plt.tight_layout()

    if plt_show:
        plt.show()
