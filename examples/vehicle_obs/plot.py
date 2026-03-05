"""
plot.py — vehicle plotting helpers (with obstacle visualization + clearance)

This file is designed for the soeampc vehicle example with obstacle avoidance.

Supported plots
---------------
1) plot_vehicle_ol_grid_2x3(...)
   - 2 rows x 3 columns overview for ONE MPC horizon (or multiple horizons):
       Row 1: delta, a_cmd, XY (px vs py with obstacles)
       Row 2: psi, v, clearance-to-obstacles (min clearance over horizon)
   - Works with either decision variable v (input_is_v=True) or applied u (input_is_v=False).
   - If obstacles are provided, circles are drawn in XY and clearance is plotted.

2) plot_vehicle_ol_grid_3x2(...)
   - 3 rows x 2 columns overview:
       Inputs: delta, a_cmd
       States: px, py
       States: psi, v
   - Optional obstacles in a separate XY figure if show_xy=True (or you can use plot_vehicle_ol_xy).

3) plot_vehicle_cl(...)
   - Closed-loop style time-series + XY plot.
   - Optional obstacle overlay in XY.
   - Optional clearance-over-time subplot (min clearance at each time step).

4) plot_vehicle_ol_xy(...)
   - Just an XY plot with obstacles + optional clearance print.

5) plot_feas / plot_feas_notfeas / plot_ctdistro
   - Convenience utilities (as in your existing file).

Obstacle conventions
--------------------
- obstacles can be:
    None
    OR list of obstacles [(ox,oy), ...] for a single trajectory
    OR list per trajectory: [ [(ox,oy), ...], [(ox,oy), ...], ... ] when plotting multiple trajectories
- r_safe can be:
    None (no circles/clearance)
    OR scalar float
    OR list per trajectory [r1, r2, ...] matching obstacles-per-trajectory

Inactive obstacles:
- In your sampling code you encode inactive obstacles far away (e.g. 1e6,1e6).
- Here we don't know FAR; we assume you already filtered them out before passing to plotting.
  (But we also provide a soft filter helper if you want.)

"""

import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Small helpers
# =============================================================================

def _ylabel_for_batch(full_labels, batch):
    """
    Y-axis label for a subplot batch.
    - single variable: full label (incl units)
    - multiple: show only symbols, e.g. "$p_x$ / $p_y$"
    """
    if len(batch) == 1:
        return full_labels[batch[0]]
    short = [full_labels[j].split(" ")[0] for j in batch]  # math part only
    return " / ".join(short)


def _as_list_per_traj(value, n_traj, default=None):
    """
    Normalize an input that can be:
      - None
      - scalar
      - list length n_traj
    into a list length n_traj.
    """
    if value is None:
        return [default for _ in range(n_traj)]
    if isinstance(value, (float, int, np.floating, np.integer)):
        return [float(value) for _ in range(n_traj)]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == n_traj:
            return list(value)
        # if a single list of obstacles was passed (not nested) and n_traj==1
        if n_traj == 1:
            return [value]
    # fallback
    return [default for _ in range(n_traj)]


def _normalize_obstacles(obstacles, n_traj):
    """
    Normalize obstacles argument into a list per trajectory:
      obstacles_per_traj = [ [(ox,oy), ...], [(ox,oy), ...], ... ]
    """
    if obstacles is None:
        return [[] for _ in range(n_traj)]

    # If user passed a single obstacle list for a single trajectory:
    # obstacles = [(ox,oy), ...]
    if n_traj == 1 and len(obstacles) > 0 and isinstance(obstacles[0], (tuple, list, np.ndarray)) and len(obstacles[0]) == 2 and not isinstance(obstacles[0][0], (tuple, list, np.ndarray)):
        return [list(obstacles)]

    # If user passed list-per-trajectory:
    # obstacles = [ [(ox,oy)...], [(ox,oy)...], ... ]
    if isinstance(obstacles, (list, tuple)) and len(obstacles) == n_traj:
        out = []
        for obs_i in obstacles:
            if obs_i is None:
                out.append([])
            else:
                out.append(list(obs_i))
        return out

    # Fallback: treat as no obstacles
    return [[] for _ in range(n_traj)]


def _reconstruct_u_from_v(mpc, X, V):
    """
    Reconstruct applied input:
        u = Kdelta*x + v
    If Kdelta is not present, fallback to v as u.
    """
    if hasattr(mpc, "Kdelta"):
        Kdelta = np.asarray(mpc.Kdelta)  # (nu, nx)
        U = np.zeros_like(V)
        for k in range(V.shape[0]):
            U[k, :] = (Kdelta @ X[k, :]) + V[k, :]
        return U

    # Fallback: treat V as already applied input
    return np.array(V, copy=True)


def _draw_obstacle_circles(ax, obstacles, r_safe, label_prefix="obs"):
    """
    Draw obstacles as circles on the given axis.
    obstacles: list of (ox,oy)
    r_safe: float
    """
    if obstacles is None or len(obstacles) == 0:
        return

    if r_safe is None:
        # draw only centers
        for i, (ox, oy) in enumerate(obstacles):
            ax.scatter([ox], [oy], s=60)
            ax.text(ox, oy, f" {label_prefix}{i+1}")
        return

    th = np.linspace(0.0, 2.0 * np.pi, 200)
    for i, (ox, oy) in enumerate(obstacles):
        cx = ox + r_safe * np.cos(th)
        cy = oy + r_safe * np.sin(th)
        ax.plot(cx, cy, linewidth=2)
        ax.scatter([ox], [oy], s=60)
        ax.text(ox, oy, f" {label_prefix}{i+1}")


def _min_clearance_over_horizon(X, obstacles, r_safe):
    """
    Compute the minimum clearance along a horizon:
        clearance_k = min_i ( distance((px_k,py_k), obs_i) - r_safe )
    Returns:
        clearance: array length N+1 (same as X length)
        min_clearance: scalar
    If no obstacles or r_safe is None -> returns None, None.
    """
    if obstacles is None or len(obstacles) == 0 or r_safe is None:
        return None, None

    px = X[:, 0]
    py = X[:, 1]
    clearance = np.full(px.shape, np.inf, dtype=float)

    for (ox, oy) in obstacles:
        d = np.sqrt((px - ox) ** 2 + (py - oy) ** 2) - float(r_safe)
        clearance = np.minimum(clearance, d)

    return clearance, float(np.min(clearance))


def _legend_if_any(ax, **kwargs):
    """Add legend only if there are labeled artists."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(**kwargs)


# =============================================================================
# Open-loop (horizon) plots
# =============================================================================

def plot_vehicle_ol_xy(
    X,
    obstacles=None,
    r_safe=None,
    title=None,
    plt_show=True,
    ax=None,
):
    """
    Plot one horizon XY trajectory with obstacles.
    X: shape (N+1,4) or (N+1,nx)
    obstacles: list of (ox,oy)
    r_safe: float
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.set_aspect("equal")
    ax.plot(X[:, 0], X[:, 1], marker="o", linewidth=2, label="trajectory")
    ax.scatter([X[0, 0]], [X[0, 1]], marker="x", s=80, label="start")

    _draw_obstacle_circles(ax, obstacles, r_safe)

    ax.grid(True)
    ax.set_xlabel(r"$p_x$ [m]")
    ax.set_ylabel(r"$p_y$ [m]")
    if title is not None:
        ax.set_title(title)

    _legend_if_any(ax, loc="best")

    if plt_show:
        plt.show()
    return ax


def plot_vehicle_ol_grid_2x3(
    mpc,
    Vtraj,                 # list of (N,2)   decision v or applied u
    Xtraj,                 # list of (N+1,4)
    labels,
    plt_show=True,
    limits=None,           # {"umin":[...], "umax":[...]} for applied u (recommended)
    input_is_v=True,       # True: Vtraj is decision v (reconstruct u). False: Vtraj is applied u.
    obstacles=None,        # None OR list-per-traj of obstacle lists
    r_safe=None,           # None OR scalar OR list-per-traj
    show_xy=True,
    show_clearance=True,
    print_clearance=True,
    title=None,
):
    """
    2x3 grid for horizon visualization (per trajectory):
      Row 1: delta(t), a_cmd(t), XY
      Row 2: psi(t), v(t), clearance(t)

    Notes:
    - XY panel uses px,py from X.
    - Clearance panel uses obstacles/r_safe if provided.
    """
    if limits is None:
        limits = {}

    n_traj = len(Vtraj)
    obstacles_per_traj = _normalize_obstacles(obstacles, n_traj)
    r_safe_per_traj = _as_list_per_traj(r_safe, n_traj, default=None)

    # time grid over horizon
    t = np.linspace(0.0, mpc.Tf, mpc.N + 1)  # N+1

    # labels (vehicle)
    xlab = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\psi$ [rad]", r"$v$ [m/s]"]
    ulab = [r"$\delta$ [rad]", r"$a_{\mathrm{cmd}}$ [m/s$^2$]"]

    fig, axs = plt.subplots(2, 3, sharex="col", figsize=(14, 7))
    axs = np.asarray(axs)
    if title is not None:
        fig.suptitle(title)

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    # Panels:
    # (0,0) delta, (0,1) a_cmd, (0,2) XY
    # (1,0) psi,   (1,1) v,     (1,2) clearance
    ax_delta = axs[0, 0]
    ax_a = axs[0, 1]
    ax_xy = axs[0, 2]
    ax_psi = axs[1, 0]
    ax_v = axs[1, 1]
    ax_clr = axs[1, 2]

    # Prepare XY axes
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel(xlab[0])
    ax_xy.set_ylabel(xlab[1])
    if show_xy:
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator
        ax_xy.xaxis.set_major_locator(MaxNLocator(nbins=6))  # ~6 Major-Ticks
        ax_xy.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_xy.xaxis.set_minor_locator(AutoMinorLocator(2))  # 2 minor intervals pro major
        ax_xy.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax_xy.grid(True, which="major")
        ax_xy.grid(True, which="minor", alpha=0.3)

        # WICHTIG: Minor tick labels AUS (sonst Textsalat)
        ax_xy.tick_params(axis="x", which="minor", labelbottom=False)
        ax_xy.tick_params(axis="y", which="minor", labelleft=False)

    if show_clearance:
        ax_clr.grid(True)
        ax_clr.set_xlabel("time [s]")
        ax_clr.set_ylabel("min clearance [m]")
        ax_clr.axhline(0.0, linestyle="--", linewidth=1.5)

    # time-series panels
    for ax in [ax_delta, ax_a, ax_psi, ax_v]:
        ax.grid(True)

    ax_delta.set_ylabel(ulab[0])
    ax_a.set_ylabel(ulab[1])
    ax_psi.set_ylabel(xlab[2])
    ax_v.set_ylabel(xlab[3])
    ax_psi.set_xlabel("time [s]")
    ax_v.set_xlabel("time [s]")

    # Plot each trajectory
    for i, (V, X) in enumerate(zip(Vtraj, Xtraj)):
        ls = linestyles[i % len(linestyles)]
        name = labels[i] if labels is not None else f"traj {i}"

        # reconstruct applied input u if needed
        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X, V)  # (N,2)
        else:
            U = np.array(V, copy=True)

        # delta, a_cmd (step plot with N+1 points)
        ax_delta.step(t, np.append([U[0, 0]], U[:, 0]), linestyle=ls, label=name)
        ax_a.step(t, np.append([U[0, 1]], U[:, 1]), linestyle=ls, label=name)

        # psi, v
        ax_psi.plot(t[:X.shape[0]], X[:, 2], linestyle=ls, label=name)
        ax_v.plot(t[:X.shape[0]], X[:, 3], linestyle=ls, label=name)

        # XY
        if show_xy:
            # nur einmal labeln, damit die Legende nicht pro Traj doppelt wird
            traj_label  = "trajectory" if i == 0 else None
            start_label = "start"      if i == 0 else None
            goal_label  = "goal"       if i == 0 else None

            # trajectory line
            ax_xy.plot(X[:, 0], X[:, 1], linestyle=ls, linewidth=2, label=traj_label)

            # start = O (offen), goal = X
            ax_xy.plot(X[0, 0],  X[0, 1],
                       marker="o", ms=10, mfc="none", mew=2,
                       linestyle="None", label=start_label)
            ax_xy.plot(X[-1, 0], X[-1, 1],
                       marker="x", ms=10, mew=2,
                       linestyle="None", label=goal_label)

            # obstacles
            obs_i = obstacles_per_traj[i]
            r_i = r_safe_per_traj[i]
            _draw_obstacle_circles(ax_xy, obs_i, r_i)

        # clearance
        if show_clearance:
            obs_i = obstacles_per_traj[i]
            r_i = r_safe_per_traj[i]
            clearance, min_clr = _min_clearance_over_horizon(X, obs_i, r_i)
            if clearance is not None:
                ax_clr.plot(t[:X.shape[0]], clearance, linestyle=ls, label=name)
                if print_clearance:
                    print(f"[plot_vehicle_ol_grid_2x3] {name} min clearance = {min_clr:+.3f} m")
            else:
                # No obstacles: show NaN line or skip
                pass

        # Optionally: bounds on applied inputs
        if "umin" in limits and "umax" in limits and limits["umin"] is not None and limits["umax"] is not None:
            umin = limits["umin"]
            umax = limits["umax"]
            if umin[0] is not None:
                ax_delta.plot([t[0], t[-1]], [umin[0], umin[0]], linestyle="dashed", alpha=0.7)
            if umax[0] is not None:
                ax_delta.plot([t[0], t[-1]], [umax[0], umax[0]], linestyle="dashed", alpha=0.7)
            if umin[1] is not None:
                ax_a.plot([t[0], t[-1]], [umin[1], umin[1]], linestyle="dashed", alpha=0.7)
            if umax[1] is not None:
                ax_a.plot([t[0], t[-1]], [umax[1], umax[1]], linestyle="dashed", alpha=0.7)

    # Legends: keep them compact
    _legend_if_any(ax_delta, loc="best")
    _legend_if_any(ax_a, loc="best")
    _legend_if_any(ax_psi, loc="best")
    _legend_if_any(ax_v, loc="best")
    if show_xy:
        _legend_if_any(ax_xy, loc="best")
    if show_clearance:
        _legend_if_any(ax_clr, loc="best")

    fig.tight_layout()

    if plt_show:
        plt.show()

    return fig, axs


def plot_vehicle_ol_grid_3x2(
    mpc,
    Vtraj,          # list of (N,2)   decision v or applied u
    Xtraj,          # list of (N+1,4)
    labels,
    plt_show=True,
    limits=None,    # {"umin":[...], "umax":[...]} for applied u
    input_is_v=True,
    # optional additional XY/clearance (separate fig)
    obstacles=None,
    r_safe=None,
    show_xy=False,
    show_clearance=False,
):
    """
    3x2 multi-subplot (time-series only):
      Row 1: delta, a_cmd
      Row 2: p_x, p_y
      Row 3: psi, v

    If show_xy=True: also calls plot_vehicle_ol_xy for each trajectory.
    If show_clearance=True and obstacles provided: prints min clearance and optionally plots clearance separately.
    """
    if limits is None:
        limits = {}

    n_traj = len(Vtraj)
    obstacles_per_traj = _normalize_obstacles(obstacles, n_traj)
    r_safe_per_traj = _as_list_per_traj(r_safe, n_traj, default=None)

    t = np.linspace(0.0, mpc.Tf, mpc.N + 1)

    xlab = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\psi$ [rad]", r"$v$ [m/s]"]
    ulab = [r"$\delta$ [rad]", r"$a_{\mathrm{cmd}}$ [m/s$^2$]"]

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axs = np.asarray(axs)

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    for i, (V, X) in enumerate(zip(Vtraj, Xtraj)):
        ls = linestyles[i % len(linestyles)]
        name = labels[i] if labels is not None else f"traj {i}"

        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X, V)
        else:
            U = np.array(V, copy=True)

        # inputs
        axs[0, 0].step(t, np.append([U[0, 0]], U[:, 0]), linestyle=ls, label=name)
        axs[0, 1].step(t, np.append([U[0, 1]], U[:, 1]), linestyle=ls, label=name)

        # states
        axs[1, 0].plot(t[:X.shape[0]], X[:, 0], linestyle=ls, label=name)
        axs[1, 1].plot(t[:X.shape[0]], X[:, 1], linestyle=ls, label=name)
        axs[2, 0].plot(t[:X.shape[0]], X[:, 2], linestyle=ls, label=name)
        axs[2, 1].plot(t[:X.shape[0]], X[:, 3], linestyle=ls, label=name)

        # optional XY + clearance
        obs_i = obstacles_per_traj[i]
        r_i = r_safe_per_traj[i]
        if show_xy:
            plot_vehicle_ol_xy(X, obstacles=obs_i, r_safe=r_i, title=f"{name} XY", plt_show=True)

        if show_clearance:
            clearance, min_clr = _min_clearance_over_horizon(X, obs_i, r_i)
            if clearance is not None:
                print(f"[plot_vehicle_ol_grid_3x2] {name} min clearance = {min_clr:+.3f} m")

    # bounds on applied inputs
    if "umin" in limits and "umax" in limits and limits["umin"] is not None and limits["umax"] is not None:
        umin = limits["umin"]
        umax = limits["umax"]
        if umin[0] is not None:
            axs[0, 0].plot([t[0], t[-1]], [umin[0], umin[0]], linestyle="dashed", alpha=0.7)
        if umax[0] is not None:
            axs[0, 0].plot([t[0], t[-1]], [umax[0], umax[0]], linestyle="dashed", alpha=0.7)
        if umin[1] is not None:
            axs[0, 1].plot([t[0], t[-1]], [umin[1], umin[1]], linestyle="dashed", alpha=0.7)
        if umax[1] is not None:
            axs[0, 1].plot([t[0], t[-1]], [umax[1], umax[1]], linestyle="dashed", alpha=0.7)

    # labels
    axs[0, 0].set_ylabel(ulab[0])
    axs[0, 1].set_ylabel(ulab[1])
    axs[1, 0].set_ylabel(xlab[0])
    axs[1, 1].set_ylabel(xlab[1])
    axs[2, 0].set_ylabel(xlab[2])
    axs[2, 1].set_ylabel(xlab[3])
    axs[2, 0].set_xlabel("time [s]")
    axs[2, 1].set_xlabel("time [s]")

    for ax in axs.ravel():
        ax.grid(True)
        _legend_if_any(ax, loc="best")

    fig.tight_layout()

    if plt_show:
        plt.show()

    return fig, axs


# =============================================================================
# Closed-loop plots (time-series + XY + optional clearance)
# =============================================================================

def plot_vehicle_cl(
    mpc,
    Utraj,                    # list: each (T,nu)  (either decision v or applied u depending on input_is_v)
    Xtraj,                    # list: each (T+1,nx)
    feasible,                 # list: each (T,) with 0 marking infeasible, or None
    labels,
    plt_show=True,
    limits=None,
    path=None,
    filename=None,
    xy_axes=(0, 1),
    input_is_v=True,
    obstacles=None,           # None OR list-per-traj of obstacle lists
    r_safe=None,              # None OR scalar OR list-per-traj
    show_clearance=True,
):
    """
    Closed-loop style plot:
      - time series of inputs and states (stacked subplots)
      - XY plot with infeasible markers
      - Optional clearance-over-time plot if obstacles provided

    Important:
    - If input_is_v=True, Utraj is decision variable v and we reconstruct applied u using mpc.Kdelta.
      For closed-loop rollouts you might already have applied u; then set input_is_v=False.
    """
    if limits is None:
        limits = {}

    n_traj = len(Utraj)
    obstacles_per_traj = _normalize_obstacles(obstacles, n_traj)
    r_safe_per_traj = _as_list_per_traj(r_safe, n_traj, default=None)

    nx = mpc.nx
    nu = Utraj[0].shape[1] if n_traj > 0 else 2
    Ts = mpc.Tf / mpc.N

    # Determine longest rollout for time axis
    N_sim_max = int(np.max(np.array([len(Utraj[i]) + 1 for i in range(n_traj)])))
    t = np.linspace(0.0, (N_sim_max - 1) * Ts, N_sim_max)

    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    colors = ["r", "g", "b", "c", "m", "y", "k", "darkred", "navy", "darkgreen"]

    # Batching
    ubatches = [[0, 1]]
    xbatches = [[0, 1], [2], [3]]

    xlabels = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\psi$ [rad]", r"$v$ [m/s]"]
    ulabels = [r"$\delta$ [rad]", r"$a$ [m/s$^2$]"]

    # If clearance is shown, add one extra subplot row
    want_clearance = show_clearance and any(len(o) > 0 for o in obstacles_per_traj) and any(rs is not None for rs in r_safe_per_traj)
    n_rows = len(ubatches) + len(xbatches) + (1 if want_clearance else 0)

    plt.clf()

    # ----------------------------
    # Inputs
    # ----------------------------
    for k in range(len(ubatches)):
        ax = plt.subplot(n_rows, 1, k + 1)
        batch = ubatches[k]

        for i in range(n_traj):
            V_or_U = Utraj[i]
            X = Xtraj[i]
            f = feasible[i] if feasible is not None else np.ones(V_or_U.shape[0], dtype=int)

            # reconstruct applied u if needed
            if input_is_v:
                U = _reconstruct_u_from_v(mpc, X, V_or_U)
            else:
                U = np.array(V_or_U, copy=True)

            for j in batch:
                ax.step(
                    t[:U.shape[0] + 1],
                    np.append([U[0, j]], U[:, j]),
                    label=f"{labels[i]} {ulabels[j]}",
                    color=colors[j],
                    linestyle=linestyles[i % len(linestyles)],
                )

                infeas_idx = (f == 0)
                if np.any(infeas_idx):
                    ax.plot(
                        t[:U.shape[0]][infeas_idx],
                        U[infeas_idx, j],
                        marker="x",
                        linestyle="None",
                        markersize=7,
                        color="red",
                    )

        # bounds
        for j in batch:
            if "umin" in limits and limits["umin"][j] is not None:
                ax.plot([t[0], t[-1]], [limits["umin"][j], limits["umin"][j]],
                        linestyle="dashed", color=colors[j], alpha=0.7)
            if "umax" in limits and limits["umax"][j] is not None:
                ax.plot([t[0], t[-1]], [limits["umax"][j], limits["umax"][j]],
                        linestyle="dashed", color=colors[j], alpha=0.7)

        ax.grid(True)
        ax.set_ylabel(_ylabel_for_batch(ulabels, batch))
        _legend_if_any(ax, loc="center left", bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    # ----------------------------
    # States
    # ----------------------------
    for k in range(len(xbatches)):
        batch = xbatches[k]
        ax = plt.subplot(n_rows, 1, len(ubatches) + k + 1)

        for i in range(n_traj):
            X = Xtraj[i]
            for j in batch:
                ax.plot(
                    t[:X.shape[0]],
                    X[:, j],
                    label=f"{labels[i]} {xlabels[j]}",
                    color=colors[j],
                    linestyle=linestyles[i % len(linestyles)],
                )

        # bounds
        for j in batch:
            if "xmin" in limits and limits["xmin"][j] is not None:
                ax.plot([t[0], t[-1]], [limits["xmin"][j], limits["xmin"][j]],
                        linestyle="dashed", color=colors[j], alpha=0.7)
            if "xmax" in limits and limits["xmax"][j] is not None:
                ax.plot([t[0], t[-1]], [limits["xmax"][j], limits["xmax"][j]],
                        linestyle="dashed", color=colors[j], alpha=0.7)

        ax.grid(True)
        ax.set_ylabel(_ylabel_for_batch(xlabels, batch))
        _legend_if_any(ax, loc="center left", bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    # ----------------------------
    # Clearance over time (optional)
    # ----------------------------
    if want_clearance:
        ax = plt.subplot(n_rows, 1, n_rows)

        for i in range(n_traj):
            X = Xtraj[i]
            obs_i = obstacles_per_traj[i]
            r_i = r_safe_per_traj[i]
            if obs_i is None or len(obs_i) == 0 or r_i is None:
                continue

            clearance, min_clr = _min_clearance_over_horizon(X, obs_i, r_i)
            if clearance is None:
                continue

            ax.plot(
                t[:X.shape[0]],
                clearance,
                label=f"{labels[i]} min clearance",
                linestyle=linestyles[i % len(linestyles)],
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.5)
        ax.grid(True)
        ax.set_ylabel("min clearance [m]")
        ax.set_xlabel("time [s]")
        _legend_if_any(ax, loc="center left", bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    plt.subplots_adjust(hspace=0.35)
    plt.xlabel("time [s]")

    if plt_show:
        plt.show()

    # Save time-series plot if requested
    if (path is not None) and (filename is not None):
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.gcf().set_size_inches(20, 12)
        plt.savefig(os.path.join(path, f"{filename}.png"), dpi=300)
        try:
            import tikzplotlib
            tikzplotlib.save(os.path.join(path, f"{filename}_double.tex"),
                             axis_height="1.4in", axis_width="6.8in")
            tikzplotlib.save(os.path.join(path, f"{filename}_single.tex"),
                             axis_height="1.4in", axis_width="3.4in")
        except Exception:
            pass

    # ----------------------------
    # XY plot (with obstacles)
    # ----------------------------
    plt.clf()
    ax = plt.gca()
    ax.set_aspect("equal")

    axis_x, axis_y = xy_axes
    ax.set_xlabel(xlabels[axis_x])
    ax.set_ylabel(xlabels[axis_y])

    for i in range(n_traj):
        X = Xtraj[i]
        f = feasible[i] if feasible is not None else np.ones(X.shape[0] - 1, dtype=int)

        ax.plot(
            X[:, axis_x], X[:, axis_y],
            label=labels[i],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
        )

        # mark infeasible steps
        Xk = X[:-1, :]
        infeas_idx = (f == 0)
        if np.any(infeas_idx):
            ax.plot(
                Xk[infeas_idx, axis_x],
                Xk[infeas_idx, axis_y],
                marker=".",
                linestyle="None",
                markersize=8,
                color="red",
            )

        # obstacles for this rollout
        obs_i = obstacles_per_traj[i]
        r_i = r_safe_per_traj[i]
        _draw_obstacle_circles(ax, obs_i, r_i)

    ax.grid(True)
    _legend_if_any(ax, loc=2)

    if (path is not None) and (filename is not None):
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.gcf().set_size_inches(10, 8)
        plt.savefig(os.path.join(path, f"{filename}_xy.png"), dpi=300)
        try:
            import tikzplotlib
            tikzplotlib.save(os.path.join(path, f"{filename}_xy.tex"), axis_width="3.4in")
        except Exception:
            pass

    if plt_show:
        plt.show()

def plot_vehicle_cl_grid_2x3(
    mpc,
    Utraj,                    # list: each (T,nu)  (decision v OR applied u)
    Xtraj,                    # list: each (T+1,nx)
    feasible=None,            # list: each (T,) with 0 marking infeasible, OR None
    labels=None,
    plt_show=True,
    limits=None,              # {"umin":[...], "umax":[...], "xmin":[...], "xmax":[...]} optional
    input_is_v=True,          # True: Utraj is v_dec -> reconstruct applied u = Kdelta*x + v
    obstacles=None,           # None OR list-per-traj: [ [(ox,oy),...], ... ]
    r_safe=None,              # None OR scalar OR list-per-traj
    xy_axes=(0, 1),
    show_clearance=True,
    mark_infeasible=True,
):
    """
    Closed-loop plot in the SAME 2x3 layout as plot_vehicle_ol_grid_2x3:

      Row 1: delta(t), a_cmd(t), XY
      Row 2: psi(t), v(t), clearance(t)

    Notes
    -----
    - If input_is_v=True, Utraj is the solver decision variable v_dec, we reconstruct applied u:
          u = Kdelta*x + v
      using the state X at the same time index.
    - XY uses X[:,px], X[:,py].
    - Clearance uses obstacles + r_safe and is computed over the closed-loop time series.
    """

    if limits is None:
        limits = {}

    n_traj = len(Utraj)
    if labels is None:
        labels = [f"traj {i}" for i in range(n_traj)]

    # normalize obstacle inputs (reuse your helpers)
    obstacles_per_traj = _normalize_obstacles(obstacles, n_traj)
    r_safe_per_traj = _as_list_per_traj(r_safe, n_traj, default=None)

    # time grid (closed-loop sample time)
    Ts = mpc.Tf / mpc.N
    T_max = max(X.shape[0] for X in Xtraj)          # X is (T+1, nx)
    t = np.linspace(0.0, (T_max - 1) * Ts, T_max)  # length T+1

    # subplot labels
    xlabels = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\psi$ [rad]", r"$v$ [m/s]"]
    ulabels = [r"$\delta$ [rad]", r"$a_{\mathrm{cmd}}$ [m/s$^2$]"]

    fig, axs = plt.subplots(2, 3, sharex="col", figsize=(14, 7))
    axs = np.asarray(axs)

    ax_delta = axs[0, 0]
    ax_a     = axs[0, 1]
    ax_xy    = axs[0, 2]
    ax_psi   = axs[1, 0]
    ax_v     = axs[1, 1]
    ax_clr   = axs[1, 2]

    # styling
    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    # prepare axes
    for ax in [ax_delta, ax_a, ax_psi, ax_v]:
        ax.grid(True)

    ax_delta.set_ylabel(ulabels[0])
    ax_a.set_ylabel(ulabels[1])
    ax_psi.set_ylabel(xlabels[2])
    ax_v.set_ylabel(xlabels[3])
    ax_psi.set_xlabel("time [s]")
    ax_v.set_xlabel("time [s]")

    # XY panel
    ax_xy.set_aspect("equal")
    ax_xy.grid(True)
    ax_xy.set_xlabel(xlabels[xy_axes[0]])
    ax_xy.set_ylabel(xlabels[xy_axes[1]])

    # clearance panel
    if show_clearance:
        ax_clr.grid(True)
        ax_clr.set_xlabel("time [s]")
        ax_clr.set_ylabel("min clearance [m]")
        ax_clr.axhline(0.0, linestyle="--", linewidth=1.5)

    # bounds on applied inputs (optional)
    def _plot_input_bounds():
        if "umin" in limits and limits["umin"] is not None:
            umin = limits["umin"]
            if umin[0] is not None:
                ax_delta.plot([t[0], t[-1]], [umin[0], umin[0]], linestyle="dashed", alpha=0.7)
            if umin[1] is not None:
                ax_a.plot([t[0], t[-1]], [umin[1], umin[1]], linestyle="dashed", alpha=0.7)
        if "umax" in limits and limits["umax"] is not None:
            umax = limits["umax"]
            if umax[0] is not None:
                ax_delta.plot([t[0], t[-1]], [umax[0], umax[0]], linestyle="dashed", alpha=0.7)
            if umax[1] is not None:
                ax_a.plot([t[0], t[-1]], [umax[1], umax[1]], linestyle="dashed", alpha=0.7)

    _plot_input_bounds()

    # plot each trajectory
    for i in range(n_traj):
        ls = linestyles[i % len(linestyles)]
        name = labels[i]

        X = np.asarray(Xtraj[i])
        V_or_U = np.asarray(Utraj[i])  # (T,nu)
        T_i = X.shape[0]               # (T+1)

        # feasibility mask
        if feasible is None:
            fmask = np.ones((T_i - 1,), dtype=int)
        else:
            fmask = np.asarray(feasible[i]).reshape(-1)
            if fmask.shape[0] != (T_i - 1):
                # be robust: clip/pad to match
                fmask = fmask[: (T_i - 1)]
                if fmask.shape[0] < (T_i - 1):
                    fmask = np.pad(fmask, (0, (T_i - 1) - fmask.shape[0]), constant_values=1)

        # reconstruct applied input u if needed
        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X[:-1, :], V_or_U)  # uses X[k] with k=0..T-1
        else:
            U = np.array(V_or_U, copy=True)

        # Inputs as step plots (need T+1 points for steps)
        ax_delta.step(t[:T_i], np.append([U[0, 0]], U[:, 0]), linestyle=ls, label=name)
        ax_a.step(t[:T_i],     np.append([U[0, 1]], U[:, 1]), linestyle=ls, label=name)

        # mark infeasible input points (optional)
        if mark_infeasible and np.any(fmask == 0):
            idx = np.where(fmask == 0)[0]
            ax_delta.plot(t[1:T_i][idx], U[idx, 0], marker="x", linestyle="None", markersize=6)
            ax_a.plot(t[1:T_i][idx],     U[idx, 1], marker="x", linestyle="None", markersize=6)

        # States
        ax_psi.plot(t[:T_i], X[:, 2], linestyle=ls, label=name)
        ax_v.plot(t[:T_i],   X[:, 3], linestyle=ls, label=name)

        # XY trajectory
        ax_xy.plot(X[:, xy_axes[0]], X[:, xy_axes[1]], linestyle=ls, linewidth=2, label=name)
        _draw_obstacle_circles(ax_xy, obstacles_per_traj[i], r_safe_per_traj[i])

        # Clearance over time (closed-loop)
        if show_clearance:
            obs_i = obstacles_per_traj[i]
            r_i = r_safe_per_traj[i]
            clearance, min_clr = _min_clearance_over_horizon(X, obs_i, r_i)
            if clearance is not None:
                ax_clr.plot(t[:T_i], clearance, linestyle=ls, label=name)
                # optional print
                # print(f"[plot_vehicle_cl_grid_2x3] {name} min clearance = {min_clr:+.3f} m")

    # legends
    _legend_if_any(ax_delta, loc="best")
    _legend_if_any(ax_a, loc="best")
    _legend_if_any(ax_psi, loc="best")
    _legend_if_any(ax_v, loc="best")
    _legend_if_any(ax_xy, loc="best")
    if show_clearance:
        _legend_if_any(ax_clr, loc="best")
    if show_clearance:
        ax_clr.set_xlim(0.0, mpc.Tf)
    fig.tight_layout()

    if plt_show:
        plt.show()

    return fig, axs


# =============================================================================
# Feasibility / distribution utilities (unchanged style, with minor safety)
# =============================================================================

def plot_feas(
    xfeas,
    yfeas,
    xlim=None,
    ylim=None,
    xlabel=r"$p_x$ [m]",
    ylabel=r"$p_y$ [m]",
    title=None,
    plt_show=True,
):
    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    ax.set_aspect("equal")

    if title is not None:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.scatter(xfeas, yfeas, marker=",", color="blue")

    if plt_show:
        plt.show()


def plot_feas_notfeas(
    feas,
    notfeas,
    xlim=None,
    ylim=None,
    xlabel=r"$p_x$ [m]",
    ylabel=r"$p_y$ [m]",
    plt_show=True,
    save_tex_path=None,
):
    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    if feas is not None and len(feas) > 0:
        plt.scatter(feas[:, 0], feas[:, 1], marker=".", color="blue")
    if notfeas is not None and len(notfeas) > 0:
        plt.scatter(notfeas[:, 0], notfeas[:, 1], marker=".", color="red")

    if save_tex_path is not None:
        try:
            import tikzplotlib
            tikzplotlib.save(save_tex_path, axis_height="2.4in", axis_width="2.4in")
        except Exception:
            pass

    if plt_show:
        plt.show()


def plot_ctdistro(ct, plt_show=True):
    plt.clf()
    computetimes = np.asarray(ct) * 1000.0

    if computetimes.size == 0:
        print("plot_ctdistro: empty input.")
        return

    eps = 1e-12
    cmin = max(computetimes.min(), eps)
    cmax = max(computetimes.max(), cmin * 1.001)

    logbins = np.geomspace(cmin, cmax, 20)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=2.1)

    plt.hist(computetimes, density=False, bins=logbins)
    locs, _ = plt.yticks()
    plt.yticks(locs, np.round(locs / len(computetimes) * 100, 1))
    plt.ylabel("fraction [%]")
    plt.grid(True)

    # remove auto legend if any
    lgnd = plt.legend()
    if lgnd is not None:
        lgnd.remove()

    plt.xlabel("compute time [ms]")
    plt.tight_layout()

    if plt_show:
        plt.show()