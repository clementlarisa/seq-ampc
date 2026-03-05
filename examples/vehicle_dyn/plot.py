# plot.py ( vehicle model dy)
# ============================================================
# model:
#   State  x = [px, py, psi, v, r, beta, a]   (nx = 7)
#   Input  u = [delta, a_cmd]                (nu = 2)
#
# Notes:
# - In dataset generation, the ACADOS solver variable is called "v" (nu=2),
#   and the applied input is u = Kdelta @ x + v.
# - These plotting utilities work for nx=7 and (optionally) reconstruct applied
#   inputs where useful.
#
# Key change vs your version:
# - Each state subplot gets proper y-axis label(s) with variable + units.
#   (No more generic "states".)

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Small helpers
# ============================================================

def _get_vehicle_labels_new():
    """Labels and units for the NEW 7-state vehicle model."""
    xlabels = [
        r"$p_x$ [m]",
        r"$p_y$ [m]",
        r"$\psi$ [rad]",
        r"$v$ [m/s]",
        r"$r$ [rad/s]",
        r"$\beta$ [rad]",
        r"$a$ [m/s$^2$]",
    ]
    ulabels = [
        r"$\delta$ [rad]",
        r"$a_{\mathrm{cmd}}$ [m/s$^2$]",
    ]
    vlabels = [
        r"$v_\delta$ [rad]",
        r"$v_a$ [m/s$^2$]",
    ]
    return xlabels, ulabels, vlabels


def _labels_for_batch(labels, batch):
    """
    Create a compact ylabel for a batch.
    - For single variable: return full label with units.
    - For multiple: join variable symbols (still readable), units remain in legend.
    """
    if len(batch) == 1:
        return labels[batch[0]]
    # Example: "$p_x$ / $p_y$"
    short = []
    for j in batch:
        # Take the math part (before space) if present
        parts = labels[j].split(" ")
        short.append(parts[0])
    return " / ".join(short)


def _reconstruct_u_from_v(mpc, X, V):
    """
    Reconstruct applied input u = Kdelta*x + v.

    Parameters
    ----------
    mpc : MPC object (expected to have attribute Kdelta or stabilizing_feedback_controller)
    X   : array (N+1, nx)
    V   : array (N, nu)

    Returns
    -------
    U : array (N, nu) applied input
    """
    if hasattr(mpc, "Kdelta"):
        Kdelta = mpc.Kdelta  # (nu, nx)
        U = np.zeros_like(V)
        for k in range(V.shape[0]):
            U[k, :] = (Kdelta @ X[k, :]) + V[k, :]
        return U

    if hasattr(mpc, "stabilizing_feedback_controller"):
        U = np.zeros_like(V)
        for k in range(V.shape[0]):
            u_fb = np.asarray(mpc.stabilizing_feedback_controller(X[k, :])).reshape(-1)
            U[k, :] = u_fb + V[k, :]
        return U

    # Fallback: treat V as already applied input
    return np.array(V, copy=True)


def plot_vehicle_ol_grid_3x3_7state(
    mpc,
    Vtraj,          # list of (N,2)   (usually solver variable v)
    Xtraj,          # list of (N+1,7)
    labels,
    plt_show=True,
    limits=None,    # optional: {"umin":[...], "umax":[...]} for applied u
    input_is_v=True # True: Vtraj is v -> reconstruct u; False: Vtraj already is u
):
    if limits is None:
        limits = {}

    # time grid over horizon
    t = np.linspace(0, mpc.Tf, mpc.N + 1)  # length N+1

    # state labels (indices: 0..6)
    xlab = [
        r"$p_x$ [m]",
        r"$p_y$ [m]",
        r"$\psi$ [rad]",
        r"$v$ [m/s]",
        r"$r$ [rad/s]",
        r"$\beta$ [rad]",
        r"$a$ [m/s$^2$]",
    ]
    # applied input labels u (indices: 0..1)
    ulab = [r"$\delta$ [rad]", r"$a_{\mathrm{cmd}}$ [m/s$^2$]"]

    # 3x3 grid mapping: (row,col) -> ("u"/"x", index)
    grid = [
        [("u", 0), ("u", 1), ("x", 4)],  # delta, a_cmd, r
        [("x", 0), ("x", 1), ("x", 5)],  # px,   py,   beta
        [("x", 2), ("x", 3), ("x", 6)],  # psi,  v,    a
    ]

    fig, axs = plt.subplots(3, 3, sharex=True)
    axs = np.asarray(axs)

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    for i, (V, X) in enumerate(zip(Vtraj, Xtraj)):
        ls = linestyles[i % len(linestyles)]
        name = labels[i]

        # reconstruct applied input u for plotting
        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X, V)  # (N,2)
        else:
            U = np.array(V, copy=True)

        # fill grid
        for rr in range(3):
            for cc in range(3):
                kind, idx = grid[rr][cc]
                ax = axs[rr, cc]

                if kind == "u":
                    # step plot for piecewise-constant inputs, need N+1 points
                    ax.step(t, np.append([U[0, idx]], U[:, idx]), linestyle=ls, label=name)
                    ax.set_ylabel(ulab[idx])
                else:
                    ax.plot(t[:X.shape[0]], X[:, idx], linestyle=ls, label=name)
                    ax.set_ylabel(xlab[idx])

    # optional input bounds on delta/a_cmd only
    if "umin" in limits and "umax" in limits and limits["umin"] is not None and limits["umax"] is not None:
        umin = limits["umin"]
        umax = limits["umax"]

        # delta bounds on (0,0)
        if umin[0] is not None:
            axs[0, 0].plot([t[0], t[-1]], [umin[0], umin[0]], linestyle="dashed", alpha=0.7)
        if umax[0] is not None:
            axs[0, 0].plot([t[0], t[-1]], [umax[0], umax[0]], linestyle="dashed", alpha=0.7)

        # a_cmd bounds on (0,1)
        if umin[1] is not None:
            axs[0, 1].plot([t[0], t[-1]], [umin[1], umin[1]], linestyle="dashed", alpha=0.7)
        if umax[1] is not None:
            axs[0, 1].plot([t[0], t[-1]], [umax[1], umax[1]], linestyle="dashed", alpha=0.7)

    # cosmetics
    for ax in axs.ravel():
        ax.grid(True)

    axs[2, 0].set_xlabel("time [s]")
    axs[2, 1].set_xlabel("time [s]")
    axs[2, 2].set_xlabel("time [s]")

    # keep legends compact: one legend per top-row subplot
    axs[0, 0].legend(loc="best")
    axs[0, 1].legend(loc="best")
    axs[0, 2].legend(loc="best")

    fig.tight_layout()

    if plt_show:
        plt.show()

    return fig, axs

# ============================================================
# Closed-loop plot (time-series + XY)
# ============================================================

def plot_vehicle_cl(
    mpc, Utraj, Xtraj, feasible, labels,
    plt_show=True, limits=None, path=None, filename=None,
    xy_axes=(0, 1),
    input_is_v=True,   # True: Utraj is solver variable v; False: Utraj is applied u
):
    """
    Closed-loop trajectory plot for the  vehicle example.

    Parameters
    ----------
    input_is_v:
        If True, title/ylabel indicates decision variable v.
        If False, indicates applied input u.
        (Your code base often passes solver variable v as Utraj.)
    """
    if limits is None:
        limits = {}

    plt.clf()
    Ntrajs = len(Utraj)

    nx = mpc.nx
    nu = Utraj[0].shape[1] if Ntrajs > 0 else 2

    Ts = mpc.Tf / mpc.N

    # Longest rollout for consistent time axis
    N_sim_max = int(np.max(np.array([len(Utraj[i]) + 1 for i in range(Ntrajs)])))
    t = np.linspace(0, (N_sim_max - 1) * Ts, N_sim_max)

    # Styles
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:gray', 'tab:pink']

    xlabels, ulabels, vlabels = _get_vehicle_labels_new()

    # Batch states into sensible groups (7-state)
    #  - positions
    #  - angles (psi, beta)
    #  - speed/accel (v, a)
    #  - yaw rate
    xbatches = [[0, 1], [2, 5], [3, 6], [4]]
    ubatches = [[0, 1]]

    batches = len(ubatches) + len(xbatches)

    # ----------------------------
    # Inputs (step plots)
    # ----------------------------
    for k in range(len(ubatches)):
        plt.subplot(batches, 1, k + 1)
        batch = ubatches[k]

        for i in range(Ntrajs):
            U = Utraj[i]
            f = feasible[i] if feasible is not None else np.ones(U.shape[0], dtype=int)

            for j in batch:
                lab = vlabels[j] if input_is_v else ulabels[j]
                plt.step(
                    t[:U.shape[0] + 1],
                    np.append([U[0, j]], U[:, j]),
                    label=f"{labels[i]} {lab}",
                    color=colors[j],
                    linestyle=linestyles[i % len(linestyles)]
                )

                infeas_idx = (f == 0)
                if np.any(infeas_idx):
                    plt.plot(
                        t[:U.shape[0]][infeas_idx],
                        U[infeas_idx, j],
                        marker='x',
                        linestyle='None',
                        markersize=7,
                        color='red'
                    )

        # bounds
        for j in batch:
            if "umin" in limits and limits["umin"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["umin"][j], limits["umin"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)
            if "umax" in limits and limits["umax"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["umax"][j], limits["umax"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)

        plt.grid(True)
        plt.ylabel("decision v" if input_is_v else "applied u")
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    # ----------------------------
    # States (line plots)
    # ----------------------------
    for k in range(len(xbatches)):
        batch = xbatches[k]
        plt.subplot(batches, 1, len(ubatches) + k + 1)

        for i in range(Ntrajs):
            X = Xtraj[i]
            for j in batch:
                plt.plot(
                    t[:X.shape[0]],
                    X[:, j],
                    label=f"{labels[i]} {xlabels[j]}",
                    color=colors[j],
                    linestyle=linestyles[i % len(linestyles)]
                )

        # bounds
        for j in batch:
            if "xmin" in limits and limits["xmin"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["xmin"][j], limits["xmin"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)
            if "xmax" in limits and limits["xmax"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["xmax"][j], limits["xmax"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)

        plt.grid(True)
        #  proper ylabel for each batch
        plt.ylabel(_labels_for_batch(xlabels, batch))
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    plt.subplots_adjust(hspace=0.4)
    plt.xlabel("time [s]")

    if plt_show:
        plt.show()

    # Save time-series plot
    if (path is not None) and (filename is not None):
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.gcf().set_size_inches(20, 12)
        plt.savefig(os.path.join(path, f"{filename}.png"), dpi=300)
        try:
            import tikzplotlib
            tikzplotlib.save(os.path.join(path, f"{filename}_double.tex"),
                             axis_height='1.4in', axis_width='6.8in')
            tikzplotlib.save(os.path.join(path, f"{filename}_single.tex"),
                             axis_height='1.4in', axis_width='3.4in')
        except Exception:
            pass

    # ----------------------------
    # XY plot (px vs py default)
    # ----------------------------
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal')

    axis_x, axis_y = xy_axes
    plt.xlabel(xlabels[axis_x])
    plt.ylabel(xlabels[axis_y])

    for i in range(Ntrajs):
        X = Xtraj[i]
        f = feasible[i] if feasible is not None else np.ones(X.shape[0] - 1, dtype=int)

        plt.plot(
            X[:, axis_x], X[:, axis_y],
            label=labels[i],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)]
        )

        # infeasible markers (align with steps)
        Xk = X[:-1, :]
        infeas_idx = (f == 0)
        if np.any(infeas_idx):
            plt.plot(
                Xk[infeas_idx, axis_x],
                Xk[infeas_idx, axis_y],
                marker='.',
                linestyle='None',
                markersize=8,
                color='red'
            )

    plt.grid(True)
    plt.legend(loc=2)

    if (path is not None) and (filename is not None):
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.gcf().set_size_inches(10, 8)
        plt.savefig(os.path.join(path, f"{filename}_xy.png"), dpi=300)
        try:
            import tikzplotlib
            tikzplotlib.save(os.path.join(path, f"{filename}_xy.tex"), axis_width='3.4in')
        except Exception:
            pass

    if plt_show:
        plt.show()


# ============================================================
# Open-loop plots (over MPC horizon)
# ============================================================

def plot_vehicle_ol_V(mpc, Vtraj, labels, plt_show=True):
    """
    Open-loop plot of the *decision variables* v over the MPC horizon for nu=2.

    Parameters
    ----------
    Vtraj:
        List of arrays (N, nu) (predicted v sequence).
    """
    plt.clf()
    t = np.linspace(0, mpc.Tf, mpc.N + 1)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    _, _, vlabels = _get_vehicle_labels_new()

    for i in range(len(Vtraj)):
        V = Vtraj[i]
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
    plt.legend(loc=1)

    if plt_show:
        plt.show()


def plot_vehicle_ol(mpc, Vtraj, Xtraj, labels, plt_show=True, limits=None):
    """
    Open-loop plot of predicted trajectories over the MPC horizon.

    Reconstructs applied input:
        u = Kdelta*x + v
    """
    if limits is None:
        limits = {}

    plt.clf()
    t = np.linspace(0, mpc.Tf, mpc.N + 1)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:gray', 'tab:pink']

    xlabels, ulabels, _ = _get_vehicle_labels_new()

    ubatches = [[0, 1]]
    xbatches = [[0, 1], [2, 5], [3, 6], [4]]
    batches = len(ubatches) + len(xbatches)

    # reconstruct applied U for each traj
    U_applied_list = []
    for V, X in zip(Vtraj, Xtraj):
        U_applied_list.append(_reconstruct_u_from_v(mpc, X, V))

    # ----------------------------
    # Applied inputs u (step)
    # ----------------------------
    for k in range(len(ubatches)):
        plt.subplot(batches, 1, k + 1)
        batch = ubatches[k]

        for i in range(len(U_applied_list)):
            U = U_applied_list[i]
            for j in batch:
                plt.step(
                    t,
                    np.append([U[0, j]], U[:, j]),
                    label=f"{labels[i]} {ulabels[j]}",
                    color=colors[j],
                    linestyle=linestyles[i % len(linestyles)]
                )

        for j in batch:
            if "umin" in limits and limits["umin"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["umin"][j], limits["umin"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)
            if "umax" in limits and limits["umax"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["umax"][j], limits["umax"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)

        plt.grid(True)
        plt.ylabel(_labels_for_batch(ulabels, batch))
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    # ----------------------------
    # States (line)
    # ----------------------------
    for k in range(len(xbatches)):
        batch = xbatches[k]
        plt.subplot(batches, 1, len(ubatches) + k + 1)

        for i in range(len(Xtraj)):
            X = Xtraj[i]
            for j in batch:
                plt.plot(
                    t[:X.shape[0]],
                    X[:, j],
                    label=f"{labels[i]} {xlabels[j]}",
                    color=colors[j],
                    linestyle=linestyles[i % len(linestyles)]
                )

        for j in batch:
            if "xmin" in limits and limits["xmin"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["xmin"][j], limits["xmin"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)
            if "xmax" in limits and limits["xmax"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["xmax"][j], limits["xmax"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)

        plt.grid(True)
        plt.ylabel(_labels_for_batch(xlabels, batch))
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    plt.subplots_adjust(hspace=0.4)
    plt.xlabel("time [s]")

    if plt_show:
        plt.show()


# ============================================================
# Feasibility scatter plots + compute time hist
# ============================================================

def plot_feas(xfeas, yfeas, xlim=None, ylim=None,
              xlabel=r"$p_x$ [m]", ylabel=r"$p_y$ [m]",
              title=None, plt_show=True):
    """Scatter plot of feasible sampled initial states in 2D (default px vs py)."""
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
    """Scatter plot of feasible (blue) vs infeasible (red) points.

    Accepts either:
      - arrays shape (N,2) for points, or
      - two 1D arrays (N,) is NOT supported directly here; pass column_stack first.
    """
    def _as_points(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 1:
            # if user accidentally passed x only, we cannot guess y
            raise ValueError(
                "plot_feas_notfeas expects points of shape (N,2). "
                "You passed a 1D array. Use np.column_stack((x,y)) before calling."
            )
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

def plot_vehicle_cl_grid_3x3(
    mpc,
    Utraj,         # list of (T,2)   usually solver var v, or applied u
    Xtraj,         # list of (T+1,7)
    feasible,      # list of (T,) with 1 feasible, 0 infeasible (or None)
    labels,
    plt_show=True,
    limits=None,   # optional: {"umin":[...], "umax":[...]} (bounds for the plotted input)
    path=None,
    filename=None,
    xy_axes=(0, 1),
    input_is_v=True,   # True: Utraj is decision v -> we plot it as-is in the grid
):
    """
    Closed-loop style plots for 7-state vehicle, as a 3x3 grid + separate XY plot.

    3x3 grid layout:
      (0,0) input0    (0,1) input1    (0,2) r
      (1,0) p_x       (1,1) p_y       (1,2) beta
      (2,0) psi       (2,1) v         (2,2) a

    Notes
    -----
    - This function plots Utraj "as given" (either v or u). If you want *applied u*
      but your dataset stores *v*, reconstruct u before calling, or add reconstruction here.
    - The time axis is based on Ts = Tf/N from the MPC object (same as your current plot_vehicle_cl).
    """
    if limits is None:
        limits = {}

    # labels
    xlabels = [
        r"$p_x$ [m]",
        r"$p_y$ [m]",
        r"$\psi$ [rad]",
        r"$v$ [m/s]",
        r"$r$ [rad/s]",
        r"$\beta$ [rad]",
        r"$a$ [m/s$^2$]",
    ]
    ulabels = [
        r"$\delta$ [rad]",
        r"$a_{\mathrm{cmd}}$ [m/s$^2$]",
    ]
    vlabels = [
        r"$v_\delta$ [rad]",
        r"$v_a$ [m/s$^2$]",
    ]
    inplabels = vlabels if input_is_v else ulabels

    Ntrajs = len(Utraj)
    if Ntrajs == 0:
        return

    Ts = mpc.Tf / mpc.N

    # longest rollout length defines the common time axis
    T_max = int(np.max([U.shape[0] for U in Utraj]))  # number of control steps
    t_u = np.linspace(0, (T_max - 1) * Ts, T_max)     # for U (length T)
    t_x = np.linspace(0, T_max * Ts, T_max + 1)       # for X (length T+1)

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    # grid mapping: (row,col) -> ("u"/"x", index)
    grid = [
        [("u", 0), ("u", 1), ("x", 4)],  # input0, input1, r
        [("x", 0), ("x", 1), ("x", 5)],  # px, py, beta
        [("x", 2), ("x", 3), ("x", 6)],  # psi, v, a
    ]

    fig, axs = plt.subplots(3, 3, sharex=True)
    axs = np.asarray(axs)

    for i in range(Ntrajs):
        U = Utraj[i]
        X = Xtraj[i]
        f = feasible[i] if feasible is not None else np.ones(U.shape[0], dtype=int)

        ls = linestyles[i % len(linestyles)]
        name = labels[i]

        # per-traj time slices (in case shorter than max)
        Tu = U.shape[0]
        Tx = X.shape[0]  # should be Tu+1
        t_u_i = t_u[:Tu]
        t_x_i = t_x[:Tx]

        # draw everything in the 3x3 grid
        for rr in range(3):
            for cc in range(3):
                kind, idx = grid[rr][cc]
                ax = axs[rr, cc]

                if kind == "u":
                    # step plot needs Tu+1 points
                    ax.step(
                        t_x_i[:Tu + 1],
                        np.append([U[0, idx]], U[:, idx]),
                        linestyle=ls,
                        label=name,
                    )
                    ax.set_ylabel(inplabels[idx])

                    # infeasible markers on inputs (red x)
                    infeas_idx = (f == 0)
                    if np.any(infeas_idx):
                        ax.plot(
                            t_u_i[infeas_idx],
                            U[infeas_idx, idx],
                            marker="x",
                            linestyle="None",
                            markersize=7,
                        )

                    # optional bounds for input plots
                    if "umin" in limits and limits["umin"] is not None and limits["umin"][idx] is not None:
                        ax.plot([t_x_i[0], t_x_i[min(Tu, len(t_x_i)-1)]],
                                [limits["umin"][idx], limits["umin"][idx]],
                                linestyle="dashed", alpha=0.7)
                    if "umax" in limits and limits["umax"] is not None and limits["umax"][idx] is not None:
                        ax.plot([t_x_i[0], t_x_i[min(Tu, len(t_x_i)-1)]],
                                [limits["umax"][idx], limits["umax"][idx]],
                                linestyle="dashed", alpha=0.7)

                else:
                    ax.plot(
                        t_x_i,
                        X[:, idx],
                        linestyle=ls,
                        label=name,
                    )
                    ax.set_ylabel(xlabels[idx])

                    # infeasible markers on states (use X[:-1] aligned to f)
                    infeas_idx = (f == 0)
                    if np.any(infeas_idx) and Tx >= 2:
                        ax.plot(
                            t_u_i[infeas_idx],
                            X[:-1, idx][infeas_idx],
                            marker=".",
                            linestyle="None",
                            markersize=8,
                        )

                ax.grid(True)

    # x-labels only on bottom row
    axs[2, 0].set_xlabel("time [s]")
    axs[2, 1].set_xlabel("time [s]")
    axs[2, 2].set_xlabel("time [s]")

    # keep legends compact (top row only)
    axs[0, 0].legend(loc="best")
    axs[0, 1].legend(loc="best")
    axs[0, 2].legend(loc="best")

    fig.tight_layout()

    if (path is not None) and (filename is not None):
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.set_size_inches(14, 9)
        fig.savefig(os.path.join(path, f"{filename}_cl_grid.png"), dpi=300)

    if plt_show:
        plt.show()

    # ----------------------------
    # XY plot (separate)
    # ----------------------------
    plt.figure()
    axis_x, axis_y = xy_axes
    plt.xlabel(xlabels[axis_x])
    plt.ylabel(xlabels[axis_y])
    plt.gca().set_aspect("equal")

    for i in range(Ntrajs):
        X = Xtraj[i]
        f = feasible[i] if feasible is not None else np.ones(X.shape[0] - 1, dtype=int)

        plt.plot(X[:, axis_x], X[:, axis_y], linestyle=linestyles[i % len(linestyles)], label=labels[i])

        infeas_idx = (f == 0)
        if np.any(infeas_idx):
            Xk = X[:-1, :]
            plt.plot(
                Xk[infeas_idx, axis_x],
                Xk[infeas_idx, axis_y],
                marker=".",
                linestyle="None",
                markersize=8,
            )

    plt.grid(True)
    plt.legend(loc="best")

    if (path is not None) and (filename is not None):
        plt.gcf().set_size_inches(8, 6)
        plt.savefig(os.path.join(path, f"{filename}_cl_xy.png"), dpi=300)

    if plt_show:
        plt.show()


def plot_ctdistro(ct, plt_show=True):
    """Histogram of solver compute times in milliseconds using log-spaced bins."""
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
    plt.ylabel('fraction [%]')
    plt.grid(True)

    lgnd = plt.legend()
    if lgnd is not None:
        lgnd.remove()

    plt.xlabel('compute time [ms]')
    plt.tight_layout()

    if plt_show:
        plt.show()
