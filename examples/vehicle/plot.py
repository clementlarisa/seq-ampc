import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Vehicle plotting helpers (kinematic bicycle)
# State: x = [px, py, psi, v]          (nx = 4)
# Input: u = [delta, a]               (nu = 2)
# ============================================================


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


def plot_vehicle_cl(mpc, Utraj, Xtraj, feasible, labels,
                    plt_show=True, limits=None, path=None, filename=None,
                    xy_axes=(0, 1)):
    """
    Closed-loop trajectory plot for the vehicle example.

    Parameters
    ----------
    mpc:
        MPC object (used for nx, N, Tf).
    Utraj:
        List of input trajectories. Each entry i is an array of shape (T_i, nu),
        where T_i is the simulation length for that rollout.
    Xtraj:
        List of state trajectories. Each entry i is an array of shape (T_i+1, nx).
    feasible:
        List of feasibility indicators per rollout. Each entry i is an array of length T_i
        where f[k] == 0 marks an *infeasible step*.
    labels:
        List of strings, one per rollout (e.g. ["MPC", "NN+filter"]).
    plt_show:
        If True, calls plt.show().
    limits:
        Optional dict with keys "umin","umax","xmin","xmax".
    path, filename:
        If given, save plots as PNG + TikZ.
    xy_axes:
        Which state indices to use for the XY plot. Default (0,1) => px vs py.
    """
    if limits is None:
        limits = {}

    plt.clf()
    Ntrajs = len(Utraj)

    nx = mpc.nx
    nu = Utraj[0].shape[1] if Ntrajs > 0 else 2

    Ts = mpc.Tf / mpc.N

    # Determine the longest rollout length, to build a time axis large enough
    N_sim_max = int(np.max(np.array([len(Utraj[i]) + 1 for i in range(Ntrajs)])))
    t = np.linspace(0, (N_sim_max - 1) * Ts, N_sim_max)

    # Style setup
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'darkred', 'navy', 'darkgreen']

    # Vehicle-specific batching:
    ubatches = [[0, 1]]
    xbatches = [[0, 1], [2], [3]]

    # ✅ labels WITH units (used in legend + ylabels)
    xlabels = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\psi$ [rad]", r"$v$ [m/s]"]
    ulabels = [r"$\delta$ [rad]", r"$a$ [m/s$^2$]"]

    batches = len(ubatches) + len(xbatches)

    # ----------------------------
    # Plot inputs (step plots)
    # ----------------------------
    for k in range(len(ubatches)):
        plt.subplot(batches, 1, k + 1)
        batch = ubatches[k]

        for i in range(Ntrajs):
            U = Utraj[i]
            f = feasible[i] if feasible is not None else np.ones(U.shape[0], dtype=int)

            for j in batch:
                plt.step(
                    t[:U.shape[0] + 1],
                    np.append([U[0, j]], U[:, j]),
                    label=f"{labels[i]} {ulabels[j]}",
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

        plt.grid()
        # ✅ proper ylabel for the input batch
        plt.ylabel(_ylabel_for_batch(ulabels, batch))
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    # ----------------------------
    # Plot states (line plots)
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

        plt.grid()
        # ✅ proper ylabel for each state subplot
        plt.ylabel(_ylabel_for_batch(xlabels, batch))
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    plt.subplots_adjust(hspace=0.4)
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
                             axis_height='1.4in', axis_width='6.8in')
            tikzplotlib.save(os.path.join(path, f"{filename}_single.tex"),
                             axis_height='1.4in', axis_width='3.4in')
        except Exception:
            pass

    # ----------------------------
    # Additional XY plot (px vs py by default)
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

    plt.grid()
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

def _reconstruct_u_from_v(mpc, X, V):
    """u = Kdelta*x + v (fallback: treat V as u if Kdelta not available)."""
    if hasattr(mpc, "Kdelta"):
        Kdelta = np.asarray(mpc.Kdelta)  # (nu,nx)
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

    return np.array(V, copy=True)


def plot_vehicle_ol_grid_3x2(
    mpc,
    Vtraj,          # list of (N,2)   (usually solver variable v)
    Xtraj,          # list of (N+1,4)
    labels,
    plt_show=True,
    limits=None,    # optional: {"umin":[...], "umax":[...]} for applied u
    input_is_v=True # if True, Vtraj is v, we reconstruct u; if False, Vtraj is already u
):
    """
    3x2 multi-subplot:
      [delta] [a_cmd]
      [p_x ] [p_y  ]
      [psi ] [v    ]
    """
    if limits is None:
        limits = {}

    # time grid over horizon
    t = np.linspace(0, mpc.Tf, mpc.N + 1)  # length N+1

    # labels
    xlab = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\psi$ [rad]", r"$v$ [m/s]"]
    ulab = [r"$\delta$ [rad]", r"$a_{\mathrm{cmd}}$ [m/s$^2$]"]
    vlab = [r"$v_\delta$ [rad]", r"$v_a$ [m/s$^2$]"]

    # create grid
    fig, axs = plt.subplots(3, 2, sharex=True)
    axs = np.asarray(axs)

    # helper to keep legends readable
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']

    for i, (V, X) in enumerate(zip(Vtraj, Xtraj)):
        ls = linestyles[i % len(linestyles)]
        name = labels[i]

        # reconstruct applied input u for plotting
        if input_is_v:
            U = _reconstruct_u_from_v(mpc, X, V)  # (N,2)
        else:
            U = np.array(V, copy=True)

        # ----- inputs -----
        # step plot needs N+1 points -> prepend first value
        axs[0, 0].step(t, np.append([U[0, 0]], U[:, 0]), linestyle=ls, label=name)
        axs[0, 1].step(t, np.append([U[0, 1]], U[:, 1]), linestyle=ls, label=name)

        # ----- states -----
        axs[1, 0].plot(t[:X.shape[0]], X[:, 0], linestyle=ls, label=name)  # px
        axs[1, 1].plot(t[:X.shape[0]], X[:, 1], linestyle=ls, label=name)  # py
        axs[2, 0].plot(t[:X.shape[0]], X[:, 2], linestyle=ls, label=name)  # psi
        axs[2, 1].plot(t[:X.shape[0]], X[:, 3], linestyle=ls, label=name)  # v

    # ----- bounds on applied inputs (optional) -----
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

    # axis labels
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

    # one legend per column (keeps it compact)
    axs[0, 0].legend(loc="best")
    axs[0, 1].legend(loc="best")

    fig.tight_layout()

    if plt_show:
        plt.show()
    return fig, axs

def plot_vehicle_ol_V(mpc, Vtraj, labels, plt_show=True):
    """
    Open-loop plot of the *decision variables* v over the MPC horizon for nu=2.
    """
    plt.clf()
    t = np.linspace(0, mpc.Tf, mpc.N + 1)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['r', 'g', 'b', 'c', 'm']

    # ✅ labels with units (decision variables)
    vlabels = [r"$v_\delta$ [rad]", r"$v_a$ [m/s$^2$]"]

    for i in range(len(Vtraj)):
        V = Vtraj[i]
        for j in range(2):
            plt.step(
                t, np.append([V[0, j]], V[:, j]),
                label=f"{labels[i]} {vlabels[j]}",
                color=colors[j],
                linestyle=linestyles[i % len(linestyles)]
            )

    plt.grid()
    plt.ylabel(_ylabel_for_batch(vlabels, [0, 1]))
    plt.legend(loc=1)

    if plt_show:
        plt.show()


def plot_vehicle_ol(mpc, Vtraj, Xtraj, labels, plt_show=True, limits=None):
    """
    Open-loop plot of predicted trajectories over the MPC horizon.

    NOTE:
    Your snippet ended mid-function originally. Below is a COMPLETE working
    implementation with proper ylabels for inputs and states.

    This version plots the *decision variables* v as inputs (since Vtraj is v).
    If you want applied u instead, you need Kdelta and X to reconstruct u.
    """
    if limits is None:
        limits = {}

    plt.clf()
    t = np.linspace(0, mpc.Tf, mpc.N + 1)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    ubatches = [[0, 1]]
    xbatches = [[0, 1], [2], [3]]
    batches = len(ubatches) + len(xbatches)

    xlabels = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\psi$ [rad]", r"$v$ [m/s]"]
    vlabels = [r"$v_\delta$ [rad]", r"$v_a$ [m/s$^2$]"]

    # ----------------------------
    # Inputs (decision v)
    # ----------------------------
    for k in range(len(ubatches)):
        plt.subplot(batches, 1, k + 1)
        batch = ubatches[k]

        for i in range(len(Vtraj)):
            V = Vtraj[i]
            for j in batch:
                plt.step(
                    t,
                    np.append([V[0, j]], V[:, j]),
                    label=f"{labels[i]} {vlabels[j]}",
                    color=colors[j],
                    linestyle=linestyles[i % len(linestyles)]
                )

        # optional bounds (if you pass limits for v)
        for j in batch:
            if "umin" in limits and limits["umin"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["umin"][j], limits["umin"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)
            if "umax" in limits and limits["umax"][j] is not None:
                plt.plot([t[0], t[-1]], [limits["umax"][j], limits["umax"][j]],
                         linestyle='dashed', color=colors[j], alpha=0.7)

        plt.grid()
        plt.ylabel(_ylabel_for_batch(vlabels, batch))
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    # ----------------------------
    # States
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

        plt.grid()
        plt.ylabel(_ylabel_for_batch(xlabels, batch))
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    plt.subplots_adjust(hspace=0.4)
    plt.xlabel("time [s]")

    if plt_show:
        plt.show()


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
    plt.scatter(xfeas, yfeas, marker=',', color='blue')

    if plt_show:
        plt.show()


def plot_feas_notfeas(feas, notfeas, xlim=None, ylim=None,
                      xlabel=r"$p_x$ [m]", ylabel=r"$p_y$ [m]",
                      plt_show=True, save_tex_path=None):
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

    if feas is not None and len(feas) > 0:
        plt.scatter(feas[:, 0], feas[:, 1], marker='.', color='blue')
    if notfeas is not None and len(notfeas) > 0:
        plt.scatter(notfeas[:, 0], notfeas[:, 1], marker='.', color='red')

    if save_tex_path is not None:
        try:
            import tikzplotlib
            tikzplotlib.save(save_tex_path, axis_height='2.4in', axis_width='2.4in')
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
    plt.ylabel('fraction [%]')
    plt.grid()

    lgnd = plt.legend()
    if lgnd is not None:
        lgnd.remove()

    plt.xlabel('compute time [ms]')
    plt.tight_layout()

    if plt_show:
        plt.show()
