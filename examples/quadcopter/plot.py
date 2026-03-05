from pathlib import Path
import matplotlib

# Headless-safe backend (must come before pyplot import)
try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np


# Keep original visual choices
_LINESTYLES_CL = ['solid', 'dotted', 'dashed']          # original closed-loop
_LINESTYLES_OL = ['solid', 'dotted', 'dashed', 'dashdot']
_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'darkred', 'navy', 'darkgreen']


def _safe_linestyle(i, styles):
    return styles[i % len(styles)]


def _safe_color(i):
    return _COLORS[i % len(_COLORS)]


def _as2d(a):
    a = np.asarray(a)
    return a if a.ndim == 2 else None


def _safe_show(plt_show):
    if plt_show:
        try:
            plt.show()
        except Exception as e:
            print(f"[WARN] plt.show() failed: {e}")


def _safe_tikz_save(filepath, **kwargs):
    """
    Optional tikz export. Never raises.
    Tries tikzplotlib-patched / tikzplotlib-compatible import names.
    """
    # Try modern patched package first (install as tikzplotlib_patched or tikzplotlib depending on package behavior)
    candidates = ["tikzplotlib", "tikzplotlib_patched", "matplot2tikz", "matplotlib2tikz"]
    for name in candidates:
        try:
            mod = __import__(name)
            if hasattr(mod, "save"):
                mod.save(str(filepath), **kwargs)
                return True
        except Exception:
            continue
    print(f"[WARN] TikZ export skipped/failed for '{filepath}'.")
    return False


def _safe_limit(limits, key, j):
    if limits is None:
        return None
    vals = limits.get(key, None)
    if vals is None:
        return None
    try:
        return vals[j]
    except Exception:
        return None


def _bad_mask(feasible, n):
    """
    Return mask of length n where feasible == 0 (infeasible points).
    Safe for length mismatches.
    """
    mask = np.zeros(n, dtype=bool)
    if feasible is None or n <= 0:
        return mask
    f = np.asarray(feasible).reshape(-1)
    m = min(n, len(f))
    if m > 0:
        mask[:m] = (f[:m] == 0)
    return mask


def _ensure_dir(path):
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_quadcopter_cl(mpc, Utraj, Xtraj, feasible, labels, plt_show=True, limits=None, path=None, filename=None):
    """
    Closed-loop plotting: preserves original layout/appearance, but robust to
    modern Matplotlib/NumPy and exporter failures.
    """
    if limits is None:
        limits = {}

    plt.clf()

    Ntrajs = min(len(Utraj), len(Xtraj), len(feasible), len(labels))
    if Ntrajs == 0:
        print("[WARN] plot_quadcopter_cl: no trajectories.")
        return

    Ts = mpc.Tf / mpc.N
    lengths = []
    for i in range(Ntrajs):
        U = _as2d(Utraj[i])
        lengths.append((U.shape[0] + 1) if U is not None and U.shape[0] > 0 else 1)

    N_sim_max = int(np.max(np.asarray(lengths)))
    t = np.linspace(0, (N_sim_max - 1) * Ts, N_sim_max)

    linestyles = _LINESTYLES_CL
    colors = _COLORS

    xbatches = [[0, 1, 2], [3, 4, 5], [6, 8], [7, 9]]
    ubatches = [[0, 1], [2]]

    xlabels = ["$x_1$", "$x_2$", "$x_3$", "$v_1$", "$v_2$", "$v_3$", "$\\phi_1$", "$\\omega_1$", "$\\phi_2$", "$\\omega_2$"]
    ulabels = ["$u_1$", "$u_2$", "$u_3$"]

    batches = len(xbatches) + len(ubatches)

    # Input subplots
    for k, batch in enumerate(ubatches):
        plt.subplot(batches, 1, k + 1)

        for i in range(Ntrajs):
            U = _as2d(Utraj[i])
            X = _as2d(Xtraj[i])  # kept for parity with original
            f = feasible[i]
            if U is None or U.shape[0] == 0:
                continue

            for j in batch:
                if j >= U.shape[1]:
                    continue

                y = np.append([U[0, j]], U[:, j])
                tt = t[:len(y)]
                plt.step(
                    tt, y,
                    label=labels[i] + " " + ulabels[j],
                    color=colors[j],
                    linestyle=_safe_linestyle(i, linestyles),
                )

                # Mark infeasible inputs aligned to U rows
                mask = _bad_mask(f, U.shape[0])
                if np.any(mask):
                    plt.plot(
                        t[:U.shape[0]][mask],
                        U[:U.shape[0], j][mask],
                        marker='x',
                        linestyle='None',
                        markersize=8,
                        color='red'
                    )

        plt.grid()

        # Preserve original look for limits (dashed horizontal lines)
        for j in batch:
            umin = _safe_limit(limits, "umin", j)
            umax = _safe_limit(limits, "umax", j)

            # Original code used colors[batch[0]-j]; preserve behavior
            c_idx = batch[0] - j
            c = colors[c_idx]

            if umin is not None:
                plt.plot([t[0], t[-1]], [umin, umin], linestyle='dashed', color=c, alpha=0.7)
            if umax is not None:
                plt.plot([t[0], t[-1]], [umax, umax], linestyle='dashed', color=c, alpha=0.7)

        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    # State subplots
    for k, batch in enumerate(xbatches):
        plt.subplot(batches, 1, len(ubatches) + k + 1)

        for i in range(Ntrajs):
            X = _as2d(Xtraj[i])
            if X is None or X.shape[0] == 0:
                continue

            for j in batch:
                if j >= X.shape[1]:
                    continue
                plt.plot(
                    t[:X.shape[0]],
                    X[:, j],
                    label=labels[i] + " " + xlabels[j],
                    color=colors[j],
                    linestyle=_safe_linestyle(i, linestyles),
                )

        for j in batch:
            xmin = _safe_limit(limits, "xmin", j)
            xmax = _safe_limit(limits, "xmax", j)

            c_idx = batch[0] - j
            c = colors[c_idx]

            if xmin is not None:
                plt.plot([t[0], t[-1]], [xmin, xmin], linestyle='dashed', color=c, alpha=0.7)
            if xmax is not None:
                plt.plot([t[0], t[-1]], [xmax, xmax], linestyle='dashed', color=c, alpha=0.7)

        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=False, framealpha=1)

    plt.subplots_adjust(hspace=0.4)
    plt.xlabel("time [s]")

    outdir = _ensure_dir(path) if (path is not None and filename is not None) else None

    if outdir is not None:
        fig = plt.gcf()
        fig.set_size_inches(20, 15)
        fig.savefig(outdir / f"{filename}.png", dpi=300, bbox_inches="tight")
        _safe_tikz_save(outdir / f"{filename}_double.tex", axis_height='1.4in', axis_width='6.8in')
        _safe_tikz_save(outdir / f"{filename}_single.tex", axis_height='1.4in', axis_width='3.4in')

    _safe_show(plt_show)

    # XY projection plot (preserve original style)
    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal')
    axis_x, axis_y = 0, 1
    plt.xlabel(xlabels[axis_x])
    plt.ylabel(xlabels[axis_y])

    ymins = []
    ymaxs = []

    for i in range(Ntrajs):
        X = _as2d(Xtraj[i])
        if X is None or X.shape[0] == 0 or X.shape[1] <= max(axis_x, axis_y):
            continue
        f = feasible[i]

        plt.plot(
            X[:, axis_x], X[:, axis_y],
            label=labels[i],
            linestyle=_safe_linestyle(i, linestyles),
            color=colors[i]
        )

        ymins.append(np.min(X[:, axis_y]))
        ymaxs.append(np.max(X[:, axis_y]))

        # Original code uses X[:-1] with feasibility mask
        X_short = X[:-1, :]
        if X_short.shape[0] > 0:
            mask = _bad_mask(f, X_short.shape[0])
            if np.any(mask):
                plt.plot(
                    X_short[mask, axis_x],
                    X_short[mask, axis_y],
                    marker='.',
                    linestyle='None',
                    markersize=8,
                    color='red'
                )

    if ymins and ymaxs:
        plt.plot(
            [0.145, 0.145],
            [np.min(ymins), np.max(ymaxs)],
            linestyle='dashed',
            color='red'
        )

    plt.tight_layout()
    plt.legend(loc=2)

    if outdir is not None:
        fig = plt.gcf()
        fig.set_size_inches(20, 15)
        fig.savefig(outdir / f"{filename}_xy.png", dpi=300, bbox_inches="tight")
        _safe_tikz_save(outdir / f"{filename}_xy.tex", axis_width='3.4in')


def plot_quadcopter_ol_V(mpc, Vtraj, labels, plt_show=True):
    plt.clf()

    t = np.linspace(0, mpc.Tf, mpc.N + 1)
    Ntrajs = min(len(Vtraj), len(labels))

    linestyles = _LINESTYLES_OL
    colors = _COLORS
    ulabels = ["u_1", "u_2", "u_3"]

    for i in range(Ntrajs):
        V = _as2d(Vtraj[i])
        if V is None or V.shape[0] == 0:
            continue
        for j in range(min(3, V.shape[1])):
            y = np.append([V[0, j]], V[:, j])
            plt.step(
                t[:len(y)],
                y,
                label=labels[i] + " " + ulabels[j],
                color=colors[j],
                linestyle=_safe_linestyle(i, linestyles)
            )

    plt.grid()
    plt.ylabel('inputs v')
    plt.legend(loc=1)
    _safe_show(plt_show)


def plot_quadcopter_ol(mpc, Utraj, Xtraj, labels, plt_show=True, limits=None):
    if limits is None:
        limits = {}

    t = np.linspace(0, mpc.Tf, mpc.N + 1)
    Ntrajs = min(len(Utraj), len(Xtraj), len(labels))

    linestyles = _LINESTYLES_OL
    looselydashed = (0, (5, 10))
    colors = _COLORS

    xbatches = [[0, 1, 2], [3, 4, 5], [6, 8], [7, 9]]
    ubatches = [[0, 1], [2]]

    xlabels = ["x_1", "x_2", "x_3", "v_1", "v_2", "v_3", "phi_1", "omega_1", "phi_2", "omega_2"]
    ulabels = ["u_1", "u_2", "u_3"]

    batches = len(xbatches) + len(ubatches)

    # Inputs
    for k, batch in enumerate(ubatches):
        plt.subplot(batches, 1, k + 1)
        for i in range(Ntrajs):
            V = _as2d(Utraj[i])  # original naming
            X = _as2d(Xtraj[i])
            if V is None or X is None or V.shape[0] == 0:
                continue

            n = min(V.shape[0], X.shape[0])
            if n == 0:
                continue

            U = np.asarray([mpc.stabilizing_feedback_controller(X[j], V[j]) for j in range(n)])
            U = _as2d(U)
            if U is None or U.shape[0] == 0:
                continue

            for j in batch:
                if j >= U.shape[1]:
                    continue
                y = np.append([U[0, j]], U[:, j])
                plt.step(
                    t[:len(y)],
                    y,
                    label=labels[i] + " " + ulabels[j],
                    color=colors[j],
                    linestyle=_safe_linestyle(i, linestyles)
                )

        plt.grid()
        plt.ylabel('inputs u')

        for j in batch:
            umin = _safe_limit(limits, "umin", j)
            umax = _safe_limit(limits, "umax", j)
            c_idx = batch[0] - j
            c = colors[c_idx]
            if umin is not None:
                plt.hlines(umin, t[0], t[-1], linestyles=looselydashed, color=c, alpha=0.7)
            if umax is not None:
                plt.hlines(umax, t[0], t[-1], linestyles=looselydashed, color=c, alpha=0.7)

        plt.legend(loc=1)

    # States
    for k, batch in enumerate(xbatches):
        plt.subplot(batches, 1, len(ubatches) + k + 1)
        for i in range(Ntrajs):
            X = _as2d(Xtraj[i])
            if X is None or X.shape[0] == 0:
                continue
            for j in batch:
                if j >= X.shape[1]:
                    continue
                plt.plot(
                    t[:X.shape[0]],
                    X[:, j],
                    label=labels[i] + " " + xlabels[j],
                    color=colors[j],
                    linestyle=_safe_linestyle(i, linestyles)
                )

        plt.ylabel('$x$')
        for j in batch:
            xmin = _safe_limit(limits, "xmin", j)
            xmax = _safe_limit(limits, "xmax", j)
            c_idx = batch[0] - j
            c = colors[c_idx]
            if xmin is not None:
                plt.hlines(xmin, t[0], t[-1], linestyles=looselydashed, color=c, alpha=0.7)
            if xmax is not None:
                plt.hlines(xmax, t[0], t[-1], linestyles=looselydashed, color=c, alpha=0.7)

        plt.grid()
        plt.legend(loc=1)

    plt.subplots_adjust(hspace=0.4)
    _safe_show(plt_show)


def plot_feas(xfeas, yfeas, xlim=None, ylim=None, plt_show=True):
    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)

    if xlim is not None:
        plt.xlim(1.2 * xlim)
    if ylim is not None:
        plt.ylim(1.2 * ylim)

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()
    plt.scatter(xfeas, yfeas, marker=',', color='blue')
    _safe_show(plt_show)


def plot_feas_notfeas(feas, notfeas, xlim, ylim, plt_show=True):
    plt.clf()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=3.4)

    plt.xlim(1.2 * xlim)
    plt.ylim(1.2 * ylim)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()

    feas = _as2d(feas)
    notfeas = _as2d(notfeas)
    if feas is not None and feas.shape[1] >= 2:
        plt.scatter(feas[:, 0], feas[:, 1], marker='.', color='blue')
    if notfeas is not None and notfeas.shape[1] >= 2:
        plt.scatter(notfeas[:, 0], notfeas[:, 1], marker='.', color='red')

    _ensure_dir("figures")
    _safe_tikz_save("figures/offlineapproximationtestgrid.tex", axis_height='2.4in', axis_width='2.4in')
    _safe_show(plt_show)


def plot_ctdistro(ct, plt_show=True):
    plt.clf()
    computetimes = np.asarray(ct).reshape(-1) * 1000.0
    computetimes = computetimes[np.isfinite(computetimes)]
    computetimes = computetimes[computetimes > 0]

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(w=3.4, h=2.1)

    if computetimes.size == 0:
        print("[WARN] plot_ctdistro: no positive finite values.")
        plt.xlabel('compute time [ms]')
        plt.ylabel('fraction [%]')
        plt.grid()
        _safe_show(plt_show)
        return

    if computetimes.min() == computetimes.max():
        bins = 20
    else:
        bins = np.geomspace(computetimes.min(), computetimes.max(), 20)

    plt.hist(computetimes, density=False, bins=bins)

    locs, _ = plt.yticks()
    plt.yticks(locs, np.round(locs / len(computetimes) * 100, 1))
    plt.ylabel('fraction [%]')
    plt.grid()

    # Original code removes an auto-legend (which may not exist)
    try:
        lgnd = plt.legend()
        if lgnd is not None:
            lgnd.remove()
    except Exception:
        pass

    plt.xlabel('compute time [ms]')
    plt.tight_layout()
    _safe_show(plt_show)