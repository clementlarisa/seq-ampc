#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_dataset.py — inspect + plot saved datasets with plot_vehicle_ol_grid_3x3

Targets the signature used in examples/vehicle_dyn_obs/samplempc.py:

plot_vehicle_ol_grid_3x3(
    mpc,
    Vtraj=[Uh],
    Xtraj=[Xh],
    labels=[...],
    plt_show=True,
    limits=limits,
    input_is_v=True,
    obstacles=[obs_h],
    r_safe=[r_h],
    show_xy=True,
    show_clearance=True,
    print_clearance=True,
    title=...
)

Usage:
  python3 eval_dataset.py --dataset latest
  python3 eval_dataset.py --dataset /abs/path/to/datasets/vehicle_8state_obs_...

Assumptions:
  - run from examples/vehicle_dyn_obs/ OR provide --root accordingly
  - mpc_parameters/ exists under root
  - dataset folder contains X.txt, U.txt, x0.txt (common), and optionally computetimes.txt
  - optionally: P_obstacles.txt (Nsaved,5) and N_active.txt (Nsaved,)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import scipy.linalg

# ---------------- Project imports ----------------
# Ensure repo root on PYTHONPATH (two levels up from examples/* by default)
def _ensure_repo_on_path(root: Path) -> None:
    repo = root.resolve().parents[1]  # examples/vehicle_8state_obs -> repo root
    if str(repo) not in sys.path:
        sys.path.append(str(repo))


def _read_txt_scalar(path: Path) -> float:
    return float(np.genfromtxt(path, delimiter=","))


def _read_txt_mat(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    return np.reshape(np.genfromtxt(path, delimiter=","), shape)


def _load_optional_txt(path: Path, dtype=float) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.loadtxt(path, dtype=dtype)
    return arr


# ---------------- Obstacle helpers (match your conventions) ----------------
FAR = 100.0  # must match generator sentinel

def obstacles_list_from_p(p: np.ndarray) -> List[Tuple[float, float]]:
    p = np.asarray(p, dtype=float).reshape(-1)
    o1 = (float(p[0]), float(p[1]))
    o2 = (float(p[2]), float(p[3]))
    obs = []
    if abs(o1[0]) < FAR * 0.1 and abs(o1[1]) < FAR * 0.1:
        obs.append(o1)
    if abs(o2[0]) < FAR * 0.1 and abs(o2[1]) < FAR * 0.1:
        obs.append(o2)
    return obs


# ---------------- Dataset loading ----------------
def _try_load_dataset_arrays(ds_dir: Path, nx: int, nu: int, N: int) -> Dict[str, Any]:
    """
    Robust loader for common seqampc txt exports.

    Expected (typical):
      x0.txt: (Nsaved, nx)
      X.txt : either (Nsaved, (N+1)*nx) or (Nsaved*(N+1), nx)
      U.txt : either (Nsaved, N*nu) or (Nsaved*N, nu)
      computetimes.txt: (Nsaved,) optional

    Also optional:
      P_obstacles.txt: (Nsaved, 5)
      N_active.txt: (Nsaved,)
    """
    out: Dict[str, Any] = {}

    # required-ish
    x0_path = ds_dir / "x0.txt"
    X_path  = ds_dir / "X.txt"
    U_path  = ds_dir / "U.txt"

    if not x0_path.exists():
        raise FileNotFoundError(f"Missing {x0_path}")
    if not X_path.exists():
        raise FileNotFoundError(f"Missing {X_path}")
    if not U_path.exists():
        raise FileNotFoundError(f"Missing {U_path}")

    x0 = np.loadtxt(x0_path)
    if x0.ndim == 1:
        x0 = x0.reshape(1, -1)
    if x0.shape[1] != nx:
        raise RuntimeError(f"x0.txt has shape {x0.shape}, expected (*,{nx})")

    Ns = x0.shape[0]

    Xraw = np.loadtxt(X_path)
    Uraw = np.loadtxt(U_path)

    # X reshape
    X = None
    if Xraw.ndim == 1:
        # single row
        if Xraw.size == (N + 1) * nx:
            X = Xraw.reshape(1, N + 1, nx)
        else:
            raise RuntimeError(f"X.txt 1D size={Xraw.size} not compatible with (N+1)*nx={(N+1)*nx}")
    elif Xraw.ndim == 2:
        if Xraw.shape[0] == Ns and Xraw.shape[1] == (N + 1) * nx:
            X = Xraw.reshape(Ns, N + 1, nx)
        elif Xraw.shape[1] == nx and (Xraw.shape[0] % (N + 1) == 0):
            Ns2 = Xraw.shape[0] // (N + 1)
            if Ns2 != Ns:
                # allow mismatch, but warn-ish by storing both
                pass
            X = Xraw.reshape(Ns2, N + 1, nx)
        else:
            raise RuntimeError(f"X.txt shape={Xraw.shape} not compatible with Ns={Ns}, N={N}, nx={nx}")
    else:
        raise RuntimeError(f"X.txt has unsupported ndim={Xraw.ndim}")

    # U reshape
    U = None
    if Uraw.ndim == 1:
        if Uraw.size == N * nu:
            U = Uraw.reshape(1, N, nu)
        else:
            raise RuntimeError(f"U.txt 1D size={Uraw.size} not compatible with N*nu={N*nu}")
    elif Uraw.ndim == 2:
        if Uraw.shape[0] == Ns and Uraw.shape[1] == N * nu:
            U = Uraw.reshape(Ns, N, nu)
        elif Uraw.shape[1] == nu and (Uraw.shape[0] % N == 0):
            Ns2 = Uraw.shape[0] // N
            U = Uraw.reshape(Ns2, N, nu)
        else:
            raise RuntimeError(f"U.txt shape={Uraw.shape} not compatible with Ns={Ns}, N={N}, nu={nu}")
    else:
        raise RuntimeError(f"U.txt has unsupported ndim={Uraw.ndim}")

    # optional
    ct = _load_optional_txt(ds_dir / "computetimes.txt", dtype=float)
    if ct is not None and ct.ndim == 0:
        ct = np.array([float(ct)])
    if ct is not None and ct.ndim == 1 and ct.shape[0] != X.shape[0]:
        # tolerate, but keep
        pass

    P_obs = _load_optional_txt(ds_dir / "P_obstacles.txt", dtype=float)
    N_active = _load_optional_txt(ds_dir / "N_active.txt", dtype=int)

    out["x0"] = x0
    out["X"] = X
    out["U"] = U
    out["computetimes"] = ct
    out["P_obstacles"] = P_obs
    out["N_active"] = N_active
    out["Nsaved"] = int(X.shape[0])
    return out


# ---------------- MPC build (lightweight) ----------------
def build_mpc_from_parameters(root: Path):
    """
    Builds MPCQuadraticCostLxLu with the same parameter convention as samplempc_obs.py,
    but WITHOUT creating an Acados solver.

    Returns: mpc, meta dict
    """
    # imports here after sys.path setup
    from dynamics.f import f
    from seqampc.mpcproblem import MPCQuadraticCostLxLu

    pdir = root / "mpc_parameters"

    # infer dims from stored matrices
    # Q,P are (nx,nx), R is (nu,nu)
    # read Q first to get nx
    Q_flat = np.genfromtxt(pdir / "Q.txt", delimiter=",")
    # Q is square; if saved row-major flat, reshape needs nx known => infer nx by sqrt
    if Q_flat.ndim == 1:
        n2 = Q_flat.size
        nx = int(round(np.sqrt(n2)))
        if nx * nx != n2:
            raise RuntimeError(f"Q.txt flat size {n2} is not a perfect square.")
        Q = Q_flat.reshape(nx, nx)
    else:
        Q = np.array(Q_flat, dtype=float)
        nx = Q.shape[0]
        if Q.shape[1] != nx:
            raise RuntimeError(f"Q.txt is not square: {Q.shape}")

    # R => nu
    R_flat = np.genfromtxt(pdir / "R.txt", delimiter=",")
    if R_flat.ndim == 1:
        m2 = R_flat.size
        nu = int(round(np.sqrt(m2)))
        if nu * nu != m2:
            raise RuntimeError(f"R.txt flat size {m2} is not a perfect square.")
        R = R_flat.reshape(nu, nu)
    else:
        R = np.array(R_flat, dtype=float)
        nu = R.shape[0]
        if R.shape[1] != nu:
            raise RuntimeError(f"R.txt is not square: {R.shape}")

    # P
    P = _read_txt_mat(pdir / "P.txt", (nx, nx))

    # N and Tf (you used N=30 hard-coded in sample; keep as param here)
    Tf = _read_txt_scalar(pdir / "Tf.txt")
    N = 30

    # gains
    K = _read_txt_mat(pdir / "K.txt", (nx, nu)).T        # (nu,nx)
    Kdelta = _read_txt_mat(pdir / "Kdelta.txt", (nx, nu)).T  # (nu,nx)

    # constraints
    # We need nconstr from Lu columns; easiest: load raw then infer.
    Lu_raw = np.genfromtxt(pdir / "Lu.txt", delimiter=",")
    # file stored as (nu, nconstr) flattened? In sample: reshape((nu,nconstr)).T
    # => total entries = nu*nconstr
    if Lu_raw.ndim == 1:
        total = Lu_raw.size
        if total % nu != 0:
            raise RuntimeError(f"Lu.txt size {total} not divisible by nu={nu}")
        nconstr = total // nu
        Lu = Lu_raw.reshape(nu, nconstr).T
    else:
        # if already 2D, assume it's (nconstr, nu) or (nu, nconstr)
        A = np.array(Lu_raw, dtype=float)
        if A.shape[1] == nu:
            Lu = A
            nconstr = A.shape[0]
        elif A.shape[0] == nu:
            Lu = A.T
            nconstr = Lu.shape[0]
        else:
            raise RuntimeError(f"Cannot interpret Lu.txt shape {A.shape} with nu={nu}")

    # Lx: stored (nx, nconstr) then transposed -> (nconstr,nx)
    Lx_raw = np.genfromtxt(pdir / "Lx.txt", delimiter=",")
    if Lx_raw.ndim == 1:
        total = Lx_raw.size
        if total != nx * nconstr:
            raise RuntimeError(f"Lx.txt size {total} != nx*nconstr={nx*nconstr}")
        Lx = Lx_raw.reshape(nx, nconstr).T
    else:
        A = np.array(Lx_raw, dtype=float)
        if A.shape == (nconstr, nx):
            Lx = A
        elif A.shape == (nx, nconstr):
            Lx = A.T
        else:
            raise RuntimeError(f"Cannot interpret Lx.txt shape {A.shape}, expected {(nconstr,nx)} or {(nx,nconstr)}")

    # Ls (1, nconstr) then transposed -> (nconstr,1)
    Ls_raw = np.genfromtxt(pdir / "Ls.txt", delimiter=",")
    if Ls_raw.ndim == 1:
        if Ls_raw.size != nconstr:
            # could be 1*nconstr flat -> ok if equal; else error
            raise RuntimeError(f"Ls.txt size {Ls_raw.size} != nconstr={nconstr}")
        Ls = Ls_raw.reshape(1, nconstr).T
    else:
        A = np.array(Ls_raw, dtype=float)
        if A.shape == (nconstr, 1):
            Ls = A
        elif A.shape == (1, nconstr):
            Ls = A.T
        else:
            raise RuntimeError(f"Cannot interpret Ls.txt shape {A.shape} with nconstr={nconstr}")

    alpha_f = _read_txt_scalar(pdir / "alpha.txt")

    # warm-start S trajectory (same idea as sample)
    rho = _read_txt_scalar(pdir / "rho_c.txt")
    w_bar = _read_txt_scalar(pdir / "wbar.txt")
    from scipy.integrate import odeint
    Sinit = odeint(lambda y, t: -rho * y + w_bar, 0, np.linspace(0, Tf, N + 1))

    # terminal shrink used in sample (for info/plot limits; MPC object needs alpha_f anyway)
    alpha_s = _read_txt_scalar(pdir / "alpha_s.txt")
    alpha_reduced = alpha_f - alpha_s * (1 - np.exp(-rho * Tf)) / rho * w_bar

    mpc = MPCQuadraticCostLxLu(
        f, nx, nu, N, Tf,
        Q, R, P, alpha_f,
        K, Lx, Lu, Kdelta,
        alpha_reduced=alpha_reduced,
        S=Sinit,
        Ls=Ls
    )
    mpc.name = str(root.name)

    meta = dict(nx=nx, nu=nu, N=N, Tf=Tf, nconstr=nconstr, alpha_f=alpha_f,
                rho=rho, w_bar=w_bar, alpha_reduced=alpha_reduced)
    return mpc, meta


# ---------------- CLI helpers ----------------
def _parse_indices(spec: str, N: int) -> List[int]:
    """
    "0,5,10" or "0 5 10" or "0:8" (inclusive) or "0:20:2" or "all"
    Allows negative indices.
    """
    s = spec.strip().lower()
    if s == "all":
        return list(range(N))
    s = s.replace(" ", ",")
    parts = [p for p in s.split(",") if p != ""]
    idx: List[int] = []
    for p in parts:
        if ":" in p:
            toks = p.split(":")
            if len(toks) not in (2, 3):
                raise ValueError(f"Bad slice: {p}")
            a = int(toks[0]) if toks[0] != "" else 0
            b_incl = int(toks[1]) if toks[1] != "" else (N - 1)
            step = int(toks[2]) if (len(toks) == 3 and toks[2] != "") else 1
            if step == 0:
                raise ValueError("step must be nonzero")
            b_excl = b_incl + (1 if step > 0 else -1)
            idx.extend(list(range(a, b_excl, step)))
        else:
            idx.append(int(p))

    out: List[int] = []
    for i in idx:
        if i < 0:
            i = N + i
        if not (0 <= i < N):
            raise ValueError(f"Index {i} out of bounds for N={N}")
        out.append(i)

    # unique preserve order
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def _print_dataset_summary(ds: Dict[str, Any], meta: Dict[str, Any], ds_dir: Path) -> None:
    print("\n================= DATASET SUMMARY =================")
    print("Dataset dir:", ds_dir)
    print(f"Nsaved = {ds['Nsaved']}")
    print(f"X shape = {ds['X'].shape}  (Nsaved, N+1, nx)")
    print(f"U shape = {ds['U'].shape}  (Nsaved, N, nu)  [decision input v]")
    print(f"x0 shape= {ds['x0'].shape}")

    if ds.get("computetimes") is not None:
        ct = ds["computetimes"]
        print(f"computetimes shape = {ct.shape}")
        if ct.size > 0:
            print(f"computetimes: min={np.min(ct):.6g}, mean={np.mean(ct):.6g}, max={np.max(ct):.6g}")

    P_obs = ds.get("P_obstacles")
    N_act = ds.get("N_active")
    if P_obs is not None:
        print(f"P_obstacles shape = {P_obs.shape} (expected Nsaved,5)")
    if N_act is not None:
        print(f"N_active shape    = {N_act.shape}")
        try:
            uniq, cnt = np.unique(N_act.astype(int), return_counts=True)
            dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
            print("N_active distribution:", dist)
        except Exception:
            pass

    print("\n--- MPC meta (from mpc_parameters) ---")
    for k in ["nx", "nu", "N", "Tf", "nconstr", "alpha_f", "alpha_reduced", "rho", "w_bar"]:
        if k in meta:
            print(f"{k:>13s}: {meta[k]}")
    print("===================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-d", type=str, required=True,
                    help="Dataset folder name under <root>/datasets/ OR an absolute path OR 'latest'.")
    ap.add_argument("--root", type=str, default=".",
                    help="Root folder containing mpc_parameters/ and datasets/ (default: current dir).")
    ap.add_argument("--no-interactive", action="store_true",
                    help="Only print summary, no plotting prompt.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    _ensure_repo_on_path(root)

    # imports after path
    from plot import plot_vehicle_ol_grid_3x3  # signature as in your sample

    # build mpc from parameters
    mpc, meta = build_mpc_from_parameters(root)
    nx, nu, N = meta["nx"], meta["nu"], meta["N"]

    # resolve dataset dir
    ds_arg = args.dataset
    ds_dir = Path(ds_arg).expanduser()
    if not ds_dir.is_absolute():
        ds_dir = (root / "datasets" / ds_arg).resolve()
    else:
        ds_dir = ds_dir.resolve()

    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {ds_dir}")

    # load arrays
    ds = _try_load_dataset_arrays(ds_dir, nx=nx, nu=nu, N=N)
    _print_dataset_summary(ds, meta, ds_dir)

    if args.no_interactive:
        return

    # plot limits (match samplempc_obs.py)
    import math
    delta_max = 25.0 * math.pi / 180.0
    v_min, v_max = 0.5, 25.0

    # input bounds from Lu (same method you used in sample)
    Lu = mpc.Lu  # (nconstr, nu) as stored in MPCQuadraticCostLxLu
    nxconstr = 0
    # u = Kdelta*x + v, and v is the decision variable
    # For "limits" in plot, we show bounds of applied u (umin/umax) as you did.
    diag0 = Lu[nxconstr + 0, 0]
    diag1 = Lu[nxconstr + 1, 1]
    diag0b = Lu[nxconstr + nu + 0, 0]
    diag1b = Lu[nxconstr + nu + 1, 1]
    if min(abs(diag0), abs(diag1), abs(diag0b), abs(diag1b)) < 1e-12:
        # fallback (still plot)
        umin = np.array([-1.0, -1.0], dtype=float)
        umax = np.array([+1.0, +1.0], dtype=float)
    else:
        umax = np.array([1.0 / Lu[nxconstr + i, i] for i in range(nu)], dtype=float)
        umin = np.array([1.0 / Lu[nxconstr + nu + i, i] for i in range(nu)], dtype=float)

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

    # interactive prompt
    Nsaved = ds["Nsaved"]

    print("Interactive plotting:")
    print("  indices:  0,5,10   or  0 5 10")
    print("  range:    0:8      (inclusive) or 0:20:2")
    print("  all:      all")
    print("  quit:     q\n")

    while True:
        try:
            spec = input(f"Plot which samples? (Nsaved={Nsaved}) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break

        if spec.lower() in ("q", "quit", "exit"):
            break
        if spec == "":
            continue

        try:
            idx = _parse_indices(spec, Nsaved)
        except Exception as e:
            print("[ERROR] parse:", e)
            continue

        if len(idx) == 0:
            print("[WARN] no indices.")
            continue

        # chunk into 3x3 (=9) figures
        chunk = 9
        chunks = [idx[i:i + chunk] for i in range(0, len(idx), chunk)]

        Xdataset = ds["X"]
        Udataset = ds["U"]
        P_obs = ds.get("P_obstacles")
        N_act = ds.get("N_active")

        for ci, inds in enumerate(chunks):
            Vtraj = []
            Xtraj = []
            labels = []
            obstacles = []
            r_safe = []

            for j in inds:
                Xh = Xdataset[j]      # (N+1, nx)
                Uh = Udataset[j]      # (N, nu) decision variable v

                Vtraj.append(Uh)
                Xtraj.append(Xh)

                if N_act is not None:
                    na = int(np.asarray(N_act[j]).reshape(-1)[0])
                    labels.append(f"idx={j} (n_active={na})")
                else:
                    labels.append(f"idx={j}")

                if P_obs is not None:
                    p = P_obs[j].reshape(-1)
                    obs = obstacles_list_from_p(p)
                    obstacles.append(obs)
                    r_safe.append(float(p[4]))
                else:
                    obstacles.append([])
                    r_safe.append(1.0)

            title = f"{ds_dir.name} | samples {inds[0]}..{inds[-1]} ({ci+1}/{len(chunks)})"

            plot_vehicle_ol_grid_3x3(
                mpc,
                Vtraj=Vtraj,
                Xtraj=Xtraj,
                labels=labels,
                plt_show=True,
                limits=limits,
                input_is_v=True,
                obstacles=obstacles,
                r_safe=r_safe,
                show_xy=True,
                show_clearance=True,
                print_clearance=True,
                title=title,
            )

        print("Done.\n")


if __name__ == "__main__":
    main()
