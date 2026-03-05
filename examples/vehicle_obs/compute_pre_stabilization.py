import os
from pathlib import Path
import numpy as np
import scipy.linalg
from casadi import SX, Function, jacobian, vertcat
import matplotlib.pyplot as plt
# Make sure we can import your dynamics
from dynamics.f import f


def _flatten_for_txt(K_nu_by_nx: np.ndarray) -> str:
    """
    Your project stores gains as (nx, nu) flattened, and then loads and transposes:
        K = reshape(txt, (nx, nu)).T

    So we must write K.T (shape nx x nu) flattened row-major.
    """
    K_T = K_nu_by_nx.T  # (nx, nu)
    return ",".join([f"{v:.6f}" for v in K_T.reshape(-1, order="C")])


def _write_txt(path: Path, arr_string: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(arr_string + "\n")


def _linearize_continuous(nx: int, nu: int, x_star: np.ndarray, u_star: np.ndarray):
    """
    Linearize your (continuous-time) dynamics xdot = f(x,u) around (x*, u*):
        xdot ≈ A (x-x*) + B (u-u*)

    Returns A, B as numpy arrays.
    """
    x = SX.sym("x", nx, 1)
    u = SX.sym("u", nu, 1)

    # f(x,u) in your repo returns an iterable of components; stack into a vector
    fx = f(x, u)
    fvec = vertcat(*fx)

    A_sym = jacobian(fvec, x)
    B_sym = jacobian(fvec, u)

    A_fun = Function("A_fun", [x, u], [A_sym])
    B_fun = Function("B_fun", [x, u], [B_sym])

    A = np.array(A_fun(x_star, u_star)).astype(float)
    B = np.array(B_fun(x_star, u_star)).astype(float)
    return A, B


def _c2d_zoh(Ac: np.ndarray, Bc: np.ndarray, dt: float):
    """
    Discretize continuous (Ac,Bc) with zero-order hold:
        [Ad Bd; 0 I] = expm([Ac Bc; 0 0] dt)
    """
    nx = Ac.shape[0]
    nu = Bc.shape[1]
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = Ac
    M[:nx, nx:] = Bc
    Md = scipy.linalg.expm(M * dt)
    Ad = Md[:nx, :nx]
    Bd = Md[:nx, nx:]
    return Ad, Bd


def _dlqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """
    Discrete LQR:
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = (R + B' P B)^-1 (B' P A)
    Returns K (nu x nx), P.
    """
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def compute_and_export_lqr(
    Tf: float = None,
    N: int = 10,
    # operating point (for vehicle x=[px,py,psi,v], u=[delta,a])
    px0: float = 0.0,
    py0: float = 0.0,
    psi0: float = 0.0,
    v0: float = 8.0,
    delta0: float = 0.0,
    a0: float = 0.0,
    # scaling for Kinit and Kdelta
    kinit_scale: float = 1.0,
    kdelta_scale: float = 0.5,
):
    """
    Computes LQR gain for the vehicle model by linearizing at (x*,u*),
    discretizing with dt=Tf/N, and solving discrete LQR.

    Exports:
      - mpc_parameters/K.txt
      - mpc_parameters/Kinit.txt  (= kinit_scale * K)
      - mpc_parameters/Kdelta.txt (= kdelta_scale * K)
    """
    fp = Path(os.path.dirname(__file__))
    param_dir = fp / "mpc_parameters"

    # dimensions for vehicle
    nx = 4
    nu = 2

    # load Tf, Q, R if not provided
    if Tf is None:
        Tf_path = param_dir / "Tf.txt"
        if Tf_path.exists():
            Tf = float(np.genfromtxt(Tf_path, delimiter=","))
        else:
            raise FileNotFoundError(f"Tf.txt not found at {Tf_path}. Provide Tf explicitly.")

    dt = Tf / N

    Q_path = param_dir / "Q.txt"
    R_path = param_dir / "R.txt"
    if not Q_path.exists() or not R_path.exists():
        raise FileNotFoundError("Need Q.txt and R.txt in mpc_parameters/ to compute LQR consistently.")

    Q = np.reshape(np.genfromtxt(Q_path, delimiter=","), (nx, nx)).astype(float)
    R = np.reshape(np.genfromtxt(R_path, delimiter=","), (nu, nu)).astype(float)

    # operating point
    x_star = np.array([px0, py0, psi0, v0], dtype=float).reshape(nx, 1)
    u_star = np.array([delta0, a0], dtype=float).reshape(nu, 1)

    # 1) linearize continuous-time model
    Ac, Bc = _linearize_continuous(nx, nu, x_star, u_star)

    # 2) discretize
    Ad, Bd = _c2d_zoh(Ac, Bc, dt)

    # 3) LQR
    K, P = _dlqr(Ad, Bd, Q, R)

    # basic sanity prints
    print("=== LQR computed ===")
    print("Tf =", Tf, "N =", N, "dt =", dt)
    print("x* =", x_star.ravel())
    print("u* =", u_star.ravel())
    print("K (nu x nx) =\n", K)
    eig_cl = np.linalg.eigvals(Ad - Bd @ K)
    print("eig(Ad-BdK) =", eig_cl)

    # 4) Plots to check what the LQR is doing (step/reference jump)
    plot_lqr_step_responses(Ad, Bd, K, dt)

    plot_eigs_discrete(Ad, Bd, K, title="Vehicle LQR eigenvalues (open vs closed loop)")

    # 4) export in repo format (flatten K.T)


def simulate_linear_cl(Ad, Bd, K, x0, x_ref=None, u_ref=None, T=50):
    """
    Simulate discrete-time closed-loop:
        u = u_ref - K (x - x_ref)
        x_{k+1} = Ad x_k + Bd u_k
    """
    nx = Ad.shape[0]
    nu = Bd.shape[1]

    if x_ref is None:
        x_ref = np.zeros((nx, 1))
    if u_ref is None:
        u_ref = np.zeros((nu, 1))

    X = np.zeros((T + 1, nx))
    U = np.zeros((T, nu))

    x = x0.reshape(nx, 1)
    X[0, :] = x[:, 0]

    for k in range(T):
        u = u_ref - K @ (x - x_ref)
        x = Ad @ x + Bd @ u

        U[k, :] = u[:, 0]
        X[k + 1, :] = x[:, 0]

    return X, U


def plot_lqr_step_responses(Ad, Bd, K, dt):
    # --- scenario A: lateral offset in py ---
    x0 = np.array([0.0, 1.0, 0.2, 8.0])  # px, py, psi, v (matches your state order)
    X, U = simulate_linear_cl(Ad, Bd, K, x0, x_ref=np.array([0, 0, 0, 8]).reshape(-1, 1), T=60)

    tX = np.arange(X.shape[0]) * dt
    tU = np.arange(U.shape[0]) * dt

    plt.figure()
    plt.plot(tX, X[:, 0], label="px")
    plt.plot(tX, X[:, 1], label="py")
    plt.plot(tX, X[:, 2], label="psi")
    plt.plot(tX, X[:, 3], label="v")
    plt.grid(True);
    plt.legend()
    plt.xlabel("time [s]");
    plt.ylabel("state")
    plt.title("LQR closed-loop response (linear model) - lateral offset")

    plt.figure()
    plt.step(tU, U[:, 0], where="post", label="delta")
    plt.step(tU, U[:, 1], where="post", label="a")
    plt.grid(True);
    plt.legend()
    plt.xlabel("time [s]");
    plt.ylabel("input")
    plt.title("LQR control inputs - lateral offset")

    # --- scenario B: speed error only ---
    x0 = np.array([0.0, 0.0, 0.0, 4.0])  # start slower than v_ref=8
    X, U = simulate_linear_cl(Ad, Bd, K, x0, x_ref=np.array([0, 0, 0, 8]).reshape(-1, 1), T=60)

    tX = np.arange(X.shape[0]) * dt
    tU = np.arange(U.shape[0]) * dt

    plt.figure()
    plt.plot(tX, X[:, 3], label="v")
    plt.grid(True);
    plt.legend()
    plt.xlabel("time [s]");
    plt.ylabel("speed [m/s]")
    plt.title("LQR closed-loop response (linear model) - speed step")

    plt.figure()
    plt.step(tU, U[:, 1], where="post", label="a")
    plt.grid(True);
    plt.legend()
    plt.xlabel("time [s]");
    plt.ylabel("acceleration")
    plt.title("LQR acceleration command - speed step")

    plt.show()


def plot_eigs_discrete(Ad, Bd=None, K=None, title="Discrete-time eigenvalues"):
    """
    Plot eigenvalues in the complex plane for discrete-time systems.
    - If K is given: plots eig(Ad - Bd K) as closed-loop.
    - Always plots eig(Ad) as open-loop (dashed).
    Also draws the unit circle (stability boundary).
    """
    eig_ol = np.linalg.eigvals(Ad)

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # unit circle
    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(th), np.sin(th), "k--", linewidth=1, label="unit circle")

    # open-loop eigs
    plt.scatter(eig_ol.real, eig_ol.imag, marker="o", label="open-loop eig(Ad)")

    # closed-loop eigs
    if (Bd is not None) and (K is not None):
        Acl = Ad - Bd @ K
        eig_cl = np.linalg.eigvals(Acl)
        plt.scatter(eig_cl.real, eig_cl.imag, marker="x", s=60, label="closed-loop eig(Ad-BdK)")

    plt.axhline(0, color="0.7", linewidth=1)
    plt.axvline(0, color="0.7", linewidth=1)
    plt.grid(True)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(title)
    plt.legend()
    plt.show()





if __name__ == "__main__":
    # simple CLI without Fire to avoid extra dependency in this helper
    compute_and_export_lqr()

