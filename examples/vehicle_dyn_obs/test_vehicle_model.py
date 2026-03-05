import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0) Utility: nice printing
# ============================================================
def print_kv(label, value, unit="", width=40, fmt="{:,.3f}"):
    """Helper for aligned console output."""
    if unit:
        print(f"{label:{width}} = {fmt.format(value)} {unit}")
    else:
        print(f"{label:{width}} = {fmt.format(value)}")


# ============================================================
# 1) Vehicle + tire parameters (centralized)
# ============================================================
def get_params():
    """
    Collect all parameters in one place so they are consistent across:
      - dynamics model
      - stiffness conversions
      - steady-circle analytic evaluation
    """
    p = {}
    p["g"] = 9.81               # [m/s^2]
    p["lf"] = 1.35              # [m] CoM -> front axle
    p["L"]  = 2.5657            # [m] wheelbase
    p["lr"] = p["L"] - p["lf"]  # [m] CoM -> rear axle
    p["m"]  = 1679 + 82 + 95    # [kg] total mass
    p["Iz"] = 2458.0            # [kg*m^2] yaw inertia
    p["h"]  = 0.48              # [m] CoM height (for longitudinal load transfer term)
    p["mu"] = 1.0               # [-] friction scaling in this formulation
    p["C_Sf"] = 21.92 / 1.0489  # [-] cornering stiffness factor (Althoff/CommonRoad style)
    p["C_Sr"] = 21.92 / 1.0489  # [-]
    p["eps_v"] = 0.5            # [m/s] speed regularization for 1/v terms
    p["tau_a"] = 0.25           # [s] PT1 time constant for acceleration
    return p


# ============================================================
# 2) Cornering stiffness conversions
# ============================================================
def cornering_stiffness_no_load_transfer(C_Sf, C_Sr, mu, m, g, lf, lr):
    """
    Convert stiffness factors C_S* (dimensionless) into Ca [N/rad]
    using static normal loads (ax = 0).
      Cf = mu * C_Sf * Fzf
      Cr = mu * C_Sr * Fzr
    """
    L = lf + lr
    Fzf = m * g * lr / L
    Fzr = m * g * lf / L
    Cf = mu * C_Sf * Fzf
    Cr = mu * C_Sr * Fzr
    return Cf, Cr, Fzf, Fzr


def cornering_stiffness_with_load_transfer(C_Sf, C_Sr, mu, m, g, lf, lr, h, ax):
    """
    Same conversion but with longitudinal load transfer (simple model):
      Fzf = (m*g*lr - m*ax*h)/L
      Fzr = (m*g*lf + m*ax*h)/L
    """
    L = lf + lr
    Fzf = (m * g * lr - m * ax * h) / L
    Fzr = (m * g * lf + m * ax * h) / L
    Cf = mu * C_Sf * Fzf
    Cr = mu * C_Sr * Fzr
    return Cf, Cr, Fzf, Fzr


# ============================================================
# 3) Dynamics model (7 states with PT1 for acceleration)
# ============================================================
def f_numpy(x, u, p):
    """
    States:
      x = [px, py, psi, v, r, beta, a]
        px    [m]     global x position
        py    [m]     global y position
        psi   [rad]   yaw angle
        v     [m/s]   speed magnitude (>=0)
        r     [rad/s] yaw rate
        beta  [rad]   slip angle at CoM
        a     [m/s^2] filtered longitudinal acceleration (state)

    Inputs:
      u = [delta, a_cmd]
        delta [rad]    steering angle
        a_cmd [m/s^2]  commanded longitudinal acceleration (filtered by PT1)

    Notes:
      - v_safe = sqrt(v^2 + eps_v^2) prevents singularities in 1/v and 1/v^2.
      - a follows a_cmd with a first-order lag (PT1): a_dot = (a_cmd - a)/tau_a.
      - The dynamic single-track (Althoff/CommonRoad) form uses a (filtered) in the
        load-transfer-like terms (g*lr - a*h), (g*lf + a*h).
    """
    g   = p["g"]
    lf  = p["lf"]
    lr  = p["lr"]
    L   = p["L"]
    m   = p["m"]
    Iz  = p["Iz"]
    h   = p["h"]
    mu  = p["mu"]
    C_Sf = p["C_Sf"]
    C_Sr = p["C_Sr"]
    eps_v = p["eps_v"]
    tau_a = p["tau_a"]

    px, py, psi, v, r, beta, a = x
    delta, a_cmd = u

    # speed regularization for denominators
    v_safe = np.sqrt(v*v + eps_v*eps_v)

    # --- global kinematics ---
    # Velocity vector in global frame uses heading (psi) and slip (beta)
    px_dot  = v * np.cos(beta + psi)
    py_dot  = v * np.sin(beta + psi)
    psi_dot = r

    # --- longitudinal (PT1 filtered) ---
    v_dot = a
    a_dot = (a_cmd - a) / tau_a

    # --- yaw + slip dynamics (Althoff-style) ---
    # All occurrences of "ax" replaced by filtered "a"
    r_dot = (
        -mu * m / (v_safe * Iz * L)
        * (lf**2 * C_Sf * (g*lr - a*h) + lr**2 * C_Sr * (g*lf + a*h)) * r
        + mu * m / (Iz * L)
        * (lr * C_Sr * (g*lf + a*h) - lf * C_Sf * (g*lr - a*h)) * beta
        + mu * m / (Iz * L)
        * (lf * C_Sf * (g*lr - a*h)) * delta
    )

    beta_dot = (
        (mu / (v_safe*v_safe * L)
            * (C_Sr * (g*lf + a*h) * lr - C_Sf * (g*lr - a*h) * lf) - 1.0) * r
        - mu / (v_safe * L)
            * (C_Sr * (g*lf + a*h) + C_Sf * (g*lr - a*h)) * beta
        + mu / (v_safe * L)
            * (C_Sf * (g*lr - a*h)) * delta
    )

    return np.array([px_dot, py_dot, psi_dot, v_dot, r_dot, beta_dot, a_dot], dtype=float)


# ============================================================
# 4) Maneuvers (u = [delta, a_cmd])
# ============================================================
def maneuver_overtake_bump(t):
    """
    Lane change with 2 pulses (not plateau).
    """
    delta = np.zeros_like(t)
    a_cmd = np.zeros_like(t)

    # accel schedule
    a_cmd[(t >= 0.0) & (t < 1.0)] = +2.0
    a_cmd[(t >= 1.0) & (t < 6.5)] = 0.0
    a_cmd[(t >= 6.5) & (t <= t[-1])] = -2.0

    # steering pulses
    d1 = 4.0 * np.pi / 180.0
    d2 = 1.15 * d1
    T1 = 0.55
    Tgap = 0.25
    T2 = 0.65

    t1s = 2.0
    t1e = t1s + T1
    t2s = t1e + Tgap
    t2e = t2s + T2

    delta[(t >= t1s) & (t < t1e)] = +d1
    delta[(t >= t2s) & (t < t2e)] = -d2

    return np.vstack([delta, a_cmd]).T


def maneuver_overtake_plateau(t):
    """
    Overtake with plateau (two lane changes):
      S-left (+ then -), hold, S-right (- then +)
    """
    delta = np.zeros_like(t)
    a_cmd = np.zeros_like(t)

    a_cmd[(t >= 0.0) & (t < 1.0)] = +2.0
    a_cmd[(t >= 1.0) & (t < 6.5)] = 0.0
    a_cmd[(t >= 6.5) & (t <= t[-1])] = -2.0

    d = 4.0 * np.pi / 180.0
    T = 0.45
    G = 0.10
    hold = 1.00

    # left lane-change S (+ then -)
    t1 = 2.0
    delta[(t >= t1) & (t < t1 + T)] = +d
    delta[(t >= t1 + T + G) & (t < t1 + 2*T + G)] = -d

    # hold on plateau
    t2 = t1 + 2*T + G + hold

    # right lane-change S (- then +)
    delta[(t >= t2) & (t < t2 + T)] = -d
    delta[(t >= t2 + T + G) & (t < t2 + 2*T + G)] = +d

    return np.vstack([delta, a_cmd]).T


def maneuver_spiral_tightening(t):
    """
    Spiral: delta ramps up, a_cmd = 0.
    """
    delta = np.zeros_like(t)
    a_cmd = np.zeros_like(t)

    # keep speed roughly constant
    a_cmd[:] = 0.0

    delta_max = 10.0 * np.pi / 180.0
    t0 = 0.5
    t1 = t[-1] * 0.9

    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)  # smoothstep
    delta[:] = delta_max * s

    return np.vstack([delta, a_cmd]).T


def maneuver_steady_circle(t, p, R=50.0, v_target=10.0, t_ramp=3.0):
    """
    Steady circle:
      - delta ramps to delta_ss ~ atan(L/R) (kinematic approx)
      - a_cmd is 0 (assume you start at v_target), or optionally ramp v with a_cmd.

    IMPORTANT:
      - Because this is dynamic (slip), achieved radius differs from R.
      - For meaningful circle evaluation: set initial v close to v_target.
    """
    delta = np.zeros_like(t)
    a_cmd = np.zeros_like(t)

    L = p["L"]
    delta_ss = np.arctan(L / R)

    # smooth delta ramp
    t0 = 0.5
    t1 = t_ramp
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    delta[:] = delta_ss * s

    # Keep v approximately constant by commanding 0 accel (start at v_target)
    a_cmd[:] = 0.0

    return np.vstack([delta, a_cmd]).T


# ============================================================
# 5) Simulation
# ============================================================
def simulate(maneuver_fn, dt, T, x0, p):
    """
    Simple forward simulation with explicit Euler (fast + clear).
    """
    N = int(T / dt) + 1
    t = np.linspace(0.0, T, N)

    x = np.zeros((N, 7))
    x[0, :] = x0

    u = maneuver_fn(t)

    for k in range(N - 1):
        xdot = f_numpy(x[k, :], u[k, :], p)
        x[k + 1, :] = x[k, :] + dt * xdot

    return t, x, u


# ============================================================
# 6) Plotting (multi-subplot, twin axis for inputs)
# ============================================================
def plot_run(t, x, u, title=""):
    px, py, psi, v, r, beta, a = x.T
    delta, a_cmd = u.T

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(title, fontsize=12)

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(px, py)
    ax1.set_xlabel("px [m]")
    ax1.set_ylabel("py [m]")
    ax1.grid(True)
    ax1.set_title("Trajectory (x-y)")

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t, psi)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("psi [rad]")
    ax2.grid(True)
    ax2.set_title("Yaw angle")

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(t, v)
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("v [m/s]")
    ax3.grid(True)
    ax3.set_title("Speed")

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(t, r)
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("r [rad/s]")
    ax4.grid(True)
    ax4.set_title("Yaw rate")

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(t, beta)
    ax5.set_xlabel("t [s]")
    ax5.set_ylabel("beta [rad]")
    ax5.grid(True)
    ax5.set_title("Slip angle")

    # inputs + accel state
    ax6 = fig.add_subplot(3, 2, 6)
    l1, = ax6.plot(t, delta, label="delta [rad]", color="tab:blue")
    ax6.set_xlabel("t [s]")
    ax6.set_ylabel("delta [rad]", color="tab:blue")
    ax6.tick_params(axis="y", labelcolor="tab:blue")
    ax6.grid(True)
    ax6.set_title("Inputs / accel")

    ax6b = ax6.twinx()
    l2, = ax6b.plot(t, a_cmd, label="a_cmd [m/s^2]", color="tab:orange")
    l3, = ax6b.plot(t, a, label="a (state) [m/s^2]", color="tab:green", linestyle="--")
    ax6b.set_ylabel("accel [m/s^2]", color="tab:orange")
    ax6b.tick_params(axis="y", labelcolor="tab:orange")

    ax6.legend([l1, l2, l3],
               ["delta [rad]", "a_cmd [m/s^2]", "a (state) [m/s^2]"],
               loc="best")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ============================================================
# 7) Plausibility checks (generic)
# ============================================================
def plausibility_checks(t, x, u, mode_name=""):
    px, py, psi, v, r, beta, a = x.T
    delta, a_cmd = u.T

    print(f"\n=== Plausibility checks {('(' + mode_name + ')') if mode_name else ''} ===")

    if np.all(np.isfinite(x)) and np.all(np.isfinite(u)):
        print("[OK] Finite values (no NaN/Inf).")
    else:
        print("[WARN] NaN/Inf detected!")

    v_min, v_max = float(np.min(v)), float(np.max(v))
    print_kv("v range", v_min, "m/s", fmt="{:,.3f}")
    print_kv("v range (max)", v_max, "m/s", fmt="{:,.3f}")
    if v_min < -1e-6:
        print("[WARN] v became negative (v is treated as magnitude here).")

    # psi_dot consistency with r
    dpsi = np.diff(psi) / np.diff(t)
    err = dpsi - r[:-1]
    rmse = float(np.sqrt(np.mean(err**2)))
    print_kv("RMSE(dpsi/dt - r)", rmse, "rad/s", fmt="{:,.3e}")

    # slip range
    beta_abs_max = float(np.max(np.abs(beta)))
    print_kv("max |beta|", beta_abs_max, "rad", fmt="{:,.4f}")

    # PT1 lag
    lag_metric = float(np.mean(np.abs(a_cmd - a)))
    print_kv("mean |a_cmd - a| (PT1 lag)", lag_metric, "m/s^2", fmt="{:,.3f}")

    print("=== End checks ===\n")


# ============================================================
# 8) Circle evaluation from simulation
#    (estimate achieved radius from curvature, and do analytic "like your snippet")
# ============================================================
def estimate_radius_from_trajectory(px, py, psi, beta, v, eps=1e-6):
    """
    Estimate radius using curvature:
      curvature kappa = d(theta_path)/ds
    where theta_path = psi + beta is direction of velocity vector.

    We compute:
      dtheta/dt = d(psi+beta)/dt
      ds/dt = v
      => kappa = (dtheta/dt) / v
      => R = 1 / kappa

    Returns:
      R_inst: instantaneous radius over time (length N-1)
      R_med: robust median radius (ignoring huge values)
    """
    theta = psi + beta
    dtheta = np.diff(theta)
    dv = (v[:-1] + v[1:]) * 0.5
    kappa = dtheta / np.maximum(dv, eps)  # curvature
    # avoid division by zero: if kappa ~0 => radius huge
    R_inst = 1.0 / np.where(np.abs(kappa) > eps, kappa, np.nan)

    # robust median on finite entries
    finite = np.isfinite(R_inst)
    R_med = float(np.nanmedian(np.abs(R_inst[finite]))) if np.any(finite) else np.nan
    return R_inst, R_med


def circle_analytic_report(p, R, V=20.0):
    """
    Produce a printout similar to your snippet, using:
      ay = V^2 / R
      Fyf, Fyr split by axle mass (static distribution)
      alpha_f = Fyf / Cf
      alpha_r = Fyr / Cr

    This is a *linear steady-state back-of-envelope* consistency check.
    """
    g = p["g"]
    lf = p["lf"]
    lr = p["lr"]
    L  = p["L"]
    m  = p["m"]
    mu = p["mu"]
    C_Sf = p["C_Sf"]
    C_Sr = p["C_Sr"]

    # axle masses from static load distribution
    mf = m * lr / L  # [kg] front axle equivalent mass
    mr = m * lf / L  # [kg] rear axle equivalent mass

    # Ca [N/rad] from factors
    Cf, Cr, Fzf, Fzr = cornering_stiffness_no_load_transfer(C_Sf, C_Sr, mu, m, g, lf, lr)

    # lateral acceleration + forces
    ay = V**2 / R
    Fyf = mf * ay
    Fyr = mr * ay

    # slip angles (linear)
    alpha_f = Fyf / Cf
    alpha_r = Fyr / Cr

    print()
    print(f"--- Analytic steady-circle check (linear) for R={R:.1f} m, V={V:.1f} m/s ---")
    print_kv("Lateral acceleration", ay, "m/s^2", fmt="{:,.2f}")
    print_kv("Mass on the front axle (mf)", mf, "kg", fmt="{:,.1f}")
    print_kv("Mass on the rear axle (mr)", mr, "kg", fmt="{:,.1f}")
    print_kv("Static normal load front (Fzf)", Fzf, "N", fmt="{:,.0f}")
    print_kv("Static normal load rear (Fzr)", Fzr, "N", fmt="{:,.0f}")
    print_kv("Cornering stiffness front (Cf)", Cf, "N/rad", fmt="{:,.0f}")
    print_kv("Cornering stiffness rear (Cr)", Cr, "N/rad", fmt="{:,.0f}")
    print_kv("Lateral force front (Fyf)", Fyf, "N", fmt="{:,.0f}")
    print_kv("Lateral force rear (Fyr)", Fyr, "N", fmt="{:,.0f}")
    print_kv("Slip angle front (alpha_f)", alpha_f, "rad", fmt="{:,.4f}")
    print_kv("Slip angle rear (alpha_r)", alpha_r, "rad", fmt="{:,.4f}")
    print("--- End analytic check ---\n")


def circle_sim_report(t, x, R_target, label=""):
    """
    Evaluate achieved radius from simulated trajectory and print a report.
    Uses:
      - median of instantaneous radius from curvature (psi+beta, v)
      - also reports average speed and average yaw rate magnitude
    """
    px, py, psi, v, r, beta, a = x.T

    # ignore initial transient (first 30% of time) for steady assessment
    k0 = int(0.30 * len(t))
    R_inst, R_med = estimate_radius_from_trajectory(px[k0:], py[k0:], psi[k0:], beta[k0:], v[k0:])

    v_mean = float(np.mean(v[k0:]))
    r_mean = float(np.mean(np.abs(r[k0:])))

    print(f"\n--- Simulation circle report {label} ---")
    print_kv("Target radius", R_target, "m", fmt="{:,.2f}")
    print_kv("Estimated radius (median |R|)", R_med, "m", fmt="{:,.2f}")
    if np.isfinite(R_med):
        print_kv("Radius error (est - target)", R_med - R_target, "m", fmt="{:,.2f}")
    print_kv("Mean speed (steady window)", v_mean, "m/s", fmt="{:,.2f}")
    print_kv("Mean |yaw rate| (steady window)", r_mean, "rad/s", fmt="{:,.3f}")
    print("--- End simulation circle report ---\n")


# ============================================================
# 9) Main: run all maneuvers + evaluations
# ============================================================
if __name__ == "__main__":
    p = get_params()

    # -------------------------
    # Overtake bump
    # -------------------------
    dt = 0.01
    t, x, u = simulate(
        maneuver_fn=maneuver_overtake_bump,
        dt=dt, T=8.0,
        x0=np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        p=p
    )
    plot_run(t, x, u, title="Overtake / Lane Change (bump) with PT1 accel state")
    plausibility_checks(t, x, u, mode_name="Overtake bump")

    # -------------------------
    # Spiral tightening
    # -------------------------
    t, x, u = simulate(
        maneuver_fn=maneuver_spiral_tightening,
        dt=dt, T=12.0,
        x0=np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        p=p
    )
    plot_run(t, x, u, title="Tightening Spiral with PT1 accel state")
    plausibility_checks(t, x, u, mode_name="Spiral")

    # -------------------------
    # Overtake plateau (two lane changes)
    # -------------------------
    t, x, u = simulate(
        maneuver_fn=maneuver_overtake_plateau,
        dt=dt, T=8.0,
        x0=np.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0]),
        p=p
    )
    plot_run(t, x, u, title="Overtake / Lane Change (plateau) with PT1 accel state")
    plausibility_checks(t, x, u, mode_name="Overtake plateau")

    # -------------------------
    # Steady circles: R=50m and R=100m
    # -------------------------
    # For circles, start directly at target speed and set a_cmd=0.
    # Increase horizon so you get a clear steady segment.
    for R in [50.0, 100.0]:
        T_circle = 30.0
        v_target = 20.0

        # create a maneuver closure that has access to params p
        circle_fn = lambda tt, R=R: maneuver_steady_circle(tt, p, R=R, v_target=v_target, t_ramp=4.0)

        t, x, u = simulate(
            maneuver_fn=circle_fn,
            dt=dt, T=T_circle,
            x0=np.array([0.0, 0.0, 0.0, v_target, 0.0, 0.0, 0.0]),
            p=p
        )
        plot_run(t, x, u, title=f"Steady Circle (target R={R:.0f} m, v≈{v_target:.0f} m/s)")
        plausibility_checks(t, x, u, mode_name=f"Circle R={R:.0f}")

        # Analytic report like your snippet
        circle_analytic_report(p, R=R, V=v_target)

        # Simulation radius estimate report
        circle_sim_report(t, x, R_target=R, label=f"(R={R:.0f} m)")

    # -------------------------
    # Print static cornering stiffness
    # -------------------------
    Cf, Cr, Fzf, Fzr = cornering_stiffness_no_load_transfer(
        p["C_Sf"], p["C_Sr"], p["mu"], p["m"], p["g"], p["lf"], p["lr"]
    )
    print("--- Static cornering stiffness (from factors) ---")
    print_kv("Cf", Cf, "N/rad", fmt="{:.3f}")
    print_kv("Cr", Cr, "N/rad", fmt="{:.3f}")
    print("--- End ---")
