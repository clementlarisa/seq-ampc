# f.py
# Single-track (Althoff/CommonRoad-style) with PT1 longitudinal accel state
# States: x = [px, py, psi, v, r, beta, a]
# Inputs: u = [delta, a_cmd]

from casadi import sin, cos, sqrt

def f(x, u):
    """
    States (7):
      x = [px, py, psi, v, r, beta, a]
        px    [m]     global x position
        py    [m]     global y position
        psi   [rad]   yaw angle
        v     [m/s]   speed magnitude
        r     [rad/s] yaw rate (psi_dot)
        beta  [rad]   slip angle at CoM
        a     [m/s^2] filtered longitudinal acceleration (state)

    Inputs:
      u = [delta, a_cmd]
        delta [rad]    steering angle (front wheel / bicycle)
        a_cmd [m/s^2]  commanded longitudinal acceleration

    Notes:
      - PT1 for accel: a_dot = (a_cmd - a)/tau_a
      - v_dot = a  (use filtered accel as speed derivative)
      - v_safe = sqrt(v^2 + eps_v^2) for regularization of 1/v and 1/v^2
    """

    # --- hardcoded parameters (as requested) ---
    g   = 9.81               # [m/s^2]
    lf  = 1.35               # [m]
    L   = 2.5657             # [m]
    lr  = L - lf             # [m]
    m   = 1679 + 82 + 95      # [kg]
    Iz  = 2458.0             # [kg*m^2]
    h   = 0.48               # [m]
    mu  = 1.0                # [-]
    C_Sf = 21.92 / 1.0489    # [-] cornering stiffness factor (front)
    C_Sr = 21.92 / 1.0489    # [-] cornering stiffness factor (rear)

    eps_v = 0.5              # [m/s] speed regularization
    tau_a = 0.25             # [s] PT1 time constant (tune as needed)

    # --- unpack states/inputs ---
    px, py, psi, v, r, beta, a = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    delta, a_cmd = u[0], u[1]

    # --- safe speed for denominators ---
    v_safe = sqrt(v*v + eps_v*eps_v)

    # --- global kinematics ---
    px_dot  = v * cos(beta + psi)
    py_dot  = v * sin(beta + psi)
    psi_dot = r

    # --- longitudinal PT1 ---
    v_dot = a
    a_dot = (a_cmd - a) / tau_a

    # --- yaw / slip dynamics (Althoff-style) ---
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

    return [px_dot, py_dot, psi_dot, v_dot, r_dot, beta_dot, a_dot]
