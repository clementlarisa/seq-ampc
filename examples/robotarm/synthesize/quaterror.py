import casadi as ca
import numpy as np

def quat_conjugate(q):
    """
    Conjugate (inverse for unit quaternion) of q = [w, x, y, z].
    For a unit quaternion, inv(q) = [w, -x, -y, -z].
    """
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])

def quat_multiply(q1, q2):
    """
    Hamilton product of two quaternions q1 and q2.
    q1, q2 = [w, x, y, z].
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return ca.vertcat(w, x, y, z)

def random_rotation_quaternion(max_degrees=30.0):
    """
    Generates a random rotation quaternion with a rotation magnitude 
    less than 'max_degrees' around a random axis.
    """
    # Pick a random unit axis
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    
    # Pick a random angle (in radians) up to the given maximum
    angle = np.random.uniform(0, np.radians(max_degrees))
    half_angle = angle / 2
    
    # Convert axis-angle to quaternion [w, x, y, z]
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return ca.DM([w, xyz[0], xyz[1], xyz[2]])

def quat_log(q, eps=1e-12):
    """
    Map a unit quaternion q into its Lie algebra (R^3) via the SO(3) log.

    If q = [w, x, y, z] corresponds to a rotation of angle theta around axis u,
    then log(q) = theta * u in R^3.

    For q close to the identity, we use a small-epsilon approximation to avoid
    numerical issues with arccos and division by sin(theta/2).
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Clamp w to avoid numerical issues outside [-1, 1].
    w_clamp = ca.fmax(ca.fmin(w, 1.0), -1.0)
    
    theta = 2.0 * ca.acos(w_clamp)  # rotation angle in [0, 2π], but effectively [0, π] for unit quaternions

    # sin(theta/2) = sqrt(1 - w^2). If angle=0 => q is identity => log(q)=0.
    sin_half = ca.sqrt(ca.fmax(1.0 - w_clamp*w_clamp, 0.0))

    # We build the axis * angle in a safe, piecewise way
    # If sin_half < eps => use a series expansion ~ 2*(x, y, z).
    small_angle = (sin_half < eps)
    
    # The "normal" case: log(q) = theta/sin(theta/2) * (x, y, z)
    # But we implement it with if_else for numerical smoothness
    log_normal = (theta / sin_half) * ca.vertcat(x, y, z)
    # log_small  = ca.vertcat(x, y, z) * (2.0 / (1e-12 + ca.sqrt(x*x + y*y + z*z)))  \ 
    #  if_else_for_axis = ca.if_else(small_angle, 0 * log_normal, log_normal)
    # However, a simpler approach near the identity is just 0 * ...
    # or a first-order expansion: log(q) ~ 2*(x, y, z) when angle ~ 0.
    log_small_approx = 2.0 * ca.vertcat(x, y, z)
    
    # We combine them carefully:
    log_q = ca.if_else(small_angle, log_small_approx, log_normal)

    return log_q

def quaternion_error_squared(quat, quat_ref):
    """
    Returns the scalar error || log(inv(quat) * quat_ref) ||, i.e. the
    geodesic distance on SO(3) between the orientations represented by
    'quat' and 'quat_ref'.

    Both are assumed to be unit quaternions.
    """
    # inv(quat) = conj(quat) for a unit quaternion
    q_diff = quat_multiply(quat_conjugate(quat), quat_ref)
    # log on SO(3) -> 3-vector
    log_q = quat_log(q_diff)
    # Norm of that 3-vector is the rotation angle in [0, π]
    return ca.sum1(log_q * log_q)

def build_quaternion_error_function():
    """
    Build a CasADi function quat_error(quat, quat_ref) -> scalar error.
    """
    quat_sym = ca.SX.sym("quat", 4)
    quat_ref_sym = ca.SX.sym("quat_ref", 4)
    err_expr = quaternion_error_squared(quat_sym, quat_ref_sym)
    return ca.Function("quat_error", [quat_sym, quat_ref_sym], [err_expr])

if __name__ == "__main__":
    # Example usage
    quat_error = build_quaternion_error_function()

    import numpy as np

    # Identity quaternion
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])

    # 90-deg rotation about z-axis -> angle = pi/2
    # unit quaternion: w=cos(pi/4), x=0, y=0, z=sin(pi/4)
    q_90z = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
    
    # Evaluate the error between identity and 90° about z
    err_val = quat_error(q_identity, q_90z).full()
    print(f"Quaternion error between identity and 90° Z-rotation: {np.sqrt(err_val.item())} rad")

    # For reference, we expect ~ pi/2 = 1.570796...