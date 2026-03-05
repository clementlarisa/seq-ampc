from casadi import sin, cos, tan

def f(x, u):
    """
    Kinematic bicycle (continuous time)
    x = [px, py, psi, v]
    u = [delta, a]
    """

    # wheelbase C3
    L = 2.5657   # C3 wheelbase [m] (adjust to your vehicle)

    # wheelbase 1:10
    #L = 2.7  # wheelbase [m] (adjust to your vehicle)

    px  = x[0]
    py  = x[1]
    psi = x[2]
    v   = x[3]

    delta = u[0]
    a     = u[1]

    return [
        v * cos(psi),
        v * sin(psi),
        v / L * tan(delta),
        a
    ]
