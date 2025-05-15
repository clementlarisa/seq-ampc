import numpy as np
import sympy
import cvxpy as cp
from itertools import product
from math import sqrt, exp
from scipy.linalg import solve_continuous_lyapunov, expm
# If you need to do discrete-time conversions, you can also use:
# from scipy.signal import cont2discrete

from mpc import get_robot_fk
import casadi as cs
###############################################################################
# 1) SYMBOLIC DEFINITIONS
###############################################################################
forward_kineamtics_fcn, limits = get_robot_fk()
q_symbolic = cs.SX.sym("q", 7)
dq_symbolic = cs.SX.sym("dq", 7)
q_dq_symbolic = cs.vertcat(q_symbolic,dq_symbolic)
postion_symbolic, orientation_symbolic = forward_kineamtics_fcn(q_symbolic)
jacobian_forward_kinematics_fcn = cs.Function(
    "jacobian_forward_kinematics",
    [q_symbolic, dq_symbolic],
    [cs.jacobian(postion_symbolic, q_dq_symbolic), cs.jacobian(orientation_symbolic, q_dq_symbolic)]    
)

limits['q_center'] = (np.array(limits["q_max"])-np.array(limits["q_min"]))/2+np.array(limits["q_min"])
limits['q'] = (np.array(limits["q_max"])-np.array(limits["q_min"]))/2

def y_func(q_dq):
    q = q_dq[:7]
    position, orientation = forward_kineamtics_fcn(q-limits['q_center'])
    return position

def dy_func(q_dq):
    q = q_dq[:7]
    position, orientation = jacobian_forward_kinematics_fcn(q-limits['q_center'], q_dq[7:])
    return position

###############################################################################
# 2) SETUP SYSTEM MATRICES AND CONSTRAINTS
###############################################################################
rho_c   = 10.0  # Exponential rate to enforce in the LMI
c_max   = 1.0
w_c_max = 6.0
T       = 0.12
N       = 3
d_w_max = 1.0
d_w_max_model = 0.25
C_par   = 2
C_obst  = 2
consider_obstacles = False

# Disturbance polyhedron: w in [-d_w_max, d_w_max]^4
# We only incorporate it in E * V', but effectively the set is {0,0,0,0} x [-d_w_max, d_w_max]^4.
# But the code you had uses E to pick the velocity part. We'll just build corners for the 4-d "w" directly.
import itertools
w_corners = []
for wvals in itertools.product([-d_w_max, d_w_max], repeat=7):
    w_corners.append(np.array(wvals).reshape(-1,1))
vertices_dist = w_corners


# Construct Lx (28x14) for state constraints
Lx = []
for i in range(7):  # x1 to x7
    Lx.append(np.eye(14)[i] / limits['q'][i])  # x_i <= q_max_i
    Lx.append(np.eye(14)[i] / -limits['q'][i])  # x_i >= q_min_i
# For x8 (index 7), using dq[3]
for i in range(7):
    Lx.append(np.eye(14)[i+7] / limits["dq"][i])   # x8 <= dq[3]
    Lx.append(np.eye(14)[i+7] / -limits["dq"][i])  # x8 >= -dq[3]
Lx = np.array(Lx)

# Construct Lu (14x7) for control (acceleration) constraints
Lu = []
for i in range(7):  # u1 to u4
    Lu.append(np.eye(7)[i] / limits["ddq"][i])
    Lu.append(np.eye(7)[i] / -limits["ddq"][i])
Lu = np.array(Lu)


# Continuous-time linear system
A_c = np.zeros((14,14))
A_c[0:7,7:14] = np.eye(7)
B_c = np.zeros((14,7))
B_c[7:14,:]   = np.eye(7)

###############################################################################
# 3) FORMULATE LMIs IN CVXPY
###############################################################################
X_0 = cp.Variable((14,14), symmetric=True)
Y_0 = cp.Variable((7,14))

# We want X_0 to be positive definite. In practice we do X_0 >> 0:
# constraints = [X_0 >> 0]
constraints = []

# LMI 1: A*X + B*Y + sym + 2*rho_c*X <= 0
#   => (A_c * X_0 + B_c * Y_0) + its transpose + 2*rho_c * X_0 <= 0
term = A_c @ X_0 + B_c @ Y_0
constraints += [ term + term.T + 2*rho_c*X_0 << 0 ]

# LMI 2: for j in 1..len(Lx): 
#   [ c_max^2,      Lx_j * X ]
#   [ (Lx_j * X)^T,     X    ] >= 0
# Because this is a block (1+8) x (1+8) matrix
for j in range(Lx.shape[0]):
    Lxj = Lx[j,:].reshape(1,14)  # 1 x 8
    top_right = Lxj @ X_0       # (1 x 8)
    blk = cp.bmat([
        [ [[c_max**2]],     top_right ],
        [ top_right.T,      X_0       ]
    ])
    constraints += [ blk >> 0 ]

# LMI 3: for j in 1..size(Lu,1):
#   [ c_max^2,    Lu_j * Y ]
#   [ (Lu_j * Y)^T,   X    ] >= 0
for j in range(Lu.shape[0]):
    Luj = Lu[j,:].reshape(1,7)    # 1 x 4
    top_right = Luj @ Y_0         # (1 x 8)
    blk = cp.bmat([
        [ [[c_max**2]],   top_right ],
        [ top_right.T,    X_0       ]
    ])
    constraints += [ blk >> 0 ]

# LMI 4: Disturbance constraint: for i in range(# corners)
#  [ X,               w_i ]
#  [ w_i^T,    w_c_max^2 ]
for wv in vertices_dist:
    # wv is 4x1, but the code lumps it with an 8x8 X.  
    # Actually in the MATLAB code, the dimension is 8 in the top-left block
    # and wv is (8 or 4?). They used E*(poly_dist.V)', effectively
    # wv is dimension 4, so the block is 8+1 x 8+1 if we embed wv in R^8 somehow.
    # Original code does:
    #  [X, vertices_dist(:,i)]
    #  [vertices_dist(:,i).T, w_c_max^2]
    # means the w_i is 8x1 in the code. But here, your E picks out the last 4 states of dimension 8.
    # If you want to replicate that, do E = [zeros(4,4); I_4].
    # But that is 8x4 => E * wv(4x1) = 8x1.
    # Let's do the same:
    E = np.vstack([np.zeros((7,7)), np.eye(7)])  # 8 x 4
    wv_8 = E @ wv  # 8x1
    blk = cp.bmat([
        [ X_0,              wv_8 ],
        [ wv_8.T, [[w_c_max**2]] ]
    ])
    constraints += [ blk >> 0 ]

# Objective is -log(det(X_0)) in MATLAB => maximize log_det(X_0) in Python.
# obj = cp.Maximize(cp.log_det(X_0))
obj = cp.Maximize(cp.trace(X_0))

prob = cp.Problem(obj, constraints)
print("Starting optimization...")
result = prob.solve(solver=cp.MOSEK, verbose=False)  # or another solver you have
# Check feasibility
if prob.status not in ["optimal", "optimal_inaccurate"]:
    raise ValueError("No suitable solution found by the solver.")

X_0_val = X_0.value
Y_0_val = Y_0.value

# Compute final P_delta and K_delta
P_delta = np.linalg.inv(X_0_val)
K_delta = Y_0_val @ P_delta

print("P_delta = ")
print(P_delta)
print("K_delta = ")
print(K_delta)

# Quick check on magnitude of eigenvalues of P_delta:
if np.max(np.abs(np.linalg.eigvals(P_delta))) > 1e5:
    raise ValueError("No suitable solution (P_delta ill-conditioned).")

###############################################################################
# 4) COMPUTE c_j-VALUES
###############################################################################
# We define a helper to invert the sqrt of P_delta
# i.e. P_delta^(-1/2) for norms:
evals, evecs = np.linalg.eig(P_delta)
if min(evals) <= 0:
    raise ValueError("P_delta is not positive definite as expected.")
D_half_inv = np.diag(1.0/np.sqrt(evals))
P_delta_mhalf = evecs @ D_half_inv @ np.linalg.inv(evecs)

# -> con_x: for each row in Lx, norm(P_delta^(-1/2)*Lx_j^T)
con_x = []
for j in range(Lx.shape[0]):
    vec = Lx[j,:].reshape((14)).T  # 8x1
    # P_delta^(-1/2) * vec => 8x1
    norm_val = np.linalg.norm(P_delta_mhalf @ vec, 2)
    con_x.append(norm_val)
con_x = np.array(con_x)

# -> con_u: for each row in Lu, norm(P_delta^(-1/2)*K_delta^T*Lu_j^T)
con_u = []
for j in range(Lu.shape[0]):
    vec = Lu[j,:].reshape((1,7)).T  # 4x1
    # K_delta^T has shape 8x4 => K_delta^T * vec => 8x1
    kv = (K_delta.T @ vec)
    norm_val = np.linalg.norm(P_delta_mhalf @ kv, 2)
    con_u.append(norm_val)
con_u = np.array(con_u)

# Now we compute the partial derivatives of the constraints for the outputs:
def Pdelta_mhalf_norm(dg):
    """Compute || P_delta^(-1/2) * dg ||_2."""
    return np.linalg.norm(P_delta_mhalf @ dg.reshape((-1,1)), 2)

# def Pdelta_mhalf_norm(dg_4):
#     """
#     dg_4 is shape (4,) containing partial derivatives wrt q1..q4 only.
#     We embed into an 8D vector, because P_delta_mhalf is 8x8.
#     """
#     # Create an 8-dimensional vector with zeros in the velocity part
#     dg_8 = np.zeros((8,))
#     dg_8[:4] = dg_4  # put the 4-element gradient in the first half

#     # Now multiply by the 8x8 matrix
#     return np.linalg.norm(P_delta_mhalf @ dg_8.reshape(-1,1), 2)

cj1 = 0.0
cj2 = 0.0
cj3 = 0.0

# Ranges for x1..x4
num_per_dim = 3
x_vals = [np.linspace(-q, q, num_per_dim) for q in limits['q']]

counter = 0
for x_tuple in product(*x_vals):
    x = np.array(x_tuple + (0.,) * 7)
    # Some obstacle sets
    for x_obst in [0.2]:
        for y_obst in [0.0]:
            for z_obst in [1.0]:
                # Evaluate y, dy at these points
                y_here  = np.array(y_func(x), dtype=float).flatten()
                dy_here = np.array(dy_func(x), dtype=float)
                
                ### Below code is what Julian did
                # dy_here is 4x4, but only the first 3 of y are relevant? Actually y_here has length 4.

                # dg1 = -dy(3,:) - 2*y(2)*dy(2,:)
                # Be mindful that in Python indexing is 0-based:
                # y_here(2) means the third element, etc.
                # The code uses y_here(2) etc. We'll replicate exactly:
                #  => in code: y_here(2) is the second index in 1-based, so that's y_here[1] in 0-based.
                # However, from your symbolic, y_here has length 4. The '3rd' is y_here[2].
                
                # MATLAB code: "dg1 = -dy_here(3,:) - 2*y_here(2)*dy_here(2,:)"
                # They used indexing: (3,:) => row=3 => Python is row=2
                # and y_here(2) => python y_here[1].
                # We need to be absolutely sure the indexing matches the original meaning:
                # y_here(3) is the 'z' coordinate, i.e. index=2 in python. 
                # y_here(2) is the 'y' coordinate, index=1 in python.
                
                # # So let's do exactly that:
                # dg1_4 = -dy_here[2,:] - 2.0*y_here[1]*dy_here[1,:]  # shape (4,)
                # c = Pdelta_mhalf_norm(dg1_4)
                # cj1 = max(cj1, c)

                # if consider_obstacles:
                #     # dg2 = dy(3,:) - 2*C_par*(y(1)-x_obst)*dy(1,:) - ...
                #     dg2 = (dy_here[2,:]
                #            - 2*C_par*(y_here[0]-x_obst)*dy_here[0,:]
                #            - 2*C_par*(y_here[1]-y_obst)*dy_here[1,:])
                #     c2 = Pdelta_mhalf_norm(dg2)
                #     cj2 = max(cj2, c2)

                #     dg3 = (-dy_here[0,:]
                #            - 2*C_obst*(y_here[1]-y_obst)*dy_here[1,:]
                #            - 2*C_obst*(y_here[2]-z_obst)*dy_here[2,:])
                #     c3 = Pdelta_mhalf_norm(dg3)
                #     cj3 = max(cj3, c3)
                
                ### We want to use interpolated SDFs instead, so what we need to do is 
                # dg/dx = dSDF/dp dp/dx
                # max gradient dSDF/dp of any SDF is 1, therefore max of |dg/dx| <= |dFK/dx|
                c = Pdelta_mhalf_norm(dy_here[0,:])
                cj1 = max(cj1, c)
                c = Pdelta_mhalf_norm(dy_here[1,:])
                cj2 = max(cj2, c)
                c = Pdelta_mhalf_norm(dy_here[2,:])
                cj3 = max(cj3, c)
                counter += 1

con_g = np.array([cj1, cj2, cj3])

print("Max constraints in x, u, g (c_j values):")
print("con_x =", con_x)
print("con_u =", con_u)
print("con_g =", con_g)
c_xmax = np.max(con_x)
c_umax = np.max(con_u)
c_gmax = np.max(con_g)
c_max_final = max(c_xmax, c_umax)
print(f"Overall c_max from x- and u- constraints = {c_xmax}, {c_umax}.")
print(f"Maximum among all g => {c_gmax}.")
print("=> c_max = ", c_max_final)

###############################################################################
# 5) COMPUte LESS CONSERVATIVE RHO, ETC.
###############################################################################
# Check the continuous-time closed-loop matrix
A_cl = A_c + B_c @ K_delta

# We want the largest real part of
# P_delta^(-1/2)*(A_cl^T + A_cl)*P_delta^(1/2). Then divided by 2 with a negative sign
# This is an approximation of the same approach in the code:
#   -max(eig( P_delta^(-1/2)*(A+B*K_delta)'*P_delta^(1/2) + ... ))/2
mat_test = (P_delta_mhalf @ A_cl.T @ np.linalg.inv(P_delta_mhalf)
            + np.linalg.inv(P_delta_mhalf) @ A_cl @ P_delta_mhalf)
eigs_test = np.linalg.eigvals(mat_test)
rho_c_practice = -0.5 * np.max(eigs_test.real)
print("rho_c_practice =", rho_c_practice)

# Check if the LMI was indeed satisfied: 
if np.max(eigs_test.real) > -2*rho_c + 1e-6:
    raise ValueError("Violated the LMI condition on decay rate")

###############################################################################
# 6) DISTURBANCE CROSSBAR
###############################################################################
w_c_practice = 0.0
for wv in vertices_dist:
    E = np.vstack([np.zeros((7,7)), np.eye(7)])  # 8 x 4
    wv_8 = E @ wv  # 8x1
    val = (wv_8.T @ P_delta @ wv_8).item()
    w_c_practice = max(w_c_practice, sqrt(val))

print("w_c_practice =", w_c_practice)
if w_c_practice > w_c_max + 1e-6:
    raise ValueError("Disturbance bound is bigger than expected w_c_max")

# Now for the model set d_w_max_model
w_corners_model = []
for wvals in itertools.product([-d_w_max_model, d_w_max_model], repeat=7):
    w_corners_model.append(np.array(wvals).reshape(-1,1))

w_c_model = 0.0
for wv in w_corners_model:
    E = np.vstack([np.zeros((7,7)), np.eye(7)])  # 8 x 4
    wv_8 = E @ wv  # 8x1
    val = (wv_8.T @ P_delta @ wv_8).item()
    w_c_model = max(w_c_model, sqrt(val))

print("w_c_model =", w_c_model)

###############################################################################
# 7) TERMINAL SET INGREDIENTS, ETC.
###############################################################################
# In your code: w_max_allowed = rho_c / c_max
w_max_allowed = rho_c / c_max_final
print("w_max_allowed = ", w_max_allowed)

# (We skip the loop over obstacles for 'w_max_allowed_output' for brevity;
#  you can replicate the same logic if needed.)

if w_max_allowed <= w_c_max:
    print("Error: Disturbance w_max is too big for the polytopic constraints.")

# Terminal tube size s_T = (1 - exp(-rho_c * T))/rho_c * w_c_practice
s_T = (1 - exp(-rho_c*T))/rho_c * w_c_practice
print("s_T =", s_T)

# Terminal cost matrix: solve Lyapunov eqn for (A_cl + kappa I)^T * P_f + P_f*(A_cl + kappa I) = -(Q+K^T R K)
# We'll replicate:
Q = np.eye(14)
R = np.eye(7)
kappa = 0.0
A_cl_kappa = A_cl + kappa*np.eye(14)

# SciPy's solve_continuous_lyapunov solves: A^T P + P A = -Q
# So we define Q_for_lyap = (Q + K^T R K)
Q_for_lyap = Q + K_delta.T @ R @ K_delta
P_f = solve_continuous_lyapunov(A_cl_kappa.T, -Q_for_lyap)
print("P_f (terminal cost) =")
print(P_f)

# ###############################################################################
# # 8) DISCRETIZATION
# ###############################################################################
# h = T/N
# # continuous A_c, B_c => discrete A_d, B_d via matrix exponential or cont2discrete:
# A_d  = expm(A_c * h)
# # B_d  = integral_0^h exp(A_c*t) dt * B_c.
# # A quick way in Python:
# eye8 = np.eye(14)
# def integral_expm(A, dt):
#     # approximate the integral_0^dt exp(A tau) dTau * B
#     # we can do a simple approach with matrix fraction decomposition or use a series expansion.
#     # For small dt it might be fine to do (A^-1)(exp(A*dt) - I).
#     # Provided A is invertible, but here A is singular. We'll use a standard trick:
#     return np.linalg.inv(A) @ (expm(A*dt) - eye8)

# # For the system, A_c has rank deficiency, so the standard formula might still be fine:
# # but let's handle the zero check:
# try:
#     B_d = integral_expm(A_c, h) @ B_c
# except np.linalg.LinAlgError:
#     # If A is singular, we can do a small Pade approach or use cont2discrete
#     from scipy.signal import cont2discrete
#     AdBd = cont2discrete((A_c, B_c, np.zeros((14,)), np.zeros((7,))), h, method='zoh')
#     A_d, B_d = AdBd[0], AdBd[1]

# print("A_d =\n", A_d)
# print("B_d =\n", B_d)

# # Closed-loop matrix discrete
# A_cl_c = A_c + B_c @ K_delta
# A_cl_d = expm(A_cl_c * h)
# try:
#     B_cl_d = integral_expm(A_cl_c, h) @ B_c
# except np.linalg.LinAlgError:
#     from scipy.signal import cont2discrete
#     AdBd = cont2discrete((A_cl_c, B_c, np.zeros((14,)), np.zeros((7,))), h, method='zoh')
#     A_cl_d, B_cl_d = AdBd[0], AdBd[1]

# print("A_cl_d =\n", A_cl_d)
# print("B_cl_d =\n", B_cl_d)

# # For real-time (1ms) sampling, do the same:
# h2 = 0.001
# A_d_1ms = expm(A_c * h2)
# try:
#     B_d_1ms = integral_expm(A_c, h2) @ B_c
# except np.linalg.LinAlgError:
#     from scipy.signal import cont2discrete
#     AdBd = cont2discrete((A_c, B_c, np.zeros((14,)), np.zeros((7,))), h2, method='zoh')
#     A_d_1ms, B_d_1ms = AdBd[0], AdBd[1]

# A_cl_d_1ms = expm(A_cl_c * h2)
# try:
#     B_cl_d_1ms = integral_expm(A_cl_c, h2) @ B_c
# except np.linalg.LinAlgError:
#     from scipy.signal import cont2discrete
#     AdBd = cont2discrete((A_cl_c, B_c, np.zeros((14,)), np.zeros((7,))), h2, method='zoh')
#     A_cl_d_1ms, B_cl_d_1ms = AdBd[0], AdBd[1]

# print("A_d_1ms =\n", A_d_1ms)
# print("B_d_1ms =\n", B_d_1ms)
# print("A_cl_d_1ms =\n", A_cl_d_1ms)
# print("B_cl_d_1ms =\n", B_cl_d_1ms)

###############################################################################
# 9) OPTIONAL: SAVE RESULTS (NUMPY .npy or .mat EQUIVALENT)
###############################################################################
# For instance:
# np.save('K_delta_8.npy', K_delta)
# np.save('A_cl_d_8.npy', A_cl_d)
# np.save('B_cl_d_8.npy', B_cl_d)
K_delta[np.abs(K_delta) < 1e-9] = 0.0
P_delta[np.abs(P_delta) < 1e-9] = 0.0
np.savez("matrices.npz", K_delta=K_delta, P_delta=P_delta, con_x=con_x, con_u=con_u, con_g=np.max(con_g))

print("\nDone.")