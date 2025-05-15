import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

def compute_lqr_gain(n, q_weight, dq_weight, ddq_weight):
    
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.zeros((n, n)), np.zeros((n, n))]
    ])
    B = np.block([
        [np.zeros((n, n))],
        [np.eye(n)]
    ])

    Q = np.block([
        [q_weight*np.eye(n), np.zeros((n, n))],
        [np.zeros((n, n)), dq_weight * np.eye(n)]
    ])
    
    R = ddq_weight * np.eye(n)

    P = solve_continuous_are(A, B, Q, R)

    K = np.linalg.inv(R) @ (B.T @ P)
    
    return K, P

def compute_lqr_terminal_cost(n, q_weight, dq_weight, ddq_weight, K_delta):
    
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.zeros((n, n)), np.zeros((n, n))]
    ])
    B = np.block([
        [np.zeros((n, n))],
        [np.eye(n)]
    ])

    Q = np.block([
        [q_weight*np.eye(n), np.zeros((n, n))],
        [np.zeros((n, n)), dq_weight * np.eye(n)]
    ])
    
    R = ddq_weight * np.eye(n)
    
    Q_for_lyap = Q + K_delta.T @ R @ K_delta
    P_f = solve_continuous_lyapunov(A.T, -Q_for_lyap)
    return P_f

if __name__ == '__main__':
    K = compute_lqr_gain()
    print("LQR gain matrix K:")
    print(K)