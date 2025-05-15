import fire
import h5py
import numpy as np
from kinematics import *
from constants import *
import matplotlib.pyplot as plt

def load_plot_hdf5(filename):
    
    with h5py.File(filename, mode="r") as f:
        X_sim = f["/X_sim"][...]
        Y_sim = f["/Y_sim"][...]
        U_sim = f["/U_sim"][...]

        print("X:", X_sim.shape, X_sim.dtype, type(X_sim))
        print("Y:", Y_sim.shape, Y_sim.dtype, type(Y_sim))
        print("U:", U_sim.shape, U_sim.dtype, type(U_sim))
    
    Y_real = np.stack([np.concatenate(kinematics(q[:7])).flatten() for q in X_sim], axis=0)
    
    x_min = np.concatenate([q_min, -np.array(dq_min_max)])
    x_max = np.concatenate([q_max, +np.array(dq_min_max)])
    
    u_min = -np.array(ddq_min_max)
    u_max = +np.array(ddq_min_max)
    x_mean = (x_min + x_max) / 2
    x_range = x_max - x_min
    
    u_mean = (u_min + u_max) / 2
    u_range = u_max - u_min
    
    X_norm = 2 * (X_sim - x_mean) / x_range
    U_norm = 2 * (U_sim - u_mean) / u_range
    
    N_data = X_norm.shape[0]
    t = np.arange(N_data)*dt
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, X_norm[:, :7])
    axes[0].set_ylabel('X[:,:7]')
    axes[0].legend([f'q{i}' for i in range(7)])
    axes[0].set_ylim(-1.05, 1.05)
    axes[1].plot(t, X_norm[:, 7:])
    axes[1].set_ylabel('X[:,7:]')
    axes[1].legend([f'dq{i}' for i in range(X_norm.shape[1] - 7)])
    axes[1].set_ylim(-1.05, 1.05)
    axes[2].step(t, U_norm, where='post')
    axes[2].set_ylabel('U')
    axes[2].legend([f'u{i}' for i in range(U_norm.shape[1])])
    axes[2].set_ylim(-1.05, 1.05)
    lines = axes[3].plot(t, Y_real[:, :3], label=[f'y_real{i}' for i in range(3)])
    for i, line in enumerate(lines):
        axes[3].plot(t, Y_sim[:, i], linestyle=':', color=line.get_color(), label=f'y_sim{i}')
    axes[3].set_ylabel('Y[:,:3]')
    axes[3].legend([f'y{i}' for i in range(3)])
    axes[3].set_xlabel('time [s]')

    plt.tight_layout()
    plt.show()
    
if __name__=="__main__":
    load_plot_hdf5("logs/sim_latest.h5")