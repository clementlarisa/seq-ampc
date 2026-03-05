import jax.numpy as jnp
import jax
import equinox as eqx
import functools
import tqdm
import numpy as np
import os

import time
import matplotlib.pyplot as plt


# from dataloader import JaxDataLoader, AmpcDataset
from dataset import MemoryAmpcDataset, H5AmpcDataset, compute_normalization_stats_memory, SubsetDataset
from dataloader_async import JaxDataLoader

from neuralnetwork import make_model, save_model, load_model, normalize, denormalize
from odesolver import IntegratorExplicitEulerFixed, IntegratorExplicitFixed, FuncPrestabilized, FuncPwConstClip, FuncPwConst
from neuralnetwork import L1, L_mpc_imitate, in_ellipse, compute_learning_rate, in_box, L_quadratic_sum

from utils import plot_histogram, remove_outliers_based_on_integrator_error

def evaluate_robustness(
    load_model_path="/share/mihaela-larisa.clement/soeampc-data/models/20250503-115751/71.eqx",
    example='franka',
    Integrator=IntegratorExplicitEulerFixed,
    seed=42,
    batch_size=10000
    ):
    
    print(f"\n####### Evaluating Model {load_model_path} ########\n\n")
    
    folder_path = load_model_path
    if folder_path.endswith(".eqx"):
        folder_path = folder_path[:-4]
    os.makedirs(folder_path, exist_ok=True)
    
    data_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    if example == 'franka':
        from datasets.franka.franka import load_dataset, x_min, x_max, u_min, u_max
        from datasets.franka.constants import N
        from datasets.franka.constants import dt
        from datasets.franka.constants import alpha_min as alpha
        from datasets.franka.constants import Q as Q_
        from datasets.franka.constants import P_delta as P_
        from datasets.franka.constants import R as R_
        from datasets.franka.constants import con_g, con_u, con_x, rho_c, w_c
        from datasets.franka.constants import sdf_3d, occupancy_3d_resolution
        from datasets.franka.franka import f_jax as continuous_time_dynamics
        from datasets.franka.sdf_jax import create_jax_interpolant_from_sdf_3d
        from datasets.franka.prestabilization_jax import prestabilization
        from datasets.franka.kinematics_jax import kinematics as forward_kinematics
        nx=14
        nu=7
        ny=7
        Q = np.diag(Q_)
        R = np.diag(R_)
        P_delta = jax.numpy.array(P_)
        P = Q
        

        
        # state_constraint_single_fcn = functools.partial(in_box, x_min=np.array(x_min), x_max=np.array(x_max))
        state_constraint_single_fcn = lambda x,s: in_box(x, x_min=np.array(x_min)+con_x*s, x_max=np.array(x_max)-con_x*s)
        f_sdf = create_jax_interpolant_from_sdf_3d(sdf_3d, occupancy_3d_resolution)
        def output_constraint_single_fcn(x, s):
            position, orientation = forward_kinematics(x)
            return f_sdf(position[0], position[1], position[2]) + 0.02 + con_g*s <= 0
        def output_constraint_fcn(X,S):
            return jax.numpy.min(jax.vmap(output_constraint_single_fcn)(X, S))
        def state_constraint_fcn(X,S):
            output_constr = output_constraint_fcn(X,S)
            state_constr = jax.numpy.min(jax.vmap(state_constraint_single_fcn)(X,S))
            return jnp.minimum(output_constr, state_constr)
            # return 
        def terminal_constraint_fcn(X_pred, X, S):
            return in_ellipse(X_pred[-1,:].at[:7].set(0), P_delta, alpha-S[-1])
        
        # terminal_set_fcn = functools.partial(in_ellipse, P=P_, alpha=)
        dynamics_func = FuncPwConst(dt=dt,nu=nu,f=continuous_time_dynamics,N=N)
        def discrete_system_dyanmics(x,u):
            return x + dt*continuous_time_dynamics(x,u)

        # Load full dataset into memory
        X, U, Y = load_dataset()
        assert U.shape[1:] == (N, nu)
        assert Y.shape[1:] == (ny,)
        assert X.shape[1:] == (N+1, nx)
        
        # plot_histogram(jnp.squeeze(X[:,0,:]), folder_path, "histogram_X_before.pdf")
        # plot_histogram(jnp.squeeze(Y), folder_path, "histogram_Y_before.pdf")
        # plot_histogram(jnp.squeeze(U[:,0,:]), folder_path, "histogram_U_before.pdf")
        
        def integrator(x0, U):
            def step(x, u):
                x_next = discrete_system_dyanmics(x, u)
                return x_next, x_next  # carry x_next, store x_next
            
            _, X = jax.lax.scan(step, x0, U)
            
            X_full = jnp.vstack([x0[None, :], X])
            return X_full
        
        X, U, Y = remove_outliers_based_on_integrator_error(
            X, U, Y,
            # Integrator(N=N, dt=dt, func=dynamics_func, nu=nu),
            integrator,
            batch_size=10000,
            threshold=0.01,
            save_histogram_path=None)
            # save_histogram_path=f"{folder_path}/histogram_integratorerror.pdf")
        
        # plot_histogram(jnp.squeeze(X[:,0,:]), folder_path, "histogram_X_after.pdf")
        # plot_histogram(jnp.squeeze(Y), folder_path, "histogram_Y_after.pdf")
        # plot_histogram(jnp.squeeze(U[:,0,:]), folder_path, "histogram_U_after.pdf")
        
        # Full dataset object (memory based)
        full_dataset = MemoryAmpcDataset(X, U, Y)

    # Compute normalization
    model, hyperparams = load_model(load_model_path)
    norm_stats = hyperparams["normalization_parameters"]
    norm_stats = {k: np.array(v) for k, v in norm_stats.items()}
    x_scale = norm_stats["x_scale"]
    u_scale = norm_stats["u_scale"]
    y_scale = norm_stats["y_scale"]
    
    x_offset = norm_stats["x_offset"]
    u_offset = norm_stats["u_offset"]
    y_offset = norm_stats["y_offset"]

    print(f"\nScaling:\n\t{x_scale=}\n\t{u_scale=}\n\t{y_scale=}\n\t{x_offset=}\n\t{u_offset=}\n\t{y_offset=}\n")

    # integrator = Integrator(N=N, dt=dt, func=dynamics_func, nu=nu)

    data_loader = JaxDataLoader(full_dataset, rng_key=data_key, batch_size=batch_size, shuffle=False)

    @eqx.filter_jit
    def compute_constraint_satisfaction(model, X, U, Y, S):
        x0_normalized = jax.vmap(functools.partial(normalize, scale=x_scale, offset=x_offset))(X[:,0,:])
        Y_normalized = jax.vmap(functools.partial(normalize, scale=y_scale, offset=y_offset))(Y)
        U_pred_norm = jax.vmap(model)(jnp.concatenate([x0_normalized, Y_normalized], axis=-1))
        U_pred = jax.vmap(jax.vmap(functools.partial(denormalize, scale=u_scale, offset=u_offset)))(U_pred_norm)
        X_pred = jax.vmap(integrator)(X[:,0,:], U_pred)
        
        state_constraint_fcn_S = functools.partial(state_constraint_fcn, S=S)
        terminal_constraint_fcn_S = functools.partial(terminal_constraint_fcn, S=S)
        in_constr = jax.vmap(state_constraint_fcn_S)(X_pred)
        in_terminal = jax.vmap(terminal_constraint_fcn_S)(X_pred, X)
        frac_in_constr = jax.numpy.mean(in_constr)
        frac_in_terminal = jax.numpy.mean(in_terminal)
        frac_feas = jnp.mean(jnp.minimum(in_constr, in_terminal))
        return frac_feas, frac_in_constr, frac_in_terminal
    

    num_samples = 21
    w_cs = np.linspace(0, 6, num_samples).tolist()
    results_frac_feas = []
    results_frac_in_constr = []
    results_frac_in_terminal = []
    
    
    for w_c in w_cs:
        print(f"\n++++++++++++ Testing robustness with {w_c=} ++++++++++++")
        S = np.zeros(N+1)
        for k in range(N):
            sdot = -rho_c*S[k]+w_c
            S[k+1] = S[k] + sdot*dt
        
        frac_feas = []
        frac_constr = []
        frac_terminal = []
        for batch_nr, batch in enumerate(tqdm.tqdm(data_loader, desc=f'eval {w_c}')):
            X_,U_,Y_ = batch
            ff, fic, fit = compute_constraint_satisfaction(model, X_, U_, Y_, S)
            frac_feas.append(ff)
            frac_constr.append(fic)
            frac_terminal.append(fit)
        results_frac_feas.append( np.mean(frac_feas)*100 )
        results_frac_in_constr.append( np.mean(frac_constr)*100 )
        results_frac_in_terminal.append( np.mean(frac_terminal)*100 )
        print(f"Mean points feasible: {float(results_frac_feas[-1]):.5f}, in state constraints {float(results_frac_in_constr[-1]):.5f}, in terminal constraint {float(results_frac_in_terminal[-1]):.5f}")
        print("")
    

    
    plt.plot(w_cs, results_frac_feas, label="Feasible", marker='o')
    plt.plot(w_cs, results_frac_in_constr, label="In Constraint Set", marker='s')
    plt.plot(w_cs, results_frac_in_terminal, label="In Terminal Set", marker='^')
    plt.xlabel("w_cs")
    plt.ylabel("Fraction")
    plt.title("Constraint Satisfaction vs w_cs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder_path}/constraint_satisfaction_vs_wcs.pdf", format="pdf")
    
        
    data = np.column_stack([w_cs, results_frac_feas, results_frac_in_constr, results_frac_in_terminal])
    np.savetxt(f"{folder_path}/constraint_satisfaction_data.dat", data,
            header="w_cs frac_feas frac_in_constr frac_in_terminal",
            comments='',
            fmt="%.6f", delimiter="\t")
    
    print(f"\nDone writing results to {folder_path}")
        
def evaluate_closed_loop(
    load_model_path="/share/mihaela-larisa.clement/soeampc-data/models/20250503-115751/71.eqx",
    example='franka',
    Integrator=IntegratorExplicitEulerFixed,
    seed=42,
    corrupt_reference = True,
    noisy = True
    ):
    folder_path = load_model_path
    if folder_path.endswith(".eqx"):
        folder_path = folder_path[:-4]
    os.makedirs(folder_path, exist_ok=True)
    
    data_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    if example == 'franka':
        from datasets.franka.franka import load_dataset_as_rollouts, x_min, x_max, u_min_jax, u_max_jax
        from datasets.franka.constants import N
        from datasets.franka.constants import q_center
        from datasets.franka.constants import dt
        from datasets.franka.constants import alpha_min as alpha
        from datasets.franka.constants import Q as Q_
        from datasets.franka.constants import P_delta as P_
        from datasets.franka.constants import R as R_
        from datasets.franka.constants import K_delta as K
        from datasets.franka.constants import con_g, con_u, con_x, rho_c, w_c
        from datasets.franka.constants import sdf_3d, occupancy_3d_resolution
        from datasets.franka.franka import f_jax as continuous_time_dynamics
        from datasets.franka.sdf_jax import create_jax_interpolant_from_sdf_3d
        from datasets.franka.prestabilization_jax import prestabilization
        from datasets.franka.kinematics_jax import kinematics as forward_kinematics
        nx=14
        nu=7
        ny=7
        Q = np.diag(Q_)
        R = np.diag(R_)
        P_delta = jax.numpy.array(P_)
        P = Q
        
        noise_level=0.5
        min_noise_level=0.001
        
        # state_constraint_single_fcn = functools.partial(in_box, x_min=np.array(x_min), x_max=np.array(x_max))
        
        # state_constraint_single_fcn = lambda x,s: in_box(x, x_min=np.array(x_min)+con_x*s, x_max=np.array(x_max)-con_x*s)
        state_constraint_single_fcn = lambda x,s: jax.numpy.logical_and(x >= np.array(x_min)+con_x*s,  x <= np.array(x_max)-con_x*s)
        f_sdf = create_jax_interpolant_from_sdf_3d(sdf_3d, occupancy_3d_resolution)
        def output_constraint_single_fcn(x, s):
            position, orientation = forward_kinematics(x)
            return f_sdf(position[0], position[1], position[2]) + 0.02 + con_g*s <= 0
        def output_constraint_fcn(X,S):
            return jax.numpy.min(jax.vmap(output_constraint_single_fcn)(X, S))
        def state_constraint_fcn(X,S):
            output_constr = output_constraint_fcn(X,S)
            state_constr_q =  jax.numpy.min(jax.vmap( state_constraint_single_fcn)(X,S)[:7])
            state_constr_dq = jax.numpy.min(jax.vmap(state_constraint_single_fcn)(X,S)[7:])
            return output_constr, state_constr_q, state_constr_dq
            # return 
        def terminal_constraint_fcn(X_pred, X, S):
            return in_ellipse(X_pred[-1,:].at[:7].set(0), P_delta, alpha-S[-1])
        
        def compute_cost(X, U):
            X_s = X[-1,:].at[:7].set(0)
            return jnp.sum(
                jax.vmap(lambda x,u: L_quadratic_sum(x, X_s, Q) + L_quadratic_sum(u, jnp.zeros_like(u), R))(X[:-1,:], U)
                ) + L_quadratic_sum(X[-1,:], X_s, P_delta)

        def distance_to_obstacle(x):
            position, orientation = forward_kinematics(x)
            return -(f_sdf(position[0], position[1], position[2]) + 0.02)
        
        def distance_to_reference(x, position_reference):
            position, orientation = forward_kinematics(x)
            return jnp.linalg.norm(position.flatten()-position_reference.flatten())
        
        cost_eval = lambda x,xref,u: L_quadratic_sum(x, xref, Q) + L_quadratic_sum(u, jnp.zeros_like(u), R)
        
        # terminal_set_fcn = functools.partial(in_ellipse, P=P_, alpha=)
        # dynamics_func = FuncPwConst(dt=dt,nu=nu,f=continuous_time_dynamics,N=N)
        # dynamics_func_onestep = FuncPwConst(dt=dt,nu=nu,f=continuous_time_dynamics,N=2)
        def discrete_system_dyanmics(x,u):
            return x + dt*continuous_time_dynamics(x,u)


        # Load full dataset into memory
        X, U, Y, common_fields = load_dataset_as_rollouts()
        N_sim = common_fields["N_sim"]
        N_data = X.shape[0]
        assert X.shape == (N_data, N_sim, N+1, nx)
        assert U.shape == (N_data, N_sim, N, nu)
        assert Y.shape == (N_data, N_sim, ny,)
        
        if corrupt_reference:
            Y[:,25:35, 1] += 0.3
            Y[:,25:35, 2] -= 0.3

    # Compute normalization
    model, hyperparams = load_model(load_model_path)
    norm_stats = hyperparams["normalization_parameters"]
    norm_stats = {k: np.array(v) for k, v in norm_stats.items()}
    x_scale = norm_stats["x_scale"]
    u_scale = norm_stats["u_scale"]
    y_scale = norm_stats["y_scale"]
    
    x_offset = norm_stats["x_offset"]
    u_offset = norm_stats["u_offset"]
    y_offset = norm_stats["y_offset"]

    print(f"\nScaling:\n\t{x_scale=}\n\t{u_scale=}\n\t{y_scale=}\n\t{x_offset=}\n\t{u_offset=}\n\t{y_offset=}\n")

    # integrator = Integrator(N=N, dt=dt, func=dynamics_func, nu=nu)
    # integrator_onestep = Integrator(N=2, dt=dt, func=dynamics_func_onestep, nu=nu)
    def integrator(x0, U):
        def step(x, u):
            x_next = discrete_system_dyanmics(x, u)
            return x_next, x_next  # carry x_next, store x_next
        
        _, X = jax.lax.scan(step, x0, U)
        
        X_full = jnp.vstack([x0[None, :], X])
        return X_full

    @eqx.filter_jit
    def compute_constraint_satisfaction(X, S):
        state_constraint_fcn_S = functools.partial(state_constraint_fcn, S=S)
        terminal_constraint_fcn_S = functools.partial(terminal_constraint_fcn, S=S)
        in_output_constr, in_constr_q, in_constr_dq = jax.vmap(state_constraint_fcn_S)(X)
        in_terminal = jax.vmap(terminal_constraint_fcn_S)(X, jnp.zeros_like(X))
        return in_output_constr, in_constr_q, in_constr_dq, in_terminal
    
    @eqx.filter_jit
    def simulate_batch(model, x0, Y):
        x0_normalized = jax.vmap(functools.partial(normalize, scale=x_scale, offset=x_offset))(x0)
        Y_normalized = jax.vmap(functools.partial(normalize, scale=y_scale, offset=y_offset))(Y)
        U_pred_norm = jax.vmap(model)(jnp.concatenate([x0_normalized, Y_normalized], axis=-1))
        U_pred = jax.vmap(jax.vmap(functools.partial(denormalize, scale=u_scale, offset=u_offset)))(U_pred_norm)
        X_pred = jax.vmap(integrator)(x0, U_pred)
        return X_pred, U_pred
    
    @eqx.filter_jit
    def safety_augmented(model, x0, Y, X_cand, U_cand, cand_cost, S):
        X_pred, U_pred = simulate_batch(model, x0,Y)
        in_output_constr, in_constr_q, in_constr_dq, in_terminal = compute_constraint_satisfaction(X_pred, S)
        pred_cost = jax.vmap(compute_cost)(X_pred, U_pred)
        cost_decrease = pred_cost <= cand_cost
        update = jnp.logical_and(jnp.logical_and(jnp.logical_and(in_constr_q, in_constr_dq), in_terminal), in_output_constr)
        mask_X = update[:, None, None]
        mask_U = update[:, None, None]
        mask_cost = update
        X_cand = jnp.where(mask_X, X_pred, X_cand)
        U_cand = jnp.where(mask_U, U_pred, U_cand)
        cand_cost = jnp.where(mask_cost, pred_cost, cand_cost)
            
        return X_cand, U_cand, cand_cost, X_pred, U_pred, pred_cost, update, in_output_constr, in_constr_q, in_constr_dq, in_terminal,  cost_decrease
        
    @eqx.filter_jit    
    def shift_append_terminal(X_cand, U_cand):
        X_cand_next = jnp.zeros_like(X_cand)
        U_cand_next = jnp.zeros_like(U_cand)
        X_cand_next = X_cand_next.at[:, :-1, :].set(X_cand[:, 1:, :])
        U_cand_next = U_cand_next.at[:, :-1, :].set(U_cand[:, 1:, :])
        # U_cand_terminal = jax.vmap(lambda x: K@x.at[:7].set(0))(X_cand[:,-1,:]) #TODO!!!
        # U_cand_terminal = jax.vmap(lambda x: K@(jnp.concatenate([x[:7]-q_center, x[7:]])))(X_cand[:,-1,:]) #TODO!!!
        U_cand_terminal = jax.vmap(lambda x: K@(jnp.concatenate([x[:7]-q_center, x[7:]])))(X_cand[:,-1,:]) #TODO!!!
        # U_cand_terminal = jax.vmap(lambda x: K@x.at[:7].set(0))(X_cand[:,-1,:]) #TODO!!!
        U_cand_next = U_cand_next.at[:, -1, :].set(U_cand_terminal)
        # U_cand_terminal = U_cand_next[:,-1,:]
        X_cand_terminal = jax.vmap(discrete_system_dyanmics)(X_cand[:, -1, :], U_cand_terminal)
        X_cand_next = X_cand_next.at[:, -1, :].set(X_cand_terminal)
        return X_cand_next, U_cand_next
    
    def test_naive(key):
        print(f"\n\nTesting Naive")
        X_sim = []
        U_sim = []
        X_cand_cl = []
        U_cand_cl = []
        X_sim.append(jnp.array(X[:,0,0,:]))
        for i in tqdm.tqdm(range(N_sim), desc="Simulation"):
            X_pred, U_pred = simulate_batch(model, X_sim[-1], Y[:,i,:])
            X_cand_cl.append(X_pred)
            U_cand_cl.append(U_pred)
            if noisy:
                key, sub_key = jax.random.split(key)
                disturbance = jax.random.uniform(sub_key, shape=U_pred[:, 0, :].shape, minval=-noise_level, maxval=noise_level)*(jnp.abs(X_pred[:,0,7:])+min_noise_level)
                u_disturbed = U_pred[:,0,:]+disturbance
                X_sim.append(jax.vmap(discrete_system_dyanmics)(X_pred[:,0,:], u_disturbed))
                U_sim.append(u_disturbed)
            else:
                X_sim.append(X_pred[:,1,:])
                U_sim.append(U_pred[:,0,:])
            
        
        X_sim = jnp.transpose(jnp.array(X_sim), axes=(1,0,2))
        U_sim = jnp.transpose(jnp.array(U_sim), axes=(1,0,2))
        X_cand_cl = jnp.transpose(jnp.array(X_cand_cl), axes=(1, 0, 2, 3))
        U_cand_cl = jnp.transpose(jnp.array(U_cand_cl), axes=(1, 0, 2, 3))
        
        print("Computing closed loop statistics:")
        
        cl_in_output = jnp.squeeze(jax.vmap(lambda X: jax.vmap(functools.partial(output_constraint_single_fcn, s=0))(X))(X_sim))
        cl_in_state = jax.vmap(
            lambda X: jax.vmap( lambda x: jnp.all(state_constraint_single_fcn(x,s=0)))(X)
            )(X_sim)
        cl_safe = jnp.mean(jnp.minimum(cl_in_output, cl_in_state))
        print(f"\tclosed loop safe simulations: {cl_safe*100:.5f}%")
        
        mean_joint_position_error = jnp.mean(jnp.abs(X_sim[:,:-1,:7] - X[:,:,0,:7]))
        print(f"\tmean joint position error: {np.rad2deg(mean_joint_position_error):.5f} deg")
        
        print(f"\n\n")
        
            # list for cl timestep, [N_batch, N_MPC, nx]
        print("\n\nSaving dataset for plotting")
        res_all = []
        N_export = 10
        for i in tqdm.tqdm(range(min(N_data, N_export)), desc="Convert"):
            res_cl = []
            for j in range(N_sim):
                q_opt   = np.array(X_cand_cl[i, j, :, :7])
                dq_opt  = np.array(X_cand_cl[i, j, :, 7:])
                ddq_opt = np.array(jax.vmap(prestabilization)(X_cand_cl[i, j, :-1], U_cand_cl[i,j]))
                v_opt   = np.array(U_cand_cl[i,j,:,:])
                q_s_opt = X_cand_cl[i,j,-1,:].at[7:].set(0)
                
                q_mpc   = np.array(X[i,j,:,:7])
                dq_mpc  = np.array(X[i,j,:,7:])
                ddq_mpc = np.array(jax.vmap(prestabilization)(X[i, j, :-1], U[i,j]))
                q_s_mpc = np.array(X[i,j,-1,:7])                
                
                d_obst = distance_to_obstacle(X_cand_cl[i,j,0,:])
                d_to_ref = distance_to_reference(X_cand_cl[i,j,0,:], Y[i, j, :3])
                cost_state = cost_eval(X_cand_cl[i,j,0], q_s_opt, U_cand_cl[i,j,0])

                
                res = {
                    'q_opt':   np.array(q_opt), # N_mpc x 7
                    'dq_opt':  np.array(dq_opt), # N_mpc x 7
                    'ddq_opt': np.array(ddq_opt), # N_mpc x 7
                    'v_opt':   np.array(v_opt), # N_mpc x 7
                    'q_s_opt': np.array(q_s_opt), # 7
                    'q_mpc':   q_mpc,
                    'dq_mpc':  dq_mpc,
                    'ddq_mpc': ddq_mpc,
                    'q_s_mpc': q_s_mpc,
                    'd_obst': d_obst,
                    'd_to_ref': d_to_ref,
                    'cost_state': cost_state,
                    'position_reference': np.array(Y[i, j, :3]),
                    'orientation_reference': np.array(Y[i, j, 3:]),
                }
                res_cl.append(res)
            res_all.append(res_cl)
            exp = {
                "data": res_all,
                "N_sim": N_sim,
                "N_mpc": N,
                "dt": dt,
                "limits": common_fields["limits"],
            }
            folder_path = load_model_path
            if folder_path.endswith(".eqx"):
                folder_path = folder_path[:-4]
            os.makedirs(folder_path, exist_ok=True)
            np.savez_compressed(f'{folder_path}/cl_eval_naive{f"_corrupted" if corrupt_reference else ""}.npz', allow_pickle=True, **exp)
        
    test_naive(model_key)
    
    
    def test_safety(key):
        # extract initial mpc solution for each rollout
        X_cand, U_cand = jnp.array(X[:,0,:,:]), jnp.array(U[:,0,:,:])
        
        X_sim = []
        U_sim = []
        update_cl           = []
        in_output_constr_cl = []
        in_constr_q_cl      = []
        in_constr_dq_cl     = []
        in_terminal_cl   = []
        cost_decrease_cl = []
        cand_cost = np.inf
        
        X_pred_cl, U_pred_cl, X_cand_cl, U_cand_cl = [], [], [], []
        
        # extract initial cl state
        print(f"\nTesting Safety Augmentation")
        X_sim.append(X_cand[:,0,:])
        for i in tqdm.tqdm(range(N_sim), desc="Simulation"):
            S = np.zeros(N+1)
            X_cand, U_cand, cand_cost, X_pred, U_pred, pred_cost, update, in_output_constr, in_constr_q, in_constr_dq, in_terminal,  cost_decrease = \
                safety_augmented(
                    model,
                    X_sim[-1],
                    Y[:,i,:],
                    X_cand,
                    U_cand,
                    cand_cost,
                    S)
            
            update_cl.append(update)
            in_output_constr_cl.append(in_output_constr)
            in_constr_q_cl.append(in_constr_q)
            in_constr_dq_cl.append(in_constr_dq)
            in_terminal_cl.append(in_terminal)
            cost_decrease_cl.append(cost_decrease)
            
            if noisy:
                key, sub_key = jax.random.split(key)
                disturbance = jax.random.uniform(sub_key, shape=U_cand[:, 0, :].shape, minval=-noise_level, maxval=noise_level)*(jnp.abs(X_cand[:,0,7:])+min_noise_level)
                u_disturbed = U_cand[:,0,:]+disturbance
                X_sim.append(jax.vmap(discrete_system_dyanmics)(X_cand[:,0,:], u_disturbed))
                U_sim.append(u_disturbed)
            else:
                X_sim.append(X_cand[:,1,:])
                U_sim.append(U_cand[:,0,:])
            
            X_pred_cl.append(X_pred)
            U_pred_cl.append(U_pred)
            X_cand_cl.append(X_cand)
            U_cand_cl.append(U_cand)
            
            X_cand, U_cand = shift_append_terminal(X_cand, U_cand)
        
        # list of cl timestep, [N_batch, N_mpc, nx], we want [N_batch, N_sim, N_mpc, nx]
        X_pred_cl = jnp.transpose(jnp.array(X_pred_cl), axes=(1, 0, 2, 3))
        U_pred_cl = jnp.transpose(jnp.array(U_pred_cl), axes=(1, 0, 2, 3))
        X_cand_cl = jnp.transpose(jnp.array(X_cand_cl), axes=(1, 0, 2, 3))
        U_cand_cl = jnp.transpose(jnp.array(U_cand_cl), axes=(1, 0, 2, 3))
        
        # list of cl timesteps, [N_batch, nx], we want [N_batch, N_sim, nx]
        X_sim = jnp.transpose(jnp.array(X_sim), axes=(1,0,2))
        U_sim = jnp.transpose(jnp.array(U_sim), axes=(1,0,2))
        
        # list of cl timestep, [N_batch], we want [N_batch, N_sim]
        update_cl            = jnp.transpose(jnp.array(update_cl, dtype=jnp.bool), axes=(1,0))
        in_output_constr_cl  = jnp.transpose(jnp.array(in_output_constr_cl, dtype=jnp.bool), axes=(1,0))
        in_constr_q_cl   = jnp.transpose(jnp.array(in_constr_q_cl, dtype=jnp.bool), axes=(1,0))
        in_constr_dq_cl  = jnp.transpose(jnp.array(in_constr_dq_cl, dtype=jnp.bool), axes=(1,0))
        in_terminal_cl   = jnp.transpose(jnp.array(in_terminal_cl, dtype=jnp.bool), axes=(1,0))
        in_constr_cl = jnp.logical_and(jnp.logical_and(in_output_constr_cl, in_constr_q_cl), in_constr_dq_cl)
        cost_decrease_cl = jnp.transpose(jnp.array(cost_decrease_cl, dtype=jnp.bool), axes=(1,0))
        
        print("\n\nComputing closed loop statistics")
        cl_in_output = jnp.squeeze(jax.vmap(lambda X: jax.vmap(functools.partial(output_constraint_single_fcn, s=0))(X))(X_sim))
        cl_in_state = jax.vmap(lambda X: jax.vmap(functools.partial(lambda x,s: jnp.all(state_constraint_single_fcn(x,s)), s=0))(X))(X_sim)
        cl_safe = jnp.mean(jnp.minimum(cl_in_output, cl_in_state))
        
        
        candidate_applied = jnp.mean(jnp.mean(~update_cl, axis=1), axis=0)
        # when solution was rrejected
        mask = ~update_cl
        reason_output_constr= jnp.mean((~in_output_constr_cl[mask]).astype(jnp.float32))
        reason_q_constr= jnp.mean((~in_constr_q_cl[mask]).astype(jnp.float32))
        reason_dq_constr= jnp.mean((~in_constr_dq_cl[mask]).astype(jnp.float32))
        reason_constr= jnp.mean((~in_constr_cl[mask]).astype(jnp.float32))
        reason_term= jnp.mean((~in_terminal_cl[mask]).astype(jnp.float32))
        reason_cost= jnp.mean((~cost_decrease_cl[mask]).astype(jnp.float32))
        
        print(f"\tclosed loop safe simulations: {cl_safe*100:.5f}%")
        print(f"\taverage closed loop candidate applied rate: {candidate_applied*100:.5f}%")
        print(f"\treasons:")
        print(f"\t\tstate constr all: {reason_constr}")
        print(f"\t\tstate constr output: {reason_output_constr}")
        print(f"\t\tstate constr q:      {reason_q_constr}")
        print(f"\t\tstate constr dq:     {reason_dq_constr}")
        
        print(f"\t\tterminal constr: {reason_term}")
        print(f"\t\tcost:            {reason_cost}")
            
        # list for cl timestep, [N_batch, N_MPC, nx]
        print("\n\nSaving dataset for plotting")
        res_all = []
        N_export = 10
        for i in tqdm.tqdm(range(min(N_data, N_export)), desc="Convert"):
            res_cl = []
            for j in range(N_sim):
                q_opt   = np.array(X_cand_cl[i, j, :, :7])
                dq_opt  = np.array(X_cand_cl[i, j, :, 7:])
                ddq_opt = np.array(jax.vmap(lambda x,u : jnp.maximum(jnp.minimum(K@(jnp.concatenate([x[:7]-q_center, x[7:]])) + u, u_max_jax), u_min_jax))(X_cand_cl[i, j, :-1], U_cand_cl[i,j]))
                v_opt   = np.array(U_cand_cl[i,j,:,:])
                q_s_opt = X_cand_cl[i,j,-1,:].at[7:].set(0)
                
                q_pred   = np.array(X_pred_cl[i, j, :, :7])
                dq_pred  = np.array(X_pred_cl[i, j, :, 7:])
                ddq_pred = np.array(jax.vmap(prestabilization)(X_pred_cl[i, j, :-1], U_pred_cl[i,j]))
                v_pred   = np.array(U_pred_cl[i,j,:,:])
                
                q_mpc   = np.array(X[i,j,:,:7])
                dq_mpc  = np.array(X[i,j,:,7:])
                ddq_mpc = np.array(jax.vmap(prestabilization)(X[i, j, :-1], U[i,j]))
                q_s_mpc = np.array(X[i,j,-1,:7])                

                d_obst = distance_to_obstacle(X_cand_cl[i,j,0,:])
                d_to_ref = distance_to_reference(X_cand_cl[i,j,0,:], Y[i, j, :3])
                cost_state = cost_eval(X_cand_cl[i,j,0], q_s_opt, U_cand_cl[i,j,0])
                
                res = {
                    'q_opt':   np.array(q_opt), # N_mpc x 7
                    'dq_opt':  np.array(dq_opt), # N_mpc x 7
                    'ddq_opt': np.array(ddq_opt), # N_mpc x 7
                    'v_opt':   np.array(v_opt), # N_mpc x 7
                    'q_s_opt': np.array(q_s_opt), # 7
                    'q_pred': q_pred,
                    'dq_pred': dq_pred,
                    'ddq_pred': ddq_pred,
                    'v_pred': v_pred,
                    'q_mpc':   q_mpc,
                    'dq_mpc':  dq_mpc,
                    'ddq_mpc': ddq_mpc,
                    'q_s_mpc': q_s_mpc,
                    'd_obst': d_obst,
                    'd_to_ref': d_to_ref,
                    'cost_state': cost_state,
                    'position_reference': np.array(Y[i, j, :3]),
                    'orientation_reference': np.array(Y[i, j, 3:]),
                    'update': float(update_cl[i,j]),
                    'in_constr': float(in_output_constr_cl[i,j]),
                    'in_constr': float(in_constr_q_cl[i,j]),
                    'in_constr': float(in_constr_dq_cl[i,j]),
                    'in_terminal': float(in_terminal_cl[i,j]),
                    'cost_decrease': float(cost_decrease_cl[i,j]),                    
                }
                res_cl.append(res)
            res_all.append(res_cl)
        exp = {
            "data": res_all,
            "N_sim": N_sim,
            "N_mpc": N,
            "dt": dt,
            "limits": common_fields["limits"],
        }
        
        folder_path = load_model_path
        if folder_path.endswith(".eqx"):
            folder_path = folder_path[:-4]
        os.makedirs(folder_path, exist_ok=True)
        np.savez_compressed(f'{folder_path}/cl_eval_safety{f"_corrupted" if corrupt_reference else ""}.npz', allow_pickle=True, **exp)
    
    test_safety(model_key)


def convert_dataset_for_plotting(folder_path, idx):
    for corrupted in [True, False]:
        naive_path = f"{folder_path}/cl_eval_naive{"_corrupted" if corrupted else ""}.npz"
        safe_path = f"{folder_path}/cl_eval_safety{"_corrupted" if corrupted else ""}.npz"
        naive = np.load(naive_path, allow_pickle=True)
        safe = np.load(safe_path, allow_pickle=True)
        
        naive_data = naive["data"][()]
        safe_data = safe["data"][()]
        dt = naive["dt"].item()
        N_sim = naive["N_sim"].item()

        assert dt == safe["dt"].item(), "Mismatched dt between datasets"

        naive_traj = naive_data[idx]
        safe_traj = safe_data[idx]

        lines = ["% time  mpc_dist  nn_dist  aug_dist  mpc_cost  nn_cost  aug_cost"]

        for j in range(N_sim):
            time = j * dt

            # From naive (NN)
            nn_d_to_obst = float(naive_traj[j]['d_obst'][0])
            nn_c = float(naive_traj[j]['cost_state'])
            nn_d_to_ref = float(naive_traj[j]['d_to_ref'])

            # From safe (augmented NN)
            aug_d_to_obst = float(safe_traj[j]['d_obst'][0])
            aug_c = float(safe_traj[j]['cost_state'])
            aug_d_to_ref = float(safe_traj[j]['d_to_ref'])

            # From MPC baseline
            # mpc_d = safe_traj[j]['d_obst']  # Assuming safe uses same candidate traj as baseline
            mpc_d = 0
            # mpc_c = cost_eval(safe_traj[j]['q_mpc'][0], safe_traj[j]['q_s_mpc'], safe_traj[j]['v_opt'][0])  # If needed
            mpc_c = 0
            
            # Append line
            lines.append(f"{time:.5f} {mpc_d:.5f} {nn_d_to_obst:.5f} {aug_d_to_obst:.5f} {mpc_c:.5f} {nn_c:.5f} {aug_c:.5f} {mpc_d:.5f} {nn_d_to_ref:.5f} {aug_d_to_ref:.5f}")

        # Write to .dat
        output_path = f"{folder_path}/cl_eval{"_corrupted" if corrupted else ""}_{idx}.dat"
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

        print(f"Data saved to {output_path}")

if __name__=="__main__":
    load_model_path = "/share/mihaela-larisa.clement/soeampc-data/models/20250510-140709/399.eqx"
    evaluate_robustness(load_model_path=load_model_path)
    evaluate_closed_loop(load_model_path=load_model_path, corrupt_reference=False, noisy=False)
    evaluate_closed_loop(load_model_path=load_model_path, corrupt_reference=True, noisy=True)
    evaluate_closed_loop(load_model_path=load_model_path, corrupt_reference=True, noisy=False)