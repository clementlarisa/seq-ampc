import casadi as cs
from urdf2casadi import urdfparser as u2c
import numpy as np
import tqdm
import trimesh
from scipy.spatial.transform import Rotation as R
import fire
import h5py
import time

from quaterror import *
from plotting import *
from sdf import *

from lqr import *


def get_robot_fk_slow(urdf_filename="franka_panda_urdf/panda.urdf", base_link="panda_link0", end_link="panda_grasptarget"):
    robot_urdf = u2c.URDFparser()
    robot_urdf.from_file(urdf_filename)
    fk_dict = robot_urdf.get_forward_kinematics(base_link, end_link) # quat is scalar-last form
    
    q = fk_dict["q"]
    position = fk_dict["T_fk"](q)[:3,3]
    quat = fk_dict["quaternion_fk"](q)
    x,y,z,w = quat[0],quat[1],quat[2],quat[3]
    orientation = cs.vertcat(w,x,y,z)
    
    forward_kinematics_fcn = cs.Function("kinematics",
                [q],
                [position, orientation],
                ["q"],
                ["position", "orientation"]
                )
    
    max_effort, max_vel = robot_urdf.get_dynamics_limits(base_link, end_link)

    limits={}
    limits['q_min']  = fk_dict['lower']
    limits['q_max']  = fk_dict['upper']
    limits['dq'] = max_vel
    limits['tau'] = max_effort
    limits['ddq'] = [15, 7.5, 10, 12.5, 15, 20, 20]

    return forward_kinematics_fcn, limits

def get_robot_fk():
    limits = np.load("franka_limits.npz")
    forward_kineamtics_fcn = cs.Function.load("franka_fk.casadi")
    return forward_kineamtics_fcn, limits

def get_sdf():
    casadi_sdf = cs.Function.load("sdf.casadi")
    return casadi_sdf

def export_data_to_pyfile(data, py_filename='exported_data.py', threshold=1000):
    base_name = py_filename.split(".py")[0]
    npz_filename = base_name + '.npz'
    
    large_arrays = {}
    with open(py_filename, 'w') as f:
        f.write('import numpy as np\n')
        f.write('import os\n\n')

        # First pass: identify large arrays
        for key, value in data.items():
            if isinstance(value, np.ndarray) and value.size >= threshold:
                large_arrays[key] = value
            else:
                # Inline export
                if isinstance(value, np.ndarray):
                    array_str = np.array2string(value, separator=', ', threshold=np.inf)
                    f.write(f'{key} = np.array({array_str})\n\n')
                else:
                    f.write(f'{key} = {repr(value)}\n\n')

        if large_arrays:
            np.savez_compressed(npz_filename, **large_arrays)
            f.write(f'# Load large arrays from NPZ file\n')
            f.write(f'_npz_data = np.load(os.path.join(os.path.dirname(__file__), "{npz_filename}"))\n')
            for key in large_arrays:
                f.write(f'{key} = _npz_data["{key}"]\n')

def export_all_constants():
    fk, limits = get_robot_fk_slow()
    q = cs.SX.sym("q", 7)
    position, orientation = fk(q)
    position_sparse = cs.sparsify(position)
    orientation_sparse =cs.sparsify(orientation)
    fk_sparse_fcn = cs.Function("kinematics",
                [q],
                [position_sparse, orientation_sparse],
                ["q"],
                ["position", "orientation"]
                )
    fk_sparse_fcn.save("franka_fk.casadi")
    np.savez_compressed("franka_limits.npz", **limits)
        
    cg = cs.CodeGenerator(fk_sparse_fcn.name())
    cg.add(fk_sparse_fcn)
    cg.generate("./tmp/")
    import numpysadi
    numpysadi.casadi_to_python_code_from_c(fk_sparse_fcn, f"tmp/{fk_sparse_fcn.name()}.py")
    numpysadi.compare_casadi_and_python_function(fk_sparse_fcn, f"tmp/{fk_sparse_fcn.name()}.py")
    numpysadi.casadi_to_python_code_from_c(fk_sparse_fcn, f"tmp/{fk_sparse_fcn.name()}_jax.py", numpy_module="jax.numpy")
    
    with np.load("matrices.npz") as data:
        K_delta = data["K_delta"]
        P_delta = data["P_delta"]
        con_x = data["con_x"]
        con_u = data["con_u"]
        con_g = data["con_g"]
    

    limits_q_center = (np.array(limits["q_max"])-np.array(limits["q_min"]))/2+np.array(limits["q_min"])
    limits["q_center"] = limits_q_center
    P_f = compute_lqr_terminal_cost(7, q_weight, dq_weight, ddq_weight, K_delta)
    alpha_min = w_c/rho_c
    
    x = cs.SX.sym("x", 14)
    u = cs.SX.sym("v", 7)
    ddq = cs.sparsify(K_delta)@(x-cs.vertcat(limits_q_center, np.zeros(7))) + u
    prestab_fcn =cs.Function("prestabilization", [x,u], [ddq], ["x", "u"], ["ddq"])
    cg = cs.CodeGenerator(prestab_fcn.name())
    cg.add(prestab_fcn)
    cg.generate("./tmp/")
    import numpysadi
    numpysadi.casadi_to_python_code_from_c(prestab_fcn, f"tmp/{prestab_fcn.name()}.py")
    numpysadi.compare_casadi_and_python_function(prestab_fcn, f"tmp/{prestab_fcn.name()}.py")
    numpysadi.casadi_to_python_code_from_c(prestab_fcn, f"tmp/{prestab_fcn.name()}_jax.py", numpy_module="jax.numpy")
    
    occupancy_3d_resolution = 0.04
    casadi_sdf, occupancy_3d, obstacle_list, sdf_3d = make_sdf_env(occupancy_3d_resolution)
    np.savez_compressed("sdf.npz", occupancy_3d=occupancy_3d, obstacle_list=obstacle_list)
    casadi_sdf.save("sdf.casadi")
    
    # N = 100
    test_positions = np.random.uniform(low=-1.0, high=1.0, size=(100, 3))
    sdf_outputs = []
    for pos in test_positions:
        val = casadi_sdf(*pos).full().item()
        sdf_outputs.append(val)
    sdf_outputs = np.array(sdf_outputs)

    # Save to HDF5
    h5_sdf_filename = "tmp/sdf.h5"
    with h5py.File(h5_sdf_filename, "w") as f:
        f.create_dataset("sdf_3d", data=sdf_3d)
        f.create_dataset("sdf_flat", data=np.transpose(sdf_3d, (2, 1, 0)).ravel(order='F'))
        f.create_dataset("dims", data=np.array(sdf_3d.shape[::-1]))
        f.create_dataset("resolution", data=occupancy_3d_resolution)
        f.create_dataset("test_positions", data=test_positions)
        f.create_dataset("sdf_outputs", data=sdf_outputs)
    print(f"Exported to {h5_sdf_filename}")
    
    data = {
        "K_delta": K_delta,
        "P_delta": P_delta,
        "con_x": con_x[::2],
        "con_u": con_u[::2],
        "con_g": con_g,
        "P_f": P_f,
        "alpha_min": alpha_min,
        "N": N,
        "dt": dt,
        "Q": [q_weight for _ in range(nq)] + [dq_weight for _ in range(ndq)],
        "R": [ddq_weight for _ in range(nddq)],
        "dt": dt,
        "N": N,
        "rho_c": rho_c,
        "w_c": w_c,
        "sdf_3d": sdf_3d,
        "occupancy_3d_resolution": occupancy_3d_resolution,
        "q_min": limits["q_min"],
        "q_max": limits["q_max"],
        "dq_min_max": limits["dq"],
        "ddq_min_max": limits["ddq"],
        "tau_min_max": limits["tau"],
        "q_center": limits["q_center"]
        }
    
    export_data_to_pyfile(data, "tmp/constants.py")
    
    

N  = 20
dt = 0.05
q_weight = 1
dq_weight = 1e-3
ddq_weight = 4e-3
rho_c = 10
w_c = 6
weight_orientation_error = 5e1
weight_position_error = 1e2

nq = 7
ndq = nq
nddq = nq


def mpc_problem(forward_kineamtics_fcn, limits, sdf_fcn = None, silent=True, solver="ipopt", max_iter=50, discretization="rk4"):
    
    opti = cs.Opti()
    
    q = opti.variable(nq, N+1 )
    dq = opti.variable(ndq, N+1 )
    v = opti.variable(nddq, N)
    
    q_s = opti.variable(nq)
    dq_s = np.zeros(ndq)
    
    n_position = 3
    n_orientation = 4
    y_position_ref = opti.parameter(n_position)
    y_orientation_ref = opti.parameter(n_orientation)
    
    q_init = opti.parameter(nq)
    dq_init = opti.parameter(ndq)
    
    quat_err_fcn = build_quaternion_error_function()
    
    cost = 0
    
    with np.load("matrices.npz") as data:
        K_delta = data["K_delta"]
        P_delta = data["P_delta"]
        con_x = data["con_x"]
        con_u = data["con_u"]
        con_g = data["con_g"]
    P_f = compute_lqr_terminal_cost(7, q_weight, dq_weight, ddq_weight, K_delta)

    limits_q_center = (np.array(limits["q_max"])-np.array(limits["q_min"]))/2+np.array(limits["q_min"])
    ddq = cs.sparsify(K_delta)@(cs.vertcat(q[:,:-1], dq[:,:-1])-cs.vertcat(limits_q_center, np.zeros(7))) + v
    
    s = np.zeros((1, N+1))
    for k in range(N):
        sdot = -rho_c*s[:,k]+w_c
        s[:,k+1] = s[:,k] + sdot*dt
    alpha_min = w_c/rho_c

    opti.subject_to(q[2,3:]==0)
    cost += 1e4*cs.sum2(q[2,:])
    
    for k in range(N):
        
        if discretization=="euler":
            q_next = q[:,k] + dt*dq[:,k] + 0.5*dt**2*ddq[:,k]
            dq_next = dq[:,k] + dt*ddq[:,k]            
        elif discretization == "rk4":
            qk = q[:, k]
            dqk = dq[:, k]
            ddqk = ddq[:, k]
            k1_q = dqk
            k1_dq = ddqk
            k2_q = dqk + 0.5 * dt * k1_dq
            k2_dq = ddqk  # constant
            k3_q = dqk + 0.5 * dt * k2_dq
            k3_dq = ddqk  # constant
            k4_q = dqk + dt * k3_dq
            k4_dq = ddqk  # constant
            q_next = qk + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            dq_next = dqk + (dt / 6.0) * (k1_dq + 2 * k2_dq + 2 * k3_dq + k4_dq)
        
        opti.subject_to( q[:,k+1] ==  q_next)
        opti.subject_to(dq[:,k+1] == dq_next)
        
        opti.subject_to(opti.bounded( -np.array(limits['ddq']) + con_u[::2]*s[:,k] , ddq[:,k], limits['ddq']-con_u[::2]*s[:,k]))
        
        if k > 0:
            opti.subject_to(opti.bounded(  limits['q_min'] + con_x[::2][:7]*s[:,k], q[:,k], limits['q_max'] - con_x[::2][:7]*s[:,k]))
            opti.subject_to(opti.bounded( -np.array(limits['dq'])  + con_x[::2][7:]*s[:,k], dq[:,k],  limits['dq']- con_x[::2][7:]*s[:,k]))
            
            if sdf_fcn is not None:
                position_k, orientation_k = forward_kineamtics_fcn(q[:,k])
                extra_distance = 0.02
                opti.subject_to(sdf_fcn(position_k[0],position_k[1],position_k[2]) + extra_distance + con_g*s[:,k] <= 0)
        
        cost += q_weight*cs.sum1((q[:,k]-q_s)**2)
        cost += dq_weight*cs.sum1((dq[:,k]-dq_s)**2)
        cost += ddq_weight*cs.sum1((ddq[:,k])**2)
    
    k = N
    opti.subject_to(opti.bounded(  limits['q_min'] + con_x[::2][:7]*s[:,k], q[:,k], limits['q_max'] - con_x[::2][:7]*s[:,k]))
    opti.subject_to(opti.bounded( -np.array(limits['dq'])  + con_x[::2][7:]*s[:,k], dq[:,k],  limits['dq']- con_x[::2][7:]*s[:,k]))
            
    # initial constraint
    opti.subject_to( q[:,0] ==  q_init)
    opti.subject_to(dq[:,0] == dq_init)
    
    # terminal constraint
    x_err = cs.vertcat(q[:,N], dq[:,N]) - cs.vertcat(q_s, dq_s)
    opti.subject_to(  x_err == 0)
    

    # virtual reference
    position, orientation = forward_kineamtics_fcn(q_s)
    position_error_squared = cs.sum1((position-y_position_ref)**2)
    orientation_error_squared = quat_err_fcn(orientation, y_orientation_ref)
    cost += weight_orientation_error*orientation_error_squared
    cost += weight_position_error*position_error_squared
    opti.subject_to(opti.bounded(  limits['q_min'] + con_x[::2][:7]*s[:,k], q_s,   limits['q_max']-con_x[::2][:7]*s[:,k]))
    # opti.subject_to(position==y_position_ref)
    if sdf_fcn is not None:
        opti.subject_to(sdf_fcn(position[0],position[1],position[2]) + extra_distance + con_g*s[:,k] <= 0)
    
    opti.minimize(cost)
    if solver=="ipopt":
        p_opts = {
            "verbose": False,
            "print_time": 0
        }
        s_opts = {
            "max_iter": max_iter,
            "print_level": 0,
            "print_level": 0, 
            "tol": 1e-4,
            "acceptable_tol": 1e-4,
            "constr_viol_tol": 1e-4,
            "acceptable_constr_viol_tol": 1e-4,
            "dual_inf_tol": 1e-4,
            "compl_inf_tol": 1e-4,
            "acceptable_iter": 10,
            # "hessian_approximation": "limited-memory",
            "mu_strategy": "adaptive",
            "linear_solver": "mumps",
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-6,
            "warm_start_mult_bound_push": 1e-6,
            # "bound_relax_factor": 1e-9
        }
        if silent:
            p_opts["print_time"] = 0
            s_opts["print_level"] = 0
            s_opts["sb"] = "yes"
        opti.solver('ipopt', p_opts, s_opts)
    elif solver=="sqpmethod":
        p_opts = {
            "verbose": False,
            "print_time": 0,
            "print_header":False,
            "print_iteration":False,
            "print_status": False,
            "max_iter": max_iter,
            "convexify_strategy": "eigen-reflect",
            "qpsol": "osqp"  # Specify OSQP as the QP solver for SQP
        }
        s_opts = {
            "error_on_fail":False,
            "osqp.max_iter": 100,
            "osqp.verbose": False
        }
        if silent:
            p_opts["print_time"] = 0
            # s_opts["print_level"] = 0
        p_opts['qpsol_options'] = s_opts
        opti.solver('sqpmethod', p_opts)
    
    def run(q_initial, dq_initial, position_reference, orientation_reference, warmstart=None):
        opti.set_value(y_position_ref, position_reference)
        opti.set_value(y_orientation_ref, orientation_reference)
        opti.set_value(q_init, q_initial)
        opti.set_value(dq_init, dq_initial)
        
        if warmstart is not None:
            opti.set_initial(q,  warmstart['q'])
            opti.set_initial(dq,  np.zeros((ndq, N+1)))
            opti.set_initial(v,  np.zeros((nddq, N)))
            opti.set_initial(q_s,  warmstart['q_s'])
        
        start_time = time.time()
        sol = opti.solve_limited()
        solve_time = time.time()-start_time
        q_opt = sol.value(q)
        dq_opt = sol.value(dq)
        ddq_opt = sol.value(ddq)
        v_opt = sol.value(v)
        stats = sol.stats()
        q_s_opt = sol.value(q_s)
        
        return {
            'q_opt':   np.array(q_opt).T, # N_mpc x 7
            'dq_opt':  np.array(dq_opt).T, # N_mpc x 7
            'ddq_opt': np.array(ddq_opt).T, # N_mpc x 7
            'v_opt':   np.array(v_opt).T, # N_mpc x 7
            'q_s_opt': np.array(q_s_opt), # 7
            'iter':    stats["iter_count"],
            'solve_time': solve_time
        }
        
    return run

def random_config(limits):
    q_min = np.array(limits["q_min"])
    q_max = np.array(limits["q_max"])
    return np.random.rand(q_min.shape[0])*(q_max-q_min) + q_min


def make_random_test_sequence_lr(total_length=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()  # uses /dev/urandom or OS entropy
    position_references = []
    orientation_references = []

    cube_centers = [
        np.array([0.0, -0.5-0.25/2, 0.35+0.1/2]),
        np.array([0.0, -0.4, 0.7])
        # np.array([0.0,  0.3, 0.25])
    ]
    cube_sizes = [
        np.array([0.2, 0.2, 0.1]),
        np.array([0.2, 0.2, 0.2])
    ]
    half_sizes = [cube_size / 2.0 for cube_size in cube_sizes]

    base_rotation = [
        R.from_euler('XYZ', [2.6, 0, 1.5707]),
        R.from_euler('XYZ', [2.6, 0, 1.5707])
        # R.from_euler('XYZ', [-2.6, 0, -1.5707])
    ]

    for i in range(total_length):
        # Randomly select one of the two cubes
        cube_index = rng.choice([0, 1])
        if i == 0:
            cube_index = 0
        cube_center = cube_centers[cube_index]

        # Sample a position within the cube
        pos_offset = rng.uniform(-half_sizes[cube_index], half_sizes[cube_index])
        pos = cube_center + pos_offset

        # Small random perturbation around the base rotation
        # This can be interpreted as a "noise" rotation composed with the base
        # perturbation = R.from_rotvec(rng.uniform(-0.087, 0.087, size=3))  # 0.1 rad stddev noise
        # rot = perturbation * base_rotation[cube_index]
        rot = base_rotation[cube_index]
        quat_xyzw = rot.as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        position_references += [pos]
        orientation_references += [quat_wxyz]

    return position_references, orientation_references


def make_sdf_env(occupancy_3d_resolution = 0.04):
    # occupancy_3d = create_occupancy_map_3d(x_dim=2, y_dim=2, z_dim=2, resolution=0.1)
    def make_trimesh_at_position(extents, translation):
        b = trimesh.creation.box(extents=extents)
        b.apply_translation(translation)
        return b
    
    eket = trimesh.load("ikea_eket.stl")
    eket.apply_translation(np.array([0,-0.5,0]))
    
    obstacle_list = [
        eket,
        # make_trimesh_at_position(np.array([0.5,0.01,0.5]), np.array([0,0.525,0.25])),
        make_trimesh_at_position(np.array([0.5,0.2,1]), np.array([0,-0.85,0.5])),
        make_trimesh_at_position(np.array([0.75,2,0.1]), np.array([0,0,0.-0.05]))
    ]
    sdf_3d = compute_signed_distance_field_3d_from_trimeshes(size_x=2,
                                                             size_y=2,
                                                             size_z=2,
                                                             res=occupancy_3d_resolution,
                                                             trimesh_list=obstacle_list)
    # just for plotting
    occupancy_3d = (sdf_3d + occupancy_3d_resolution/2 > 0).astype(np.uint8)
    occupancy_3d = None
    
    _, casadi_sdf = create_casadi_interpolant_from_sdf_3d(sdf_3d, occupancy_3d_resolution)
    return casadi_sdf, occupancy_3d, obstacle_list, sdf_3d

def random_cl_experiment(total_length=100, noise_level_q_init=0.1, noise_level_q=0.005, noise_level_dq=0.05, noise_level_ddq=0.2, dq_rand_scale=True):
    rng = np.random.default_rng()
    
    # load precomputed fk and sdf
    fk_fcn, limits = get_robot_fk()
    casadi_sdf = get_sdf()

    # casadi_sdf = None
    run_fcn = mpc_problem(fk_fcn, limits, sdf_fcn=casadi_sdf, solver="ipopt", max_iter=200)
    
    scale = np.array([1, 1, 0, 1, 1, 1, 1])
    q_initial = np.array([-1.57079, -0.785398, 0.0, -2, 0, 1.5708, 0.785398]) + rng.uniform(-1, 1, size=7) * noise_level_q_init * scale
    position_home, orientation_home = fk_fcn(q_initial)
    position_home = np.array(position_home).flatten()
    orientation_home = np.array(orientation_home).flatten()   

    position_references, orientation_references = make_random_test_sequence_lr(total_length, rng)
    
    dq_initial = np.zeros(7)
    
    warmstart = {
        "q": np.tile(q_initial[:,None], (1,N+1)),
        # "q_s": q_ref
        "q_s": q_initial
    }
    
    res_cl = []
    assert len(position_references) == len(orientation_references), "Position and orientation references should be same len!"
    N_sim = total_length
    idx_ref = 0
    pbar = tqdm.tqdm(range(N_sim))
    for i in pbar:
        res = run_fcn(
            q_initial=q_initial,
            dq_initial=dq_initial,
            position_reference=position_references[idx_ref],
            orientation_reference=orientation_references[idx_ref],
            warmstart=warmstart)
        res_cl.append(res)
        ddq_initial = res_cl[-1]["ddq_opt"][0,:].T + rng.uniform(-1, 1, size=7) * scale * noise_level_ddq
        dq_initial  = res_cl[-1]["dq_opt"][0,:].T + rng.uniform(-1, 1, size=7) * scale * noise_level_dq
        q_initial   = res_cl[-1]["q_opt"][0,:].T + rng.uniform(-1, 1, size=7) * scale * noise_level_q #+ res_cl[-1]["dq_opt"][0,:].T + np.random.multivariate_normal(np.zeros(7), noise_level_q*np.eye(7))
        
        q_initial += dt*dq_initial
        dq_initial += dt*ddq_initial
        
        # if dq_rand_scale:
        if dq_rand_scale and rng.choice([True,False]):
            dq_initial = rng.uniform(np.zeros_like(dq_initial), np.ones_like(dq_initial))*dq_initial
        
        warmstart = None
        
        res_cl[-1]["position_reference"] = np.array(position_references[idx_ref])
        res_cl[-1]["orientation_reference"] = np.array(orientation_references[idx_ref])
        
        pr_, _ = fk_fcn(res_cl[-1]["q_s_opt"])
        progress = np.linalg.norm(np.array(pr_).flatten() - position_references[idx_ref].flatten())
        pbar.set_postfix({"progress": f"{progress:.4f}"})
        pr_init_, _ = fk_fcn(q_initial)
        # delta_to_target = np.linalg.norm(np.array(pr_init_).flatten() - position_references[idx_ref].flatten()) <= 0.02
        delta_to_target = np.linalg.norm(np.array(pr_init_).flatten() - np.array(pr_).flatten()) <= 0.05
        # delta_velocity = np.linalg.norm(res_cl[-1]["dq_opt"][0,:].T) <= 0.001
        if delta_to_target:
            idx_ref += 1
    
    print(f"Used: {idx_ref+1=} different references")
    exp={
        "limits": limits,
        "data": res_cl,
        "dt": dt,
        "N_mpc": N,
        "N_sim": len(res_cl),
    }
    
    print_solve_time_statistics(exp)
    
    return exp

def run_save(length, filename):
    exp = random_cl_experiment(length)
    np.savez_compressed(f'{filename}.npz', allow_pickle=True, **exp)
    
def run_save_animate(length, filename, noisy=True):
    if noisy:
        exp = random_cl_experiment(length)
    else:
        exp = random_cl_experiment(length, noise_level_q_init=0, noise_level_ddq=0, noise_level_dq=0, dq_rand_scale=False)
    np.savez_compressed(f'{filename}.npz', allow_pickle=True, **exp )
    
    occupancy_3d_resolution=0.04
    _, occupancy_3d, obstacle_list, _ = make_sdf_env(occupancy_3d_resolution)
    exp["occupancy_3d"] = occupancy_3d
    exp["obstacle_list"] = obstacle_list
    exp["occupancy_3d_resolution"] = occupancy_3d_resolution
    animate(exp, f'{filename}.mp4', show_plot=False)
    
    iters = []
    solve_times = []
    for d in exp["data"]:
        iters.append(d['iter'])
        solve_times.append(d['solve_time'])
    iters = np.array(iters)
    solve_times = np.array(solve_times)
    plt.close()
    plt.hist(iters, bins=50, edgecolor='black')
    plt.xlabel('Iters')
    plt.ylabel('Frequency')
    plt.title('Iters')
    plt.savefig(f'{filename}_iters.pdf', format='pdf')
    plt.close()
    plt.hist(solve_times, bins=50, edgecolor='black')
    plt.xlabel('solve times [s]')
    plt.ylabel('Frequency')
    plt.title('Solve Times')
    plt.savefig(f'{filename}_solve_times.pdf', format='pdf')
    plt.close()

if __name__=="__main__":
    
    fire.Fire({
        "run_save_animate": run_save_animate,
        "run_save": run_save,
        "export_all_constants": export_all_constants
    })