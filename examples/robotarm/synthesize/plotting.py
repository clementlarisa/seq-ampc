import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.transform import Rotation as R
import tqdm
from pathlib import Path
import fire
import shutil
import subprocess

from pyrender_mpc import *

def animate(exp, outfile=None, save_plot=True, show_plot=True):

    plt.rc('axes', labelsize=16)    # font size of the x and y labels
    plt.rc('xtick', labelsize=16)   # font size of the tick labels
    plt.rc('ytick', labelsize=16)   # font size of the tick labels
    plt.rc('legend', fontsize=16)   # font size of the tick labels

    fig = plt.figure(layout='constrained', figsize=(20, 11.25))
    left_fig, right_fig = fig.subfigures(1,2,wspace=0)
    
    ax_3dplot = right_fig.add_subplot()
    ax_q, ax_dq, ax_ddq = left_fig.subplots(3,1,sharex=True)
    
    def normalize_q(q):
        q_min = np.array(exp["limits"]["q_min"]).reshape(1,-1)
        q_max = np.array(exp["limits"]["q_max"]).reshape(1,-1)
        return 2 * (q - q_min) / (q_max - q_min) - 1
    
    q_cl   = np.zeros((exp["N_sim"], 7))
    dq_cl  = np.zeros((exp["N_sim"], 7))
    ddq_cl = np.zeros((exp["N_sim"], 7))
    
    for i in range(exp["N_sim"]):
        q_cl[i,:]   = exp["data"][i]["q_opt"][0,:]
        dq_cl[i,:]  = exp["data"][i]["dq_opt"][0,:]
        ddq_cl[i,:] = exp["data"][i]["ddq_opt"][0,:]
    
    t = np.linspace(0,exp["N_sim"]*exp["dt"],exp["N_sim"]+1)
    t_MPC = np.linspace(0,exp["N_mpc"]*exp["dt"], exp["N_mpc"]+1)
    
    def pose_to_transform(position, quaternion):
        w,x,y,z = quaternion.flatten()
        rot_matrix = R.from_quat([x,y,z,w]).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = position.flatten()
        return transform
    
    n_ghosts = exp["N_mpc"]
    T_ref = pose_to_transform(exp["data"][0]["position_reference"],exp["data"][0]["orientation_reference"])
    render_frame_fcn = make_renderer(
        n_ghosts=n_ghosts,
        ghost_alpha=int(255/n_ghosts)*1.5,
        # size=(1080,1080),
        size=(320,320),
        plot_reference_cf=T_ref,
        plot_voxelmap=exp["occupancy_3d"] if "occupancy_3d" in exp.keys() else None,
        plot_voxelmap_resolution=exp["occupancy_3d_resolution"] if "occupancy_3d_resolution" in exp.keys() else None,
        plot_obstacle_list=exp["obstacle_list"] if "obstacle_list" in exp.keys() else None
        )
        
    def update_robot_rendering_frame(i):
        ax_3dplot.clear()
        traj_ol = exp['data'][i]["q_opt"]
        T_ref = pose_to_transform(exp["data"][i]["position_reference"],exp["data"][i]["orientation_reference"])
        frame = render_frame_fcn(traj_ol[::(exp["N_mpc"]+1)//(n_ghosts+1)], q_cl[:i+1,:], plot_reference_cf=T_ref)
        ax_3dplot.imshow(frame)
        ax_3dplot.axis('off')
    
    def update_ddq_graph(i):
        ax_ddq.clear()
        ax_ddq.step(t[:i+1], ddq_cl[:i+1,0]/exp["limits"]["ddq"][0], color='C0', linewidth=3, label=r"$\ddot{q}_1$")
        ax_ddq.step(t[:i+1], ddq_cl[:i+1,1]/exp["limits"]["ddq"][1], color='C1', linewidth=3, label=r"$\ddot{q}_2$")
        ax_ddq.step(t[:i+1], ddq_cl[:i+1,2]/exp["limits"]["ddq"][2], color='C2', linewidth=3, label=r"$\ddot{q}_3$")
        ax_ddq.step(t[:i+1], ddq_cl[:i+1,3]/exp["limits"]["ddq"][3], color='C3', linewidth=3, label=r"$\ddot{q}_4$")
        ax_ddq.step(t[:i+1], ddq_cl[:i+1,3]/exp["limits"]["ddq"][4], color='C4', linewidth=3, label=r"$\ddot{q}_5$")
        ax_ddq.step(t[:i+1], ddq_cl[:i+1,3]/exp["limits"]["ddq"][5], color='C5', linewidth=3, label=r"$\ddot{q}_6$")
        ax_ddq.step(t[:i+1], ddq_cl[:i+1,3]/exp["limits"]["ddq"][6], color='C6', linewidth=3, label=r"$\ddot{q}_7$")
        
        t_now = i*exp["dt"]
        ax_ddq.step(t_MPC+t_now, np.append(exp['data'][i]["ddq_opt"][0,0],exp['data'][i]["ddq_opt"][:,0])/exp["limits"]["ddq"][0], color='C0', alpha=0.5, linestyle=(0,(5,1)))
        ax_ddq.step(t_MPC+t_now, np.append(exp['data'][i]["ddq_opt"][0,1],exp['data'][i]["ddq_opt"][:,1])/exp["limits"]["ddq"][1], color='C1', alpha=0.5, linestyle=(0,(5,1)))
        ax_ddq.step(t_MPC+t_now, np.append(exp['data'][i]["ddq_opt"][0,2],exp['data'][i]["ddq_opt"][:,2])/exp["limits"]["ddq"][2], color='C2', alpha=0.5, linestyle=(0,(5,1)))
        ax_ddq.step(t_MPC+t_now, np.append(exp['data'][i]["ddq_opt"][0,3],exp['data'][i]["ddq_opt"][:,3])/exp["limits"]["ddq"][3], color='C3', alpha=0.5, linestyle=(0,(5,1)))
        ax_ddq.step(t_MPC+t_now, np.append(exp['data'][i]["ddq_opt"][0,4],exp['data'][i]["ddq_opt"][:,4])/exp["limits"]["ddq"][4], color='C4', alpha=0.5, linestyle=(0,(5,1)))
        ax_ddq.step(t_MPC+t_now, np.append(exp['data'][i]["ddq_opt"][0,5],exp['data'][i]["ddq_opt"][:,5])/exp["limits"]["ddq"][5], color='C5', alpha=0.5, linestyle=(0,(5,1)))
        ax_ddq.step(t_MPC+t_now, np.append(exp['data'][i]["ddq_opt"][0,6],exp['data'][i]["ddq_opt"][:,6])/exp["limits"]["ddq"][6], color='C6', alpha=0.5, linestyle=(0,(5,1)))
        
        ax_ddq.set_ylabel('joint acceleration normalized')
        ax_ddq.set_yticks([-1,1])
        ax_ddq.set_yticklabels([r"$\ddot{q}_{min}$", r"$\ddot{q}_{max}$"])
        
        ax_ddq.fill_between([t[0], t[-1]+t_MPC[-1]], [1, 1], [1.1, 1.1] , color="k", alpha=0.1, linewidth=0)
        ax_ddq.fill_between([t[0], t[-1]+t_MPC[-1]], [-1, -1], [-1.1,-1.1], color="k", alpha=0.1, linewidth=0)
        
        ax_ddq.set_ylim([-1.1, 1.1])
        ax_ddq.set_xlim([t[0],t[-1]+t_MPC[-1]])
        ax_ddq.axvline(x = i*exp["dt"], color = 'b', linewidth=3)
        ax_ddq.grid()
    
    def update_dq_graph(i):
        ax_dq.clear()
        ax_dq.plot(t[:i+1], dq_cl[:i+1, 0]/exp["limits"]["dq"][0], color='C0', linewidth=3)
        ax_dq.plot(t[:i+1], dq_cl[:i+1, 1]/exp["limits"]["dq"][1], color='C1', linewidth=3)
        ax_dq.plot(t[:i+1], dq_cl[:i+1, 2]/exp["limits"]["dq"][2], color='C2', linewidth=3)
        ax_dq.plot(t[:i+1], dq_cl[:i+1, 3]/exp["limits"]["dq"][3], color='C3', linewidth=3)
        ax_dq.plot(t[:i+1], dq_cl[:i+1, 4]/exp["limits"]["dq"][4], color='C4', linewidth=3)
        ax_dq.plot(t[:i+1], dq_cl[:i+1, 5]/exp["limits"]["dq"][5], color='C5', linewidth=3)
        ax_dq.plot(t[:i+1], dq_cl[:i+1, 6]/exp["limits"]["dq"][6], color='C6', linewidth=3)
        
        t_now = i*exp["dt"]
        ax_dq.plot(t_MPC+t_now, exp['data'][i]["dq_opt"][:,0]/exp["limits"]["dq"][0], color='C0', alpha=0.5, linestyle=(0,(5,1)))
        ax_dq.plot(t_MPC+t_now, exp['data'][i]["dq_opt"][:,1]/exp["limits"]["dq"][1], color='C1', alpha=0.5, linestyle=(0,(5,1)))
        ax_dq.plot(t_MPC+t_now, exp['data'][i]["dq_opt"][:,2]/exp["limits"]["dq"][2], color='C2', alpha=0.5, linestyle=(0,(5,1)))
        ax_dq.plot(t_MPC+t_now, exp['data'][i]["dq_opt"][:,3]/exp["limits"]["dq"][3], color='C3', alpha=0.5, linestyle=(0,(5,1)))
        ax_dq.plot(t_MPC+t_now, exp['data'][i]["dq_opt"][:,4]/exp["limits"]["dq"][4], color='C4', alpha=0.5, linestyle=(0,(5,1)))
        ax_dq.plot(t_MPC+t_now, exp['data'][i]["dq_opt"][:,5]/exp["limits"]["dq"][5], color='C5', alpha=0.5, linestyle=(0,(5,1)))
        ax_dq.plot(t_MPC+t_now, exp['data'][i]["dq_opt"][:,6]/exp["limits"]["dq"][6], color='C6', alpha=0.5, linestyle=(0,(5,1)))
        
        ax_dq.fill_between([t[0], t[-1]+t_MPC[-1]], [1, 1], [1.1, 1.1] , color="k", alpha=0.1, linewidth=0)
        ax_dq.fill_between([t[0], t[-1]+t_MPC[-1]], [-1, -1], [-1.1,-1.1], color="k", alpha=0.1, linewidth=0)
        
        ax_dq.set_ylabel('joint velocity normalized')
        ax_dq.set_yticks([-1,1])
        ax_dq.set_yticklabels([r"$\dot{q}_{min}$", r"$\dot{q}_{max}$"])
        
        ax_dq.set_ylim([-1.1,1.1])
        ax_dq.set_xlim([t[0],t[-1]+t_MPC[-1]])
        ax_dq.axvline(x = i*exp["dt"], color = 'b', linewidth=3)
        ax_dq.grid()
    
    def update_q_graph(i):
        ax_q.clear()
        
        q_cl_norm = normalize_q(q_cl[:i+1,:])
        ax_q.plot(t[:i+1], q_cl_norm[:, 0], color='C0', linewidth=3, label="$q_1$")
        ax_q.plot(t[:i+1], q_cl_norm[:, 1], color='C1', linewidth=3, label="$q_2$")
        ax_q.plot(t[:i+1], q_cl_norm[:, 2], color='C2', linewidth=3, label="$q_3$")
        ax_q.plot(t[:i+1], q_cl_norm[:, 3], color='C3', linewidth=3, label="$q_4$")
        ax_q.plot(t[:i+1], q_cl_norm[:, 4], color='C4', linewidth=3, label="$q_5$")
        ax_q.plot(t[:i+1], q_cl_norm[:, 5], color='C5', linewidth=3, label="$q_6$")
        ax_q.plot(t[:i+1], q_cl_norm[:, 6], color='C6', linewidth=3, label="$q_7$")
        
        t_now = i*exp["dt"]
        q_ol_norm = normalize_q(exp['data'][i]["q_opt"][:,:])
        ax_q.plot(t_MPC+t_now, q_ol_norm[:,0], color='C0', alpha=0.5, linestyle=(0,(5,1)))
        ax_q.plot(t_MPC+t_now, q_ol_norm[:,1], color='C1', alpha=0.5, linestyle=(0,(5,1)))
        ax_q.plot(t_MPC+t_now, q_ol_norm[:,2], color='C2', alpha=0.5, linestyle=(0,(5,1)))
        ax_q.plot(t_MPC+t_now, q_ol_norm[:,3], color='C3', alpha=0.5, linestyle=(0,(5,1)))
        ax_q.plot(t_MPC+t_now, q_ol_norm[:,4], color='C4', alpha=0.5, linestyle=(0,(5,1)))
        ax_q.plot(t_MPC+t_now, q_ol_norm[:,5], color='C5', alpha=0.5, linestyle=(0,(5,1)))
        ax_q.plot(t_MPC+t_now, q_ol_norm[:,6], color='C6', alpha=0.5, linestyle=(0,(5,1)))
        
        ax_q.fill_between([t[0], t[-1]+t_MPC[-1]], [1, 1], [1.1, 1.1] , color="k", alpha=0.1, linewidth=0)
        ax_q.fill_between([t[0], t[-1]+t_MPC[-1]], [-1, -1], [-1.1,-1.1], color="k", alpha=0.1, linewidth=0)
        
        ax_q.set_ylabel('joint position normalized')
        ax_q.set_yticks([-1,1])
        ax_q.set_yticklabels([r"$q_{min}$", r"$q_{max}$"])
        
        ax_q.set_ylim([-1.1,1.1])
        # ax_q.set_xlim([t[0],t[-1]+t_MPC[-1]])
        ax_q.set_xlim([t[0],t[-1]+t_MPC[-1]])
        ax_q.axvline(x = i*exp["dt"], color = 'b', linewidth=3)
        ax_q.grid()
        ax_q.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    def update_graphs(i):
        update_q_graph(i)
        update_dq_graph(i)
        update_ddq_graph(i)
    
        
    def update_plot(i):
        # update_3d_plot(i)
        update_robot_rendering_frame(i)
        update_graphs(i)


    steps = exp["N_sim"]
    t_end = steps * exp["dt"]
    dt_ms = int(t_end / steps * 1000)
    fps = int(1000 / dt_ms)

    # Create temp frame directory
    if outfile is None:
        outfile = "animation_output.mp4"
    frame_dir = outfile.replace(".mp4", "")
    os.makedirs(frame_dir, exist_ok=True)

    print("[INFO] Rendering frames...")
    for i in tqdm.tqdm(range(steps)):
        update_plot(i)
        frame_path = os.path.join(frame_dir, f"frame_{i:05d}.jpg")
        fig.savefig(frame_path, dpi=96)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found in PATH. Please install it or make sure it's available in your environment.")

    print("[INFO] Combining frames into video...")
    cmd = [
        ffmpeg_path,
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'frame_%05d.jpg'),
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        outfile
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[INFO] Removing temporary frame folder...")
    shutil.rmtree(frame_dir)
    
    
    
def print_solve_time_statistics(exp):
    
    data = exp['data']
    iters = []
    solve_times = []
    for s in data:
        iters.append(s["iter"])
        solve_times.append(s["solve_time"])
        
    iters = np.array(iters)
    solve_times = np.array(solve_times)
    mean_iters = np.mean(iters)
    mean_solve_time = np.mean(solve_times)
    max_solve_time = np.max(solve_times)
    mean_iter_time = np.sum(solve_times)/np.sum(iters)
    print(f"\nSolver time statistics:")
    print(f"tmean no of iterations: {mean_iters}")
    print(f"tmean solve time:       {mean_solve_time*1000:.3f}ms")
    print(f"tmax solve time:        {max_solve_time*1000:.3f}ms")
    print(f"tmea iter time:         {mean_iter_time*1000:.3f}ms")


def plot_npz(filename, idx=0):
    dataset_path = Path(filename)
    print(f"Loading dataset from {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    
    exp = {key: data[key] for key in data.files}
    if 'data' in exp:
        exp['data'] = exp['data'][idx]
    exp['limits'] = np.load("franka_limits.npz")
    
    from mpc import make_sdf_env
    occupancy_3d_resolution=0.04
    _, occupancy_3d, obstacle_list, _ = make_sdf_env(occupancy_3d_resolution)
    exp["occupancy_3d"] = occupancy_3d
    exp["obstacle_list"] = obstacle_list
    exp["occupancy_3d_resolution"] = occupancy_3d_resolution
    
    export_folder =str(dataset_path)[:-4]
    print(f"Exporting to: {export_folder}/{idx}.mp4")
    os.makedirs(export_folder, exist_ok=True)
    animate(exp, f"{export_folder}/{idx}.mp4", show_plot=False)
    

if __name__=="__main__":
    fire.Fire({
        "plot_npz": plot_npz
    })