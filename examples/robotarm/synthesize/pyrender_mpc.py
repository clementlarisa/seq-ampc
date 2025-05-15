import os
import math
import numpy as np
import imageio
import trimesh
import pyribbit
import tqdm
from urchin import URDF
from scipy.spatial.transform import Rotation as R

NUM_PANDA_JOINTS = 7

import os
if os.getenv('PYOPENGL_PLATFORM') is None:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    print("PYOPENGL_PLATFORM was not set. Now set to 'egl'.")
else:
    print("PYOPENGL_PLATFORM is already set to:", os.environ['PYOPENGL_PLATFORM'])

def compute_camera_pose(azimuth, elevation, distance, target=np.array([0.0, 0.0, 0.0])):
    eye = np.array([
        target[0] + distance * np.cos(azimuth) * np.cos(elevation),
        target[1] + distance * np.sin(azimuth),
        target[2] + distance * np.cos(azimuth) * np.sin(elevation)
    ])
    
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    up = np.array([0.0, 0.0, 1.0])
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    R_cam = np.column_stack((right, true_up, -forward))
    
    T = np.eye(4)
    T[:3, :3] = R_cam
    T[:3, 3] = eye
    
    return T

def make_pose_xyzrpy(x, y, z, roll, pitch, yaw):
    rotation = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    return pose

def create_coordinate_frame_mesh(size=0.1, radius=0.005, sections=32):
    def create_axis_cylinder(color):
        cyl = trimesh.creation.cylinder(radius=radius, height=size, sections=sections)
        cyl.apply_translation([0, 0, size/2])
        num_faces = cyl.faces.shape[0]
        cyl.visual.face_colors = np.tile(np.array(color), (num_faces, 1))
        return cyl

    cyl_z = create_axis_cylinder([0, 0, 255, 255])
    
    cyl_x = create_axis_cylinder([255, 0, 0, 255])
    T_x = np.eye(4)
    T_x[:3, :3] = R.from_euler('y', np.pi/2).as_matrix()
    cyl_x.apply_transform(T_x)
    
    cyl_y = create_axis_cylinder([0, 255, 0, 255])
    T_y = np.eye(4)
    T_y[:3, :3] = R.from_euler('x', -np.pi/2).as_matrix()
    cyl_y.apply_transform(T_y)
    
    frame_mesh = trimesh.util.concatenate([cyl_x, cyl_y, cyl_z])
    return frame_mesh

def build_scene(show_coord_frames=False):
    scene = pyribbit.Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]) * 0.6)
    
    light = pyribbit.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5)
    scene.add(light, pose=make_pose_xyzrpy(0, 0, 2, 0, 0, 0))
    point_light = pyribbit.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    point_light_node = scene.add(point_light, pose=make_pose_xyzrpy(0,0,1.4,0,0,0))
    cam_light = pyribbit.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    
    cam = pyribbit.PerspectiveCamera(yfov=np.pi / 3.0)
    azimuth = np.deg2rad(15)
    elevation = np.deg2rad(30)
    distance = 1.2
    target = np.array([0,-0.3,0.3])
    cam_pose = compute_camera_pose(azimuth, elevation, distance, target)
    scene.add(cam, pose=cam_pose)
    cam_light_node = scene.add(cam_light, pose=cam_pose)
    
    boxf_trimesh = trimesh.creation.box(extents=np.array([0.75,2,0.1]))
    boxf_face_colors = np.tile(np.array([0.5, 0.5, 0.5, 1.0]), (boxf_trimesh.faces.shape[0],1))
    boxf_trimesh.visual.face_colors = boxf_face_colors
    boxf_mesh = pyribbit.Mesh.from_trimesh(boxf_trimesh, smooth=False)
    boxf_node = pyribbit.Node(mesh=boxf_mesh, translation=np.array([0,0,-0.05]))
    scene.add_node(boxf_node)
    
    return scene

def add_coordinate_frame_to_scene(scene, size=0.1, radius=0.005):
    cf_mesh = create_coordinate_frame_mesh(size=size, radius=radius)
    cf_pyribbit = pyribbit.Mesh.from_trimesh(cf_mesh, smooth=False)
    coordinate_frame_node = scene.add(cf_pyribbit)
    return coordinate_frame_node

def add_robot_to_scene(scene, robot, alpha):
    fk_meshes = robot.visual_trimesh_fk()
    base_pose = np.eye(4)
    node_map = {}
    for tm, pose in fk_meshes.items():
        if tm.visual.kind == "texture":
            tm.visual.material.diffuse[-1] = alpha
        elif tm.visual.kind == 'face':
            if tm.visual.face_colors.ndim == 2:
                tm.visual.face_colors[:, -1] = alpha
            else:
                tm.visual.face_colors[-1] = alpha
        elif tm.visual.kind == 'vertex':
            if tm.visual.vertex_colors.ndim == 2:
                tm.visual.vertex_colors[:, -1] = alpha
            else:
                tm.visual.vertex_colors[-1] = alpha

        world_pose = base_pose @ pose
        mesh = pyribbit.Mesh.from_trimesh(tm, smooth=False)
        node = scene.add(mesh, pose=world_pose)
        node_map[tm] = node
    
    return node_map

def update_robot_pose(robot, robot_obj, cfg):
    fk = robot.visual_trimesh_fk(cfg=cfg)
    for mesh in fk:
        pose = fk[mesh]
        robot_obj[mesh].matrix = pose

def create_thick_line_mesh(points, radius=0.003, color=(255, 0, 0, 255), sections=16):

    if len(points) < 2:
        return None
    
    cylinder_list = []
    for i in range(len(points) - 1):
        p0 = np.array(points[i], dtype=float)
        p1 = np.array(points[i+1], dtype=float)
        seg_vec = p1 - p0
        seg_length = np.linalg.norm(seg_vec)
        if seg_length < 1e-9:
            continue
        
        cyl = trimesh.creation.cylinder(
            radius=radius,
            height=seg_length,
            sections=sections
        )
        
        face_count = cyl.faces.shape[0]
        cyl.visual.face_colors = np.tile(np.array(color, dtype=np.uint8),
                                         (face_count, 1))
        
        seg_dir = seg_vec / seg_length
        z_axis = np.array([0, 0, 1], dtype=float)

        axis = np.cross(z_axis, seg_dir)
        angle = np.arccos(np.clip(np.dot(z_axis, seg_dir), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-9:
            T_rotate = np.eye(4)
            if angle > 1e-6:  # ~180 deg
                T_rotate[:3,:3] = R.from_euler('x', np.pi).as_matrix()
        else:
            axis = axis / np.linalg.norm(axis)
            R_mat = R.from_rotvec(axis * angle).as_matrix()
            T_rotate = np.eye(4)
            T_rotate[:3,:3] = R_mat
        
        T_translate = np.eye(4)
        T_translate[:3, 3] = p0
        
        transform = T_translate @ T_rotate
        cyl.apply_transform(transform)
        
        cylinder_list.append(cyl)
    
    if not cylinder_list:
        return None
    
    mesh = trimesh.util.concatenate(cylinder_list)
    return mesh

def extrude_circle_along_path(path_points, circle_radius=1.0, circle_segments=16, color=(255, 0, 0, 255)):

    path_points = np.asarray(path_points, dtype=float)
    n_path = len(path_points)
    if n_path < 2:
        raise ValueError("Path must contain at least two points.")
    
    n_circle = circle_segments

    tangents = []
    for i in range(n_path):
        if i == 0:
            tangent = path_points[i+1] - path_points[i]
        elif i == n_path - 1:
            tangent = path_points[i] - path_points[i-1]
        else:
            forward = path_points[i+1] - path_points[i]
            backward = path_points[i] - path_points[i-1]
            tangent = 0.5 * (forward + backward)
        
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-8:
            raise ValueError("Degenerate segment or repeated path point detected.")
        tangents.append(tangent / tangent_norm)
    
    tangents = np.array(tangents)

    global_up = np.array([0.0, 0.0, 1.0])
    
    normals = []
    binormals = []
    for i in range(n_path):
        t = tangents[i]
        normal = np.cross(t, global_up)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            alt_up = np.array([0.0, 1.0, 0.0])
            normal = np.cross(t, alt_up)
            norm_len = np.linalg.norm(normal)
        
        normal /= norm_len
        binormal = np.cross(t, normal)
        binormal /= np.linalg.norm(binormal)

        normals.append(normal)
        binormals.append(binormal)
        
    normals = np.array(normals)
    binormals = np.array(binormals)

    angles = np.linspace(0, 2.0 * np.pi, n_circle, endpoint=False)
    all_rings = []
    for i in range(n_path):
        center = path_points[i]
        n = normals[i]
        b = binormals[i]
        
        ring_points = []
        for theta in angles:
            x_local = circle_radius * np.cos(theta)
            y_local = circle_radius * np.sin(theta)
            ring_point = center + x_local * n + y_local * b
            ring_points.append(ring_point)
        all_rings.append(ring_points)

    all_rings = np.array(all_rings)  # shape: (n_path, n_circle, 3)

    vertices = all_rings.reshape((-1, 3))

    faces = []
    for i in range(n_path - 1):
        ring_start = i * n_circle
        next_ring_start = (i + 1) * n_circle

        for j in range(n_circle):
            i0 = ring_start + j
            i1 = ring_start + (j + 1) % n_circle
            i2 = next_ring_start + j
            i3 = next_ring_start + (j + 1) % n_circle

            faces.append([i0, i1, i2])
            faces.append([i2, i1, i3])

    faces = np.array(faces, dtype=np.int32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mesh.visual.face_colors = np.tile(color, (len(faces), 1))
    
    return mesh

def make_renderer(n_ghosts=5, ghost_alpha=int(0.2*255), plot_reference_cf=None, size=(1920,1080), plot_voxelmap=None, plot_voxelmap_resolution=1.0, plot_obstacle_list=None):
    robot = URDF.load("franka_panda_urdf/panda.urdf")  
    width, height = size
    renderer = pyribbit.OffscreenRenderer(viewport_width=width, viewport_height=height)
    scene = build_scene()
    
    main_alpha = 255              # Main robot: fully opaque
    main_robot = add_robot_to_scene(scene, robot, main_alpha)
    ghost_list = []
    for i in range(n_ghosts):
        ghost_list.append(add_robot_to_scene(scene, robot, ghost_alpha))
    
    current_cf = add_coordinate_frame_to_scene(scene)
    virtual_reference_cf = add_coordinate_frame_to_scene(scene)
    if plot_reference_cf is not None:
        reference_cf = add_coordinate_frame_to_scene(scene)
        reference_cf.matrix = plot_reference_cf
    
    # world_cf = add_coordinate_frame_to_scene(scene, 1, 0.01)
        
    if plot_voxelmap is not None:
        Nz, Ny, Nx = plot_voxelmap.shape
        offset_x = Nx * plot_voxelmap_resolution / 2.0
        offset_y = Ny * plot_voxelmap_resolution / 2.0
        offset_z = Nz * plot_voxelmap_resolution / 2.0
        
        cube = trimesh.creation.box(extents=[plot_voxelmap_resolution, plot_voxelmap_resolution, plot_voxelmap_resolution])
        cube.visual.vertex_colors = [253,212,143, 128]
        indices = np.argwhere(plot_voxelmap == 1)
        tfs = np.tile(np.eye(4), (len(indices), 1, 1))
        for i, (iz, iy, ix) in enumerate(indices):
            cx = (ix + 0.5) * plot_voxelmap_resolution - offset_x
            cy = (iy + 0.5) * plot_voxelmap_resolution - offset_y
            cz = (iz + 0.5) * plot_voxelmap_resolution - offset_z
            tfs[i, :3, 3] = [cx, cy, cz]
        mesh = pyribbit.Mesh.from_trimesh(cube, poses=tfs)
        scene.add(mesh)
    
    if plot_obstacle_list is not None:
        for o in plot_obstacle_list:
            scene.add(pyribbit.Mesh.from_trimesh(o))
        
    for l in robot.links:
        if l.name == "panda_grasptarget":
            gripper_link = l
            break
        
    line_node = scene.add(
        pyribbit.Mesh.from_trimesh(trimesh.creation.box([0,0,0])),  # dummy initial mesh
        pose=np.eye(4)
    )

    def render_frame(traj_ol, traj_cl=None, plot_reference_cf=None):
        main_cfg = {f"panda_joint{i+1}": traj_ol[0, i] for i in range(NUM_PANDA_JOINTS)}
        update_robot_pose(robot, main_robot, main_cfg)
        for i in range(n_ghosts):
            ghost_cfg = {f"panda_joint{j+1}": traj_ol[0 + i + 1, j] for j in range(NUM_PANDA_JOINTS)}
            update_robot_pose(robot, ghost_list[i], ghost_cfg)
        
        fk_main = robot.link_fk(cfg=main_cfg)
        current_cf.matrix = fk_main[gripper_link]
            
        fk_ghost = robot.link_fk(cfg=ghost_cfg)
        virtual_reference_cf.matrix = fk_ghost[gripper_link]
        
        if plot_reference_cf is not None:
            reference_cf.matrix = plot_reference_cf
        
        if traj_cl is not None:
            ee_positions = []
            for i in range(traj_cl.shape[0]):
                cl_cfg = {f"panda_joint{j+1}": traj_cl[i, j] for j in range(NUM_PANDA_JOINTS)}
                fk_cl = robot.link_fk(cfg=cl_cfg)
                ee_pos = fk_cl[gripper_link][:3, 3]
                ee_positions.append(ee_pos)
            if len(ee_positions) > 1:
                filtered_positions = [ee_positions[0]]
                for i in range(1, len(ee_positions)):
                    prev = filtered_positions[-1]
                    curr = ee_positions[i]
                    tangent = curr - prev
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm >= 1e-6:
                        filtered_positions.append(curr)
                    else:
                        pass

                if len(filtered_positions) > 1:
                    tube_mesh = extrude_circle_along_path(filtered_positions, 0.005, color=[255, 0, 0, 255])
                    if tube_mesh is not None:
                        line_node.mesh = pyribbit.Mesh.from_trimesh(tube_mesh, smooth=False)
        flags = pyribbit.constants.RenderFlags.SHADOWS_DIRECTIONAL | pyribbit.constants.RenderFlags.SHADOWS_SPOT
        rendered_image, _ = renderer.render(scene, flags=flags)
        return rendered_image
    
    return render_frame


if __name__ == "__main__":
    n_ghosts = 20
    N_mpc = 20
    render_frame_fcn = make_renderer(n_ghosts=n_ghosts)
    
    num_steps = 100
    traj = np.linspace(-np.pi, np.pi, num_steps).reshape(num_steps, 1)
    traj = np.repeat(traj, NUM_PANDA_JOINTS, axis=1)
    traj[:,1] = np.pi/2
    # traj[:,4:] = 0
    traj[:,5] = np.pi
    
    frames = []
    for idx in tqdm.tqdm(range(num_steps - N_mpc)):
        rendered_image = render_frame_fcn(traj[idx:idx+N_mpc+1:(N_mpc+1)//(n_ghosts+1)])
        frames.append(rendered_image)
    
    video_filename = "panda_trajectory.mp4"
    imageio.mimwrite(video_filename, frames, fps=20)
    print(f"Video saved to {video_filename}")