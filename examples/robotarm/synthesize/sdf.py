import numpy as np
from scipy import ndimage
import casadi as ca
import trimesh
import tqdm

def create_occupancy_map_3d(x_dim, y_dim, z_dim, resolution):
    nx = int(x_dim / resolution)
    ny = int(y_dim / resolution)
    nz = int(z_dim / resolution)
    occupancy = np.zeros((nz, ny, nx), dtype=np.uint8)
    z0, z1 = nz // 4, nz // 4 + 5
    y0, y1 = ny // 4, ny // 4 + 10
    x0, x1 = nx // 4, nx // 4 + 8
    occupancy[z0:z1, y0:y1, x0:x1] = 1
    return occupancy

def compute_signed_distance_field_3d(occupancy_3d):
    dist_to_obs = ndimage.distance_transform_edt(occupancy_3d == 0)
    dist_to_free = ndimage.distance_transform_edt(occupancy_3d == 1)
    sdf = np.where(occupancy_3d == 0, dist_to_obs, -dist_to_free)
    return sdf

def compute_signed_distance_field_3d_from_trimeshes(size_x, size_y, size_z, res, trimesh_list):
    Nx = int(size_x / res)
    Ny = int(size_y / res)
    Nz = int(size_z / res)
    x_coords = np.linspace(-size_x/2 + res/2, size_x/2 - res/2, Nx)
    y_coords = np.linspace(-size_y/2 + res/2, size_y/2 - res/2, Ny)
    z_coords = np.linspace(-size_z/2 + res/2, size_z/2 - res/2, Nz)
    voxel_centers = np.zeros((Nx*Ny*Nz, 3))
    idx = 0
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                voxel_centers[idx] = [x_coords[i], y_coords[j], z_coords[k]]
                idx += 1
    sdf_3d_flat = -np.inf*np.ones(len(voxel_centers), dtype=bool)
    for m in tqdm.tqdm(trimesh_list):
        dist = trimesh.proximity.ProximityQuery(m).signed_distance(voxel_centers)
        sdf_3d_flat = np.maximum(sdf_3d_flat, dist)

    sdf_3d = sdf_3d_flat.reshape((Nz, Ny, Nx))
    return sdf_3d


def create_casadi_interpolant_from_sdf_3d(sdf_3d, resolution):
    nz, ny, nx = sdf_3d.shape
    dx = (nx - 1) * resolution
    dy = (ny - 1) * resolution
    dz = (nz - 1) * resolution
    xgrid = np.linspace(0, dx, nx) - dx/2
    ygrid = np.linspace(0, dy, ny) - dy/2
    zgrid = np.linspace(0, dz, nz) - dz/2
    data_3d_for_interp = np.transpose(sdf_3d, (2, 1, 0))
    data_flat = data_3d_for_interp.ravel(order='F')
    data_sym_flat = ca.MX.sym('sd', nx*ny*nz)
    lut = ca.interpolant('SDF_3D', 'linear', [xgrid, ygrid, zgrid])
    x_sym = ca.MX.sym('x')
    y_sym = ca.MX.sym('y')
    z_sym = ca.MX.sym('z')
    # data_sym_flat = ca.reshape(data_sym, (nx*ny*nz, 1))
    sdf_value_sym = lut(ca.vertcat(x_sym, y_sym, z_sym), data_sym_flat)
    f_sdf = ca.Function('f_sdf_3d',[x_sym, y_sym, z_sym, data_sym_flat],[sdf_value_sym],['x','y','z','sd'],['sdf_value'])
    # sdf_3d_transposed = np.transpose(sdf_3d, (2,1,0))
    f_sdf_this_map = ca.Function('f_sdf_this_map',[x_sym, y_sym, z_sym],[f_sdf(x_sym, y_sym, z_sym, data_flat)],['x','y','z'],['sdf_value'])
    return f_sdf, f_sdf_this_map

if __name__ == "__main__":
    width, height, depth = 10.0, 10.0, 5.0
    resolution = 0.1
    occupancy_3d = create_occupancy_map_3d(width, height, depth, resolution)
    sdf_3d = compute_signed_distance_field_3d(occupancy_3d)
    f_sdf, f_sdf_this_map = create_casadi_interpolant_from_sdf_3d(sdf_3d, resolution)
    val = f_sdf_this_map(0.0, 0.0, 0.0)
    print(val)