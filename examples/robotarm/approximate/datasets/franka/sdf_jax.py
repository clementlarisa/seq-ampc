import jax
import jax.numpy as jnp


def create_jax_interpolant_from_sdf_3d(sdf_3d, resolution):
    nz, ny, nx = sdf_3d.shape
    dx = (nx - 1) * resolution
    dy = (ny - 1) * resolution
    dz = (nz - 1) * resolution

    # Define grid coordinates
    xgrid = jnp.linspace(0, dx, nx) - dx / 2
    ygrid = jnp.linspace(0, dy, ny) - dy / 2
    zgrid = jnp.linspace(0, dz, nz) - dz / 2

    # Define voxel size
    voxel_size = resolution

    # Convert to array for indexing
    sdf_3d_jax = jnp.array(sdf_3d)

    def f_sdf_this_map(x, y, z):
        # Convert (x, y, z) to indices in the sdf array
        ix = (x - xgrid[0]) / voxel_size
        iy = (y - ygrid[0]) / voxel_size
        iz = (z - zgrid[0]) / voxel_size

        # Get integer part
        i0 = jnp.floor(ix).astype(int)
        j0 = jnp.floor(iy).astype(int)
        k0 = jnp.floor(iz).astype(int)

        # Get fractional part
        dx = ix - i0
        dy = iy - j0
        dz = iz - k0

        # Clamp indices to avoid overflow
        i0 = jnp.clip(i0, 0, nx - 2)
        j0 = jnp.clip(j0, 0, ny - 2)
        k0 = jnp.clip(k0, 0, nz - 2)

        def get_val(i, j, k):
            return sdf_3d_jax[k, j, i]

        # Trilinear interpolation
        c000 = get_val(i0,     j0,     k0)
        c001 = get_val(i0,     j0,     k0 + 1)
        c010 = get_val(i0,     j0 + 1, k0)
        c011 = get_val(i0,     j0 + 1, k0 + 1)
        c100 = get_val(i0 + 1, j0,     k0)
        c101 = get_val(i0 + 1, j0,     k0 + 1)
        c110 = get_val(i0 + 1, j0 + 1, k0)
        c111 = get_val(i0 + 1, j0 + 1, k0 + 1)

        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        c = c0 * (1 - dz) + c1 * dz
        return c

    return f_sdf_this_map