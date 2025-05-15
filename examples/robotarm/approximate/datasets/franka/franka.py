import numpy as np
import math
import jax
from math import *
from datasets.franka.constants import q_min, q_max, ddq_min_max, dq_min_max
from datasets.franka.prestabilization_jax import prestabilization
import os
from pathlib import Path
import tqdm
import h5py
import jax.numpy as jnp

def f_jax(x,u):
    ddq = prestabilization(x, u).flatten()
    ddq_out = jnp.maximum(jnp.minimum(ddq, u_max_jax), u_min_jax)
    # linear system
    return jax.numpy.array([
        x[7],
        x[8],
        x[9],
        x[10],
        x[11],
        x[12],
        x[13],
        ddq_out[0],
        ddq_out[1],
        ddq_out[2],
        ddq_out[3],
        ddq_out[4],
        ddq_out[5],
        ddq_out[6]])
    
x_min = q_min + [-dq for dq in dq_min_max]
x_max = q_max + dq_min_max

x_min_jax = jnp.array(x_min)
x_max_jax = jnp.array(x_max)

u_min = [-ddq for ddq in ddq_min_max]
u_max = ddq_min_max

u_min_jax = jnp.array(u_min)
u_max_jax = jnp.array(u_max)


# def load_dataset():
#     dataset_path = Path(os.path.dirname(__file__))/"data"
#     npz_files = sorted(dataset_path.glob("*.npz"))

#     if not npz_files:
#         print(f"No .npz files found in {dataset_path}")
#         return

#     common_fields = {}
#     X_list=[]
#     U_list=[]
#     for i, file in tqdm.tqdm(enumerate(npz_files)):
#         with np.load(file, allow_pickle=True) as data:
#             if i == 0:
#                 # Store all fields except "data" from the first file
#                 common_fields = {k: v for k, v in data.items() if k != "data"}


#             for rollout in data["data"]:
#                 for mpcsol in rollout:
#                     q_opt  = mpcsol["q_opt"]  # shape (N_mpc, nq)
#                     dq_opt = mpcsol["dq_opt"]  # shape (N_mpc, nq)
#                     v_opt  = mpcsol["v_opt"]    # shape (N_mpc, nq)

#                     # Stack q_opt and dq_opt along last dimension -> (N_mpc, nq+nq)
#                     x = np.concatenate([q_opt, dq_opt], axis=-1)

#                     # Store
#                     X_list.append(x)  # Each x is (N_mpc, 2*nq)
#                     U_list.append(v_opt)  # Each v_opt is (N_mpc, nq)

#     # Finally stack all together
#     X = np.stack(X_list, axis=0)  # shape (N_total, N_mpc, 2*nq)
#     U = np.stack(U_list, axis=0)  # shape (N_total, N_mpc, nq)

#     print(f"X shape: {X.shape}")
#     print(f"U shape: {U.shape}")
    
#     return X, U
    
# def load_dataset():
#     dataset_path = Path(os.path.dirname(__file__)) / "data"
#     npz_files = sorted(dataset_path.glob("*.npz"))

#     if not npz_files:
#         print(f"No .npz files found in {dataset_path}")
#         return

#     # First pass: count total number of (rollout, mpcsol) pairs
#     total_samples = 0
#     common_fields = {}

#     for i, file in tqdm.tqdm(enumerate(npz_files), desc="Counting samples"):
#         with np.load(file, allow_pickle=True) as data:
#             if i == 0:
#                 # Store all fields except "data" from the first file
#                 common_fields = {k: v for k, v in data.items() if k != "data"}

#             for rollout in data["data"]:
#                 total_samples += len(rollout)  # Each rollout is a list of mpcsols

#     # Get shape information from first sample
#     with np.load(npz_files[0], allow_pickle=True) as data:
#         sample_mpcsol = data["data"][0][0]  # First rollout, first mpcsol
#         q_opt = sample_mpcsol["q_opt"]
#         dq_opt = sample_mpcsol["dq_opt"]
#         v_opt = sample_mpcsol["v_opt"]

#         N_mpc, nq = q_opt.shape

#     # Preallocate arrays
#     X = np.zeros((total_samples, N_mpc, 2 * nq), dtype=q_opt.dtype)
#     U = np.zeros((total_samples, N_mpc-1, nq), dtype=v_opt.dtype)

#     # Second pass: load data
#     idx = 0
#     for file in tqdm.tqdm(npz_files, desc="Loading data"):
#         with np.load(file, allow_pickle=True) as data:
#             for rollout in data["data"]:
#                 for mpcsol in rollout:
#                     q_opt = mpcsol["q_opt"]
#                     dq_opt = mpcsol["dq_opt"]
#                     v_opt = mpcsol["v_opt"]

#                     x = np.concatenate([q_opt, dq_opt], axis=-1)

#                     X[idx] = x
#                     U[idx] = v_opt
#                     idx += 1

#     print(f"X shape: {X.shape}")
#     print(f"U shape: {U.shape}")

#     return X, U

def load_dataset(max_datapoints=None):
    dataset_path = Path(os.path.dirname(__file__)) / "data"
    npz_files = sorted(dataset_path.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {dataset_path}")
        return

    # First pass: count total number of (rollout, mpcsol) pairs
    total_samples = 0
    common_fields = {}

    for i, file in enumerate(tqdm.tqdm(npz_files, desc="Counting samples")):
        with np.load(file, allow_pickle=True) as data:
            if i == 0:
                common_fields = {k: v for k, v in data.items() if k != "data"}
            for rollout in data["data"]:
                total_samples += len(rollout)

    if max_datapoints is not None:
        total_samples = min(total_samples, max_datapoints)

    # Get shape information from first sample
    with np.load(npz_files[0], allow_pickle=True) as data:
        sample_mpcsol = data["data"][0][0]
        q_opt = sample_mpcsol["q_opt"]
        dq_opt = sample_mpcsol["dq_opt"]
        v_opt = sample_mpcsol["v_opt"]
        position_ref = sample_mpcsol["position_reference"]
        orientation_ref = sample_mpcsol["orientation_reference"]

        N_mpc, nq = q_opt.shape

    # Preallocate arrays
    X = np.zeros((total_samples, N_mpc, 2 * nq), dtype=q_opt.dtype)
    Y = np.zeros((total_samples, 7), dtype=position_ref.dtype)
    U = np.zeros((total_samples, N_mpc - 1, nq), dtype=v_opt.dtype)

    # Second pass: load data
    idx = 0
    for file in tqdm.tqdm(npz_files, desc="Loading data"):
        with np.load(file, allow_pickle=True) as data:
            for rollout in data["data"]:
                for mpcsol in rollout:
                    if idx >= total_samples:
                        break  # stop once we've loaded enough samples

                    q_opt = mpcsol["q_opt"]
                    dq_opt = mpcsol["dq_opt"]
                    v_opt = mpcsol["v_opt"]
                    position_ref = mpcsol["position_reference"]
                    orientation_ref = mpcsol["orientation_reference"]

                    x = np.concatenate([q_opt, dq_opt], axis=-1)
                    y = np.concatenate([position_ref, orientation_ref], axis=-1)
                    
                    X[idx] = x
                    U[idx] = v_opt
                    Y[idx] = y
                    idx += 1
                if idx >= total_samples:
                    break
        if idx >= total_samples:
            break

    print(f"Loaded {idx} samples")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"U shape: {U.shape}")

    return X, U, Y


def load_dataset_as_rollouts(max_datapoints=None):
    dataset_path = Path(os.path.dirname(__file__)) / "data"
    npz_files = sorted(dataset_path.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {dataset_path}")
        return

    # Get shape information from first sample
    with np.load(npz_files[0], allow_pickle=True) as data:
        sample_mpcsol = data["data"][0][0]
        N_sim = data["N_sim"]
        q_opt = sample_mpcsol["q_opt"]
        dq_opt = sample_mpcsol["dq_opt"]
        v_opt = sample_mpcsol["v_opt"]
        position_ref = sample_mpcsol["position_reference"]
        orientation_ref = sample_mpcsol["orientation_reference"]

        N_mpc, nq = q_opt.shape

    # First pass: count total number of (rollout, mpcsol) pairs
    total_rollouts = 0
    common_fields = {}

    for i, file in enumerate(tqdm.tqdm(npz_files, desc="Counting samples")):
        with np.load(file, allow_pickle=True) as data:
            if i == 0:
                common_fields = {k: v for k, v in data.items() if k != "data"}
            for rollout in data["data"]:
                assert len(rollout) == N_sim
            total_rollouts += len(data["data"])

    if max_datapoints is not None:
        total_rollouts = min(total_rollouts, max_datapoints)



    # Preallocate arrays
    X = np.zeros((total_rollouts, N_sim, N_mpc, 2 * nq), dtype=q_opt.dtype)
    Y = np.zeros((total_rollouts, N_sim, 7), dtype=position_ref.dtype)
    U = np.zeros((total_rollouts, N_sim, N_mpc - 1, nq), dtype=v_opt.dtype)

    # Second pass: load data
    idx = 0
    for file in tqdm.tqdm(npz_files, desc="Loading data"):
        with np.load(file, allow_pickle=True) as data:
            for rollout in data["data"]:
                if idx >= total_rollouts:
                    break
                for simstep, mpcsol in enumerate(rollout):
                    q_opt = mpcsol["q_opt"]
                    dq_opt = mpcsol["dq_opt"]
                    v_opt = mpcsol["v_opt"]
                    position_ref = mpcsol["position_reference"]
                    orientation_ref = mpcsol["orientation_reference"]

                    x = np.concatenate([q_opt, dq_opt], axis=-1)
                    y = np.concatenate([position_ref, orientation_ref], axis=-1)
                    
                    X[idx, simstep] = x
                    U[idx, simstep] = v_opt
                    Y[idx, simstep] = y
                idx += 1
                if idx >= total_rollouts:
                    break
        if idx >= total_rollouts:
            break

    print(f"Loaded {idx} samples")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"U shape: {U.shape}")

    return X, U, Y, common_fields

def load_dataset_convert_to_h5(max_size=None, h5_file_path=Path(os.path.dirname(__file__))/"dataset.h5"):
    # Check if H5 already exists
    h5_file_path = Path(h5_file_path)  # ensure it's a Path object
    if h5_file_path.exists():
        print(f"HDF5 file already exists at {h5_file_path}. Skipping conversion ✅")
        return h5_file_path
    
    dataset_path = Path(os.path.dirname(__file__)) / "data"
    npz_files = sorted(dataset_path.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {dataset_path}")
        return

    # First pass: count total number of samples
    total_samples = 0
    common_fields = {}

    for i, file in tqdm.tqdm(enumerate(npz_files), desc="Counting samples"):
        with np.load(file, allow_pickle=True) as data:
            if i == 0:
                common_fields = {k: v for k, v in data.items() if k != "data"}

            for rollout in data["data"]:
                total_samples += len(rollout)

    # Get shape info
    with np.load(npz_files[0], allow_pickle=True) as data:
        sample_mpcsol = data["data"][0][0]
        q_opt = sample_mpcsol["q_opt"]
        dq_opt = sample_mpcsol["dq_opt"]
        v_opt = sample_mpcsol["v_opt"]

        N_mpc, nq = q_opt.shape

    print(f"Total samples: {total_samples}, N_mpc: {N_mpc}, nq: {nq}")

    # Create HDF5 file
    with h5py.File(h5_file_path, "w") as f:
        dset_X = f.create_dataset(
            "X", shape=(total_samples, N_mpc, 2*nq), dtype=q_opt.dtype
        )
        dset_U = f.create_dataset(
            "U", shape=(total_samples, N_mpc-1, nq), dtype=v_opt.dtype
        )

        # Second pass: fill the datasets
        idx = 0
        for file in tqdm.tqdm(npz_files, desc="Loading and writing data"):
            with np.load(file, allow_pickle=True) as data:
                for rollout in data["data"]:
                    for mpcsol in rollout:
                        q_opt = mpcsol["q_opt"]
                        dq_opt = mpcsol["dq_opt"]
                        v_opt = mpcsol["v_opt"]

                        x = np.concatenate([q_opt, dq_opt], axis=-1)

                        dset_X[idx] = x
                        dset_U[idx] = v_opt
                        idx += 1

    print(f"Dataset saved to {h5_file_path} ✅")

    return h5_file_path