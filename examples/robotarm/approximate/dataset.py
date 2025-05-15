import queue
from torch.utils.data import Dataset
import threading
import jax
import jax.numpy as jnp
import numpy as np
import h5py
import tqdm

class MemoryAmpcDataset(Dataset):
    def __init__(self, X, U, Y):
        self.X = X
        self.U = U
        self.Y = Y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'U': self.U[idx], 'Y': self.Y[idx]}


class H5AmpcDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, "r")
        self.X = self.h5_file["X"]
        self.U = self.h5_file["U"]
        self.Y = self.h5_file["Y"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {'X': self.X[idx], 'U': self.U[idx], 'Y': self.Y[idx]}
    
class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def compute_normalization_stats_memory(X_dataset, U_dataset, Y_dataset):
    X_scale = (np.max(X_dataset, axis=0) - np.min(X_dataset, axis=0)) / 2.0
    U_scale = (np.max(U_dataset, axis=0) - np.min(U_dataset, axis=0)) / 2.0
    Y_scale = (np.max(Y_dataset, axis=0) - np.min(Y_dataset, axis=0)) / 2.0
    X_offset = np.mean(X_dataset, axis=0)
    U_offset = np.mean(U_dataset, axis=0)
    Y_offset = np.mean(Y_dataset, axis=0)

    return {
        "x_scale": X_scale[0, :],
        "x_offset": X_offset[0, :],
        "u_scale": U_scale[0, :],
        "u_offset": U_offset[0, :],
        "y_scale": Y_scale,
        "y_offset": Y_offset
    }