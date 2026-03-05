from pathlib import Path
from datetime import datetime
import os
import errno
import numpy as np
import shutil

from tqdm import tqdm

from .mpcproblem import *

def append_to_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename):
    Nsamples = np.shape(x0dataset)[0]
    p = Path("/share/mihaela-larisa.clement/soeampc-data/archive").joinpath(filename, "data")
    p.mkdir(parents=True,exist_ok=True)
    f = open(p.joinpath("x0.txt"), 'a')
    np.savetxt(f,  x0dataset,    delimiter=",")
    f.close()
    f = open(p.joinpath("X.txt"), 'a')
    np.savetxt(f,  np.reshape( Xdataset, ( Nsamples, mpc.nx*(mpc.N+1))),  delimiter=",")
    f.close()
    f = open(p.joinpath("U.txt"), 'a')
    np.savetxt(f  ,  np.reshape( Udataset, ( Nsamples, mpc.nu*mpc.N)),    delimiter=",")
    f.close()
    f = open(p.joinpath("ct.txt"), 'a')
    np.savetxt(f,  computetimes, delimiter=",")
    f.close()

def get_date_string():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def export_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, filename, barefilename=False):    
    date = get_date_string()
    Nsamples = np.shape(x0dataset)[0]

    # print("\nExporting Dataset with Nvalid",Nsamples,"feasible samples\n")
    
    datasetname = mpc.name+"_N_"+str(Nsamples)+"_"+filename+date

    if barefilename:
        datasetname=filename
    
    p = Path("/share/mihaela-larisa.clement/soeampc-data/archive").joinpath(datasetname, "data")
    p.mkdir(parents=True,exist_ok=True)
    np.savetxt(p.joinpath("x0.txt") ,  x0dataset,    delimiter=",")
    np.savetxt(p.joinpath("X.txt")  ,  np.reshape( Xdataset, ( Nsamples, mpc.nx*(mpc.N+1))),  delimiter=",")
    np.savetxt(p.joinpath("U.txt")  ,  np.reshape( Udataset, ( Nsamples, mpc.nu*mpc.N)),    delimiter=",")
    np.savetxt(p.joinpath("ct.txt") ,  computetimes, delimiter=",")

    mpc.savetxt(Path("/share/mihaela-larisa.clement/soeampc-data/archive").joinpath(datasetname, "parameters"))

    print("Exported to directory:\n\t",  Path("/share/mihaela-larisa.clement/soeampc-data/archive").joinpath(datasetname).absolute(),"\n")

    target = datasetname
    link_name=Path("/share/mihaela-larisa.clement/soeampc-data/archive").joinpath("latest")
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
    return datasetname


def import_dataset(mpc, file="latest", return_aux=False):
    """
    Imports dataset from archive/<file>/data.

    If return_aux=True, also returns:
      - P_obstacles (np.ndarray or None)
      - N_active (np.ndarray or None)

    Both auxiliary files are expected at archive/<file>/ and have a header
    in the first line, which is ignored.
    """
    root = Path("/share/mihaela-larisa.clement/soeampc-data/archive").joinpath(file)
    p = root.joinpath("data")

    x0raw = np.loadtxt(p.joinpath("x0.txt"), delimiter=",")
    Xraw = np.loadtxt(p.joinpath("X.txt"), delimiter=",")
    Uraw = np.loadtxt(p.joinpath("U.txt"), delimiter=",")
    computetimes = np.loadtxt(p.joinpath("ct.txt"), delimiter=",")

    # Robust shapes for Nsamples == 1
    # x0raw = np.atleast_2d(x0raw)
    # Xraw = np.atleast_2d(Xraw)
    # Uraw = np.atleast_2d(Uraw)
    # computetimes = np.atleast_1d(computetimes)

    Nsamples = x0raw.shape[0]
    x0dataset = x0raw.reshape(Nsamples, mpc.nx)
    Udataset = Uraw.reshape(Nsamples, mpc.N, mpc.nu)
    Xdataset = Xraw.reshape(Nsamples, mpc.N + 1, mpc.nx)

    if not return_aux:
        return x0dataset, Udataset, Xdataset, computetimes

    def _load_optional_headered_array(path: Path, expected_rows: int, expected_cols: int | None = None):
        """
        Load optional root-level file with first line header.
        Returns None if file does not exist.

        If expected_cols is None: returns 1D array of length expected_rows.
        If expected_cols is set: returns 2D array of shape (expected_rows, expected_cols).
        """
        if not path.exists():
            return None

        with open(path, "r") as f:
            lines = f.readlines()

        if len(lines) <= 1:
            # header only
            if expected_cols is None:
                return np.array([])
            return np.empty((0, expected_cols))

        data_lines = lines[1:]  # skip header
        vals = np.loadtxt(data_lines)  # add delimiter if your files are comma-separated

        if expected_cols is None:
            vals = np.atleast_1d(vals)
            if vals.shape[0] != expected_rows:
                raise ValueError(
                    f"{path.name} has {vals.shape[0]} entries after header, but dataset has {expected_rows} samples."
                )
            return vals

        # expected matrix
        vals = np.atleast_2d(vals)

        # Handle single-sample case where loadtxt can return shape (expected_cols,)
        if vals.shape == (1, expected_cols):
            pass
        elif vals.shape[0] == expected_rows and vals.shape[1] == expected_cols:
            pass
        elif vals.ndim == 2 and vals.shape[0] == expected_cols and expected_rows == 1:
            # rare transpose issue for single row; normalize
            vals = vals.reshape(1, expected_cols)
        else:
            raise ValueError(
                f"{path.name} expected shape ({expected_rows}, {expected_cols}), got {vals.shape}"
            )

        if vals.shape[0] != expected_rows or vals.shape[1] != expected_cols:
            raise ValueError(
                f"{path.name} expected shape ({expected_rows}, {expected_cols}), got {vals.shape}"
            )

        return vals

    P_obstacles = _load_optional_headered_array(
        root.joinpath("P_obstacles.txt"),
        expected_rows=Nsamples,
        expected_cols=5,
    )

    N_active = _load_optional_headered_array(
        root.joinpath("N_active.txt"),
        expected_rows=Nsamples,
        expected_cols=None,
    )

    return x0dataset, Udataset, Xdataset, computetimes, P_obstacles, N_active

def mergesamples(folder_names, new_dataset_name=get_date_string(), remove_after_merge=False):
    """
    Merge multiple sample folders into one dataset.

    Merges:
      - data/x0.txt, data/X.txt, data/U.txt, data/ct.txt (existing behavior)
      - root/P_obstacles.txt and root/N_active.txt (new behavior)
        with first line (header) skipped in each source file and a single header
        written in the merged output if the file exists in at least one source.
    """
    p = Path("/share/mihaela-larisa.clement/soeampc-data/archive")
    mpc = import_mpc(folder_names[0], MPCQuadraticCostLxLu)

    def _read_header_and_data_lines(path: Path):
        """
        Returns (header_line_or_none, data_lines_list).
        Header is the first line. Data lines are remaining lines.
        Missing file -> (None, []).
        """
        if not path.exists():
            return None, []

        with open(path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            return None, []

        header = lines[0]
        data_lines = lines[1:] if len(lines) > 1 else []
        return header, data_lines

    def _append_aux_file(src_root: Path, dst_root: Path, filename: str, state: dict):
        """
        Append src_root/filename -> dst_root/filename, skipping src header.
        Writes exactly one header into destination (taken from first found source).
        state tracks whether header was already written.
        """
        src = src_root.joinpath(filename)
        dst = dst_root.joinpath(filename)

        header, data_lines = _read_header_and_data_lines(src)
        if header is None:
            return  # file missing in this source, silently skip

        # Ensure destination parent exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Write header once
        if not state.get(filename, False):
            with open(dst, "w") as f:
                f.write(header)
                f.writelines(data_lines)
            state[filename] = True
        else:
            with open(dst, "a") as f:
                f.writelines(data_lines)

    print("file:", folder_names[0])

    # Import first dataset (with aux if available)
    x0dataset, Udataset, Xdataset, computetimes, P_obstacles, N_active = import_dataset(
        mpc, folder_names[0], return_aux=True
    )
    Nsamples = x0dataset.shape[0]

    exporttempfilename = export_dataset(
        mpc,
        x0dataset,
        Udataset,
        Xdataset,
        computetimes,
        mpc.name + "_N_temp_merged_" + str(new_dataset_name),
        barefilename=True
    )

    # Merge auxiliary root-level files into temp export folder
    temp_root = p.joinpath(exporttempfilename)
    header_written = {}
    first_root = p.joinpath(folder_names[0])

    _append_aux_file(first_root, temp_root, "P_obstacles.txt", header_written)
    _append_aux_file(first_root, temp_root, "N_active.txt", header_written)

    for f in tqdm(folder_names[1:]):
        print("file:", f)
        x0dataset, Udataset, Xdataset, computetimes, P_obstacles, N_active = import_dataset(
            mpc, f, return_aux=True
        )
        Nsamples = Nsamples + x0dataset.shape[0]

        # Existing merge for data/*.txt
        append_to_dataset(mpc, x0dataset, Udataset, Xdataset, computetimes, exporttempfilename)

        # New merge for root-level aux files
        src_root = p.joinpath(f)
        _append_aux_file(src_root, temp_root, "P_obstacles.txt", header_written)
        _append_aux_file(src_root, temp_root, "N_active.txt", header_written)

    print("\ncollected a total of ", str(Nsamples), "sample points")
    exportfilename = mpc.name + "_N_" + str(Nsamples) + "_merged_" + str(new_dataset_name)
    os.rename(p.joinpath(exporttempfilename), p.joinpath(exportfilename))
    print("\nExported merged dataset to:\n")
    print("\t", exportfilename, "\n")

    if remove_after_merge:
        print("\n\nRemoving Folders:\n")
        print("\t", folder_names)
        for f in folder_names:
            shutil.rmtree(p.joinpath(f), ignore_errors=True)

    return exportfilename

def merge_parallel_jobs(merge_list, new_dataset_name=""):
    """merges datasets matching merge_list into single dataset    
    """
    print("\n\n===============================================")
    print("Merging datasets for arrayjobids"+str(merge_list))
    print("===============================================\n")

    path=Path("/share/mihaela-larisa.clement/soeampc-data/archive")
    # print([name for name in os.listdir(p)])

    merge_folders = [folder_name for folder_name in os.listdir(path) if any(str(dataset_name) in folder_name for dataset_name in merge_list) ] 
    
    return mergesamples(merge_folders, new_dataset_name=new_dataset_name, remove_after_merge=True)

def merge_single_parallel_job(dataset_name):
    return merge_parallel_jobs([dataset_name], new_dataset_name=dataset_name)

def print_compute_time_statistics(compute_times):
    print(f"Compute time mean ={ np.mean(compute_times) :.5f} [s]")
    print(f"Compute time max 3 = { np.sort(compute_times[np.argpartition(compute_times, -3)[-3:]])} [s]")
    print(f"Compute time sum = { np.sum(compute_times)/60/60  :.5f} [core-h]")

def mpc_dataset_import(dataset_name, mpc_type=MPCQuadraticCostLxLu):
    mpc = import_mpc(dataset_name, mpc_type)
    X0, V, X, compute_times = import_dataset(mpc, dataset_name)
    return mpc, X0, V, X, compute_times

def print_dataset_statistics(dataset_name):
    mpc = import_mpc(dataset_name, MPCQuadraticCostLxLu)
    x0dataset, Udataset, Xdataset, compute_times = import_dataset(mpc, dataset_name)
    print_compute_time_statistics(compute_times)