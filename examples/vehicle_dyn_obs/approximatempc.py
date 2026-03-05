# import tensorflow.keras as keras
# import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_ENABLE_XLA"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

import fire

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

from soeampc.trainampc import (
    architecture_search,
    retrain_model,
    run_statistical_test as train_run_statistical_test,
    test_ampc,
    computetime_test_model,
    NeuralType,
)
from soeampc.mpcproblem import *
from soeampc.datasetutils import import_dataset, print_dataset_statistics

def find_approximate_mpc(neural_type="MLP", dataset="latest", rnn_units=32, dense_units = (200, 400, 600, 600, 400, 200,), retrain=False, retrain_model_name="latest"):
    # import latest dataset
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)

    # import_dataset returns: x0, U, Xtraj, ct, P_obstacles, N_active
    X, U, _, _, P_obstacles, _ = import_dataset(mpc, dataset, True)

    if P_obstacles is None:
        raise ValueError("P_obstacles.txt not found in dataset, but this training run requires it.")

    # Robustify shape for single-sample datasets
    # X = np.atleast_2d(X)
    # P_obstacles = np.atleast_2d(P_obstacles)

    # If single sample and np.atleast_2d made it (1, N) that's fine.
    # We expect P_obstacles to have 5 columns originally; drop the last one -> 4 columns.
    if P_obstacles.shape[1] < 5:
        raise ValueError(
            f"Expected P_obstacles to have at least 5 columns before dropping the last one, got shape {P_obstacles.shape}"
        )

    print("X shape before concat:", X.shape)
    print("X first row before concat:", X[0])

    print("P_obstacles shape before drop:", P_obstacles.shape)
    print("P_obstacles first row before drop:", P_obstacles[0])

    # Drop last column
    P_obstacles = P_obstacles[:, :-1]

    if P_obstacles.shape[1] != 4:
        raise ValueError(
            f"After dropping last column, expected P_obstacles to have 4 columns, got shape {P_obstacles.shape}"
        )

    # Consistency check: same number of samples
    if X.shape[0] != P_obstacles.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X.shape[0]} rows, P_obstacles has {P_obstacles.shape[0]} rows"
        )

    X_aug = np.concatenate([X, P_obstacles], axis=1)

    print("X_aug shape after concat:", X_aug.shape)
    print("X_aug first row after concat:", X_aug[0])

    input_dim = X_aug.shape[1]   # 12

    # define architectures to be tested
    architectures = np.array([
        [input_dim, 200, 400, 600, 600, 400, 200, mpc.nu*mpc.N]
    ])
    print(f"architectures: {architectures}")

    # ----------------- RNN ------------------------
    print(f"RNN Units: {rnn_units}")
    print(f"Dense Units: {dense_units}")

    hyperparameters = [
        {"learning_rate":0.001, "patience": 1000, "max_epochs": 100000, "batch_size": 1000},
    ]

    model = architecture_search(
        mpc, X_aug, U,
        neural_type=NeuralType(neural_type.lower()),
        hyperparameters=hyperparameters,
        architectures=architectures,
        dense_units=dense_units,
        rnn_units=rnn_units,
        retrain=retrain,
        retrain_model_name=retrain_model_name,
        model_name_prefix="vehicle_8state_new"
    )
    return model

def run_statistical_test(neural_type="MLP", dataset="latest", rnn_units=32, dense_units = (200, 400, 600, 600, 400, 200,), retrain_model_name="latest"):
    # import latest dataset
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)

    # import_dataset returns: x0, U, Xtraj, ct, P_obstacles, N_active
    X, U, _, _, P_obstacles, _ = import_dataset(mpc, dataset, True)

    if P_obstacles is None:
        raise ValueError("P_obstacles.txt not found in dataset, but this training run requires it.")

    # Robustify shape for single-sample datasets
    # X = np.atleast_2d(X)
    # P_obstacles = np.atleast_2d(P_obstacles)

    # If single sample and np.atleast_2d made it (1, N) that's fine.
    # We expect P_obstacles to have 5 columns originally; drop the last one -> 4 columns.
    if P_obstacles.shape[1] < 5:
        raise ValueError(
            f"Expected P_obstacles to have at least 5 columns before dropping the last one, got shape {P_obstacles.shape}"
        )

    print("X shape before concat:", X.shape)
    print("X first row before concat:", X[0])

    print("P_obstacles shape before drop:", P_obstacles.shape)
    print("P_obstacles first row before drop:", P_obstacles[0])

    # Drop last column
    P_obstacles = P_obstacles[:, :-1]

    if P_obstacles.shape[1] != 4:
        raise ValueError(
            f"After dropping last column, expected P_obstacles to have 4 columns, got shape {P_obstacles.shape}"
        )

    # Consistency check: same number of samples
    if X.shape[0] != P_obstacles.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X.shape[0]} rows, P_obstacles has {P_obstacles.shape[0]} rows"
        )

    X_aug = np.concatenate([X, P_obstacles], axis=1)

    print("X_aug shape after concat:", X_aug.shape)
    print("X_aug first row after concat:", X_aug[0])

    input_dim = X_aug.shape[1]   # 12

    # define architectures to be tested
    architectures = np.array([
        [input_dim, 200, 400, 600, 600, 400, 200, mpc.nu*mpc.N]
    ])
    print(f"architectures: {architectures}")

    # ----------------- RNN ------------------------
    print(f"RNN Units: {rnn_units}")
    print(f"Dense Units: {dense_units}")

    hyperparameters = [
        {"learning_rate":0.001, "patience": 1000, "max_epochs": 100000, "batch_size": 1000},
    ]

    model = train_run_statistical_test(
        mpc, X_aug, U,
        neural_type=NeuralType(neural_type.lower()),
        architectures=architectures,
        retrain_model_name=retrain_model_name
        )
    return model

if __name__=="__main__":
    fire.Fire({
        "find_approximate_mpc": find_approximate_mpc,
        "retrain_model": retrain_model,
        "test_ampc": test_ampc,
        "print_dataset_statistics": print_dataset_statistics,
        "computetime_test_model":   computetime_test_model,
        "run_statistical_test": run_statistical_test,
    })