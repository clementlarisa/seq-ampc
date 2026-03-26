import numpy as np

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

from seqampc.mpcproblem import *
from seqampc.datasetutils import import_dataset, print_dataset_statistics
from seqampc.trainampc import (
    architecture_search,
    retrain_model,
    run_statistical_test as train_run_statistical_test,
    test_ampc,
    computetime_test_model,
    NeuralType,
)

def find_approximate_mpc(neural_type="MLP", dataset="latest", rnn_units=32, dense_units = (200, 400, 600, 600, 400, 200,), retrain=False, retrain_model_name="latest"):
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)
    X, U, _, _, _, _ = import_dataset(mpc, dataset)

    # define architectures to be tested
    architectures = np.array([
        [mpc.nx, 200, 400, 600, 600, 400, 200, mpc.nu*mpc.N]
        ])

    # ----------------- RNN ------------------------
    print(f"RNN Units: {rnn_units}")
    print(f"Dense Units: {dense_units}")

    hyperparameters = [
                        {"learning_rate":0.001,  "patience": 1000, "max_epochs": 100000, "batch_size": 6250},
                    ]

    model = architecture_search(mpc, X, U, neural_type=NeuralType(neural_type.lower()), hyperparameters=hyperparameters, architectures=architectures,
                                dense_units=dense_units, rnn_units=rnn_units, retrain=retrain, retrain_model_name=retrain_model_name)
    return model

def run_statistical_test(neural_type="MLP", dataset="latest", rnn_units=32, dense_units = (200, 400, 600, 600, 400, 200,), retrain_model_name="latest"):
    mpc = import_mpc(dataset, MPCQuadraticCostLxLu)
    X, U, _, _, _, _ = import_dataset(mpc, dataset, True)
    # Robustify shape for single-sample datasets
    X = np.atleast_2d(X)

    input_dim = X.shape[1]   

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
        mpc, X, U,
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
        "computetime_test_model": computetime_test_model,
        "run_statistical_test": run_statistical_test,

    })