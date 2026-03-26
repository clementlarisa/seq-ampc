
import os

from seqampc.simple_rnn_ar import RNN_AR_RolloutModel

# hard-disable XLA paths that trigger PTX tempfiles
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_ENABLE_XLA"] = "0"
from enum import Enum

import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
tf.config.optimizer.set_jit(False)
import numpy as np
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
import keras.backend as backend
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import time

import math

from .RNN_rollout import RNNRolloutModel
from .datasetutils import print_compute_time_statistics, mpc_dataset_import
from .mpcproblem import MPCQuadraticCostLxLu
from .config import MODELS_DIR, LEARNING_CURVES_DIR

from pathlib import Path
from datetime import datetime
import os
import errno

class NeuralType(Enum):
    MLP = "mlp"
    LSTM = "lstm"
    RNN = "rnn"
    RNN_AR = "rnn_ar"

def import_model(modelname="latest"):
    p = MODELS_DIR.joinpath(modelname)

    # Keras v3 native format
    if p.is_file() and p.suffix == ".keras":
        print(f"Loading keras model {modelname}")
        return keras.models.load_model(p)

    # Legacy H5
    if p.is_file() and p.suffix == ".h5":
        return keras.models.load_model(p)

     # TF SavedModel directory
    if p.is_dir() and (p / "saved_model.pb").exists():
        # Use legacy Keras 2 loader
        import tf_keras  # provided by the `tf-keras` package
        return tf_keras.models.load_model(str(p))

    raise ValueError(f"Unrecognized model format at: {p}")

def export_model(model, modelname):
    """exports tensorflow keras model
    """
    p = MODELS_DIR
    model.save(p.joinpath(modelname))
    link_name=p.joinpath("latest")
    target=modelname
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def export_model_mpc(mpc, model, modelname):
    export_model(model, modelname)
    mpc.savetxt(MODELS_DIR.joinpath(modelname, "mpc_parameters"))

def import_model_mpc(modelname, mpcclass=MPCQuadraticCostLxLu):
    p = MODELS_DIR.joinpath(modelname, "mpc_parameters")
    mpc = mpcclass.genfromtxt(p)
    model = import_model(modelname)
    return mpc, model


def generate_model(traindata, neural_type, architecture, output_shape, rnn_units=32, dense_units=(50,)):
    """returns a fully connected feedforward NN normalized to traindata
    
    Input layer is normalizer to traindata, hidden layers are tanh activation,
    output layer is linear activation with N*nu neurons, where N is MPC horizon,
    reshape layer is appended.

    Args:
        traindata:
            numpy array of initial conditions, traindata.shape = (Nsamples, nx)
        architecture:
            array of layer widths: [nx, N_hidden_1, N_hidden_2, ..., N*nu]
        output_shape:
            shape_like of output, e.g. (N, nu)
    Returns:
        tensorflow keras model
    """
    if neural_type == NeuralType.MLP:
        X_normalizer = layers.Normalization(input_shape=[architecture[0], ], axis=None)
        X_normalizer.adapt(traindata)

        model = keras.Sequential()
        model.add(X_normalizer)
        for units in architecture[:-1]:
            initializer = tf.keras.initializers.GlorotNormal()
            model.add(layers.Dense(units=units, activation="tanh", kernel_initializer=initializer))

        model.add(layers.Dense(units=architecture[-1], activation="linear"))
        model.add(layers.Reshape(output_shape))

    elif neural_type == NeuralType.RNN:
        # traindata is expected shape (Nsamples, Tin, n_features) after expand_dims in architecture_search
        # output_shape is expected (N, nu) for raw Y or (1, N, nu) for expanded RNN Y
        if len(output_shape) == 3:
            # Y_train was expanded to (batch, 1, N, nu)
            rollout_horizon = int(output_shape[1])   # N
            out_dim = int(output_shape[2])           # nu
        elif len(output_shape) == 2:
            rollout_horizon = int(output_shape[0])   # N
            out_dim = int(output_shape[1])           # nu
        else:
            raise ValueError(f"Unsupported RNN output_shape={output_shape}")

        model = RNNRolloutModel(rollout_horizon, rnn_units=rnn_units, dense_units=dense_units, out_dim=out_dim)
        model.adapt_normalizer(traindata)
        model.build(traindata.shape)

    elif neural_type == NeuralType.RNN_AR:
        T = traindata.shape[1]
        model = RNN_AR_RolloutModel(T, rnn_units=rnn_units, dense_units=dense_units, out_dim=3)
        model.adapt_normalizer(traindata)
        model.build(traindata.shape)

    loss='mean_absolute_error'

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mse', 'mae'],
        jit_compile=False,
    )

    return model

def train_model(neural_type, model, X_train, Y_train, batch_size=int(1e4), max_epochs=int(1e3), patience=int(1e3), learning_rate=1e-3, rnn_units=32, dense_units=(50,), model_name_prefix=None):
    """trains a given model on X,Y dataset

    Args:
        model:
            tensorflow keras model
        X_train:
            training data of initial conditions x0
        Y_train:
            training data of predicted input sequences
        batch_size:
            batch size used for learning
        max_epochs:
            number of epochs used for learning
        patience:
            if loss is not reduced for patience epochs, training is aborted
        learning_rate:
            learning_rate used for training
    Returns:
        trained tensorflow keras model
    """

    def _fmt_units(units):
        if units is None:
            return "none"
        if isinstance(units, (list, tuple)):
            return "x".join(str(int(u)) for u in units)
        return str(int(units))

    model.optimizer.learning_rate.assign(learning_rate)

    overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = patience, restore_best_weights=True)
    rnn_str = _fmt_units(rnn_units)
    dense_str = _fmt_units(dense_units)

    base_model_name = (
        f"{neural_type}"
        f"_rnn{rnn_str}"
        f"_dense{dense_str}"
        f"_bs{batch_size}"
        f"_ep{max_epochs}"
        f"_pat{patience}"
        f"_lr{learning_rate:g}"
    )

    if model_name_prefix:
        safe_prefix = str(model_name_prefix).replace(" ", "_").replace("/", "_")
        model_name = f"{safe_prefix}_{base_model_name}"
    else:
        model_name = base_model_name

    checkpoint_filepath = MODELS_DIR.joinpath(
        model_name + "_checkpoint.keras"
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_freq="epoch",
        save_best_only=True,
    )

    learning_curve_callback = tf.keras.callbacks.CSVLogger(
        str(LEARNING_CURVES_DIR / f"{model_name}.csv")
    )
    history = model.fit(
        X_train,
        Y_train,
        verbose=1,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split = 0.1,
        callbacks=[overfitCallback, model_checkpoint_callback, learning_curve_callback]
        )
    
    return model

def retrain_model(dataset="latest", model_name="latest", batch_size=int(1e4), max_epochs=int(1e3), patience=int(1e3), learning_rate=1e-3):
    """retrains a given model on X, Y dataset
    
    Args:
        model:
            tensorflow keras class object

        architecture_string:
            string used for model saving
        batch_size:
            batch size used for learning
        max_epochs:
            number of epochs used for learning
        patience:
            if loss is not reduced for patience epochs, training is aborted
        learning_rate:
            learning_rate used for training
    Returns:
        tensorflow keras model
    """
    mpc, X, Y, _, mpc_compute_times = mpc_dataset_import(dataset)
    model = import_model(modelname=model_name)
    architecture_string=model_name.split('_',1)[0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    model = train_model(model=model, X_train=X_train, Y_train=Y_train, batch_size=batch_size, max_epochs=max_epochs, patience=patience, learning_rate=learning_rate )
    testresult, mu = statistical_test(mpc, model, X_test, Y_test)
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    modelname = architecture_string + '_mu=' + ('%.2f' % mu) + '_' + date
    export_model(model, modelname)
    return model

def run_statistical_test(mpc, X, Y, neural_type, architectures, mu_crit=0.6, p_testpoints=int(1e3), retrain_model_name="latest"):
    print("\nperforming statistical test\n")
    print(f"{retrain_model_name}")

    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(X, Y, test_size=0.1, random_state=42)

    X_train = X_train_raw
    Y_train = Y_train_raw
    X_test  = X_test_raw
    Y_test  = Y_test_raw

    output_shape=Y_train.shape[1:]
    p = MODELS_DIR
    p.mkdir(parents=True,exist_ok=True)
    for a in architectures:
        expected_input_dim = X.shape[-1]  # works for flat X before RNN expand_dims
        if a[0] != expected_input_dim:
            raise Exception(
                f'architecture input neurons ({a[0]}) do not match feature dimension ({expected_input_dim})'
            )
        if a[-1] != mpc.nu*mpc.N:
            raise Exception('architecture does not have nu*N output neurons')
        print("\n\n===============================================")
        print(f"Statistical test on {retrain_model_name}")
        print("===============================================\n")
        if neural_type in [NeuralType.RNN, NeuralType.RNN_AR]:
            X_train = np.expand_dims(X_train_raw, axis=1)
            Y_train = np.expand_dims(Y_train_raw, axis=1)
            X_test = np.expand_dims(X_test_raw, axis=1)
            Y_test = np.expand_dims(Y_test_raw, axis=1)
            model = import_model(retrain_model_name)
        elif neural_type == NeuralType.MLP:
            model = import_model(retrain_model_name)
            
        model.summary()

        testresult, mu = statistical_test(
                        mpc,
                        model,
                        X_test_raw,
                        Y_test_raw,
                        p=p_testpoints,
                        mu_crit=mu_crit
                    )
    return None

def architecture_search(mpc, X, Y, neural_type, architectures, hyperparameters, rnn_units=32, dense_units=(50,), mu_crit=0.6, p_testpoints=int(10e3), retrain=False, retrain_model_name="latest", model_name_prefix=None):
    """Crude search for "good enough" approximator for predicting Y from X

    The list of architectures is traversed until a model achieves at least the desired mu_crit.
    The hyperparameter list is traversed for each architecture, this can be used to manually schedule a decaying learning rate. The statistical_test function is invoked after each hyperparameter dict from the list and the result is saved to disk.
    The function returns the first architecture from the list that achieves mu_crit.

    Args:
        mpc:
            instance of mpc class
        X:
            array of initial conditions x0
        Y: 
            array of corresponding predicted input sequences
        architectures:
            array of archictetures, each architecture is an array of layer widths, e.g. `architectures=np.array([[mpc.nx, 10, mpc.nu*mpc.N],[mpc.nx, 20, mpc.nu*mpc.N]])`
        hyperparameters:
            array of dicts with keys "learning_rates", "patience", "max_epochs" and "batch_size"

    Returns:
        tensorflow keras model that achieves mu_crit or None, if no model achieves mu_crit.        
    """
    print("\nperforming architecture search\n")

    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(X, Y, test_size=0.1, random_state=42)

    X_train = X_train_raw
    Y_train = Y_train_raw
    X_test  = X_test_raw
    Y_test  = Y_test_raw

    output_shape=Y_train.shape[1:]
    p = MODELS_DIR
    p.mkdir(parents=True,exist_ok=True)
    for a in architectures:
        expected_input_dim = X.shape[-1]  # works for flat X before RNN expand_dims
        if a[0] != expected_input_dim:
            raise Exception(
                f'architecture input neurons ({a[0]}) do not match feature dimension ({expected_input_dim})'
            )
        if a[-1] != mpc.nu*mpc.N:
            raise Exception('architecture does not have nu*N output neurons')
        print("\n\n===============================================")
        print(f"Training {neural_type} Model")
        print("===============================================\n")
        if neural_type in [NeuralType.RNN, NeuralType.RNN_AR]:
            X_train = np.expand_dims(X_train_raw, axis=1)
            Y_train = np.expand_dims(Y_train_raw, axis=1)
            X_test = np.expand_dims(X_test_raw, axis=1)
            Y_test = np.expand_dims(Y_test_raw, axis=1)

            if retrain:
                model = import_model(retrain_model_name)
            else:
                model = generate_model(X_train, neural_type, a, output_shape, dense_units=dense_units, rnn_units=rnn_units)
        elif neural_type == NeuralType.MLP:
            if retrain:
                model = import_model(retrain_model_name)
            else:
                model = generate_model(X_train, neural_type, a, output_shape)
        model.summary()
        tic = time.time()
        for hp in hyperparameters:
            print("training with hyperparameters:",hp)
            model = train_model(
                neural_type,
                model, X_train,
                Y_train,
                batch_size=hp["batch_size"],
                max_epochs=hp["max_epochs"],
                patience=hp["patience"],
                learning_rate=hp["learning_rate"],
                rnn_units=rnn_units, dense_units=dense_units,
                model_name_prefix=model_name_prefix)
            testresult, mu = statistical_test(
                            mpc,
                            model,
                            X_test_raw,
                            Y_test_raw,
                            p=p_testpoints,
                            mu_crit=mu_crit
                        )
            date = datetime.now().strftime("%Y%m%d-%H%M%S")
            arch_str = '-'.join([str(d) for d in a])

            prefix = ""
            if model_name_prefix:
                # optional sanitization
                safe_prefix = str(model_name_prefix).replace(" ", "_").replace("/", "_")
                prefix = safe_prefix + "_"

            modelname = f"{prefix}{arch_str}_mu={mu:.2f}_{date}.keras"
            export_model(model, modelname)

            print(f"Training time so far was {time.time()-tic} [s]")
    return None


def statistical_test(mpc, model, testpoints_X, testpoints_V, p=int(1e3), delta_h=0.10, mu_crit=0.80):
    """Tests if model yields a feasible solution to mpc in mu_crit fraction of feasible set.

    Supports augmented NN inputs:
      - testpoints_X may have shape (Nsamples, n_features) with n_features >= mpc.nx
      - the model receives the full feature vector
      - MPC rollout/feasibility uses only the first mpc.nx entries as the physical state x0

    Uses Hoeffdings inequality on indicator function I. If I(x0) = 1 iff model(x0) is a feasible
    solution to mpc problem. testpoints_X should be iid samples from feasible set of mpc (or
    augmented features whose first mpc.nx entries are iid states from the feasible set).
    """
    p = min(np.shape(testpoints_X)[0], p)
    print("\noffline testing on \n\tp = ", p)

    # Basic shape guard
    if np.shape(testpoints_X)[-1] < mpc.nx:
        raise ValueError(
            f"testpoints_X feature dim {np.shape(testpoints_X)[-1]} is smaller than mpc.nx={mpc.nx}"
        )
    
    rng = np.random.default_rng(seed=42)
    p = min(testpoints_X.shape[0], p)
    ids = rng.choice(testpoints_X.shape[0], size=p, replace=False)

    testpoints_X_rand = testpoints_X[ids]
    testpoints_V_rand = testpoints_V[ids]   

    I = np.zeros(p)
    dist = np.zeros((p, mpc.nu))

    inference_times = np.zeros(p)
    forward_sim_times = np.zeros(p)

    model_input_rank = None

    try:
        input_shape = getattr(model, "input_shape", None)
        if input_shape is not None:
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            model_input_rank = len(input_shape)
    except Exception:
        model_input_rank = None

    if model_input_rank is None:
        cls_name = model.__class__.__name__
        if cls_name in ("RNNRolloutModel", "RNN_AR_RolloutModel"):
            model_input_rank = 3   # (batch, time, features)
        else:
            model_input_rank = 2   # fallback for MLP

    for j in tqdm(range(p)):
        # Full feature vector for NN inference (may include appended obstacle features)
        x_feat = np.asarray(testpoints_X_rand[j])
 
        x_feat = np.squeeze(x_feat)

        # Raw x0 for MPC simulator/feasibility (must remain the true state dimension)
        x0 = np.asarray(x_feat[:mpc.nx], dtype=float).reshape(mpc.nx,)

        # Ground-truth control sequence
        Vtrue = np.asarray(testpoints_V_rand[j])
        Vtrue = np.squeeze(Vtrue)  # ensure (N, nu)

        tic = time.time()

        # Model input formatting
        if model_input_rank == 3:
            # RNN case: (batch, T, n_features) with T=1
            x_batch = x_feat[None, None, :]
        elif model_input_rank == 2:
            # MLP case: (batch, n_features)
            x_batch = x_feat[None, :]
        else:
            raise ValueError(
                f"Unsupported model input rank: {model_input_rank}, input_shape={input_shape}"
            )

        V = model(x_batch).numpy()
        V = np.squeeze(V, axis=0)   # remove batch dim
        V = np.squeeze(V)           # remove possible T=1 dim for RNN outputs
        inference_times[j] = time.time() - tic

        V = np.asarray(V, dtype=float).reshape(mpc.N, mpc.nu)

        tic = time.time()
        X = mpc.forward_simulate_trajectory_clipped_inputs(x0, V)  # x0 is true state only
        forward_sim_times[j] = time.time() - tic

        I[j] = mpc.feasible(X, V, only_states=True)
        dist[j] = np.linalg.norm(V - Vtrue, np.inf, 0)

    mu = np.mean(I)
    print("\t mean(I) =", mu)

    epsilon = math.sqrt(-math.log(delta_h / 2) / (2 * p))
    print("\t epsilon =", epsilon)

    if len(I[I == 1]) > 0:
        worst_case_passing_dist = np.max(dist[I == 1], 0)
        print("\t worst case passing dist (I==1): V-Vtrue =", worst_case_passing_dist)
    else:
        print("\t worst case passing dist (I==1) not computed, no testpoint passed the test")

    if len(I[I == 0]) > 0:
        best_case_not_passing_dist = np.min(dist[I == 0], 0)
        print("\t best case not passing dist (I==0): V-Vtrue =", best_case_not_passing_dist)
    else:
        print("\t best case not passing dist (I==0) not computed, all points passed the test")

    print("\ninference time statistics:")
    print_compute_time_statistics(inference_times)
    print("\nforward simulation time statistics:")
    print_compute_time_statistics(forward_sim_times)

    if mu_crit <= mu - epsilon:
        print("test passed\n")
        return True, mu

    print("test failed for mu_crit <= mu - epsilon with", mu_crit, "!<=", mu - epsilon, "\n")
    return False, mu

def test_ampc(dataset="latest", model_name="latest", p=int(1e4)):
    """Manually performs statistical test on ampc with given dataset
    
    Args:
        dataset:
            name the dataset to be used
        model_name:
            name of the model to be used
        p:
            number of samples to be evaluated in statistical test

    """
    mpc, X, Y, _, mpc_compute_times = mpc_dataset_import(dataset)

    print("\nmpc compute time statistics:")
    print_compute_time_statistics(mpc_compute_times)
    
    model = import_model(modelname=model_name)
    model.summary()
    statistical_test(mpc, model, X, Y, p=p)


def computetime_test_model(dataset="latest", model_name="latest", N_samples = int(10e3)):
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print("WARNING: N_samples exceeds size of dataset, will use N_samples =", N_samples,"instead")
    model = import_model(modelname=model_name)

    tic = time.time()
    model.predict(X[:N_samples])
    duration = time.time() - tic
    print(f"mean duration .predict() was {duration/N_samples*1000} [ms]")
    
    tic = time.time()
    model.predict(X[:N_samples], batch_size=1, max_queue_size=1)
    duration = time.time() - tic
    print(f"mean duration .predict(batch_size=1) was {duration/N_samples*1000} [ms]")