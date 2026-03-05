# Vehicle Example (Kinematic Bicycle) — SOEAMPC

This folder mirrors the structure of the `examples/quadcopter` example, but uses a **kinematic bicycle model** (ground vehicle) with a **pre-stabilizing feedback** and the same SOEAMPC dataset workflow.

## Folder structure

- `dynamics/`
  - `f.py`  
    Vehicle dynamics `xdot = f(x,u)` used by MPC and simulation.
- `mpc_parameters/`
  - `Q.txt, R.txt, P.txt`  
    Quadratic cost matrices
  - `K.txt, Kinit.txt, Kdelta.txt`  
    Linear feedback gains (see notes below)
  - `Lx.txt, Lu.txt, Ls.txt`  
    Constraint matrices for one-sided constraints `C*[x;s] + D*v <= 1`
  - `alpha.txt, alpha_s.txt`  
    Terminal set size parameters
  - `Tf.txt`  
    MPC horizon length
  - `rho_c.txt, wbar.txt`  
    Parameters for the auxiliary state `s`
- `samplempc.py`  
  Generates MPC solutions for random initial states and exports datasets.
- `safeonlineevaluation.py`  
  Runs closed-loop safety online evaluation (same pattern as quadcopter).
- `plot.py`  
  Vehicle-specific plotting helpers (`plot_vehicle_ol`, `plot_vehicle_ol_V`, `plot_vehicle_cl`).
- `compute_pre_stabilization.py`  
  Linearize + compute LQR gains and export `K/Kinit/Kdelta` into `mpc_parameters/`.

---

## Model overview

State (nx=4):
- `x = [p_x, p_y, psi, v]`

Input (nu=2):
- `u = [delta, a]`  
  steering angle and longitudinal acceleration.

**Important:** In this project the OCP decision variable is **`v`** (unfortunate name clash), and the *applied* input is:
- `u = Kdelta * x + v_decision`

So ACADOS "u" corresponds to the **decision variable**, and the applied physical input is reconstructed via the pre-stabilization feedback.

---

## Requirements

- Python environment for SOEAMPC 
- `acados` + `acados_template`
- CasADi compatible with your acados build

### ACADOS build note (important)
```
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DACADOS_WITH_QPOASES=ON \
  -DCMAKE_INSTALL_PREFIX=/home/{usr}/Projects/acados

make -j"$(nproc)"
make install
```

---

## Quick start

From `examples/vehicle/`:

### 1) Compute LQR gains (K, Kinit, Kdelta)
```bash
python3 compute_pre_stabilization.py
```
### MPC ingredients
To compute the MPC ingredients, run the `offlinempcingredients.m` file in MATLAB.
The output of the MATLAB file are already available in the folder `mpc_parameters`.
```

```
matlab -nodisplay -r "run('offlinempcingredients.m')"
```

The mpc parameters should be saved in human readable `.txt` form in the folder `mpc_parameters`.

## MPC dataset generation
To generate samples of the MPC, call
```
python3 samplempc.py sample_mpc \\
    --numberofsamples=10
```
The results of this would be saved in a folder called `datasets/quadcopter_N_{numberofsamples}_{date}-{time}`.


You can similarly create a larger dataset, by calling this function in parallel
```
python3 samplempc.py parallel_sample_mpc \\
    --instances=24 \\
    --samplesperinstance=10 \\
    --prefix=Cluster_test
```
The results of this would be saved in a folder called `datasets/quadcopter_N_{instances*samplesperinstance}_merged_{prefix}_{date}-{time}`

If you downloaded the precomputed dataset for this example, you should find it under `datasets/quadcopter_N_9600000`.

## Training a NN

If you want to train an approximator on the precomputed dataset for this example, call
```
python3 approximatempc.py find_approximate_mpc --dataset=latest
```
The models will be saved in a `models` folder.

If you downloaded the pretrained NN, you should find it under `models/10-200-400-600-600-400-200-30_mu=0.12_20230104-232806`

## Testing the NN
You can run closed loop test with the model calling
```
python3 safeonlineevaluation.py closed_loop_test_on_dataset \\
    --dataset=latest \\
    --model_name=10-200-400-600-600-400-200-30_mu=0.12_20230104-232806 \\
    --N_samples=3000 \\
    --N_sim=1000
```

## Troubleshooting
### PyCharm setup
Interpreter:
Open the repository root in PyCharm.

Set interpreter to your project venv:
```soeampc/.venv/bin/python (Linux)```
Ensure PyCharm uses that interpreter for run configurations.

Enviroment variables:
```
PYTHONUNBUFFERED=1;
ACADOS_SOURCE_DIR=/home/{usr}/acados;
LD_LIBRARY_PATH=/home/{usr}/acados/lib
```