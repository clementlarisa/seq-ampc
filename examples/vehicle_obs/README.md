# Vehicle Example (Kinematic Bicycle + Obstacle Avoidance) — Sequential-AMPC

This folder has an example that uses a **kinematic bicycle model** (ground vehicle) with:

- **Pre-stabilizing LQR feedback**
- **Nonlinear obstacle avoidance constraints**
- The same **Sequential-AMPC dataset generation and approximation workflow**

---

## Folder structure

- `dynamics/`
  - `f.py`  
    Vehicle dynamics `xdot = f(x,u)` used by MPC and simulation.

- `mpc_parameters/`
  - `Q.txt, R.txt, P.txt`  
    Quadratic cost matrices
  - `K.txt, Kinit.txt, Kdelta.txt`  
    Linear feedback gains
  - `Lx.txt, Lu.txt, Ls.txt`  
    Constraint matrices for one-sided constraints  
    `C*[x;s] + D*v <= 1`
  - `alpha.txt, alpha_s.txt`  
    Terminal set size parameters
  - `Tf.txt`  
    MPC horizon length
  - `rho_c.txt, wbar.txt`  
    Parameters for auxiliary state `s`

- `samplempc.py`  
  Generates MPC solutions including obstacle constraints and exports datasets.

- `safeonlineevaluation.py`  
  Closed-loop safety evaluation (same pattern as quadcopter).

- `plot.py`  
  Vehicle plotting helpers (`plot_vehicle_ol`, `plot_vehicle_ol_V`, `plot_vehicle_cl`).

- `compute_pre_stabilization.py`  
  Linearization + LQR gain computation and export of  
  `K`, `Kinit`, `Kdelta`.

---

## Model Overview

### State (nx = 4)

```
x = [p_x, p_y, psi, v]
```

- `p_x, p_y` — position  
- `psi` — heading angle  
- `v` — longitudinal velocity  

### Input (nu = 2)

```
u = [delta, a]
```

- `delta` — steering angle  
- `a` — longitudinal acceleration  

---

## Pre-Stabilization Structure

The ACADOS decision variable is:

```
v_decision
```

The **applied physical input** is reconstructed as:

```
u = Kdelta * x + v_decision
```

Therefore:

- ACADOS "u" = optimization variable
- Applied plant input = pre-stabilized feedback + correction

This improves numerical conditioning and stabilizes dataset generation.

---

## Obstacle Avoidance Formulation

The OCP model contains **parameters**:

```
p = [o1x, o1y, o2x, o2y, r_safe]
```

Obstacle constraints are implemented as nonlinear inequalities:

```
c1 = (px - o1x)^2 + (py - o1y)^2 - r_safe^2  >= 0
c2 = (px - o2x)^2 + (py - o2y)^2 - r_safe^2  >= 0
```

This enforces that the vehicle remains **outside safety circles** around obstacles.

### Important

Obstacles are treated as **parameters**, not states.

This allows:
- Mixed obstacle configurations
- Efficient dataset generation
- Generalization in NN training

---

## Mixed Obstacle Sampling Strategy

To improve robustness of the learned NN controller, datasets are generated with mixed obstacle presence:

- 20%: 0 active obstacles  
- 40%: 1 active obstacle  
- 40%: 2 active obstacles  

Inactive obstacles are placed far away (numerically inactive).

Obstacle parameters are saved per sample in:

```
P_obstacles.txt
```

This file contains:

```
[o1x, o1y, o2x, o2y, r_safe]
```

for each dataset entry.

---

## Requirements

- Python environment for seqampc
- `acados` + `acados_template`
- Compatible CasADi version

---

## ACADOS build (with OpenMP)

### 0) System packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y git cmake build-essential pkg-config
# OpenMP runtime / headers (recommended):
sudo apt install -y libomp-dev
```

Notes:
- If you build with **GCC**, OpenMP support is typically provided via `libgomp` (often already present).
- `libomp-dev` is a safe default and also helps if you use **clang**.

### 1) Configure + build + install

From your `acados` build directory:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DACADOS_WITH_QPOASES=ON \
  -DACADOS_WITH_OPENMP=ON \
  -DCMAKE_INSTALL_PREFIX=/home/{usr}/Projects/acados

make -j"$(nproc)"
make install
```

### 2) Verify OpenMP is active (quick checks)

```bash
# Look for OpenMP linkage symbols in the installed shared libs
ldd /home/{usr}/Projects/acados/lib/libacados.so | grep -E "omp|gomp" || true
```

If this prints `libgomp.so` or `libomp.so`, OpenMP is linked.

---

## Quick Start

From `examples/vehicle_obs/`:

### 1) Compute LQR gains

```bash
python3 compute_pre_stabilization.py
```

### 2) Compute MPC ingredients (MATLAB)

Run:

```bash
matlab -nodisplay -r "run('offlinempcingredients.m')"
```

The generated `.txt` files must be stored in:

```
mpc_parameters/
```

---

## MPC Dataset Generation

### Single run

```bash
python3 samplempc.py sample_mpc \
    --numberofsamples=10
```

Results are stored in:

```
datasets/vehicle_obs_N_{numberofsamples}_{date}-{time}
```

### Parallel generation

```bash
python3 samplempc.py parallel_sample_mpc \
    --instances=24 \
    --samplesperinstance=10 \
    --prefix=Cluster_test
```

Merged results:

```
datasets/vehicle_obs_N_{instances*samplesperinstance}_merged_{prefix}_{date}-{time}
```

---

## Training a Neural Network Approximation

```bash
python3 approximatempc.py find_approximate_mpc \
    --dataset=latest
```

Models are saved in:

```
models/
```

---

## Closed-Loop Testing

```bash
python3 safeonlineevaluation.py closed_loop_test_on_dataset \
    --dataset=latest \
    --model_name=<your_model_name> \
    --N_samples=3000 \
    --N_sim=1000
```

The evaluation includes:

- Constraint satisfaction (including obstacles)
- Closed-loop feasibility
- Safety violation statistics

---

## Dataset Contents (Vehicle + Obstacles)

Each dataset entry includes:

- `X.txt` — state trajectory  
- `U.txt` — decision variable trajectory  
- `P_obstacles.txt` — obstacle parameters  
- Cost and constraint information  

The NN therefore learns:

```
(x, obstacle_parameters) → v_decision
```

---

## PyCharm Setup

### Interpreter

Open repository root.

Set interpreter to:

```
seqampc/.venv/bin/python
```

### Environment variables

```
PYTHONUNBUFFERED=1;
ACADOS_SOURCE_DIR=/home/{usr}/acados;
LD_LIBRARY_PATH=/home/{usr}/acados/lib
```

---

## Notes

- Obstacle avoidance is handled purely as nonlinear constraints.
- The NN learns an approximation of a **parametric NMPC**.
- The structure supports extension to:
  - More obstacles
  - Moving obstacles
  - Track-following references
