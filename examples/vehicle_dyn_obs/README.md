# Vehicle Example (7-State Single-Track) + Obstacles --- Sequential-AMPC

This example uses a **7-state single-track vehicle model** with a
**pre-stabilizing feedback** and **hard obstacle avoidance
constraints**.\
It follows the Sequential-AMPC workflow:

1.  Generate MPC solutions
2.  Export dataset
3.  Train neural network approximator
4.  Closed-loop evaluation

------------------------------------------------------------------------

## Model Overview

State (nx = 7):

    x = [p_x, p_y, psi, v, r, beta, a]

Input (nu = 2):

    u = [delta, a_cmd]

### Pre-Stabilization Convention

The ACADOS decision variable is **v_dec** and the applied physical input
is:

    u = Kdelta * x + v_dec

So ACADOS "u" corresponds to the decision variable.

------------------------------------------------------------------------

## Obstacle Avoidance (Hard Constraints)

Two circular obstacles are modeled via nonlinear path constraints.

Parameter vector (np = 5):

    p = [o1x, o1y, o2x, o2y, r_safe]

Constraints at every stage:

    (px - oix)^2 + (py - oiy)^2 - r_safe^2 >= 0

Inactive obstacles are disabled by setting:

    o = (FAR, FAR)

with FAR = 100.

After dataset generation, obstacle parameters are written to:

-   `P_obstacles.txt`
-   `N_active.txt`

------------------------------------------------------------------------

## Folder Structure

-   `dynamics/f.py`\
    7-state vehicle dynamics

-   `mpc_parameters/`\
    Contains Q, R, P, K, Kdelta, constraint matrices, horizon length,
    etc.

-   `samplempc_obs.py`\
    Dataset generation with obstacle constraints

-   `plot.py`\
    3×3 plotting helpers including obstacle visualization

------------------------------------------------------------------------

## Requirements

-   Python environment for seqampc
-   acados + acados_template
-   CasADi compatible with your acados build

------------------------------------------------------------------------

## ACADOS Build Example

    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DACADOS_WITH_QPOASES=ON \
      -DCMAKE_INSTALL_PREFIX=/home/{usr}/acados

    make -j"$(nproc)"
    make install

------------------------------------------------------------------------

## Generate MPC Dataset

From the folder:

    python3 samplempc_obs.py sample_mpc \
      --numberofsamples=1000 \
      --showplot=True

Parallel generation:

    python3 samplempc_obs.py parallel_sample_mpc \
      --instances=24 \
      --samplesperinstance=10000 \
      --prefix=Cluster_test

------------------------------------------------------------------------

## Train Neural Network

    python3 approximatempc.py find_approximate_mpc --dataset=latest

Models are saved in the `models/` directory.

------------------------------------------------------------------------

## Closed-Loop Testing

    python3 safeonlineevaluation.py closed_loop_test_on_dataset \
      --dataset=latest \
      --model_name=<your_model> \
      --N_samples=3000 \
      --N_sim=1000

------------------------------------------------------------------------

## Notes

-   Ensure `input_is_v=True` when plotting if your dataset stores the
    decision variable `v_dec`.
-   Obstacle circles are visualized automatically when passing
    `obstacles` and `r_safe` to plotting functions.
