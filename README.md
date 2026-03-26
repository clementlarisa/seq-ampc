# Sequential-AMPC: Safe Learning-Based Nonlinear Model Predictive Control through Recurrent Neural Network Modeling

This repository contains the implementation for the paper:

> **Towards Safe Learning-Based Non-Linear Model Predictive Control through Recurrent Neural Network Modeling**
> Mihaela-Larisa Clement, Monika Farsang, Agnes Poks, Johannes Edelmann, Manfred Plochl, Radu Grosu, Ezio Bartocci
> arXiv:2603.24503 [cs.LG], 2026
> https://arxiv.org/abs/2603.24503

We propose **Sequential-AMPC**, a sequential neural policy that generates MPC candidate control sequences by sharing parameters across the prediction horizon. For deployment, the policy is wrapped in a safety-augmented online evaluation and fallback mechanism, yielding **Safe Sequential-AMPC**.

## Attribution

This codebase builds on the [SOEAMPC](https://arxiv.org/abs/2304.09575) framework. If you use this code, please cite both papers:

```bibtex
@article{clement2026seqampc,
  title={Towards Safe Learning-Based Non-Linear Model Predictive Control through Recurrent Neural Network Modeling},
  author={Clement, Mihaela-Larisa and Farsang, M{\'o}nika and Poks, Agnes and Edelmann, Johannes and Pl{\"o}chl, Manfred and Grosu, Radu and Bartocci, Ezio},
  journal={arXiv preprint arXiv:2603.24503},
  year={2026}
}

@article{Hose_2025,
   title={Approximate Nonlinear Model Predictive Control With Safety-Augmented Neural Networks},
   volume={33},
   ISSN={2374-0159},
   url={http://dx.doi.org/10.1109/TCST.2025.3590268},
   DOI={10.1109/tcst.2025.3590268},
   number={6},
   journal={IEEE Transactions on Control Systems Technology},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Hose, Henrik and Köhler, Johannes and Zeilinger, Melanie N. and Trimpe, Sebastian},
   year={2025},
   month=nov, pages={2490–2497} }

```

## Installation

### 1. Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate seqampc
```

### 2. acados

[acados](https://docs.acados.org/installation/index.html) is required for MPC sampling. After building acados, add the Python interface to your path:

```bash
export ACADOS_INSTALL_DIR=/path/to/acados
export PYTHONPATH=$ACADOS_INSTALL_DIR/interfaces/acados_template:$PYTHONPATH
export LD_LIBRARY_PATH=$ACADOS_INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

### 3. Data directory (optional)

By default, datasets and models are read from `./data/`. To use a different location, set:

```bash
export SEQAMPC_DATA_ROOT=/path/to/data
```

## Numerical Examples

Examples from the paper are in the `examples/` folder:

- [Quadcopter](examples/quadcopter/) — 10-state quadrotor stabilization (no obstacles)
- [Vehicle kinematic + obstacles](examples/vehicle_obs/) — 4-state kinematic bicycle model with static obstacle avoidance
- [Vehicle dynamic + obstacles](examples/vehicle_dyn_obs/) — 8-state single-track model with slip and yaw dynamics, static obstacle avoidance

Each example follows the same workflow: **sample MPC solutions**, **train a neural approximation**, and **evaluate in closed loop**.

### Quadcopter

```bash
cd examples/quadcopter

# 1. Sample MPC solutions (requires acados)
python samplempc.py sample_mpc
# or parallel sampling (e.g. 40 instances, 25 samples each):
python samplempc.py parallel_sample_mpc --instances=40 --samplesperinstance=25

# 2. Merge parallel sampling jobs
python samplempc.py merge_single_parallel_job

# 3. Train neural approximation (MLP or RNN)
python approximatempc.py find_approximate_mpc --dataset=<DATASET> --neural_type=MLP
python approximatempc.py find_approximate_mpc --dataset=<DATASET> --neural_type=RNN \
    --dense_units="(200, 400, 600)" --rnn_units=256

# 4. Retrain from checkpoint
python approximatempc.py find_approximate_mpc --dataset=<DATASET> --neural_type=RNN \
    --dense_units="(200, 400, 600)" --rnn_units=256 \
    --retrain=True --retrain_model_name=<MODEL_NAME>

# 5. Run statistical test
python approximatempc.py run_statistical_test --dataset=<DATASET> --neural_type=RNN \
    --retrain_model_name=<MODEL_NAME>

# 6. Evaluate naive AMPC on dataset
python safeonlineevaluation.py evaluate_naive_ampc_on_dataset \
    --dataset=<DATASET> --model_name=<MODEL_NAME>

# 7. Closed-loop test with plots
python safeonlineevaluation.py closed_loop_test_on_dataset_plot \
    --dataset=<DATASET> --model_name=<MODEL_NAME>

# 8. Closed-loop test on dataset
python safeonlineevaluation.py closed_loop_test_on_dataset \
    --dataset=<DATASET> --model_name=<MODEL_NAME> --N_samples=1000

# 9. Closed-loop test with rejection reason breakdown
python safeonlineevaluation.py closed_loop_test_reason \
    --dataset=<DATASET> --model_name=<MODEL_NAME> --N_samples=1000
```

### Vehicle kinematic + obstacles (vehicle_obs)

```bash
cd examples/vehicle_obs

# 1. Sample MPC solutions (requires acados)
python samplempc.py parallel_sample_mpc --instances=40 --samplesperinstance=25

# 2. Merge parallel sampling jobs
python samplempc.py merge_single_parallel_job

# 3. Train neural approximation (MLP or RNN)
python approximatempc.py find_approximate_mpc --dataset=<DATASET> --neural_type=MLP
python approximatempc.py find_approximate_mpc --dataset=<DATASET> --neural_type=RNN \
    --dense_units="(200, 400, 600)" --rnn_units=256

# 4. Run statistical test
python approximatempc.py run_statistical_test --dataset=<DATASET> --neural_type=MLP \
    --retrain_model_name=<MODEL_NAME>

# 5. Closed-loop test on dataset
python safeonlineevaluation.py closed_loop_test_on_dataset_vehicle_obs \
    --dataset_dir=<DATASET> --model_name=<MODEL_NAME> --N_samples=1000 --N_sim=100
```

### Vehicle dynamic + obstacles (vehicle_dyn_obs)

```bash
cd examples/vehicle_dyn_obs

# 1. Sample MPC solutions (requires acados)
python samplempc.py parallel_sample_mpc --instances=40 --samplesperinstance=25

# 2. Merge parallel sampling jobs
python samplempc.py merge_single_parallel_job

# 3. Train neural approximation (MLP or RNN)
python approximatempc.py find_approximate_mpc --dataset=<DATASET> --neural_type=MLP
python approximatempc.py find_approximate_mpc --dataset=<DATASET> --neural_type=RNN \
    --dense_units="(200, 400, 600)" --rnn_units=256

# 4. Run statistical test
python approximatempc.py run_statistical_test --dataset=<DATASET> --neural_type=MLP \
    --retrain_model_name=<MODEL_NAME>

# 5. Closed-loop test on dataset
python safeonlineevaluation.py closed_loop_test_on_dataset_vehicle_obs \
    --dataset_dir=<DATASET> --model_name=<MODEL_NAME> --N_samples=1000 --N_sim=100

# 6. Inspect dataset interactively
python eval_dataset.py --dataset=<DATASET>
```

In all commands above, `<DATASET>` is the dataset folder name (under `$SEQAMPC_DATA_ROOT/archive/`) and `<MODEL_NAME>` is the model folder or `.keras` checkpoint name (under `$SEQAMPC_DATA_ROOT/models/`). Use `--dataset=latest` to automatically pick the most recent dataset.

## Datasets

Precomputed datasets and pretrained models are available on [Zenodo](). For the vehicle datasets, we hold out at evaluation time separate datasets.

## License

See [LICENSE](LICENSE) for details.
