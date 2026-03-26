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

## Datasets

Precomputed datasets and pretrained models are available on [Zenodo]().

## License

See [LICENSE](LICENSE) for details.
