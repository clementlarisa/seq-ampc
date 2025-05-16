# Robot arm (stand-alone example)
A 12-DoF state robot arm tracking example, third joint of the Franka is locked to avoid set-valued solutions.
You can find the pretrained models and dataset on [Zenodo]().
Due to Zenodo's limitation to 50GB per dataset, we only uploaded about 20% of samples used in the paper.

This example does not work with the provided docker container! Please use an up-to-date `python 3.13` with `casadi` and `jax` installed.

## MPC ingredients
To compute the MPC ingredients, run the [`lmi_casadi.py`](./synthesize/lmi_casadi.py) file in Python (requires MOSEK solver).
The output of the Python file are already available as [`matrices.npz`](./synthesize/matrices.npz).

## MPC dataset generation
To generate a single sample of the MPC, call
```
python mpc.py run_save --length=100 --filename=testoutput
```
If you downloaded the precomputed dataset for this example, you should find it `dataset.tar`.
Due to Zenodo limitation to 50GB per dataset, we only uploaded about 20% of samples used in the paper.


## Testing the NN
You can find the pretrained model in the zenodo datset as `.eqx` and `.h5` files.
You can run all closed loop tests with the model calling:
```
python evaluation.py
```

For the C++ implementation (fast online inference), compile the project in benchmark as
```
make main_benchmark
```

Afterwards, run
```
main_benchmark <path/to/model.h5>
```.