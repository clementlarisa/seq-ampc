import numpy as np
from tqdm import tqdm
import time

from .datasetutils import export_dataset, print_compute_time_statistics, mpc_dataset_import


def sample_dataset_from_mpc(mpc, run, sampler, outfile, verbose=False):
    """evaluates the run function of mpc for points provided by sampler and exports the generated dataset

    Args:
        mpc:
            mpc class instance
        run:
            function that takes an initial condition x0 and evaluates the mpc.
            run returns tuple (X, U, status, computetime),
            with X predicted trajectory of shape (mpc.N+1, mpc.nx),
            U predicted sequence of shape (mpc.N, mpc.nu),
            status is acados status (0: success, 2: max iter)
        sampler:
            sampler class instance
        outfile:
            path to output
        verbose:
            prints extra info
    Returns:
        tuple (`x0`, `U`, `X`, `ct`, `name`), where `x0` is array of initial states shape (`Nsamples`, `mpc.nx`),
            `U` is array of input sequences shape (`Nsamples`, `mpc.N`, `mpc.nu`) or sequences `V` such that `U = Kdelta @ X + V`,
            `X` is an array of predicted state sequence when applying `U` from `x0` shape (`Nsamples`,`mpc.N+1`, `mpc.nx`)
            `ct` is an array of solver times for each mpc problem, `names` is the path to where the dataset was saved.
    """  
    X0dataset = np.empty((sampler.Nsamples, mpc.nx), float)
    Xdataset =  np.empty((sampler.Nsamples, mpc.N+1, mpc.nx), float)
    Udataset =  np.empty((sampler.Nsamples, mpc.N, mpc.nu), float)
    computetimes = np.empty(sampler.Nsamples, float)
    sqp_iterations = np.empty(sampler.Nsamples, int)

    runtobreak = True

    print("\n\n===============================================")
    print("Evaluating MPC on grid with",sampler.Nsamples,"points")
    print("===============================================\n")
    i = np.zeros(mpc.nx)
    Nvalid = 0
    n = 0
    with tqdm(total=sampler.Nsamples) as pbar:
        while n < sampler.Nsamples:
            x0 = sampler.sample()
            X, U, status, elapsed, sqp_iteration = run(x0)
            if status == 0 or status == 2:
                if verbose:
                    print("acados status: ", status)
                if mpc.feasible(X, U, verbose=verbose, robust=True, only_states=False):
                    X0dataset[Nvalid,:] = x0
                    Xdataset[Nvalid,:,:]  = X
                    Udataset[Nvalid,:,:]  = U
                    computetimes[Nvalid] = elapsed
                    sqp_iterations[Nvalid] = sqp_iteration
                    Nvalid += 1
                    if runtobreak:
                        pbar.update(1)
                        n += 1

            if not runtobreak:
                pbar.update(1)
                n +=1

    print("Got",Nvalid,"feasible solutions for MPC")
    print("MPC compute time statistics:")
    print_compute_time_statistics(computetimes[:Nvalid])

    mean_sqp_iteration_time = np.mean(computetimes[:Nvalid]/sqp_iterations[:Nvalid])
    print(f"{mean_sqp_iteration_time=}")

    datasetname = export_dataset(
        mpc,
        X0dataset[:Nvalid,:],
        Udataset[:Nvalid,:,:],
        Xdataset[:Nvalid,:,:],
        computetimes[:Nvalid],
        outfile)
    return X0dataset[:Nvalid,:], Udataset[:Nvalid,:,:], Xdataset[:Nvalid,:,:], computetimes[:Nvalid], datasetname


def computetime_test_fwd_sim(run, dataset="latest", N_samples = int(10e3)):
    mpc, X, V, _, _ = mpc_dataset_import(dataset)
    if N_samples >= X.shape[0]:
        N_samples = X.shape[0]
        print("WARNING: N_samples exceeds size of dataset, will use N_samples =", N_samples,"instead")
    tic = time.time()
    for i in range(N_samples):
        run(X[i], V[i])
    duration = time.time() - tic
    print(f"mean duration was {duration/N_samples*1000} [ms]")


    compute_times = np.zeros(N_samples)
    for i in range(N_samples):
        tic = time.time()
        run(X[i], V[i])
        compute_times[i] = time.time() - tic
    
    print_compute_time_statistics(compute_times)