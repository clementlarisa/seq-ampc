import matplotlib.pyplot as plt
import jax.numpy
import jax.numpy as jnp
import os
import numpy as np
import tqdm

def plot_histogram(dataset_to_plot, outdir, filename):
    num_subplots = dataset_to_plot.shape[1]
    nrows = int(jax.numpy.ceil(jax.numpy.sqrt(num_subplots)))
    ncols = int(jax.numpy.ceil(num_subplots / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    axes = axes.flatten()
    for i in range(num_subplots):
        ax = axes[i]
        ax.hist(dataset_to_plot[:, i], bins=50, edgecolor='black')
        ax.set_title(f'Histogram of dataset[:, {i}]')
    for j in range(num_subplots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_outdir = outdir
    if not os.path.exists(plot_outdir):
        os.makedirs(plot_outdir)
    fig.savefig(f"{plot_outdir}/{filename}")
    plt.close(fig)
    
    
def remove_outliers_based_on_integrator_error(X, U, Y, integrator, batch_size=1024, threshold=2.0, save_histogram_path=None):
    num_samples = X.shape[0]
    abs_sums = []

    print(f"Processing {num_samples} samples in batches of {batch_size}...")

    # --- JIT compiled batch computation ---
    @jax.jit
    def compute_batch_abs_sum(X_batch, U_batch):
        X0_batch = X_batch[:, 0, :]
        X_pred_batch = jax.vmap(integrator)(X0_batch, U_batch)
        abs_sum_batch = jnp.sum(jnp.abs(X_pred_batch[:, -1, :] - X_batch[:, -1, :]), axis=-1)
        return abs_sum_batch
    # ---------------------------------------

    # First pass: compute abs_sums
    for start_idx in tqdm.tqdm(range(0, num_samples, batch_size), desc="Computing abs sums..."):
        end_idx = min(start_idx + batch_size, num_samples)
        X_batch = X[start_idx:end_idx]
        U_batch = U[start_idx:end_idx]

        abs_sum_batch = compute_batch_abs_sum(X_batch, U_batch)
        abs_sums.append(np.array(abs_sum_batch))  # move to CPU numpy

    abs_sums = np.concatenate(abs_sums, axis=0)

    mean_abs_sum = np.mean(abs_sums)
    std_abs_sum = np.std(abs_sums)
    # threshold = mean_abs_sum + threshold_std * std_abs_sum

    print(f"Mean integrator error: {mean_abs_sum:.6f}, Std: {std_abs_sum:.6f}, Threshold: {threshold:.6f}")

    if save_histogram_path is not None:
        os.makedirs(os.path.dirname(save_histogram_path), exist_ok=True)

        plt.figure()
        plt.hist(abs_sums, bins=50)
        plt.xlabel('Integrator Error (|X_pred - X_| summed over features)')
        plt.ylabel('Count')
        plt.title('Histogram of Integrator Errors (Before Cleaning)')
        plt.grid(True)
        plt.savefig(save_histogram_path.replace(".pdf", "_before.pdf"))
        plt.close()
        print(f"Saved histogram (before cleaning) to {save_histogram_path.replace('.pdf', '_before.pdf')}")

    # Second pass: select good samples
    X_clean_list = []
    Y_clean_list = []
    U_clean_list = []
    abs_sums_clean = []

    print("Filtering samples based on threshold...")

    for start_idx in tqdm.tqdm(range(0, num_samples, batch_size), desc="Filtering samples..."):
        end_idx = min(start_idx + batch_size, num_samples)
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        U_batch = U[start_idx:end_idx]

        abs_sum_batch = compute_batch_abs_sum(X_batch, U_batch)
        abs_sum_batch = np.array(abs_sum_batch)

        good_mask = abs_sum_batch <= threshold

        X_clean_list.append(X_batch[good_mask])
        Y_clean_list.append(Y_batch[good_mask])
        U_clean_list.append(U_batch[good_mask])
        abs_sums_clean.append(abs_sum_batch[good_mask])

    X_clean = np.concatenate(X_clean_list, axis=0)
    Y_clean = np.concatenate(Y_clean_list, axis=0)
    U_clean = np.concatenate(U_clean_list, axis=0)
    abs_sums_clean = np.concatenate(abs_sums_clean, axis=0)

    print(f"Removed {X.shape[0] - X_clean.shape[0]} outliers out of {X.shape[0]} samples. Remaining is {100 - (X.shape[0] - X_clean.shape[0]) / X.shape[0] * 100:.2f}%.")

    # Plot histogram of cleaned data
    if save_histogram_path is not None:
        plt.figure()
        plt.hist(abs_sums_clean, bins=50)
        plt.xlabel('Integrator Error (|X_pred - X_| summed over features)')
        plt.ylabel('Count')
        plt.title('Histogram of Integrator Errors (After Cleaning)')
        plt.grid(True)
        plt.savefig(save_histogram_path.replace(".pdf", "_after.pdf"))
        plt.close()
        print(f"Saved histogram (after cleaning) to {save_histogram_path.replace('.pdf', '_after.pdf')}")

    return X_clean, U_clean, Y_clean


def print_duration(duration):
    # Convert to hours, minutes, seconds
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print nicely formatted
    print(f"Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    