# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_learning_curve(csv_path: str | Path, save_png: str | Path | None = None):
#     csv_path = Path(csv_path)
#     df = pd.read_csv(csv_path)

#     x = df["epoch"] if "epoch" in df.columns else range(len(df))

#     plt.figure()
#     if "loss" in df.columns:
#         plt.plot(x, df["loss"], label="loss")
#     if "val_loss" in df.columns:
#         plt.plot(x, df["val_loss"], label="val_loss")

#     if "mae" in df.columns:
#         plt.plot(x, df["mae"], label="mae", linestyle="--")
#     if "val_mae" in df.columns:
#         plt.plot(x, df["val_mae"], label="val_mae", linestyle="--")

#     plt.xlabel("epoch")
#     plt.ylabel("value")
#     plt.title(csv_path.name)
#     plt.legend()
#     plt.grid(True)

#     if save_png is not None:
#         save_png = Path(save_png)
#         plt.savefig(save_png, dpi=200, bbox_inches="tight")

#     plt.show()
#     return df


# csv_path = Path("/share/mihaela-larisa.clement/soeampc-data/models/learning_curves/NeuralType.MLP_rnn256_dense200_bs10000_ep100000_pat1000_lr0.0005.csv")

# png_path = Path.cwd() / csv_path.with_suffix(".png").name

# df = plot_learning_curve(csv_path, save_png=png_path)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import re

def make_model_prefix(name: str) -> str:
    name_u = name.upper()

    if "MLP" in name_u:
        return "AMPC on "

    if "RNN" in name_u:
        # Extract rnn units (e.g., rnn256)
        rnn_match = re.search(r'_rnn(\d+)', name, flags=re.IGNORECASE)

        # Extract dense block after "_dense" up to next known suffix token or end
        dense_match = re.search(
            r'_dense([^_]+?)(?=_(?:bs|ep|pat|lr|wd|drop|opt|seed)|$)',
            name,
            flags=re.IGNORECASE
        )

        if rnn_match:
            rnn_units = rnn_match.group(1)

            if dense_match:
                dense_spec = dense_match.group(1)
                return f"Sequential-AMPC on "

            return f"RNN {rnn_units} on "

        # fallback if RNN exists but pattern not matched
        return "RNN model on "

    return ""

def earlystop_epoch_from_series(values, min_delta=0.0, patience=1000):
    """
    Simulate tf.keras.callbacks.EarlyStopping for mode='min' with min_delta>=0.
    Returns (best_index, stop_index, stopped_early_bool).
    """
    best = float("inf")
    best_idx = None
    wait = 0

    for i, v in enumerate(values):
        if pd.isna(v):
            wait += 1
        else:
            if v < best - min_delta:  # improvement for mode="min"
                best = float(v)
                best_idx = i
                wait = 0
            else:
                wait += 1

        if wait >= patience:
            return best_idx, i, True

    return best_idx, len(values) - 1, False


def make_title_from_name(name: str) -> str:
    name_u = name.upper()
    dataset = (
        "1/4 Quadcopter dataset" if "QUARTER" in name_u
        else "1/10 Quadcopter dataset" if "TENTH" in name_u
        else "ST-Vehicle Dynamic Dataset"
    )   
    title_prefix = make_model_prefix(name)
    title = f"{title_prefix}{dataset}"
    return title


def plot_learning_curve(
    csv_path: str | Path,
    save_png: str | Path | None = None,
    patience: int = 1000,
    min_delta: float = 0.0,
):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # x-axis
    x = df["epoch"].to_numpy() if "epoch" in df.columns else df.index.to_numpy()

    # Early stopping decision based on val_loss
    stop_idx = len(df) - 1
    best_idx = None
    stopped_early = False

    if "val_loss" in df.columns:
        best_idx, stop_idx, stopped_early = earlystop_epoch_from_series(
            df["val_loss"].to_numpy(),
            min_delta=min_delta,
            patience=patience,
        )

    df_plot = df.iloc[: stop_idx + 1]
    x_plot = x[: stop_idx + 1]

    plt.figure(figsize=(5, 3))

    # Plot only loss + val_loss with requested labels
    if "loss" in df_plot.columns:
        plt.plot(x_plot, df_plot["loss"], label="training loss")

    if "val_loss" in df_plot.columns:
        val_line, = plt.plot(x_plot, df_plot["val_loss"], label="validation loss")

        if best_idx is not None and best_idx <= stop_idx and not pd.isna(df["val_loss"].iloc[best_idx]):
            best_val = float(df["val_loss"].iloc[best_idx])
            best_epoch = int(x[best_idx])

            print(f"Best validation loss: {best_val:.6f}")

            plt.scatter(
                [x[best_idx]],
                [df["val_loss"].iloc[best_idx]],
                marker="o",
                color="green",
                zorder=5,
                label=f"best val loss = {best_val:.6f} @ epoch {best_epoch}",
            )

        if stopped_early:
            plt.axvline(
                x=x[stop_idx],
                linestyle="--",
                color="red",
                label=f"early stop @ {int(x[stop_idx])}",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(make_title_from_name(csv_path.name))
    plt.legend()
    plt.grid(True)

    if save_png is not None:
        save_png = Path(save_png)
        plt.savefig(save_png, dpi=300, bbox_inches="tight")

        save_pdf = save_png.with_suffix(".pdf")
        plt.savefig(save_pdf, bbox_inches="tight")  # vector PDF

    plt.show()
    return df, stop_idx, best_idx, stopped_early
    
# "NeuralType.RNN_rnn128_dense200x400x600x600x400_bs5000_ep100000_pat1000_lr0.001.csv"
csv_path = Path(
    "/share/mihaela-larisa.clement/soeampc-data/models/learning_curves/"
    # "NeuralType.MLP_rnn256_dense200_bs10000_ep100000_pat1000_lr0.0005.csv"
    # "NeuralType.MLP_rnn256_dense200_bs5000_ep100000_pat1000_lr0.001_tenth.csv"
    # "NeuralType.MLP_rnn256_dense200_bs5000_ep100000_pat1000_lr0.001_quarter.csv"
    # "NeuralType.RNN_rnn128_dense200x400x600x600_bs5000_ep100000_pat1000_lr0.001.csv"
    # "NeuralType.RNN_rnn256_dense200_bs10000_ep100000_pat1000_lr0.001.csv"
    # "NeuralType.RNN_rnn256_dense200x400x600_bs5000_ep100000_pat1000_lr0.001.csv"
    # "NeuralType.RNN_rnn256_dense200_bs5000_ep100000_pat1000_lr0.001_tenth.csv"
    # "NeuralType.RNN_rnn256_dense200_bs5000_ep100000_pat1000_lr0.001_quarter.csv"
    # "NeuralType.MLP_rnn32_dense200x400x600x600x400x200_bs1000_ep100000_pat1000_lr0.001.csv"
    # "NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001.csv"
    # "NeuralType.RNN_rnn256_dense200x400x600_bs6250_ep100000_pat1000_lr0.001.csv"
    # "NeuralType.RNN_rnn256_dense200x400x600_bs6250_ep100000_pat1000_lr0.001_tenth.csv"
    # "NeuralType.RNN_rnn256_dense200x400x600_bs6250_ep100000_pat1000_lr0.001_quarter.csv"


    # "NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001.csv"
    # "vehicle_8state__NeuralType.MLP_rnn32_dense200x400x600x600x400x200_bs1000_ep100000_pat1000_lr0.001.csv"
    # "vehicle_NeuralType.MLP_rnn32_dense200x400x600x600x400x200_bs10000_ep100000_pat1000_lr0.001.csv"
    # "vehicle_NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001.csv"
    # "NeuralType.MLP_rnn32_dense200x400x600x600x400x200_bs1000_ep100000_pat1000_lr0.001.csv"
    "vehicle_8state__NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001.csv"
)
png_path = Path.cwd() / csv_path.with_suffix(".png").name

df, stop_idx, best_idx, stopped_early = plot_learning_curve(
    csv_path, save_png=png_path, patience=1000, min_delta=0.0
)
print(f"best_idx={best_idx}, stop_idx={stop_idx}, stopped_early={stopped_early}")
if "epoch" in df.columns and best_idx is not None:
    print(
        f"best_epoch={int(df['epoch'].iloc[best_idx])}, stop_epoch={int(df['epoch'].iloc[stop_idx])}"
    )
