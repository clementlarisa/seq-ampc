#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=GPU-a100
#SBATCH --exclude=ivm-a40-q-2
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --output=RNN_%j.out
#SBATCH --error=RNN_%j.err

echo "=== Loading conda environment ==="
source /home/mihaela-larisa.clement/miniconda3/etc/profile.d/conda.sh
conda activate soeampc

export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export TF_ENABLE_XLA=0
# pick a writable temp dir (home is almost always writable)
export TMPDIR="$HOME/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" || exit 1
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

# also put caches somewhere writable
export CUDA_CACHE_PATH="$TMPDIR/cuda-cache"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" || exit 1

# prove it's writable
python - <<'PY'
import os, tempfile
print("TMPDIR env:", os.environ.get("TMPDIR"))
f = tempfile.NamedTemporaryFile(dir=os.environ["TMPDIR"], delete=True)
print("write TMPDIR: OK", f.name)
PY

    # --dataset=quadcopter_N_9600000/quadcopter_N_9600000_merged_20221223-161206 \

echo "=== Running approximatempc.py ==="
python /home/mihaela-larisa.clement/soeampc/examples/quadcopter/approximatempc.py \
    find_approximate_mpc \
    --dataset=quadcopter_N_9600000/quadcopter_N_9600000_merged_20221223-161206 \
    --neural_type=RNN \
    --dense_units="(200, 400, 600)" \
    --rnn_units=256 \
    --retrain=True \
    --retrain_model_name="NeuralType.RNN_rnn256_dense200x400x600_bs6250_ep100000_pat1000_lr0.001_checkpoint.keras"

echo "=== Job finished ==="
