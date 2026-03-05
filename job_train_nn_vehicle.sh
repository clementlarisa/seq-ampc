#!/bin/bash
#SBATCH --job-name=116rnn
#SBATCH --partition=GPU-a100
#SBATCH --exclude=ivm-a40-q-2
#SBATCH --gres=gpu:a100:1
#SBATCH --time=72:00:00
#SBATCH --output=RNN_veh_%j.out
#SBATCH --error=RNN_veh_%j.err

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
    #--dataset=vehicle_8state_obs_N_22000_merged_vehicle_8state_obs_N_22000_merged_ClusterTest10k_plus_1k_5k_6k_20260223-194502 \
    # --dense_units="(200, 400, 600)" \
    # --rnn_units=256

    #vehicle_8state_obs_N_116000_merged_vehicle_8state_obs_N_116000_20260225-093417
    #vehicle_obs_N_55000_merged_vehicle_obs_N_55000_20260225-123451
echo "=== Running approximatempc.py ==="
python /home/mihaela-larisa.clement/soeampc/examples/vehicle_dyn_obs/approximatempc.py \
    find_approximate_mpc \
    --dataset=vehicle_8state_obs_N_116000_merged_vehicle_8state_obs_N_116000_20260225-093417 \
    --neural_type=RNN \
    --dense_units="(200, 400, 600)" \
    --rnn_units=256

echo "=== Job finished ==="
