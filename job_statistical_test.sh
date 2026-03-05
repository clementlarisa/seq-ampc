#!/bin/bash
#SBATCH --job-name=stattest
#SBATCH --partition=GPU-a40
#SBATCH --exclude=ivm-a40-q-2
#SBATCH --gres=gpu:a40:1
#SBATCH --time=72:00:00
#SBATCH --output=Stattest_quad_%j.out
#SBATCH --error=Stattest_quad_%j.err

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

    #    --dataset=vehicle_8state_obs_N_32000_merged_vehicle_8state_obs_N_32000_merged_20260224-232550 \
    # --neural_type=RNN \
    # --dense_units="(200, 400, 600)" \
    # --rnn_units=256 \
    # --retrain_model_name="NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001_checkpoint.keras"
    
echo "=== Running approximatempc.py ==="
python /home/mihaela-larisa.clement/soeampc/examples/vehicle_dyn_obs/approximatempc.py \
    run_statistical_test \
    --dataset=vehicle_8state_obs_N_116000_merged_vehicle_8state_obs_N_116000_20260225-093417 \
    --neural_type=MLP \
    --retrain_model_name="vehicle_8state_116k_NeuralType.MLP_bs1000_ep100000_pat1000_lr0.001_checkpoint.keras"
    
echo "=== Job finished ==="
