#!/bin/bash
#SBATCH --job-name=closed_loop
#SBATCH --partition=GPU-a40
#SBATCH --exclude=ivm-a40-q-2
#SBATCH --gres=gpu:a40:1
#SBATCH --time=72:00:00
#SBATCH --output=closed_loop_%j.out
#SBATCH --error=closed_loop_%j.err
#SBATCH --array=0

# ------------------- CONFIG -------------------
DATASET="vehicle_8state_obs_N_32000_merged_vehicle_8state_obs_N_32000_merged_20260224-232550"
# DATASET="vehicle_8state_obs_N_116000_merged_vehicle_8state_obs_N_116000_20260225-093417"
# DATASET="vehicle_obs_N_55000_merged_vehicle_obs_N_55000_20260225-123451"

MODEL_ROOT="/share/mihaela-larisa.clement/soeampc-data/models"
#MODEL_ROOT="/share/mihaela-larisa.clement/soeampc-data/models/quadcopter/models"

MODELS=(
    # "vehicle_8state_32k_NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "vehicle_8state__NeuralType.MLP_rnn32_dense200x400x600x600x400x200_bs1000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "vehicle_8state_116k_NeuralType.MLP_bs1000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "vehicle_8state__NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "vehicle_NeuralType.MLP_bs10000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "vehicle_NeuralType.RNN_rnn256_dense200x400x600_bs1000_ep100000_pat1000_lr0.001_checkpoint.keras"
)


MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$MODEL_NAME" ]]; then
    echo "Invalid array index: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

MODEL_TAG=$(basename "$MODEL_NAME" .keras | tr '/.' '_')

# ------------------- ENV -------------------
source /home/mihaela-larisa.clement/miniconda3/etc/profile.d/conda.sh
conda activate soeampc

echo "=== Evaluating model: $MODEL_NAME ==="

# python /home/mihaela-larisa.clement/soeampc/examples/quadcopter/safeonlineevaluation.py \
#     closed_loop_test_on_dataset_plot \
#     --dataset="$DATASET" \
#     --model_name="$MODEL_NAME"

python /home/mihaela-larisa.clement/soeampc/examples/vehicle_dyn_obs/safeonlineevaluation.py \
    closed_loop_test_on_dataset_vehicle_obs \
    --dataset_dir="$DATASET" \
    --model_name="$MODEL_NAME" \
    --N_samples=1000 \
    --N_sim=1

echo "=== Finished model: $MODEL_NAME ==="
