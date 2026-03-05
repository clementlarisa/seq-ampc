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
DATASET="quadcopter_N_120000_test/quadcopter_N_120000_merged_Cluster_2023_01_06_05_51_PM_31926010_2_20230106-175146"

MODEL_ROOT="/share/mihaela-larisa.clement/soeampc-data/models"
# MODEL_ROOT="/share/mihaela-larisa.clement/soeampc-data/models/quadcopter/models"

MODELS=(
    # "$MODEL_ROOT/10-200-400-600-600-400-200-30_mu=0.12_20230104-232806"
    # "$MODEL_ROOT/NeuralType.RNN_rnn256_dense200x400x600_bs5000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.RNN_rnn256_dense200_bs10000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.RNN_rnn128_dense200x400x600x600_bs5000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.RNN_rnn128_dense200x400x600x600x400_bs5000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.MLP_rnn256_dense200_bs5000_ep100000_pat1000_lr0.0005_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.MLP_rnn256_dense200_bs5000_ep100000_pat1000_lr0.001_tenth_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.MLP_rnn256_dense200_bs5000_ep100000_pat1000_lr0.001_quarter_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.RNN_rnn256_dense200x400x600_bs5000_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.RNN_rnn256_dense200x400x600_bs6250_ep100000_pat1000_lr0.001_checkpoint.keras"
    # "$MODEL_ROOT/NeuralType.RNN_rnn256_dense200x400x600_bs6250_ep100000_pat1000_lr0.001_tenth_checkpoint.keras"
    "$MODEL_ROOT/NeuralType.RNN_rnn256_dense200x400x600_bs6250_ep100000_pat1000_lr0.001_quarter_checkpoint.keras"
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

python /home/mihaela-larisa.clement/soeampc/examples/quadcopter/safeonlineevaluation.py \
    closed_loop_test_reason \
    --dataset="$DATASET" \
    --model_name="$MODEL_NAME" \
    --N_samples=1000

echo "=== Finished model: $MODEL_NAME ==="
