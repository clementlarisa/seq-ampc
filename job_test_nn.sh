#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=GPU-a40
#SBATCH --exclude=ivm-a40-q-2
#SBATCH --gres=gpu:a40:1
#SBATCH --time=72:00:00
#SBATCH --output=eval_RNN_%j.out
#SBATCH --error=eval_RNN_%j.err
#SBATCH --array=0

# ------------------- CONFIG -------------------
DATASET="quadcopter_N_120000_test/quadcopter_N_120000_merged_Cluster_2023_01_06_05_51_PM_31926010_2_20230106-175146"

MODEL_ROOT="/share/mihaela-larisa.clement/soeampc-data/models"
#MODEL_ROOT="/share/mihaela-larisa.clement/soeampc-data/models/quadcopter/models"

MODELS=(
    "$MODEL_ROOT/NeuralType.MLP_rnn256_dense200_bs5000_ep100000_pat1000_lr0.001_tenth_checkpoint.keras"
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

python /home/mihaela-larisa.clement/soeampc/examples/quadcopter/safeonlineevaluation.py \
    evaluate_naive_ampc_on_dataset \
    --dataset="$DATASET" \
    --model_name="$MODEL_NAME"

echo "=== Finished model: $MODEL_NAME ==="
