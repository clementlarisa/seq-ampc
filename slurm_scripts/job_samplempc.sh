#!/bin/bash
#SBATCH --job-name=samplempc
#SBATCH --partition=GPU-a40
#SBATCH --exclude=ivm-a40-q-2
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --output=samplempc_%A_%a.out
#SBATCH --error=samplempc_%A_%a.err
#SBATCH --array=0

set -euo pipefail

# ------------------- ENV -------------------
source /home/mihaela-larisa.clement/miniconda3/etc/profile.d/conda.sh
conda activate soeampc

export ACADOS_SOURCE_DIR=/home/mihaela-larisa.clement/acados
export ACADOS_INCLUDE_PATH="$ACADOS_SOURCE_DIR/include"
export ACADOS_LIB_PATH="$ACADOS_SOURCE_DIR/lib"
export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib64:$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ACADOS_SOURCE_DIR/bin:${PATH:-}"

# Prevent thread oversubscription when using multiple Python processes
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Safe diagnostics
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "nproc=$(nproc)"

python - <<'PY'
import os
print("os.cpu_count() =", os.cpu_count())
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    print(k, "=", os.environ.get(k))
PY

python - <<'PY'
import os
try:
    print("sched_getaffinity =", len(os.sched_getaffinity(0)))
except Exception as e:
    print("sched_getaffinity unavailable:", e)
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
PY

echo "=== Sampling MPC ==="
echo "Python: $(which python)"
echo "ACADOS_SOURCE_DIR: $ACADOS_SOURCE_DIR"

# Sanity checks
python -c "import sys; print(sys.executable)"
python -c "import acados_template; print(acados_template.__file__)"
command -v t_renderer >/dev/null && echo "t_renderer OK"
test -f "$ACADOS_LIB_PATH/libacados.so" && echo "libacados.so OK"
test -f "$ACADOS_INCLUDE_PATH/acados_c/ocp_nlp_interface.h" && echo "acados headers OK"

# total 20k across 3 jobs:
# tasks 0 and 1 -> 417 samples/instance
# task 2       -> 416 samples/instance
if [[ "${SLURM_ARRAY_TASK_ID}" == "2" ]]; then
    SAMPLES_PER_INSTANCE=25
else
    SAMPLES_PER_INSTANCE=25
fi

INSTANCES=40
PREFIX="ClusterTestingDataset_${SLURM_ARRAY_TASK_ID}_"

echo "INSTANCES=$INSTANCES"
echo "SAMPLES_PER_INSTANCE=$SAMPLES_PER_INSTANCE"
echo "PREFIX=$PREFIX"

python /home/mihaela-larisa.clement/soeampc/examples/vehicle_obs/samplempc.py \
    parallel_sample_mpc \
    --instances=${INSTANCES} \
    --samplesperinstance=${SAMPLES_PER_INSTANCE} \
    --prefix=${PREFIX}

echo "=== Finished sampling ==="