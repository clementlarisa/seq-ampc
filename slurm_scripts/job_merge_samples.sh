#!/bin/bash
#SBATCH --job-name=merge_samples
#SBATCH --partition=GPU-a40
#SBATCH --exclude=ivm-a40-q-2
#SBATCH --time=04:00:00
#SBATCH --output=merge_samples_%j.out
#SBATCH --error=merge_samples_%j.err

set -euo pipefail

# ------------------- ENV -------------------
source /home/mihaela-larisa.clement/miniconda3/etc/profile.d/conda.sh
conda activate soeampc

# ------------------- CONFIG -------------------
NEW_DATASET_NAME="vehicle_obs_N_55000_$(date +%Y%m%d-%H%M%S)"
REMOVE_AFTER_MERGE=0   # 1=true, 0=false

ARCHIVE_ROOT="/share/mihaela-larisa.clement/soeampc-data/archive"

FOLDERS=(
#   "vehicle_8state_obs_N_6672_merged_ClusterTest_0__20260224-224711"
#   "vehicle_8state_obs_N_6672_merged_ClusterTest_1__20260224-224711"
#   "vehicle_8state_obs_N_6656_merged_ClusterTest_2__20260224-224711"
#   "vehicle_8state_obs_N_6000_20260223-063727-20260223T192306Z-1-001/vehicle_8state_obs_N_6000_20260223-063727"
#   "vehicle_8state_obs_N_5000_20260222-210617-20260223T192304Z-1-001/vehicle_8state_obs_N_5000_20260222-210617"
#   "vehicle_8state_obs_N_1000_20260221-202735-20260223T192300Z-1-001/vehicle_8state_obs_N_1000_20260221-202735"
    # "vehicle_8state_obs_N_32000_merged_vehicle_8state_obs_N_32000_merged_20260224-232550"
    # "vehicle_8state_obs_N_20000_merged_Cluster40_0__20260225-075254"
    # "vehicle_8state_obs_N_20000_merged_Cluster40_1__20260225-065254"
    # "vehicle_8state_obs_N_20000_merged_Cluster40_2__20260225-065254"
    # "vehicle_8state_obs_N_8000_merged_ClusterTestDataset_0__20260225-004730"
    # "vehicle_8state_obs_N_8000_merged_ClusterTestDataset_1__20260225-000141"
    # "vehicle_8state_obs_N_8000_merged_ClusterTestDataset_2__20260225-020243"
    "vehicle_obs_N_5000_20260225-061104"
    "vehicle_obs_N_6000_20260224-044528"
    "vehicle_obs_N_6656_merged_ClusterTest_2__20260224-225302"
    "vehicle_obs_N_6672_merged_ClusterTest_0__20260224-214724"
    "vehicle_obs_N_6672_merged_ClusterTest_1__20260224-214724"
    "vehicle_obs_N_8000_merged_ClusterTestDataset_0__20260225-003923"
    "vehicle_obs_N_8000_merged_ClusterTestDataset_1__20260224-233923"
    "vehicle_obs_N_8000_merged_ClusterTestDataset_2__20260224-233923"
)

echo "=== Starting merge ==="
echo "Archive root: $ARCHIVE_ROOT"
echo "New dataset name: $NEW_DATASET_NAME"
echo "Remove after merge: $REMOVE_AFTER_MERGE"
printf ' - %s\n' "${FOLDERS[@]}"

# Build Python list literal safely from bash array
FOLDERS_PY=$(printf '"%s",' "${FOLDERS[@]}")
FOLDERS_PY="[${FOLDERS_PY%,}]"

python - <<PY
from pathlib import Path

# Adjust this import if mergesamples is in another module
from soeampc.datasetutils import mergesamples

archive_root = Path("${ARCHIVE_ROOT}")
folder_names = ${FOLDERS_PY}
new_dataset_name = "${NEW_DATASET_NAME}"
remove_after_merge = bool(${REMOVE_AFTER_MERGE})

print("Checking source folders...")
missing = [f for f in folder_names if not (archive_root / f).exists()]
if missing:
    print("ERROR: The following folders do not exist under", archive_root)
    for m in missing:
        print(" -", m)
    raise SystemExit(1)

print("Merging folders:")
for f in folder_names:
    print(" -", f)

out = mergesamples(
    folder_names,
    new_dataset_name=new_dataset_name,
    remove_after_merge=remove_after_merge,
)

print("\\nMerged dataset:", out)
print("Merged path:", archive_root / out)
PY

echo "=== Finished merge ==="