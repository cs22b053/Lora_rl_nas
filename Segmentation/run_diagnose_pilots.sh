#!/bin/bash
#SBATCH --job-name=adl_bn_check
#SBATCH --output=pilot_bn_%A_%a.out
#SBATCH --error=pilot_bn_%A_%a.err
#SBATCH --array=0-1
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=04:00:00

set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
retrain_env="${RETRAIN_VENV:-$PWD/.venvs/retrain_cu118}"
if [[ ! -f "$retrain_env/.autodeeplab-ready" ]]; then
  echo 'Run bash setup_retrain_env.sh on the login node first.' >&2
  exit 1
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
pilots=(pilot_reference_seed0_20k pilot_selected_seed0_20k)
pilot="${pilots[${SLURM_ARRAY_TASK_ID:-0}]}"
srun --unbuffered "$retrain_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "$retrain_env/bin/python" -u diagnose_retrain_bn.py \
  --checkpoint "$pilot/latest.pth" \
  --output "$pilot/bn_diagnostic.json" \
  --batch-size 16 --calibration-batches 128 --seed 0 "$@"
