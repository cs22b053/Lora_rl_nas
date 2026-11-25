#!/bin/bash
#SBATCH --job-name=adl_reference
#SBATCH --output=reference_retrain_%j.out
#SBATCH --error=reference_retrain_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300

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

# The full reference-length schedule is 1.5M optimizer updates, not epochs.
# For a shorter fixed budget, set MAX_STEPS BEFORE the first submission; keep it
# unchanged on resume. Runtime must be measured for this larger corrected model.
# e.g. MAX_STEPS=740000 sbatch run_retrain_baseline.sh
# A short hardware/data check (retains the full LR schedule):
# sbatch run_retrain_baseline.sh --stop-after-steps 200 --val-every 200
# Resume that run with --val-every 5000 and without --stop-after-steps.
# OOM: start a NEW directory with --batch-size 8 --accumulation-steps 2.
# BN then sees 8 samples, so this is not identical to physical batch 16.
srun --unbuffered "$retrain_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "$retrain_env/bin/python" -u train.py \
  --data-dir "${RUN_DIR:-data_reference_f32}" \
  --max-steps "${MAX_STEPS:-1500000}" \
  --filter-multiplier 32 \
  --batch-size 16 \
  --crop-size 769 \
  --base-lr 0.05 \
  --weight-decay 0.0001 \
  --warmup-steps 1000 \
  --drop-path-prob 0.1 \
  --workers 8 \
  --resume latest \
  "$@"
