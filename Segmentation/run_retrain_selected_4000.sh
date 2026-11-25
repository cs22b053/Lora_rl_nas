#!/bin/bash
#SBATCH --job-name=adl_selected_4000
#SBATCH --output=selected_4000_%j.out
#SBATCH --error=selected_4000_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=48:00:00

# Run bash setup_retrain_env.sh once before submitting (isolated CUDA 11.8 build).
# Start fresh, never point RUN_DIR at a 20k-pilot directory.
# Resubmit this same command to continue after each 48-hour allocation.
# Optional initial check: sbatch run_retrain_selected_4000.sh --stop-after-steps 5000
# Then resubmit without --stop-after-steps; keep every recipe setting unchanged.
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
retrain_env="${RETRAIN_VENV:-$PWD/.venvs/retrain_cu118}"
if [[ ! -x "$retrain_env/bin/python" || ! -f "$retrain_env/.autodeeplab-ready" ]]; then
  echo 'Environment setup is incomplete. Run bash setup_retrain_env.sh on the login node first.' >&2
  exit 1
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
srun --unbuffered "$retrain_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "$retrain_env/bin/python" -u train.py \
  --data-dir "${RUN_DIR:-retrain_selected_seed0_4000_bn001}" \
  --architecture-json "${ARCHITECTURE_JSON:-architecture_seed0_halfres_reranked.json}" \
  --epochs 4000 --batch-size 16 --accumulation-steps 1 \
  --filter-multiplier 32 --crop-size 769 \
  --base-lr 0.05 --weight-decay 0.0001 --warmup-steps 1000 \
  --power 0.9 --drop-path-prob 0.1 --bn-momentum 0.01 \
  --seed 0 --workers 8 --val-every 5000 --save-every 1000 \
  --snapshot-every 50000 --resume latest "$@"
