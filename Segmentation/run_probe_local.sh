#!/bin/bash
#SBATCH --job-name=adl_local_probe
#SBATCH --output=local_probe_%A_%a.out
#SBATCH --error=local_probe_%A_%a.err
#SBATCH --array=0-4%1
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
local_env="${RETRAIN_VENV:-$PWD/.venvs/retrain_cu118}"
if [[ ! -f "$local_env/.autodeeplab-ready" ]]; then
  echo 'The private training environment is missing; run setup_retrain_env.sh on the login node.' >&2
  exit 1
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
srun --unbuffered "$local_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "$local_env/bin/python" -u train_local_probe.py \
  --manifest "${CANDIDATES_DIR:-local_candidates_v1_seed0}/manifest.json" \
  --candidate "${SLURM_ARRAY_TASK_ID:-0}" --seed "${PROBE_SEED:-0}" \
  --stop-at-step "${STOP_AT_STEP:-50000}" "$@"
