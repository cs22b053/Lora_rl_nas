#!/bin/bash
#SBATCH --job-name=adl_local_final
#SBATCH --output=local_final_%j.out
#SBATCH --error=local_final_%j.err
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
  echo 'The private training environment is missing.' >&2
  exit 1
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
srun --unbuffered "$local_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "$local_env/bin/python" -u train_local_winner.py \
  --architecture-json "${CANDIDATES_DIR:-local_candidates_v1_seed0}/winner.json" \
  --data-dir "${RUN_DIR:-retrain_local_winner_f20_500k_seed0}" --seed "${FINAL_SEED:-0}" "$@"
