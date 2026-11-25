#!/bin/bash
#SBATCH --job-name=adl_select
#SBATCH --output=search_select_%j.out
#SBATCH --error=search_select_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=12:00:00

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
srun --unbuffered "$retrain_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "$retrain_env/bin/python" -u select_search_architecture.py "$@"
