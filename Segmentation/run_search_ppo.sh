#!/bin/bash
#SBATCH --job-name=adl_cell_search
#SBATCH --output=search_corrected_%j.out
#SBATCH --error=search_corrected_%j.err
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

# Independent runs: SEARCH_SEED=1/2/3; never share an output directory.
# --smoke-test exercises all stages in a separate *_smoke directory.
# Resubmit the SAME command after each 48h allocation; --resume latest is automatic.
srun --unbuffered "$retrain_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "$retrain_env/bin/python" -u train_autodeeplab.py \
  --seed "${SEARCH_SEED:-1}" \
  --output-dir "${SEARCH_DIR:-search_lora_cvar_v2_seed${SEARCH_SEED:-1}}" \
  --epochs 100 --warmup-epochs 50 --batch-size 6 --crop-size 321 \
  --filter-multiplier 8 --lora-rank 4 --lora-lr-multiplier 2 \
  --risk-weight 0.2 --cvar-alpha 0.2 --workers 8 --resume latest "$@"
