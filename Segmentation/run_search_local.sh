#!/bin/bash
#SBATCH --job-name=adl_local_search
#SBATCH --output=local_search_%j.out
#SBATCH --error=local_search_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
export SEARCH_SEED="${SEARCH_SEED:-0}"
export SEARCH_DIR="${SEARCH_DIR:-search_local_v1_seed${SEARCH_SEED}}"
exec bash run_search_ppo.sh --search-space reference-local --local-candidates 128 \
  --candidate-seed 0 --target-filters 20 --max-final-params 10147427 "$@"
