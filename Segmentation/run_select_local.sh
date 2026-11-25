#!/bin/bash
#SBATCH --job-name=adl_local_select
#SBATCH --output=local_select_%j.out
#SBATCH --error=local_select_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=12:00:00
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
export SEARCH_SEED="${SEARCH_SEED:-0}"
local_search_dir="${SEARCH_DIR:-search_local_v1_seed${SEARCH_SEED}}"
local_candidates_dir="${CANDIDATES_DIR:-local_candidates_v1_seed${SEARCH_SEED}}"
local_env="${RETRAIN_VENV:-$PWD/.venvs/retrain_cu118}"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
bash run_select_search.sh --search-dirs "$local_search_dir" \
  --rank-images 595 --calibration-batches 32 --all-visited \
  --output "$local_candidates_dir/proxy_ranking.json" "$@"
"$local_env/bin/python" local_experiments.py prepare \
  --ranking "$local_candidates_dir/proxy_ranking.json" --search-dir "$local_search_dir" \
  --output "$local_candidates_dir" --challengers 4
