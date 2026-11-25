#!/bin/bash
#SBATCH --job-name=adl_search_ablation
#SBATCH --output=search_ablation_%A_%a.out
#SBATCH --error=search_ablation_%A_%a.err
#SBATCH --array=0-4
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300

# Five SEARCH controls; no child retraining is launched here.
# Submit with --array=0-4%1 for one GPU at a time. Repeat with SEARCH_SEED=1/2.
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
export SEARCH_SEED="${SEARCH_SEED:-0}"
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) arm=random; options=(--random-search --lora-rank 0 --risk-weight 0) ;;
  1) arm=ppo; options=(--lora-rank 0 --risk-weight 0) ;;
  2) arm=lora_ppo; options=(--lora-rank 4 --risk-weight 0) ;;
  3) arm=cvar_ppo; options=(--lora-rank 0 --risk-weight 0.2) ;;
  4) arm=lora_cvar_ppo; options=(--lora-rank 4 --risk-weight 0.2) ;;
  *) echo 'Expected array index 0 through 4' >&2; exit 1 ;;
esac
export SEARCH_DIR="search_ablation_v2_${arm}_seed${SEARCH_SEED}"
exec bash run_search_ppo.sh "${options[@]}" "$@"
