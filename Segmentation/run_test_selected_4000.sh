#!/bin/bash
#SBATCH --job-name=adl_selected_eval
#SBATCH --output=selected_eval_%j.out
#SBATCH --error=selected_eval_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000m
#SBATCH --time=12:00:00

# Full-resolution Cityscapes validation of the selected model's best checkpoint.
# Optional multi-scale/flip: --eval_scales 0.5 0.75 1.0 1.25 1.5 --eval-flip
# Optional full precision: --eval-fp32
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
export RUN_DIR="${RUN_DIR:-retrain_selected_seed0_4000_bn001}"
exec bash run_test_baseline.sh --results-dir log_selected_4000_eval "$@"
