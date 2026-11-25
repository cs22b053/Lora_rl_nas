#!/bin/bash
#SBATCH --job-name=adl_final_bn
#SBATCH --output=selected_final_bn_%j.out
#SBATCH --error=selected_final_bn_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=04:00:00

# Frozen-weight, train-only BN calibration diagnostic of the completed best model.
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
exec bash run_diagnose_pilots.sh \
  --checkpoint retrain_selected_seed0_4000_bn001/best.pth \
  --output retrain_selected_seed0_4000_bn001/bn_final_diagnostic.json "$@"
