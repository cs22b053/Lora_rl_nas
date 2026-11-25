#!/bin/bash
#SBATCH --job-name=adl_reference_4000
#SBATCH --output=reference_4000_%j.out
#SBATCH --error=reference_4000_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300

# Matched control for the completed selected-cell run: same training recipe,
# published cell, fresh initialization. Resubmit the same command to resume.
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
export RUN_DIR="${RUN_DIR:-retrain_reference_seed0_4000_bn001}"
export MAX_STEPS="${MAX_STEPS:-740000}"
exec bash run_retrain_baseline.sh --seed 0 --bn-momentum 0.01 "$@"
