#!/bin/bash
#SBATCH --job-name=adl_candidate4_500k
#SBATCH --output=candidate4_500k_%j.out
#SBATCH --error=candidate4_500k_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000m
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300

# Time-limited choice authorized by the user: candidate 4 based on seed-0
# probes only. This is not a two-seed-qualified winner export.
# Fresh full-data F20 training; never point RUN_DIR at a probe directory.
# Resubmit this identical command after each allocation to resume.
set -eo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
export RUN_DIR="${RUN_DIR:-retrain_candidate4_f20_500k_seed0_nosdp}"
export MAX_STEPS=500000
exec bash run_retrain_baseline.sh \
  --architecture-json local_candidates_v1_seed0/candidate_4.json \
  --filter-multiplier 20 --seed 0 --bn-momentum 0.01 \
  --drop-path-prob 0 --power 0.9 --accumulation-steps 1 \
  --val-every 5000 --save-every 1000 --snapshot-every 50000 "$@"
