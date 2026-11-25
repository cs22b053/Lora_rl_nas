#!/bin/bash
#SBATCH --job-name=adl_posteval
#SBATCH --output=posteval_%A_%a.out
#SBATCH --error=posteval_%A_%a.err
#SBATCH --array=0-3%1
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000m
#SBATCH --time=12:00:00

# Evaluation only. Apply the same protocol to the reference and candidate 4.
# Tasks: 0=reference best, 1=candidate4 best, 2=reference final, 3=candidate4 final.
# Default: stored versus train-only calibrated BN, single-scale FP32.
# Optional: POSTEVAL_MODE=tta sbatch --array=0-1%1 run_posteval_candidate4.sh
# TTA uses stored BN, AMP, five scales plus horizontal flips; report separately.
# Inspect commands without starting Python/GPU work: bash this_script --dry-run
set -euo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"

dry_run=0
if [[ $# -eq 1 && "$1" == --dry-run ]]; then
  dry_run=1
elif [[ $# -ne 0 ]]; then
  echo 'Only --dry-run is supported; select bn or tta with POSTEVAL_MODE.' >&2
  exit 2
fi

task="${SLURM_ARRAY_TASK_ID:-0}"
case "$task" in
  0) label=reference_best; run=retrain_reference_f20_500k_seed0_nosdp; saved=best.pth ;;
  1) label=candidate4_best; run=retrain_candidate4_f20_500k_seed0_nosdp; saved=best.pth ;;
  2) label=reference_final; run=retrain_reference_f20_500k_seed0_nosdp; saved=step_500000.pth ;;
  3) label=candidate4_final; run=retrain_candidate4_f20_500k_seed0_nosdp; saved=step_500000.pth ;;
  *) echo 'Array task must be 0, 1, 2 or 3.' >&2; exit 2 ;;
esac

checkpoint="$run/$saved"
retrain_env="${RETRAIN_VENV:-$PWD/.venvs/retrain_cu118}"
mode="${POSTEVAL_MODE:-bn}"
result_root="${POSTEVAL_ROOT:-posteval_candidate4_500k}"
case "$mode" in
  bn)
    command=("$retrain_env/bin/python" -u diagnose_retrain_bn.py
      --checkpoint "$checkpoint" --output "$result_root/bn/$label.json"
      --batch-size 16 --calibration-batches 128 --workers 8 --seed 0)
    ;;
  tta)
    command=("$retrain_env/bin/python" -u evaluate.py
      --checkpoint "$checkpoint" --eval-batch-size 1 --workers 8
      --eval_scales 0.5 0.75 1.0 1.25 1.5 --eval-flip
      --results-dir "$result_root/tta/$label")
    ;;
  *) echo 'POSTEVAL_MODE must be bn or tta.' >&2; exit 2 ;;
esac

if [[ ! -f "$checkpoint" ]]; then
  echo "Missing completed-run checkpoint: $checkpoint" >&2
  exit 1
fi
echo "Evaluation only: $label, mode=$mode, checkpoint=$checkpoint"
if [[ "$dry_run" -eq 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -x "$retrain_env/bin/python" || ! -f "$retrain_env/.autodeeplab-ready" ]]; then
  echo 'Use the existing prepared retrain environment; its ready marker is missing.' >&2
  exit 1
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
srun --unbuffered "$retrain_env/bin/python" -u check_retrain_cuda.py
srun --unbuffered "${command[@]}"
