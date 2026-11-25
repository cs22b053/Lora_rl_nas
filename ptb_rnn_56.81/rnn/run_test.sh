#!/bin/bash
#SBATCH --job-name=ptb_test
#SBATCH --output=ptb_test%j.out
#SBATCH --error=ptb_test%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi

MODEL_DIR=models/train_best_lora_cvar_v2
LOG_DIR=logs
DATA_DIR=data/penn

mkdir -p $LOG_DIR

python test.py \
  --data=$DATA_DIR \
  --save=$MODEL_DIR
