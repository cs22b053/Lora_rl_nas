#!/bin/bash
#SBATCH --job-name=ptb_train
#SBATCH --output=ptb_train%j.out
#SBATCH --error=ptb_train%j.err
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

MODEL_DIR=models/train_best_lora_cvar_1
LOG_DIR=logs
DATA_DIR=data/penn

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

# Best architecture found by LoRA + CVaR RL v2 search (loss=5.280, PPL≈197 on WS supernet)
fixed_arc="0 3 1 2 1 2 3 2 3 2 3 2 1 1 1 2 3 2 5 1 5 3"

python train.py \
  --data=$DATA_DIR \
  --save=$MODEL_DIR \
  --epochs=8000 \
  --arch="$fixed_arc" 2>&1 | tee -a $LOG_DIR/train.best_lora_cvar.log
