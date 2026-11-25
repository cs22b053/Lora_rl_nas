#!/bin/bash
#SBATCH --job-name=ptb_dyneval
#SBATCH --output=ptb_dyneval_%j.out
#SBATCH --error=ptb_dyneval_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi

# ====== Dynamic Evaluation on the BEST model (56.81 static) ======
# This adapts model weights during test time — a standard technique
# that gives ~2-3 PPL improvement. Used by AWD-LSTM, ENAS, DARTS papers.
python dynamic_eval.py \
  --data=data/penn \
  --model_path=models/train_best_lora_cvar_1

