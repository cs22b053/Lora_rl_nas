#!/bin/bash
#SBATCH --job-name=nao
#SBATCH --output=nao_train%j.out
#SBATCH --error=nao_train%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1        # CHANGED: 2 -> 1 (Your code can't use 2 GPUs)
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4       # CHANGED: 32 -> 16 (Matches num_workers=16 in python code)
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi
MODEL=NAONet_V2_36_cifar10_0
OUTPUT_DIR=exp_0/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

fixed_arc="0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 1 1 0 1 0 1 0 1 0 3 0 1 0 1 0 1 0 1 0 1"


python train_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --arch="$fixed_arc" \
  --use_aux_head \
  --cutout_size=16 | tee -a $OUTPUT_DIR/train.log