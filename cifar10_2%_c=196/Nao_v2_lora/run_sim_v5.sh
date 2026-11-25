#!/bin/bash
#SBATCH --job-name=nao_10op_v5
#SBATCH --output=nao_search_%j.out
#SBATCH --error=nao_search_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi
MODEL=search_cifar10_v0
OUTPUT_DIR=exp_1/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

python train_search.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --child_batch_size=512 \
    --child_lr_max=0.20 \
    --child_epochs=50 \
    --lora_rank=16 \
    --lora_lr_multiplier=10.0 \
    | tee -a $OUTPUT_DIR/train.log
