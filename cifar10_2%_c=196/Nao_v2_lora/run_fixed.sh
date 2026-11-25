#!/bin/bash
#SBATCH --job-name=nao_search
#SBATCH --output=nao_search_%j.out
#SBATCH --error=nao_search_%j.err
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi
MODEL=search_cifar10_fixed
OUTPUT_DIR=exp_fixed/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

python train_search.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --child_batch_size=1024 \
    --child_lr_max=0.40 \
    --child_epochs=50 \
    --lora_rank=16 \
    --lora_lr_multiplier=4.0 \
    | tee -a $OUTPUT_DIR/train.log
