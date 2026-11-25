#!/bin/bash
#SBATCH --job-name=nao_lora
#SBATCH --output=nao_search%j.out
#SBATCH --error=nao_search%j.err
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --gres=gpu:a100:1        # CHANGED: 2 -> 1 (Your code can't use 2 GPUs)
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4       # CHANGED: 32 -> 16 (Matches num_workers=16 in python code)
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi
MODEL=search_cifar10_0
OUTPUT_DIR=exp_fixed_10op/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

python train_search.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --child_batch_size=512 \
    --child_lr_max=0.2 \
    --child_epochs=50 \
    --child_layers=3 \
    --child_channels=20 \
    --lora_rank=16 \
    --lora_lr_multiplier=10.0 \
    --controller_mlp_layers=2 \
    --controller_mlp_hidden_size=200 \
    --controller_encoder_hidden_size=128 \
    --controller_encoder_emb_size=64 \
    --controller_encoder_layers=2 \
    --controller_decoder_hidden_size=128 \
    --controller_decoder_layers=2 \
    --controller_seed_arch=1000 \
    --controller_new_arch=300 \
    --controller_epochs=1000