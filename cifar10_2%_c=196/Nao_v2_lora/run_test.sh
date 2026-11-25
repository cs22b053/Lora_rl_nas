#!/bin/bash
#SBATCH --job-name=nao
#SBATCH --output=nao_search_%j.out
#SBATCH --error=nao_search_%j.err
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:a100_3g.20gb
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi
MODEL=search_cifar10_test
OUTPUT_DIR=exp_test/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

python train_search.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --child_batch_size=512 \
    --child_lr_max=0.2 \
    --child_epochs=2 \
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
    --controller_seed_arch=50 \
    --controller_new_arch=20 \
    --controller_epochs=50 \
    | tee -a $OUTPUT_DIR/train.log
