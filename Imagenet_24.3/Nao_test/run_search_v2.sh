#!/bin/bash
#SBATCH --job-name=nao_v2_search
#SBATCH --output=nao_search_%j.out
#SBATCH --error=nao_search_%j.err
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi
MODEL=search_cifar10_v2
OUTPUT_DIR=exp_v2/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

# Improved search with 4 iterations (PPO+CVaR actually runs),
# larger seed pool (2000), more child epochs (80), more new archs (500)
python train_search.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --child_batch_size=512 \
    --child_lr_max=0.2 \
    --child_epochs=80 \
    --child_layers=3 \
    --child_channels=24 \
    --lora_rank=16 \
    --lora_lr_multiplier=10.0 \
    --controller_mlp_layers=2 \
    --controller_mlp_hidden_size=200 \
    --controller_encoder_hidden_size=128 \
    --controller_encoder_emb_size=64 \
    --controller_encoder_layers=2 \
    --controller_decoder_hidden_size=128 \
    --controller_decoder_layers=2 \
    --controller_seed_arch=2000 \
    --controller_new_arch=500 \
    --controller_epochs=1000 \
    --controller_expand=10 \
    | tee -a $OUTPUT_DIR/train.log

echo "Search complete. Results in $OUTPUT_DIR"
echo "To train the best architecture, see run_train_v2.sh"
