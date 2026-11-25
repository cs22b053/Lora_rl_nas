#!/bin/bash
#SBATCH --job-name=nao_imagenet
#SBATCH --output=nao_test_imagenet_%j.out
#SBATCH --error=nao_test_imagenet_%j.err
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

# ============================================================
# CONFIGURATION
# ============================================================
# Best 10-op arch from search (top-1 from arch_pool.final, 91.12% supernet valid)
ARCH="0 1 0 4 2 3 1 8 1 8 3 3 0 8 2 8 2 1 1 8 0 4 0 3 0 7 0 0 2 8 1 3 1 8 0 1 2 0 0 2"

# Experiment name
EXPERIMENT="NAONet_V2_Imagenet"
OUTPUT_DIR="exp_v2/$EXPERIMENT"
DATA_DIR="/scratch/data/imagenet"

# Seed for reproducibility (change for multi-seed runs)
SEED=0

mkdir -p $OUTPUT_DIR

echo "============================================="
echo "Training architecture: $ARCH"
echo "Output dir: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "============================================="

python train_imagenet.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --arch="$ARCH" \
    --search_space=middle \
    --batch_size=128 \
    --epochs=250 \
    --layers=4 \
    --channels=48 \
    --lr=0.1 \
    --keep_prob=1.0 \
    --drop_path_keep_prob=1.0 \
    --l2_reg=3e-5 \
    --use_aux_head \
    --label_smooth=0.1 \
    --lazy_load \
    --seed=$SEED \
    | tee -a $OUTPUT_DIR/train.log

echo "Training complete. Results in $OUTPUT_DIR"
echo "Training Done for ImageNet dataset"
