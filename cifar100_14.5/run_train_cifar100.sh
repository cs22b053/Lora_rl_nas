#!/bin/bash
#SBATCH --job-name=nao_v2_train_c100
#SBATCH --output=nao_train_c100_%j.out
#SBATCH --error=nao_train_c100_%j.err
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

# ============================================================
# CONFIGURATION — CIFAR-100 Training (train_cifar.py)
# ============================================================
# Best 10-op arch from search (same arch that got 98% on CIFAR-10)
ARCH="0 1 0 4 2 3 1 8 1 8 3 3 0 8 2 8 2 1 1 8 0 4 0 3 0 7 0 0 2 8 1 3 1 8 0 1 2 0 0 2"

EXPERIMENT="NAONet_V2_cifar100_train"
OUTPUT_DIR="exp_v2/$EXPERIMENT"
DATA_DIR="data"

SEED=0

mkdir -p $OUTPUT_DIR

echo "============================================="
echo "Training architecture on CIFAR-100 (train_cifar.py)"
echo "Arch: $ARCH"
echo "Output dir: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "============================================="

# ============================================================
# TRAINING RECIPE (train_cifar.py)
# ============================================================
# 600 epochs, 36 channels, 6 layers — the simpler training script
# CIFAR-100 has 100 classes (set via --dataset=cifar100)
# ============================================================
python train_cifar.py \
    --data=$DATA_DIR \
    --dataset=cifar100 \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --arch="$ARCH" \
    --batch_size=128 \
    --epochs=600 \
    --layers=6 \
    --nodes=5 \
    --channels=36 \
    --cutout_size=16 \
    --lr_max=0.025 \
    --lr_min=0 \
    --keep_prob=0.6 \
    --drop_path_keep_prob=0.8 \
    --l2_reg=5e-4 \
    --use_aux_head \
    --seed=$SEED \
    | tee -a $OUTPUT_DIR/train.log

echo "CIFAR-100 training complete. Results in $OUTPUT_DIR"
