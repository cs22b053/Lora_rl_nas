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

EXPERIMENT="NAONet_V2_cifar100_train_2"
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

python train_cifar.py \
    --data=$DATA_DIR \
    --dataset=cifar100 \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --arch="$ARCH" \
    --use_aux_head \
    --cutout_size=16 | tee -a $OUTPUT_DIR/train.log