#!/bin/bash
#SBATCH --job-name=nao_v2_train
#SBATCH --output=nao_train_%j.out
#SBATCH --error=nao_train_%j.err
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
# CONFIGURATION
# ============================================================
# Best 10-op arch from search (top-1 from arch_pool.final, 91.12% supernet valid)
ARCH="0 1 0 4 2 3 1 8 1 8 3 3 0 8 2 8 2 1 1 8 0 4 0 3 0 7 0 0 2 8 1 3 1 8 0 1 2 0 0 2"

# Experiment name
EXPERIMENT="NAONet_V2_test_v0"
OUTPUT_DIR="exp_v2/$EXPERIMENT"
DATA_DIR="data"

# Seed for reproducibility (change for multi-seed runs)
SEED=0

mkdir -p $OUTPUT_DIR

echo "============================================="
echo "Training architecture: $ARCH"
echo "Output dir: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "============================================="

# ============================================================
# IMPROVED TRAINING RECIPE
# ============================================================
# vs original: +AutoAugment, +LabelSmooth, +SWA, +AuxHead
# channels 192 (vs 128), layers 8 (vs 6), epochs 1000 (vs 800)
# Removed broken warm-restart at epoch 600
# ============================================================
python test_cifar.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --search_space=middle \
    --arch="$ARCH" \
    --batch_size=96 \
    --epochs=1000 \
    --layers=8 \
    --nodes=5 \
    --channels=192 \
    --cutout_size=16 \
    --lr_max=0.030 \
    --lr_min=0 \
    --keep_prob=0.6 \
    --drop_path_keep_prob=0.8 \
    --l2_reg=5e-4 \
    --use_aux_head \
    --autoaugment \
    --label_smooth=0.1 \
    --swa \
    --swa_start_frac=0.75 \
    --seed=$SEED \
    | tee -a $OUTPUT_DIR/train.log

echo "Training complete. Results in $OUTPUT_DIR"
