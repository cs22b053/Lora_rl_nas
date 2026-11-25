#!/bin/bash
#SBATCH --job-name=nao_5op_improved
#SBATCH --output=nao_train_5op_%j.out
#SBATCH --error=nao_train_5op_%j.err
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi

# ============================================================
# QUICK WIN: Train the existing best 5-op arch (2.39% error)
# with improved training recipe. No new search needed.
# Expected: 2.39% -> ~1.9-2.1% error
# ============================================================

# Best 5-op architecture (from the LoRA search that achieved 2.39%)
ARCH="1 3 0 1 2 0 0 1 0 1 2 1 2 2 0 1 0 4 0 4 0 2 0 2 2 0 0 1 0 2 2 4 1 2 0 3 1 0 0 1"

SEED=${1:-0}  # Pass seed as first arg, default 0
EXPERIMENT="NAONet_5op_improved_seed${SEED}"
OUTPUT_DIR="exp_v2/$EXPERIMENT"
DATA_DIR="data"

mkdir -p $OUTPUT_DIR

echo "============================================="
echo "QUICK WIN: Improved training on existing best 5-op arch"
echo "Architecture: $ARCH"
echo "Output dir: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "Improvements: AutoAugment, LabelSmooth=0.1, SWA, AuxHead"
echo "Model: 192ch / 8L / 1000ep (vs 128ch / 6L / 800ep)"
echo "============================================="

python test_cifar.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --search_space=small \
    --arch="$ARCH" \
    --batch_size=96 \
    --epochs=1000 \
    --layers=8 \
    --nodes=5 \
    --channels=192 \
    --cutout_size=16 \
    --lr_max=0.025 \
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
