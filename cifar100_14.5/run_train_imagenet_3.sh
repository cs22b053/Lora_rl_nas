#!/bin/bash
#SBATCH --job-name=nao_imgnet_v4
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
# CONFIGURATION — Improved ImageNet Training
# ============================================================
# Best 10-op arch from NAO-V2 LoRA+CVaR search (98% CIFAR-10)
ARCH="0 1 0 4 2 3 1 8 1 8 3 3 0 8 2 8 2 1 1 8 0 4 0 3 0 7 0 0 2 8 1 3 1 8 0 1 2 0 0 2"

EXPERIMENT="NAONet_V2_Imagenet_cosine"
OUTPUT_DIR="exp_v2/$EXPERIMENT"
DATA_DIR="/scratch/data/imagenet"

SEED=0

mkdir -p $OUTPUT_DIR

echo "============================================="
echo "ImageNet Training — Cosine LR + Warmup + DropPath"
echo "Arch: $ARCH"
echo "Output dir: $OUTPUT_DIR"
echo "Seed: $SEED"
echo "============================================="

# ============================================================
# IMPROVED TRAINING RECIPE
# ============================================================
# Key changes vs previous run (73% plateau):
#   1. Cosine annealing LR (was: StepLR gamma=0.97)
#      - At epoch 150: cosine LR ≈ 0.02 vs StepLR 0.0017
#      - Maintains learning signal 10x longer
#   2. 5-epoch linear warmup (stabilizes early training)
#   3. Drop path prob 0.2 (linearly scheduled 0→0.2)
#      - Regularizes to close train-val gap (was 9.4%)
#   4. 300 epochs (extra 50 epochs for cosine tail)
#   5. Batch 256 + LR 0.1 (linear scaling from base LR 0.05)
#
# A100 optimizations: batch_size=256, 16 workers, 80GB RAM
# ============================================================
python train_imagenet.py \
    --data=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --arch="$ARCH" \
    --batch_size=256 \
    --eval_batch_size=256 \
    --epochs=300 \
    --layers=14 \
    --nodes=5 \
    --channels=48 \
    --lr=0.1 \
    --lr_min=0.0 \
    --lr_schedule=cosine \
    --warmup_epochs=5 \
    --label_smooth=0.1 \
    --l2_reg=3e-5 \
    --grad_clip=5.0 \
    --drop_path_prob=0.2 \
    --use_aux_head \
    --keep_prob=1.0 \
    --lazy_load \
    --seed=$SEED \
    | tee -a $OUTPUT_DIR/train.log

echo "Training complete. Results in $OUTPUT_DIR"
echo "Cosine LR + Warmup + DropPath, 14 layers, 48 channels, 300 epochs"
