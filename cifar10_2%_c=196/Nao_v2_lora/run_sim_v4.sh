#!/bin/bash
#SBATCH --job-name=nao
#SBATCH --output=nao_test_%j.out
#SBATCH --error=nao_test_%j.err
#SBATCH --partition=mediumq
#SBATCH --qos=mediumq
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40000M
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda activate /scratch/cs22b053/nao_env

unset LD_LIBRARY_PATH

nvidia-smi

# Define model name and output directory
MODEL=NAONet_V2_128_cifar10_TEST_1
OUTPUT_DIR=exp_orchid_std/$MODEL
DATA_DIR=data

mkdir -p $OUTPUT_DIR

# The same architecture string used in training
fixed_arc="0 6 0 9 0 3 0 5 3 7 0 9 0 1 1 5 3 7 5 2 0 7 0 8 1 6 0 7 2 4 2 4 4 6 3 0 2 7 4 6"

python test_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --arch="$fixed_arc" \
  --use_aux_head \
  --cutout_size=16 \
  --channels=128 \
  --layers=6 \
  --epochs=600 \
  --batch_size=64 | tee -a $OUTPUT_DIR/test.log