#!/bin/bash

# Make the script exit on any error
set -e

# Clear any existing CUDA environment variables
unset CUDA_VISIBLE_DEVICES

# Print system information
echo "-------------- System Information --------------"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Directory: $(pwd)"
echo "-------------- GPU Information --------------"
nvidia-smi
echo "--------------  MIG Information --------------"
nvidia-smi -L

# Set environment variables to help with CUDA device detection
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Get the number of MIG instances
NUM_MIGS=$(nvidia-smi -L | grep -i MIG | wc -l)
echo "Detected $NUM_MIGS MIG instances"

if [ $NUM_MIGS -eq 0 ]; then
    echo "No MIG instances detected. Please configure MIG or use standard GPUs"
    exit 1
fi

# Run the distributed training with automatically detected MIG instances
echo "-------------- Starting Training --------------"
echo "Using automatic MIG UUID detection"
echo "Starting training with $NUM_MIGS MIG instances"

# MOCO+MRL
# python train_distributed.py \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-gpus $NUM_MIGS \
#   --checkpoint-dir ../checkpoints/moco_mrl

# MOCO only
python train_distributed_moco.py \
  --batch-size 256 \
  --epochs 100 \
  --num-gpus $NUM_MIGS \
  --checkpoint-dir ../checkpoints/moco_distributed

# MRL only
# python train_distributed_mrl.py \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-gpus $NUM_MIGS \
#   --checkpoint-dir ../checkpoints/mrl_distributed

