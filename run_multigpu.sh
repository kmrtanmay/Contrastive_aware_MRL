#!/bin/bash

# Make the script exit on any error
set -e

# Clear any existing CUDA environment variables that might interfere
unset CUDA_VISIBLE_DEVICES

# Print system information
echo "-------------- System Information --------------"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Directory: $(pwd)"
echo "-------------- GPU Information --------------"
nvidia-smi

# Detect the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs available for training"

if [ $NUM_GPUS -eq 0 ]; then
    echo "No GPUs detected. Exiting."
    exit 1
fi

# Allow specifying the number of GPUs as a command-line argument
if [ $# -gt 0 ]; then
    NUM_GPUS_TO_USE=$1
    echo "Using $NUM_GPUS_TO_USE GPUs as specified"
else
    NUM_GPUS_TO_USE=$NUM_GPUS
    echo "Using all $NUM_GPUS available GPUs"
fi

# Set environment variables for better distributed training
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN

# Run the multi-GPU training script
echo "-------------- Starting Training --------------"
python train_multigpu.py \
  --batch-size 256 \
  --epochs 100 \
  --num-gpus $NUM_GPUS_TO_USE \
  --checkpoint-dir ../checkpoints/moco_mrl

echo "Training completed"