# Contrastive_aware_MRL
Fusing Contrastive Learning with Matryoshka Representation 

# MoCo-Matryoshka

This repository implements Matryoshka Representation Learning (MRL) combined with Momentum Contrast (MoCo) for self-supervised visual representation learning. The implementation is based on PyTorch and distributed training is supported.

## Overview

Matryoshka Representation Learning creates nested representations where smaller dimensions are subsets of larger ones, allowing for flexible compute-performance trade-offs at inference time. MoCo is a contrastive learning framework that builds a dynamic dictionary of encoded representations for unsupervised learning. This implementation combines these approaches to create nested contrastive representations.

## Features

- Combined Matryoshka Representation Learning and MoCo contrastive objectives
- Single-GPU and distributed multi-GPU training support
- Support for MIG (Multi-Instance GPU) partitioning
- TensorBoard integration for tracking metrics
- Evaluation with kNN and mean Average Precision (mAP)
- Flexible nesting dimensions configuration

## Installation

```bash
git clone https://github.com/kmrtanmay/main/contrastive_MRL/Contrastive_aware_MRL.git
cd Contrastive_aware_MRL
pip install -r requirements.txt
```

## Dataset

This implementation uses ImageNet-100, a subset of ImageNet with 100 classes, accessed through Hugging Face's datasets library. The dataset will be automatically downloaded when you run the training script.

## Usage

### Single-GPU Training

```bash
python train.py --batch-size 256 --epochs 100
```

### Multi-GPU Distributed Training

```bash
python train_distributed.py --batch-size 256 --epochs 100 --num-gpus 4
```

### Configuration Options

You can modify various hyperparameters using command-line arguments:

```bash
python train.py --help
```

Key parameters:

- `--batch-size`: Batch size (per GPU for distributed training)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--nesting-list`: Comma-separated list of nesting dimensions
- `--queue-size`: Size of MoCo memory queue 
- `--moco-m`: MoCo momentum coefficient for updating key encoder
- `--temp`: Temperature for softmax in contrastive loss
- `--eval-interval`: Evaluation frequency in epochs

## Project Structure

- `models/`: Model architecture and loss functions
  - `matryoshka_loss.py`: Matryoshka contrastive loss implementation
  - `moco_matry