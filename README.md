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
git clone https://github.com/your-username/moco-matryoshka.git
cd moco-matryoshka
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
  - `moco_matryoshka.py`: MoCo model architecture with Matryoshka representation
- `data/`: Dataset loading and processing
  - `imagenet100.py`: ImageNet-100 dataset loading utilities
- `utils/`: Helper functions
  - `distributed.py`: Distributed training utilities
  - `evaluation.py`: Evaluation metrics and functions
- `train.py`: Single-GPU training script
- `train_distributed.py`: Multi-GPU distributed training script

## Code Explanation

### Matryoshka Loss

The key concept is the `MoCoMatryoshkaLoss` that applies contrastive learning at each nesting dimension:

```python
# Compute loss for each nesting dimension
for i, dim in enumerate(self.nesting_list):
    # Get features up to the current nesting dimension
    q_nested = q[:, :dim]  # [N, dim]
    k_nested = k[:, :dim]  # [N, dim]
    
    # Normalize features
    q_nested = F.normalize(q_nested, dim=1)
    k_nested = F.normalize(k_nested, dim=1)
    
    # Calculate InfoNCE loss for this dimension
    # ...
    
    individual_losses.append(loss)
```

This creates a representation where features at smaller dimensions are optimized alongside the full representation.

## Model Architecture

The model uses a ResNet-50 backbone with a modified final layer to output the full dimension required for the nested representation:

```python
def create_moco_matryoshka_resnet50(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024, 2048]):
    # Create base ResNet50 models
    model_q = models.resnet50(pretrained=False)
    model_k = models.resnet50(pretrained=False)
    
    # Replace the final FC layer
    full_dim = nesting_list[-1]
    in_features = model_q.fc.in_features
    model_q.fc = nn.Linear(in_features, full_dim)
    model_k.fc = nn.Linear(in_features, full_dim)
    
    return model_q, model_k
```

## Training and Evaluation

The training procedure follows the MoCo approach with a queue of negative samples but applies the loss across all nesting dimensions. Evaluation is performed at regular intervals using kNN classification and mAP metrics.

## Distributed Training

The distributed implementation supports MIG partitioning, which allows multiple isolated GPU instances on a single physical GPU. It uses PyTorch's DistributedDataParallel (DDP) for efficient multi-GPU training.

## Citation

If you use this code in your research, please cite the original Matryoshka Representation Learning and MoCo papers:

```
@inproceedings{kusupati2022matryoshka,
  title={Matryoshka Representation Learning},
  author={Kusupati, Aditya and Wallingford, Matthew and Salakhutdinov, Ruslan and Jain, Renjie and Carreira-Perpin{\'a}n, Miguel A and Farhadi, Ali},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{he2020momentum,
  title={Momentum Contrast for Unsupervised Visual Representation Learning},
  author={He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## License

MIT