import os
import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

from models import (
    create_moco_resnet50,
    initialize_moco_momentum_encoder,
    update_moco_momentum_encoder,
    MoCoLoss
)
from data import (
    MoCoDataset,
    ImageNet100Dataset,
    load_imagenet100,
    get_moco_augmentations,
    get_evaluation_transform
)
from utils import (
    print_gpu_info,
    setup_mig_environment,
    setup,
    cleanup,
    get_device,
    wrap_ddp_model,
    create_distributed_sampler
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MoCo on ImageNet-100 with Distributed Data Parallel')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='global batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='initial learning rate (default: 0.03)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--moco-m', type=float, default=0.999,
                        help='MoCo momentum for updating key encoder (default: 0.999)')
    parser.add_argument('--queue-size', type=int, default=4096,
                        help='size of memory queue (default: 4096)')
    parser.add_argument('--dim', type=int, default=2048,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval in epochs (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers per GPU (default: 4)')
    parser.add_argument('--log-dir', type=str, default='runs/moco_distributed',
                        help='tensorboard log directory (default: runs/moco_distributed)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints/moco',
                        help='checkpoint directory (default: ../checkpoints/moco)')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='number of GPUs to use (default: 4)')
    parser.add_argument('--backend', type=str, default='gloo',
                        help='distributed backend: gloo or nccl (default: gloo)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    
    return parser.parse_args()

# Extract features function for evaluation
def extract_features(model, data_loader, device):
    """Extract features for the entire dataset"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            
            # Get features
            features = model(images)
            
            # Normalize features
            features = F.normalize(features, dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate features and labels
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels

# Evaluation functions
def evaluate_knn(train_features, train_labels, test_features, test_labels, k=20):
    """Perform kNN classification and return top-1 and top-5 accuracy"""
    # Initialize kNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    
    # Fit on training data
    knn.fit(train_features, train_labels)
    
    # Predict probabilities for test data
    probs = knn.predict_proba(test_features)
    
    # Get top-5 predictions
    top5_preds = np.argsort(-probs, axis=1)[:, :5]
    
    # Calculate top-1 accuracy
    top1_correct = (top5_preds[:, 0] == test_labels).sum()
    top1_accuracy = top1_correct / len(test_labels)
    
    # Calculate top-5 accuracy
    top5_correct = 0
    for i, label in enumerate(test_labels):
        if label in top5_preds[i]:
            top5_correct += 1
    top5_accuracy = top5_correct / len(test_labels)
    
    return top1_accuracy, top5_accuracy

def calculate_mAP(query_features, query_labels, gallery_features, gallery_labels):
    """Calculate Mean Average Precision for retrieval"""
    # Compute cosine similarity
    similarities = np.dot(query_features, gallery_features.T)
    
    # Sort gallery indices by similarity for each query
    sorted_indices = np.argsort(-similarities, axis=1)
    
    # Calculate AP for each query
    aps = []
    for i, query_label in enumerate(query_labels):
        # Get sorted gallery labels for this query
        retrieved_labels = gallery_labels[sorted_indices[i]]
        
        # Find relevant items (same class as query)
        relevant = (retrieved_labels == query_label)
        
        # If no relevant items found, skip this query
        if not relevant.any():
            continue
        
        # Calculate cumulative sum of relevant items
        cumsum_relevant = np.cumsum(relevant)
        
        # Calculate precision at each position where a relevant item is found
        precisions = cumsum_relevant[relevant] / (np.arange(len(relevant))[relevant] + 1)
        
        # Calculate average precision
        ap = precisions.mean()
        aps.append(ap)
    
    # Return mean of average precisions
    return np.mean(aps) if aps else 0.0

def run_evaluation(model, train_loader, val_loader, device, writer, epoch):
    """Run KNN and mAP evaluation"""
    print(f"Running evaluation at epoch {epoch}...")
    
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    
    # KNN evaluation
    top1_acc, top5_acc = evaluate_knn(train_features, train_labels, val_features, val_labels)
    
    # mAP evaluation
    mAP = calculate_mAP(val_features, val_labels, train_features, train_labels)
    
    # Log to TensorBoard
    writer.add_scalar('KNN/top1_acc', top1_acc, epoch)
    writer.add_scalar('KNN/top5_acc', top5_acc, epoch)
    writer.add_scalar('Retrieval/mAP', mAP, epoch)
    
    print(f"Top-1 acc: {top1_acc:.4f}, Top-5 acc: {top5_acc:.4f}, mAP: {mAP:.4f}")
    
    return top1_acc

# Create evaluation data loaders
def create_eval_dataloaders(hf_dataset, batch_size=256, num_workers=4):
    """Create train and validation dataloaders for evaluation"""
    eval_transform = get_evaluation_transform()
    
    # Create datasets
    train_dataset = ImageNet100Dataset(hf_dataset, split="train", transform=eval_transform)
    val_dataset = ImageNet100Dataset(hf_dataset, split="validation", transform=eval_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for feature extraction
        num_workers=num_workers,
        pin_memory=True
    )
    
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_on_gpu(
    rank, 
    world_size,
    args
):
    # Set environment for this process to see only one MIG device
    setup_mig_environment(rank, world_size)
    
    # Initialize distributed process group
    setup(rank, world_size)
    
    # Print GPU info for this process
    print_gpu_info()
    
    # Get device
    device = get_device(rank)
    
    # Create log directory for this rank
    rank_log_dir = os.path.join(args.log_dir, f'rank_{rank}')
    os.makedirs(rank_log_dir, exist_ok=True)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(rank_log_dir)
    
    try:
        # Create models
        model_q, model_k = create_moco_resnet50(dim=args.dim, pretrained=args.pretrained)
        
        # Move models to device
        model_q = model_q.to(device)
        model_k = model_k.to(device)
        
        # Initialize momentum encoder with same weights as query encoder
        initialize_moco_momentum_encoder(model_q, model_k)
        
        # Wrap model_q with DDP
        model_q = wrap_ddp_model(model_q, device_id=0)
        
        # Load dataset
        hf_dataset = load_imagenet100()
        
        # Get MoCo augmentation
        moco_transform = get_moco_augmentations()
        
        # Create base dataset
        base_train_dataset = ImageNet100Dataset(hf_dataset, split="train", transform=None)
        
        # Create MoCo dataset with two augmentations
        train_dataset = MoCoDataset(base_train_dataset, moco_transform)
        
        # Create distributed sampler
        train_sampler = create_distributed_sampler(train_dataset, rank, world_size)
        
        # Create data loader with distributed sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size // world_size,  # Divide batch size among GPUs
            shuffle=False,  # Sampler handles shuffling
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler
        )
        
        # Create evaluation dataloaders (only for rank 0)
        if rank == 0:
            eval_train_loader, eval_val_loader = create_eval_dataloaders(
                hf_dataset=hf_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        
        # Define optimizer - use per-GPU learning rate
        # Scale the learning rate by world_size because gradients are averaged
        optimizer = torch.optim.SGD(
            model_q.parameters(),
            lr=args.lr * world_size,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs
        )
        
        # Initialize the MoCo loss
        criterion = MoCoLoss(
            dim=args.dim,
            queue_size=args.queue_size,
            temperature=args.temp
        ).to(device)
        
        # Training loop
        best_top1 = 0.0
        for epoch in range(args.epochs):
            # Set epoch for distributed sampler
            train_sampler.set_epoch(epoch)
            
            model_q.train()
            total_loss = 0.0
            
            for batch_idx, ((im_q, im_k), _) in enumerate(train_loader):
                im_q, im_k = im_q.to(device), im_k.to(device)
                
                # Get query features
                q = model_q(im_q)
                
                # Get key features (no gradient)
                with torch.no_grad():
                    # Update momentum encoder
                    # Note: we need to access the module inside DDP
                    update_moco_momentum_encoder(model_q.module, model_k, args.moco_m)
                    
                    # Get key features
                    k = model_k(im_k)
                
                # Calculate loss
                loss = criterion(q, k)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log loss
                total_loss += loss.item()
                
                # Log batch progress (only from rank 0)
                if rank == 0 and batch_idx % 10 == 0:
                    # Log to TensorBoard
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/batch', loss.item(), step)
                    
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            # Update learning rate
            scheduler.step()
            
            # Log epoch statistics (only from rank 0)
            if rank == 0:
                avg_loss = total_loss / len(train_loader)
                writer.add_scalar('Loss/epoch', avg_loss, epoch)
                writer.add_scalar('LR/learning_rate', scheduler.get_last_lr()[0], epoch)
                
                print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                
                # Run evaluation every eval_interval epochs or on the last epoch
                if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
                    # For evaluation, use the module inside DDP
                    top1_acc = run_evaluation(
                        model_q.module, eval_train_loader, eval_val_loader, 
                        device, writer, epoch
                    )
                    
                    # Save best model
                    if top1_acc > best_top1:
                        best_top1 = top1_acc
                        print(f"New best model with Top-1 accuracy: {best_top1:.4f}")
                        torch.save({
                            'epoch': epoch,
                            'model_q_state_dict': model_q.module.state_dict(),
                            'model_k_state_dict': model_k.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                            'best_top1': best_top1,
                            'dim': args.dim,
                        }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_q_state_dict': model_q.module.state_dict(),
                        'model_k_state_dict': model_k.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'dim': args.dim,
                    }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save final model (only from rank 0)
        if rank == 0:
            torch.save({
                'model_q_state_dict': model_q.module.state_dict(),
                'model_k_state_dict': model_k.state_dict(),
                'dim': args.dim,
            }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
            
            print(f"Training completed. Best Top-1 accuracy: {best_top1:.4f}")
    
    except Exception as e:
        print(f"Process {rank} encountered error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close TensorBoard writer
        writer.close()
        
        # Always clean up
        cleanup()

def main():
    args = parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Use the specified number of GPUs or available MIG instances
    from utils.distributed import MIG_UUIDS
    world_size = min(args.num_gpus, len(MIG_UUIDS))
    print(f"Starting distributed training with {world_size} GPUs")
    
    # Spawn processes
    mp.spawn(
        train_on_gpu,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()