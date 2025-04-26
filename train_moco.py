import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tqdm import tqdm

from models import (
    create_moco_resnet50,
    initialize_moco_momentum_encoder,
    update_moco_momentum_encoder,
    MoCoLoss
)
from data import (
    setup_imagenet100_training,
    create_eval_dataloaders,
    load_imagenet100
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MoCo on ImageNet-100')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size per GPU (default: 256)')
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
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--log-dir', type=str, default='runs/moco',
                        help='tensorboard log directory (default: runs/moco)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints/moco',
                        help='checkpoint directory (default: ../checkpoints/moco)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    
    return parser.parse_args()

# Extract features function
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

# Evaluation function
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

# mAP calculation
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

# Run evaluation
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

def train_moco():
    args = parse_args()
    
    # Create log and checkpoint directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    model_q, model_k = create_moco_resnet50(dim=args.dim, pretrained=args.pretrained)
    
    # Move models to device
    model_q = model_q.to(device)
    model_k = model_k.to(device)
    
    # Initialize momentum encoder with same weights as query encoder
    initialize_moco_momentum_encoder(model_q, model_k)
    
    # Create data loader for training
    train_loader = setup_imagenet100_training(args.batch_size, args.num_workers)
    
    # Create evaluation data loaders
    hf_dataset = load_imagenet100()
    eval_train_loader, eval_val_loader = create_eval_dataloaders(
        hf_dataset=hf_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Define optimizer
    optimizer = torch.optim.SGD(
        model_q.parameters(),
        lr=args.lr,
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
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_top1 = 0.0
    for epoch in range(args.epochs):
        model_q.train()
        total_loss = 0.0
        
        for batch_idx, ((im_q, im_k), _) in enumerate(train_loader):
            im_q, im_k = im_q.to(device), im_k.to(device)
            
            # Get query features
            q = model_q(im_q)
            
            # Get key features (no gradient)
            with torch.no_grad():
                # Update momentum encoder
                update_moco_momentum_encoder(model_q, model_k, args.moco_m)
                
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
            
            # Log batch progress
            if batch_idx % 10 == 0:
                # Log to TensorBoard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/batch', loss.item(), step)
                
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch statistics
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('LR/learning_rate', scheduler.get_last_lr()[0], epoch)
        
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Run evaluation every eval_interval epochs or on the last epoch
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            top1_acc = run_evaluation(
                model_q, eval_train_loader, eval_val_loader, 
                device, writer, epoch
            )
            
            # Save best model
            if top1_acc > best_top1:
                best_top1 = top1_acc
                print(f"New best model with Top-1 accuracy: {best_top1:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_q_state_dict': model_q.state_dict(),
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
                'model_q_state_dict': model_q.state_dict(),
                'model_k_state_dict': model_k.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dim': args.dim,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'model_q_state_dict': model_q.state_dict(),
        'model_k_state_dict': model_k.state_dict(),
        'dim': args.dim,
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
    
    writer.close()
    print(f"Training completed. Best Top-1 accuracy: {best_top1:.4f}")
    
    return model_q

if __name__ == '__main__':
    train_moco()