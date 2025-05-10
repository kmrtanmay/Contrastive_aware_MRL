import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import (
    create_matryoshka_resnet50,
    MatryoshkaLoss
)
from data import (
    ImageNet100Dataset,
    load_imagenet100
)
from utils.mrl_evaluation import run_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description='Train Matryoshka Representation Learning on ImageNet-100')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,  # Reduced learning rate for stability
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nesting-list', type=str, default='8,16,32,64,128,256,512,1024,2048',
                        help='comma-separated list of nesting dimensions (default: 8,16,32,64,128,256,512,1024,2048)')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval in epochs (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--log-dir', type=str, default='runs/mrl',
                        help='tensorboard log directory (default: runs/mrl)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints/mrl',
                        help='checkpoint directory (default: ../checkpoints/mrl)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='gradient clipping value (default: 1.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing value (default: 0.1)')
    parser.add_argument('--efficient', action='store_true',
                        help='use efficient implementation of MRL')
    
    return parser.parse_args()

def train_mrl():
    args = parse_args()
    
    # Parse nesting list
    nesting_list = [int(dim) for dim in args.nesting_list.split(',')]
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_matryoshka_resnet50(nesting_list, args.pretrained)
    model = model.to(device)
    
    # Load dataset
    hf_dataset = load_imagenet100()
    
    # Define data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    train_dataset = ImageNet100Dataset(hf_dataset, split="train", transform=train_transform)
    val_dataset = ImageNet100Dataset(hf_dataset, split="validation", transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create loss function
    criterion = MatryoshkaLoss(
        nesting_list=nesting_list,
        num_classes=100,  # ImageNet-100 has 100 classes
        label_smoothing=args.label_smoothing,
        # Create weights that emphasize smaller dimensions slightly more
        relative_importance=[1.0 + 0.05 * (len(nesting_list) - i) for i in range(len(nesting_list))]
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr / 10  # Don't let LR drop too low
    )
    
    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        criterion.train()
        
        # Training statistics
        total_loss = 0.0
        epoch_losses = {f"dim_{dim}": 0.0 for dim in nesting_list}
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            features = model(inputs)
            loss, individual_losses = criterion(features, targets)
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected at batch {batch_idx}. Skipping batch.")
                continue
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            for i, dim in enumerate(nesting_list):
                epoch_losses[f"dim_{dim}"] += individual_losses[i].item()
            
            # Log progress
            if batch_idx % 10 == 0:
                # Log to TensorBoard
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/batch_total', loss.item(), step)
                for i, dim in enumerate(nesting_list):
                    writer.add_scalar(f'Loss/batch_dim_{dim}', individual_losses[i].item(), step)
                
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                dim_losses = [f"{dim}: {individual_losses[i].item():.4f}" for i, dim in enumerate(nesting_list)]
                print(f"Dimension losses: {', '.join(dim_losses)}")
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch statistics
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/epoch_total', avg_loss, epoch)
        for dim in nesting_list:
            writer.add_scalar(f'Loss/epoch_dim_{dim}', epoch_losses[f"dim_{dim}"] / len(train_loader), epoch)
        writer.add_scalar('LR/learning_rate', scheduler.get_last_lr()[0], epoch)
        
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            avg_acc = run_evaluation(
                model, criterion,
                train_loader, val_loader,
                device, writer, epoch
            )
            
            # Save best model
            if avg_acc > best_acc:
                best_acc = avg_acc
                print(f"New best model with average accuracy: {best_acc:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'nesting_list': nesting_list,
                }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'nesting_list': nesting_list,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_state_dict': criterion.state_dict(),
        'nesting_list': nesting_list,
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
    
    writer.close()
    print(f"Training completed. Best average accuracy: {best_acc:.4f}")
    
    return model, criterion

if __name__ == '__main__':
    train_mrl()