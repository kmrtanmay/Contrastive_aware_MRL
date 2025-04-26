import os
import argparse
import torch
import torch.multiprocessing as mp
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
from utils import (
    print_gpu_info,
    setup_mig_environment,
    setup,
    cleanup,
    get_device,
    wrap_ddp_model,
    create_distributed_sampler
)
from utils.mrl_evaluation import run_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description='Train Supervised Matryoshka Representation Learning on ImageNet-100 with Distributed Data Parallel')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='global batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
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
                        help='number of data loading workers per GPU (default: 4)')
    parser.add_argument('--log-dir', type=str, default='runs/mrl_distributed',
                        help='tensorboard log directory (default: runs/mrl_distributed)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints/mrl',
                        help='checkpoint directory (default: ../checkpoints/mrl)')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='number of GPUs to use (default: 4)')
    parser.add_argument('--backend', type=str, default='gloo',
                        help='distributed backend: gloo or nccl (default: gloo)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='gradient clipping value (default: 1.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing value (default: 0.1)')
    parser.add_argument('--efficient', action='store_true',
                        help='use efficient implementation of MRL')
    
    return parser.parse_args()

def train_on_gpu(rank, world_size, args):
    # Parse nesting list
    nesting_list = [int(dim) for dim in args.nesting_list.split(',')]
    
    # Set up distributed environment
    setup_mig_environment(rank, world_size)
    setup(rank, world_size)
    print_gpu_info()
    device = get_device(rank)
    
    # Create log directory for this rank
    rank_log_dir = os.path.join(args.log_dir, f'rank_{rank}')
    os.makedirs(rank_log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(rank_log_dir)
    
    try:
        # Create model
        model = create_matryoshka_resnet50(nesting_list, args.pretrained)
        model = model.to(device)
        
        # Wrap model with DistributedDataParallel
        model = wrap_ddp_model(model, device_id=0)
        
        # Load dataset
        hf_dataset = load_imagenet100()
        
        # Define data transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # Add color jitter for better generalization
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),  # Add random grayscale for robustness
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
        
        # Create distributed sampler
        train_sampler = create_distributed_sampler(train_dataset, rank, world_size)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size // world_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Create validation loader (only for rank 0)
        if rank == 0:
            val_dataset = ImageNet100Dataset(hf_dataset, split="validation", transform=val_transform)
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
        
        # Create optimizer - scale learning rate by world_size
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(criterion.parameters()),
            lr=args.lr * world_size,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=args.lr / 10  # Don't let LR drop too low
        )
        
        # Training loop
        best_acc = 0.0
        for epoch in range(args.epochs):
            # Set sampler epoch
            train_sampler.set_epoch(epoch)
            
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
                
                # Check for NaN loss and skip if found
                if torch.isnan(loss).any():
                    if rank == 0:
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
                
                # Log progress (only from rank 0)
                if rank == 0 and batch_idx % 10 == 0:
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
            
            # Log epoch statistics (only from rank 0)
            if rank == 0:
                avg_loss = total_loss / len(train_loader)
                writer.add_scalar('Loss/epoch_total', avg_loss, epoch)
                for dim in nesting_list:
                    writer.add_scalar(f'Loss/epoch_dim_{dim}', epoch_losses[f"dim_{dim}"] / len(train_loader), epoch)
                writer.add_scalar('LR/learning_rate', scheduler.get_last_lr()[0], epoch)
                
                print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                
                # Evaluation
                if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
                    # Use unwrapped model for evaluation
                    avg_acc = run_evaluation(
                        model.module, criterion,
                        train_loader, val_loader,
                        device, writer, epoch
                    )
                    
                    # Save best model
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        print(f"New best model with average accuracy: {best_acc:.4f}")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'loss_state_dict': criterion.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'nesting_list': nesting_list,
                        }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'loss_state_dict': criterion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'nesting_list': nesting_list,
                    }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save final model (only from rank 0)
        if rank == 0:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'loss_state_dict': criterion.state_dict(),
                'nesting_list': nesting_list,
            }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
            
            print(f"Training completed. Best average accuracy: {best_acc:.4f}")
    
    except Exception as e:
        print(f"Process {rank} encountered error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        writer.close()
        cleanup()

def main():
    args = parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Get MIG UUIDs
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