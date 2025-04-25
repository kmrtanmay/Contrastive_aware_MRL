import os
import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models import (
    create_moco_matryoshka_resnet50,
    initialize_moco_momentum_encoder,
    update_moco_momentum_encoder,
    MoCoMatryoshkaLoss
)
from data import (
    MoCoDataset,
    ImageNet100Dataset,
    load_imagenet100,
    get_moco_augmentations,
    get_evaluation_transform,
    create_eval_dataloaders
)
from utils import (
    print_gpu_info,
    setup_mig_environment,
    setup,
    cleanup,
    get_device,
    wrap_ddp_model,
    create_distributed_sampler,
    run_evaluation
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MoCo Matryoshka on ImageNet-100 with Distributed Data Parallel')
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
    parser.add_argument('--nesting-list', type=str, default='8,16,32,64,128,256,512,1024,2048',
                        help='comma-separated list of nesting dimensions (default: 8,16,32,64,128,256,512,1024,2048)')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval in epochs (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers per GPU (default: 4)')
    parser.add_argument('--log-dir', type=str, default='runs/matryoshka_moco_distributed',
                        help='tensorboard log directory (default: runs/matryoshka_moco_distributed)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='checkpoint directory (default: checkpoints)')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='number of GPUs to use (default: 4)')
    parser.add_argument('--backend', type=str, default='gloo',
                        help='distributed backend: gloo or nccl (default: gloo)')
    
    return parser.parse_args()

def train_on_gpu(
    rank, 
    world_size,
    args
):
    # Parse nesting list
    nesting_list = [int(dim) for dim in args.nesting_list.split(',')]
    
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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(rank_log_dir)
    
    try:
        # Create models
        model_q, model_k = create_moco_matryoshka_resnet50(nesting_list)
        
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
        
        # Initialize the MoCo Matryoshka loss
        criterion = MoCoMatryoshkaLoss(
            nesting_list=nesting_list,
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
            epoch_losses = {f"dim_{dim}": 0.0 for dim in nesting_list}
            
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
                loss, individual_losses = criterion(q, k)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log losses
                total_loss += loss.item()
                for i, dim in enumerate(nesting_list):
                    epoch_losses[f"dim_{dim}"] += individual_losses[i].item()
                
                # Log batch progress (only from rank 0)
                if rank == 0 and batch_idx % 10 == 0:
                    # Log to TensorBoard
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/batch_total', loss.item(), step)
                    for i, dim in enumerate(nesting_list):
                        writer.add_scalar(f'Loss/batch_dim_{dim}', individual_losses[i].item(), step)
                    
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                    dim_losses = [f"{dim}: {loss.item():.4f}" for dim, loss in zip(nesting_list, individual_losses)]
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
                
                # Run evaluation every eval_interval epochs or on the last epoch
                if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
                    # For evaluation, use the module inside DDP
                    avg_top1 = run_evaluation(
                        model_q.module, nesting_list, 
                        eval_train_loader, eval_val_loader, 
                        device, writer, epoch
                    )
                    
                    # Save best model
                    if avg_top1 > best_top1:
                        best_top1 = avg_top1
                        print(f"New best model with Top-1 accuracy: {best_top1:.4f}")
                        torch.save({
                            'epoch': epoch,
                            'model_q_state_dict': model_q.module.state_dict(),
                            'model_k_state_dict': model_k.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                            'best_top1': best_top1,
                            'nesting_list': nesting_list,
                        }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_q_state_dict': model_q.module.state_dict(),
                        'model_k_state_dict': model_k.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'nesting_list': nesting_list,
                    }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save final model (only from rank 0)
        if rank == 0:
            torch.save({
                'model_q_state_dict': model_q.module.state_dict(),
                'model_k_state_dict': model_k.state_dict(),
                'nesting_list': nesting_list,
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