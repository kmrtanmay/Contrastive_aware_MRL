import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from models import (
    create_moco_matryoshka_resnet50,
    initialize_moco_momentum_encoder,
    update_moco_momentum_encoder,
    MoCoMatryoshkaLoss
)
from data import (
    setup_imagenet100_training,
    create_eval_dataloaders,
    load_imagenet100
)
from utils import run_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description='Train MoCo Matryoshka on ImageNet-100')
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
    parser.add_argument('--nesting-list', type=str, default='8,16,32,64,128,256,512,1024,2048',
                        help='comma-separated list of nesting dimensions (default: 8,16,32,64,128,256,512,1024,2048)')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval in epochs (default: 5)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--log-dir', type=str, default='runs/matryoshka_moco',
                        help='tensorboard log directory (default: runs/matryoshka_moco)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='checkpoint directory (default: checkpoints)')
    
    return parser.parse_args()

def train_matryoshka_moco():
    args = parse_args()
    
    # Parse nesting list
    nesting_list = [int(dim) for dim in args.nesting_list.split(',')]
    
    # Create log and checkpoint directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    model_q, model_k = create_moco_matryoshka_resnet50(nesting_list)
    
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
    
    # Initialize the MoCo Matryoshka loss
    criterion = MoCoMatryoshkaLoss(
        nesting_list=nesting_list,
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
        epoch_losses = {f"dim_{dim}": 0.0 for dim in nesting_list}
        
        for batch_idx, ((im_q, im_k), _) in enumerate(train_loader):
            im_q, im_k = im_q.to(device), im_k.to(device)
            
            # Get query features
            q = model_q(im_q)  # [batch_size, full_dim]
            
            # Get key features (no gradient)
            with torch.no_grad():
                # Update momentum encoder
                update_moco_momentum_encoder(model_q, model_k, args.moco_m)
                
                # Get key features
                k = model_k(im_k)  # [batch_size, full_dim]
            
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
            
            # Log batch progress
            if batch_idx % 10 == 0:
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
        
        # Log epoch statistics
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/epoch_total', avg_loss, epoch)
        for dim in nesting_list:
            writer.add_scalar(f'Loss/epoch_dim_{dim}', epoch_losses[f"dim_{dim}"] / len(train_loader), epoch)
        writer.add_scalar('LR/learning_rate', scheduler.get_last_lr()[0], epoch)
        
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Run evaluation every eval_interval epochs or on the last epoch
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            avg_top1 = run_evaluation(
                model_q, nesting_list, 
                eval_train_loader, eval_val_loader, 
                device, writer, epoch
            )
            
            # Save best model
            if avg_top1 > best_top1:
                best_top1 = avg_top1
                print(f"New best model with Top-1 accuracy: {best_top1:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_q_state_dict': model_q.state_dict(),
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
                'model_q_state_dict': model_q.state_dict(),
                'model_k_state_dict': model_k.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'nesting_list': nesting_list,
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'model_q_state_dict': model_q.state_dict(),
        'model_k_state_dict': model_k.state_dict(),
        'nesting_list': nesting_list,
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))
    
    writer.close()
    print(f"Training completed. Best Top-1 accuracy: {best_top1:.4f}")
    
    return model_q

if __name__ == '__main__':
    train_matryoshka_moco()