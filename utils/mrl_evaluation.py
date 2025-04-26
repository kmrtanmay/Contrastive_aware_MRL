import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def evaluate_accuracy(model, loss_fn, data_loader, device):
    """
    Evaluate accuracy at each nesting dimension
    
    Args:
        model: Trained model
        loss_fn: MatryoshkaLoss instance
        data_loader: DataLoader for evaluation
        device: Device to use
        
    Returns:
        accuracies: Dict mapping nesting dimensions to accuracy values
    """
    model.eval()
    nesting_list = loss_fn.nesting_list
    
    correct = {dim: 0 for dim in nesting_list}
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            features = model(inputs)
            
            # Get predictions for each nesting dimension
            predictions = loss_fn.predict(features)
            
            # Update correct counts
            for i, dim in enumerate(nesting_list):
                correct[dim] += (predictions[i] == targets).sum().item()
            
            total += targets.size(0)
    
    # Calculate accuracies
    accuracies = {dim: correct[dim] / total for dim in nesting_list}
    
    return accuracies

def extract_features(model, data_loader, nesting_list, device):
    """
    Extract features at each nesting dimension
    
    Args:
        model: Trained model
        data_loader: DataLoader for the dataset
        nesting_list: List of nesting dimensions
        device: Device to use
        
    Returns:
        features: Dict mapping dimensions to feature arrays
        labels: Array of labels
    """
    model.eval()
    features = {dim: [] for dim in nesting_list}
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            
            # Get full features
            output = model(inputs)
            
            # Store features for each nesting dimension
            for dim in nesting_list:
                # Extract and normalize features for this dimension
                nested_features = F.normalize(output[:, :dim], dim=1)
                features[dim].append(nested_features.cpu().numpy())
            
            # Store labels
            labels.append(targets.numpy())
    
    # Concatenate features and labels
    for dim in nesting_list:
        features[dim] = np.concatenate(features[dim], axis=0)
    
    labels = np.concatenate(labels, axis=0)
    
    return features, labels

def run_evaluation(model, loss_fn, train_loader, val_loader, device, writer, epoch):
    """
    Run evaluation and log results to TensorBoard
    
    Args:
        model: Trained model
        loss_fn: MatryoshkaLoss instance
        train_loader: Training set dataloader
        val_loader: Validation set dataloader
        device: Device to use
        writer: TensorBoard SummaryWriter
        epoch: Current epoch
        
    Returns:
        avg_acc: Average accuracy across all dimensions
    """
    print(f"Running evaluation at epoch {epoch}...")
    nesting_list = loss_fn.nesting_list
    
    # Evaluate accuracy
    val_accuracies = evaluate_accuracy(model, loss_fn, val_loader, device)
    
    # Log accuracies to TensorBoard
    for dim, acc in val_accuracies.items():
        writer.add_scalar(f'Accuracy/dim_{dim}', acc, epoch)
        print(f"Dimension {dim}: Accuracy: {acc:.4f}")
    
    # Calculate and log average accuracy
    avg_acc = sum(val_accuracies.values()) / len(val_accuracies)
    writer.add_scalar('Accuracy/average', avg_acc, epoch)
    print(f"Average accuracy: {avg_acc:.4f}")
    
    return avg_acc