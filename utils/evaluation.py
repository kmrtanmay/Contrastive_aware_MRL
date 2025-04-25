import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

def extract_features(model, data_loader, nesting_list, device):
    """
    Extract features for the entire dataset at each nesting dimension
    
    Args:
        model: Trained model
        data_loader: DataLoader for the dataset
        nesting_list: List of dimensions for the Matryoshka representation
        device: Device to use
        
    Returns:
        all_features: Dict of nested features for each dimension
        all_labels: Labels for each image
    """
    model.eval()
    all_features = {dim: [] for dim in nesting_list}
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            
            # Get features for the full dimension
            features = model(images)  # [batch_size, full_dim]
            
            # Store features for each nesting dimension
            for dim in nesting_list:
                # Extract features up to the nesting dimension and normalize
                nested_features = F.normalize(features[:, :dim], dim=1)
                all_features[dim].append(nested_features.cpu().numpy())
            
            all_labels.append(labels.numpy())
    
    # Concatenate features for each dimension
    for dim in nesting_list:
        all_features[dim] = np.concatenate(all_features[dim], axis=0)
    
    # Concatenate labels
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_features, all_labels

def evaluate_knn(train_features, train_labels, test_features, test_labels, k=20):
    """
    Perform kNN classification and return top-1 and top-5 accuracy
    
    Args:
        train_features: Training set features
        train_labels: Training set labels
        test_features: Test set features
        test_labels: Test set labels
        k: Number of neighbors
        
    Returns:
        top1_accuracy: Top-1 accuracy
        top5_accuracy: Top-5 accuracy
    """
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
    """
    Calculate Mean Average Precision for retrieval
    
    Args:
        query_features: Query set features
        query_labels: Query set labels
        gallery_features: Gallery set features
        gallery_labels: Gallery set labels
        
    Returns:
        mAP: Mean Average Precision
    """
    # Compute cosine similarity between queries and gallery
    # Since features are already normalized, dot product equals cosine similarity
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

def run_evaluation(model, nesting_list, train_loader, val_loader, device, writer, epoch):
    """
    Run KNN and mAP evaluation at specified nesting dimensions
    Log results to TensorBoard
    
    Args:
        model: Trained model
        nesting_list: List of dimensions for the Matryoshka representation
        train_loader: Training set dataloader
        val_loader: Validation set dataloader
        device: Device to use
        writer: TensorBoard SummaryWriter
        epoch: Current epoch
        
    Returns:
        avg_top1: Average top-1 accuracy across all dimensions
    """
    print(f"Running evaluation at epoch {epoch}...")
    
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, nesting_list, device)
    val_features, val_labels = extract_features(model, val_loader, nesting_list, device)
    
    # Evaluate for each nesting dimension
    for dim in nesting_list:
        print(f"Evaluating dimension {dim}...")
        
        # KNN evaluation
        top1_acc, top5_acc = evaluate_knn(
            train_features[dim], train_labels,
            val_features[dim], val_labels
        )
        
        # m