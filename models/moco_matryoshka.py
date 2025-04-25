import torch
import torch.nn as nn
import torchvision.models as models

def create_moco_matryoshka_resnet50(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024, 2048]):
    """
    Creates a MoCo-compatible ResNet50 model with Matryoshka structure
    
    Args:
        nesting_list: List of dimensions for the Matryoshka representation
        
    Returns:
        model_q: Query encoder model
        model_k: Key encoder model (momentum updated)
    """
    # Create base ResNet50 models
    model_q = models.resnet50(pretrained=False)
    model_k = models.resnet50(pretrained=False)
    
    # Modify the final layer to output the full dimension
    full_dim = nesting_list[-1]
    
    # Replace the final FC layer with an identity layer that preserves the full feature dimension
    in_features = model_q.fc.in_features
    model_q.fc = nn.Linear(in_features, full_dim)
    model_k.fc = nn.Linear(in_features, full_dim)
    
    return model_q, model_k

def initialize_moco_momentum_encoder(model_q, model_k):
    """
    Initialize the momentum encoder with same weights as query encoder
    
    Args:
        model_q: Query encoder model
        model_k: Key encoder model to be initialized
    """
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False  # Key encoder not updated by gradient

def update_moco_momentum_encoder(model_q, model_k, momentum=0.999):
    """
    Update the momentum encoder using the momentum update rule
    
    Args:
        model_q: Query encoder model
        model_k: Key encoder model to be updated
        momentum: Momentum coefficient (default: 0.999)
    """
    with torch.no_grad():
        for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)