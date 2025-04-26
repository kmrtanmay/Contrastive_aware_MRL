import torch
import torch.nn as nn
import torchvision.models as models

def create_matryoshka_resnet50(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024, 2048], pretrained=False):
    """
    Creates a ResNet50 model with Matryoshka structure
    
    Args:
        nesting_list: List of dimensions for the Matryoshka representation
        pretrained: Whether to initialize with ImageNet weights
        
    Returns:
        model: ResNet50 model with modified output dimension
    """
    # Create base ResNet50 model
    model = models.resnet50(pretrained=False)
    
    # Modify the final layer to output the full dimension
    full_dim = nesting_list[-1]
    
    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, full_dim)
    
    # Initialize the weights properly (helps with stability)
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out')
    nn.init.zeros_(model.fc.bias)
    
    return model

class MatryoshkaFeatureExtractor(nn.Module):
    """
    Wrapper module that extracts features at different nesting dimensions
    """
    def __init__(self, base_model, nesting_list):
        super(MatryoshkaFeatureExtractor, self).__init__()
        self.base_model = base_model
        self.nesting_list = nesting_list
        
    def forward(self, x):
        # Get full features from base model
        features = self.base_model(x)
        
        # Extract features at each nesting dimension
        nested_features = {dim: features[:, :dim] for dim in self.nesting_list}
        
        return features, nested_features