import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class MatryoshkaLoss(nn.Module):
    """
    Implements supervised loss for Matryoshka Representation Learning.
    Applies cross-entropy loss across each nesting dimension.
    Based on the original implementation.
    """
    def __init__(
        self, 
        nesting_list: List[int],
        num_classes: int = 100,  # Number of classes in ImageNet-100
        relative_importance: Optional[List[float]] = None,
        label_smoothing: float = 0.0
    ):
        super(MatryoshkaLoss, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        
        # Store relative importance as a PyTorch Parameter so it gets moved to the correct device
        if relative_importance is None:
            self.register_buffer('relative_importance', None)
        else:
            self.register_buffer('relative_importance', torch.tensor(relative_importance))
        
        # Create criterion with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Create classifiers for each nesting dimension
        self.classifiers = MRL_Linear_Layer(nesting_list, num_classes, efficient=False)
            
    def forward(self, 
                features: torch.Tensor,  # features [N, full_dim]
                targets: torch.Tensor    # target labels [N]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes classification loss at each nesting dimension.
        
        Args:
            features: Features of shape [N, full_dim]
            targets: Target labels of shape [N]
            
        Returns:
            total_loss: Weighted sum of losses across all nesting dimensions
            individual_losses: List of losses at each nesting dimension
        """
        # Get logits for each nesting dimension
        nesting_logits = self.classifiers(features)
        
        # Calculate losses for each output and stack them
        individual_losses = [self.criterion(logits, targets) for logits in nesting_logits]
        losses = torch.stack(individual_losses)
        
        # Set relative_importance to 1 if not specified
        if self.relative_importance is None:
            # Create tensor of ones directly on the same device as losses
            rel_importance = torch.ones_like(losses)
        else:
            # The relative_importance buffer will be on the same device as the model
            rel_importance = self.relative_importance
        
        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        
        # Return total loss and individual losses
        return weighted_losses.sum(), individual_losses
    
    def predict(self, features):
        """
        Get predictions at each nesting dimension
        
        Args:
            features: Features of shape [N, full_dim]
            
        Returns:
            predictions: List of prediction tensors for each nesting dimension
        """
        # Get logits for each nesting dimension
        nesting_logits = self.classifiers(features)
        
        # Get predictions
        predictions = []
        for logits in nesting_logits:
            _, preds = torch.max(logits, 1)
            predictions.append(preds)
        
        return predictions


class MRL_Linear_Layer(nn.Module):
    """
    Linear layer implementation for Matryoshka Representation Learning.
    Creates classifiers for each nesting dimension.
    """
    def __init__(self, nesting_list: List[int], num_classes=1000, efficient=False, bias=True):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes  # Number of classes for classification
        self.efficient = efficient
        
        if self.efficient:
            setattr(self, f"nesting_classifier_0", nn.Linear(nesting_list[-1], self.num_classes, bias=bias))
        else:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, bias=bias))
    
    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()
    
    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()), )
                else:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias, )
            else:
                nesting_logits += (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)
        return nesting_logits