import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class MatryoshkaContrastiveLoss(nn.Module):
    """
    Implements contrastive loss for Matryoshka Representation Learning.
    Applies NCE loss across each nesting dimension and aggregates the results.
    """
    def __init__(
        self, 
        nesting_list: List[int],
        temperature: float = 0.07, 
        relative_importance: Optional[List[float]] = None
    ):
        super(MatryoshkaContrastiveLoss, self).__init__()
        self.nesting_list = nesting_list
        self.temperature = temperature
        # Set relative importance weights for each nesting dimension
        if relative_importance is None:
            self.relative_importance = torch.ones(len(nesting_list))
        else:
            self.relative_importance = torch.tensor(relative_importance)
            
    def forward(self, 
                features_q: torch.Tensor,  # query features [N, full_dim]
                features_k: torch.Tensor,  # key features [N, full_dim]
                indices_pos: torch.Tensor  # positive pair indices [N]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes contrastive loss at each nesting dimension.
        
        Args:
            features_q: Query features of shape [N, full_dim]
            features_k: Key features of shape [N or K, full_dim]
            indices_pos: Indices of positive pairs for each query [N]
            
        Returns:
            total_loss: Weighted sum of losses across all nesting dimensions
            individual_losses: List of losses at each nesting dimension
        """
        batch_size = features_q.size(0)
        device = features_q.device
        
        individual_losses = []
        
        # Compute loss for each nesting dimension
        for i, dim in enumerate(self.nesting_list):
            # Extract features up to the current nesting dimension
            q_nested = features_q[:, :dim]  # [N, dim]
            k_nested = features_k[:, :dim]  # [N or K, dim]
            
            # Normalize the features (L2 normalization)
            q_nested = F.normalize(q_nested, dim=1)
            k_nested = F.normalize(k_nested, dim=1)
            
            # Compute logits
            # Shape: [N, N or K]
            logits = torch.matmul(q_nested, k_nested.T) / self.temperature
            
            # Generate labels: positives are at positions specified by indices_pos
            labels = indices_pos
            
            # Apply InfoNCE loss
            loss = F.cross_entropy(logits, labels)
            individual_losses.append(loss)
        
        # Stack losses and apply relative importance weights
        stacked_losses = torch.stack(individual_losses)
        weighted_losses = self.relative_importance.to(device) * stacked_losses
        
        # Return total loss and individual losses
        return weighted_losses.sum(), individual_losses
    
class MoCoMatryoshkaLoss(nn.Module):
    """
    Integrates Matryoshka Representation Learning with MoCo-style contrastive learning.
    Uses a queue of negative samples for more efficient contrastive learning.
    """
    def __init__(
        self,
        nesting_list: List[int],
        queue_size: int = 65536,
        temperature: float = 0.07,
        relative_importance: Optional[List[float]] = None
    ):
        super(MoCoMatryoshkaLoss, self).__init__()
        self.nesting_list = nesting_list
        self.queue_size = queue_size
        self.temperature = temperature
        
        # Set relative importance weights for each nesting dimension
        if relative_importance is None:
            self.relative_importance = torch.ones(len(nesting_list))
        else:
            self.relative_importance = torch.tensor(relative_importance)
        
        # Initialize a separate queue for each nesting dimension
        for i, dim in enumerate(nesting_list):
            # Create a queue with the appropriate dimensions for this nesting level
            queue = torch.randn(queue_size, dim)
            # Normalize each queue entry
            queue = F.normalize(queue, dim=1)
            # Register as buffer
            self.register_buffer(f"queue_{i}", queue)
        
        # Register queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace keys in the queue for each nesting dimension
        for i, dim in enumerate(self.nesting_list):
            # Get the queue for this nesting dimension
            queue = getattr(self, f"queue_{i}")
            
            # Extract and normalize features for this dimension
            keys_dim = F.normalize(keys[:, :dim], dim=1)
            
            # Replace keys in the queue
            if ptr + batch_size <= self.queue_size:
                queue[ptr:ptr + batch_size] = keys_dim
            else:
                # Handle queue wrapping
                remaining = self.queue_size - ptr
                queue[ptr:] = keys_dim[:remaining]
                queue[:batch_size - remaining] = keys_dim[remaining:]
        
        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, q, k):
        """
        Input:
            q: Query embeddings [N, D] (from the main encoder)
            k: Key embeddings [N, D] (from the momentum encoder)
        """
        batch_size = q.shape[0]
        device = q.device
        
        individual_losses = []
        
        # Compute loss for each nesting dimension
        for i, dim in enumerate(self.nesting_list):
            # Get features up to the current nesting dimension
            q_nested = q[:, :dim]  # [N, dim]
            k_nested = k[:, :dim]  # [N, dim]
            
            # Normalize features
            q_nested = F.normalize(q_nested, dim=1)
            k_nested = F.normalize(k_nested, dim=1)
            
            # Get negative samples from the queue for this dimension
            queue_nested = getattr(self, f"queue_{i}").clone().detach()  # [queue_size, dim]
            
            # Compute logits
            # Positive logits: [N, 1]
            l_pos = torch.einsum('nc,nc->n', [q_nested, k_nested]).unsqueeze(-1)
            # Negative logits: [N, queue_size]
            l_neg = torch.einsum('nc,kc->nk', [q_nested, queue_nested])
            
            # Concatenate logits and apply temperature
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
            
            # Labels: positives are at index 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Apply InfoNCE loss
            loss = F.cross_entropy(logits, labels)
            individual_losses.append(loss)
        
        # Update queue with current batch
        self._dequeue_and_enqueue(k)
        
        # Stack losses and apply weights
        stacked_losses = torch.stack(individual_losses)
        weighted_losses = self.relative_importance.to(device) * stacked_losses
        
        return weighted_losses.sum(), individual_losses