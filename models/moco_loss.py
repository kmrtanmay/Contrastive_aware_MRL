import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCoLoss(nn.Module):
    """
    Standard MoCo contrastive loss without Matryoshka structure.
    Uses a queue of negative samples for more efficient contrastive learning.
    """
    def __init__(
        self,
        dim=2048,
        queue_size=65536,
        temperature=0.07
    ):
        super(MoCoLoss, self).__init__()
        self.dim = dim
        self.queue_size = queue_size
        self.temperature = temperature
        
        # Initialize queue
        queue = torch.randn(queue_size, dim)
        queue = F.normalize(queue, dim=1)
        self.register_buffer("queue", queue)
        
        # Register queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace keys in the queue
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Handle queue wrapping
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        
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
        

        # Normalize features
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Get queue
        queue = self.queue.clone().detach()
        
        # Compute logits
        # Positive logits: [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: [N, queue_size]
        l_neg = torch.einsum('nc,kc->nk', [q, queue])
        
        # Concatenate logits and apply temperature
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # Labels: positives are at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Apply InfoNCE loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue with current batch
        self._dequeue_and_enqueue(k)
        
        return loss