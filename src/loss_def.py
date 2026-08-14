import torch
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss adapted for DP training.
    Prevents majority class dominance under high differential privacy noise.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Focal factor calculation: (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        
        if self.pos_weight is not None:
            weight_factor = targets * self.pos_weight + (1.0 - targets)
            loss = weight_factor * focal_factor * bce_loss
        else:
            loss = focal_factor * bce_loss

        return loss.mean()