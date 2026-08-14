import torch
import torch.nn as nn

class DPMLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron designed for DP-SGD compatibility.
    Uses LayerNorm / GroupNorm / Dropout instead of BatchNorm.
    """
    def __init__(self, input_dim: int = 30, hidden_dim: int = 64):
        super(DPMLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GroupNorm(num_groups=2, num_channels=hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)