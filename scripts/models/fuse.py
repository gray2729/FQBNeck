import torch.nn as nn

class fuse(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        return self.features(x)