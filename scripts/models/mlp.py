import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, latent_dim=128, num_class=2):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_class)
        )
        
    def forward(self, x):
        return self.classifier(x)
