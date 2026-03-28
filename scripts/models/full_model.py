import torch.nn as nn

from .fft import FFT
from .cnn import CNN
from .vib import VIB
from .mlp import MLP

class FQBNeck(nn.Module):
    def __init__(self, feature_dim=256, latent_dim=128, num_class=2):
        super().__init__()
        
        self.fft = FFT()
        self.cnn = CNN(feature_dim)
        self.vib = VIB(feature_dim, latent_dim)
        self.mlp = MLP(latent_dim, num_class)
        
    def forward(self, x):
        x = self.fft(x)
        features = self.cnn(x)
        
        z, mu, logvar = self.vib(features)
        logits = self.mlp(z)
        
        return logits, mu, logvar