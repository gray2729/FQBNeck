import torch
import torch.nn as nn

from .fft import FFT
from .cnn import CNN
from .vib import VIB
from .mlp import MLP
from .fuse import fuse
    
class FQBNeck(nn.Module):
    def __init__(self, feature_dim=256, latent_dim=128, num_class=2):
        super().__init__()
        
        self.fft = FFT()
        self.rgb_cnn = CNN(input_channel=3, out_channel=latent_dim)
        self.fft_cnn = CNN(input_channel=6, out_channel=feature_dim)
        self.vib = VIB(feature_dim, latent_dim)
        self.mlp = MLP(latent_dim, num_class)
        self.fuse = fuse(latent_dim)
        
    def forward(self, x):
        rgb_features = self.rgb_cnn(x)
        
        fft_features = self.fft_cnn(self.fft(x))
        z, mu, logvar = self.vib(fft_features)
        
        features = torch.cat([z, rgb_features], dim=1)
        fused_features = self.fuse(features)
        
        logits = self.mlp(fused_features)
        
        return logits, mu, logvar