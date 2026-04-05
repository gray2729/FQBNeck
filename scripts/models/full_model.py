import torch
import torch.nn as nn

from .fft import FFT
from .cnn import CNN
from .vib import VIB
from .mlp import MLP
from .fuse import fuse
from .resnet import ResNet

class FQBNeck(nn.Module):
    def __init__(self, feature_dim=256, latent_dim=128, num_class=2):
        super().__init__()
        
        self.fft = FFT()
        #self.rgb_cnn = CNN(input_channel=3, out_channel=latent_dim)
        self.rgb_cnn = ResNet(output_channel=latent_dim, pretrained=True)
        self.fft_cnn = CNN(input_channel=6, out_channel=feature_dim)
        self.vib = VIB(feature_dim, latent_dim)
        self.mlp = MLP(latent_dim, num_class)
        self.fuse = fuse(latent_dim)
        
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )

        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
        
    def forward(self, x):
        x_rgb = (x - self.mean) / self.std
        rgb_features = self.rgb_cnn(x_rgb)
        
        fft_features = self.fft_cnn(self.fft(x))
        z, mu, logvar = self.vib(fft_features)
        
        features = torch.cat([z, rgb_features], dim=1)
        fused_features = self.fuse(features)
        
        logits = self.mlp(fused_features)
        
        return logits, mu, logvar