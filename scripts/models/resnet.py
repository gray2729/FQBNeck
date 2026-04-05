import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, output_dim=128, pretrained=True):
        super().__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        self.layers = nn.Sequential(*list(resnet.children())[:-1])
        self.project = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.project(x)
        
        return x