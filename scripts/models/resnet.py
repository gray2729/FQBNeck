import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)