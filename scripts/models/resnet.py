import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)