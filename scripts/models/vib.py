import torch
import torch.nn as nn

class VIB(nn.Module):
    def __init__(self, input_dim=256, lantent_dim=128):
        super().__init__()
        
        self.mu = nn.Linear(input_dim, lantent_dim)
        self.logvar = nn.Linear(input_dim, lantent_dim)
        
    def  forward(self, x):
        #get distribution parameters
        mu = self.mu(x)
        logvar = self.logvar(x)
        logvar = torch.clamp(logvar, -10, 10)
        
        #compute std
        std = torch.exp(0.5 * logvar)
        
        #reparameterization trick
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar