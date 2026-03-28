import torch
import torch.nn as nn

class FFT(nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        #self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        fft = torch.fft.fft2(x)
        #fft_shift = torch.fft.fftshift(fft) 
        #magnitude = torch.log(torch.abs(fft_shift) + 1)
        
        #normalize
        #mu = magnitude.mean(dim=[1, 2, 3], keepdim = True)
        #std = magnitude.std(dim=[1, 2, 3], keepdim = True)
        
        #magnitude = (magnitude-mu)/(std + 1e-8)
        #magnitude = self.bn(magnitude)
        
        real = fft.real
        imag = fft.imag

        x = torch.cat([real, imag], dim=1)
        
        magnitude = (x - x.mean(dim=[2,3], keepdim=True)) / (x.std(dim=[2,3], keepdim=True) + 1e-6)
        
        return magnitude
