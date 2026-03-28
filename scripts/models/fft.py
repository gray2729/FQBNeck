import torch
import torch.nn as nn

class FFT(nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        #self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft) 
        #magnitude = torch.log(torch.abs(fft_shift) + 1)
        
        real = fft_shift.real
        imag = fft_shift.imag

        x = torch.cat([real, imag], dim=1)
        
        #normalize
        mu = x.mean(dim=[2, 3], keepdim = True)
        std = x.std(dim=[2, 3], keepdim = True)
        
        magnitude = (x-mu)/(std + 1e-6)
        
        return magnitude
