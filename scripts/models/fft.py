import torch
import torch.nn as nn

class FFT(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft) 
        
        magnitude = torch.log(torch.abs(fft_shift) + 1)
        
        #normalize
        mu = magnitude.mean(dim=[1, 2, 3], keepdim = True)
        std = magnitude.std(dim=[1, 2, 3], keepdim = True)
        
        magnitude = (magnitude-mu)/(std + 1e-8)
        
        return magnitude
