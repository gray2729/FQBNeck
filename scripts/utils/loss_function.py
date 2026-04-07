import torch.nn.functional as funct

#def VIB_loss(logits, target, mu, logvar, beta=0.001):
def VIB_loss(logits, target, fft_mu, fft_logvar, rgb_mu, rgb_logvar, beta=0.001):
    #classification loss
    ce_loss = funct.cross_entropy(logits, target)
    
    #kl divergence
    #kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    #kl_div = kl_div.sum(dim=1).mean()
    
    fft_kl_div = -0.5 * (1 + fft_logvar - fft_mu.pow(2) - fft_logvar.exp())
    fft_kl_div = fft_kl_div.sum(dim=1).mean()
    
    rgb_kl_div = -0.5 * (1 + rgb_logvar - rgb_mu.pow(2) - rgb_logvar.exp())
    rgb_kl_div = rgb_kl_div.sum(dim=1).mean()
    
    #return ce_loss + beta * kl_div
    return ce_loss + beta * (fft_kl_div + rgb_kl_div)