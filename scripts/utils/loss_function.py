import torch.nn.functional as funct

def VIB_loss(logits, target, mu, logvar, beta=0.001):
    #classification loss
    ce_loss = funct.cross_entropy(logits, target)
    
    #kl divergence
    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_div.mean()
    
    print("CE:", ce_loss.item(), "KL:", kl_div.item())
    
    return ce_loss + beta * kl_div