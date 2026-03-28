import torch
from scripts.utils.loss_function import VIB_loss

def validate_model(model, val_loader, device, beta = 0.001):
    model.eval()
    total_loss = 0
    correct = 0
    samples = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits, mu, logvar = model(imgs)
            loss = VIB_loss(logits, labels, mu, logvar, beta)
            
            preds = logits.argmax(dim=1)
            
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            samples += batch_size
        
        avg_loss = total_loss / samples
        accuracy = correct / samples
            
        return avg_loss, accuracy
