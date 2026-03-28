import torch

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits, _, _ = model(imgs)
            preds = logits.argmax(dim=1)
            print("Probs sample:", probs[:5])         # should NOT be all [0.5, 0.5]
            print("Pred distribution:", logits.argmax(dim=1).float().mean().item())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        return correct / total
