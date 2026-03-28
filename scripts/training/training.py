from scripts.utils.loss_function import VIB_loss

def train_model(model, train_loader, optimizer, device, beta = 0.001):
    model.train()
    total_loss = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, mu, logvar = model(imgs)
        loss = VIB_loss(logits, labels, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)
        