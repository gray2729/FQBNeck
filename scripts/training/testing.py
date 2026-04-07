import torch
from scripts.utils.evaluation_metrics import evaluate_predictions

def test_model(model, test_loader, device):
    model.eval()
    
    all_pred_labels = []
    all_true_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits, _, _, _, _ = model(imgs)
            
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    results = evaluate_predictions(all_true_labels, all_pred_labels, all_probs)
    return results