from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             average_precision_score, 
                             roc_auc_score,
                             confusion_matrix)

def evaluate_predictions(true, pred, probs):
    try:
        auc = roc_auc_score(true, probs)
    except ValueError:
        auc = 0.0
        
    results = {
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, zero_division=0),
        "recall": recall_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
        "avg precision": average_precision_score(true, probs),
        "auc": auc,
        "confusion mat": confusion_matrix(true, pred).tolist(),
        "norm confusion mat": confusion_matrix(true, pred, normalize=True).tolist()
        }
    
    return results