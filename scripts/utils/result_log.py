import json
import csv
import shutil

def save_metrics(results, folder_path, dataset):
    save_path = folder_path / f"{dataset}_metrics.json"
    
    with open(save_path, "w") as file:
        json.dump(results, file, indent = 4)
        
    print(f"Saved metrics to {save_path}")
    
def save_configs(configs, folder_path):
    save_path = folder_path / "config.yaml"
    shutil.copy(configs, save_path)
    
class result_logger:
    def __init__(self, folder_path):
        self.save_path = folder_path / "losses.csv"
        
        with open(self.save_path, "w", newline = "") as file:
            csv.writer(file).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            
    def save_losses(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.save_path, "a", newline = "") as file:
            csv.writer(file).writerow([epoch, train_loss, train_acc, val_loss, val_acc])