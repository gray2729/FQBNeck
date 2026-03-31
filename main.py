import argparse
import random
import yaml

import numpy as np

import torch
import torch.optim as optim

from pathlib import Path

from scripts.data.image_loaders import create_loaders, create_train_loader
from scripts.models.full_model import FQBNeck
from scripts.training.training import train_model
from scripts.training.validation import validate_model
from scripts.training.testing import test_model
from scripts.utils.result_log import save_metrics, save_configs, result_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def create_directory(model_name):
    index = ''
    while True:
        try:
            Path(f"results/{model_name}" + index).mkdir()
            break
        except FileExistsError:
            if index:
                index = "("+str(int(index[1:-1])+1)+")"
            else:
                index = "(1)"
            pass
    
    results_path = Path(f"results/{model_name}"+index)
    print(f"Created directory: {results_path}")
    
    return results_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, 
                        help="dataset name")
    parser.add_argument("--config", type=str, required=True, 
                        help="config name")
    parser.add_argument("--process", type=str, required=True,
                        choices=[
                            "training",
                            "testing"
                        ])
    parser.add_argument("--model_name", type=str, required=True,
                        help="model saved name")
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    #set dataset path
    dataset_path = Path(f"datasets/{args.dataset}")
    
    #Set file paths
    file_path = Path(f"saved_models/{args.model_name}.pt")
    results_path = create_directory(args.model_name)
    
    #Load image data
    config_path = Path(f"scripts/configs/{args.config}.yaml")
    with open(config_path) as file:
        config = yaml.safe_load(file)
    
    #train_loader, val_loader, test_loader = create_loaders(config, dataset_path)
    train_loader = create_train_loader(config, dataset_path)
    val_loader, test_loader = create_loaders(config, dataset_path)
    
    #Either do training or testing
    if args.process == "training":
        EPOCHS = config["epochs"]
        LR = config["lr"]
        BETA = config["beta"]
        
        save_configs(config_path, results_path)
        
        logger = result_logger(results_path)
        best_val_acc = 0
        
        if file_path.exists():
            print(f"Model found, loading {args.model_name}.pt")
            model = torch.load(file_path, weights_only=False, map_location=DEVICE)
        else:   
            model = FQBNeck(feature_dim=256, latent_dim=256)    
            model = model.to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=LR)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHs, eta_min=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(0.5*EPOCHS), T_mult=2, eta_min=1e-6)
        
        print(f"Training {args.model_name}")
        for epoch in range(EPOCHS):
            beta = min(BETA, epoch / int(0.4*EPOCHS) * BETA)
            
            train_loss, train_acc = train_model(model, train_loader, optimizer, DEVICE, beta)
            val_loss, val_acc = validate_model(model, val_loader, DEVICE)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}: Train. loss = {train_loss:.4f}, Train. acc = {train_acc:.4f}, Val. loss = {val_loss:.4f}, Val. acc = {val_acc:.4f}")
            logger.save_losses(epoch, train_loss, train_acc, val_loss, val_acc)
            
            if val_acc > best_val_acc:
                print (f"New best: {val_acc:.4f}, saving model as {args.model_name}.pt")
                best_val_acc = val_acc
                torch.save(model, file_path)
        
        #Test model
        print(f"Testing {args.model_name}")
        model = torch.load(file_path, weights_only=False, map_location=DEVICE)
        results = test_model(model, test_loader, DEVICE)
        
    elif args.process == "testing":
        print(f"Testing {args.model_name}")
        model = torch.load(file_path, weights_only=False, map_location=DEVICE)
        results = test_model(model, test_loader, DEVICE)
        
    #Show and save results
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    save_metrics(results, results_path)
    
if __name__ == "__main__":
    main()