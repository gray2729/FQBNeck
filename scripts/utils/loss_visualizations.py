import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def paths(model_name, vis_type):
    parent_path = Path.cwd().parents[1]
    data_path = parent_path / "results" / f"{model_name}" / "losses.csv"
    file_path = parent_path / "figures" / "loss_curves" / f"{model_name}_{vis_type}_Curve.png"
    
    return data_path, file_path

def plot(model_name, vis_type):
    data_path, file_path = paths(model_name, vis_type)
    df = pd.read_csv(data_path)
    
    if vis_type == "Loss":
        vis_abbr = "loss"
    else:
        vis_abbr = "acc"
    
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df[f"train_{vis_abbr}"], label=f"Training {vis_type}")
    plt.plot(df["epoch"], df[f"val_{vis_abbr}"], label=f"Validation {vis_type}")
    
    plt.xlabel("Epoch")
    plt.ylabel(f"{vis_type}")
    plt.title(f"Training and Validation {vis_type}")
    
    plt.legend()
    plt.grid(False)
    
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help = "model name")
    parser.add_argument("--visualization", type=str, required=True,
                        choices=[
                            "Loss",
                            "Accuracy"
                        ])
    args = parser.parse_args()
    
    plot(args.model, args.visualization)
                        
if __name__ == "__main__":
    main()