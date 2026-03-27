import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


def plot_distribution(dataset):
    parent_path = Path.cwd().parents[1]
    dataset_path = parent_path / "datasets" / f"{dataset}"
    file_path = parent_path / "figures" / "dataset_distributions" / f"{dataset}_distribution.png"
    
    splits = ["Training", "Validation", "Testing"]
    classes = ["real", "fake"]

    # Count images
    counts = {split: [] for split in splits}

    for split in splits:
        for clss in classes:
            cls_path = dataset_path / split / clss
            counts[split].append(len(list(cls_path.glob("*"))))

    # Convert to numpy for plotting
    train_counts = counts["Training"]
    val_counts = counts["Validation"]
    test_counts = counts["Testing"]

    x = np.arange(len(classes))  # real, fake
    width = 0.25  # bar width

    plt.figure()

    plt.bar(x - width, train_counts, width, label="Train")
    plt.bar(x, val_counts, width, label="Validation")
    plt.bar(x + width, test_counts, width, label="Test")

    plt.xticks(x, ["Real", "Fake"])
    plt.ylabel("Number of Images")
    plt.title(f"Class Distribution Across {dataset}")
    plt.legend()

    plt.savefig(file_path, bbox_inches='tight')
    plt.show()

def show_dataset_samples(dataset, n=4):
    parent_path = Path.cwd().parents[1]
    dataset_path = parent_path / "datasets" / f"{dataset}"
    file_path = parent_path / "figures" / "dataset_samples"

    splits = ["Training", "Validation", "Testing"]
    classes = ["real", "fake"]

    for split in splits:
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"{split} folder not found — skipping.")
            continue

        plt.figure(figsize=(3*n, 6))
        plt.suptitle(f"{dataset_path.name} - {split}", fontsize=14)

        for row, clss in enumerate(classes):
            class_path = split_path / clss

            if not class_path.exists():
                print(f"{clss} folder missing in {split} — skipping.")
                continue

            images = [p for p in class_path.iterdir()
                      if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

            if len(images) == 0:
                print(f"No images in {class_path}")
                continue

            selected = random.sample(images, min(n, len(images)))

            for col, img_path in enumerate(selected):
                img = Image.open(img_path)

                plt.subplot(2, n, row*n + col + 1)
                plt.imshow(img)
                plt.axis("off")

                if col == 0:
                    plt.ylabel(clss.capitalize(), fontsize=12)

        plt.tight_layout()
        plt.savefig(file_path / f"{dataset}_{split}_sample.png", bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset name")
    parser.add_argument("--visualization", type=str, required=True,
                        choices=[
                            "distribution",
                            "sample"
                        ])
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.visualization == "distribution":
        plot_distribution(args.dataset)
        
    elif args.visualization == "sample":
        show_dataset_samples(args.dataset)

if __name__ == "__main__":
    main()