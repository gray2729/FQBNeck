import argparse
from PIL import Image
from pathlib import Path

def find_corrupted_images(root):
    dataset_path = Path.cwd().parents[1] / "datasets" / f"{root}"
    corrupted_files = []

    for path in dataset_path.rglob("*"):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                corrupted_files.append(path)

    return corrupted_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset name")
    
    args = parser.parse_args()
    
    corrupted = find_corrupted_images(args.dataset)
    print("Corrupted files:", corrupted)
    

if __name__ == "__main__":
    main()