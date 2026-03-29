import os
import random
import shutil

def create_small_dataset(src_root, dst_root, samples_per_class=50, seed=42):
    random.seed(seed)

    splits = ["Training", "Validation", "Testing"]
    classes = ["fake", "real"]

    for split in splits:
        for cls in classes:
            src_dir = os.path.join(src_root, split, cls)
            dst_dir = os.path.join(dst_root, split, cls)

            os.makedirs(dst_dir, exist_ok=True)

            # Get all files
            all_files = [f for f in os.listdir(src_dir)
                         if os.path.isfile(os.path.join(src_dir, f))]

            # Sample files
            sampled_files = random.sample(
                all_files,
                min(samples_per_class, len(all_files))
            )

            # Copy files
            for file in sampled_files:
                src_path = os.path.join(src_dir, file)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)

            print(f"{split}/{cls}: {len(sampled_files)} files copied")

def main():
    create_small_dataset(
        src_root="datasets/Hybrid",
        dst_root="datasets/Hybrid_Sample",
        samples_per_class=50
    )

if __name__ == "__main__":
    main()