import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TrainImageData(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size

        self.samples = self.load_paths()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2)
        ])
        
    def load_paths(self):
        samples = []

        for label, cls in enumerate(["real", "fake"]):
            cls_path = os.path.join(self.root_dir, cls)

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                samples.append((img_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # load image as tensor  
        try:
            img = Image.open(path).convert("RGB")
        except OSError:
            return self.__getitem__((idx + 1) % len(self.samples))
            
        img = transforms.functional.to_tensor(img)

        # apply transforms
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

class ImageData(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size

        self.samples = self.load_paths()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size))
        ])

    def load_paths(self):
        samples = []

        for label, cls in enumerate(["real", "fake"]):
            cls_path = os.path.join(self.root_dir, cls)

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                samples.append((img_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # load image as tensor        
        with Image.open(path) as img:
            img = img.convert("RGB")
        
        img = transforms.functional.to_tensor(img)

        # apply transforms
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
