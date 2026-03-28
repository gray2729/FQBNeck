import os
import torch
from torch.utils.data import Dataset
#from torchvision.io import read_image
from torchvision import transforms
from PIL import Image

class ImageData(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size

        self.samples = self.load_paths()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            #transforms.ConvertImageDtype(torch.float32)
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
        #img = read_image(path)
        
        with Image.open(path) as img:
            img = img.convert("RGB")
        
        img = transforms.functional.to_tensor(img)
        #if img.shape[0] == 4:
        #    img = img[:3, :, :]
            
        #if img.shape[0] == 1:
        #    img = img.repeat(3, 1, 1)

        # apply transforms
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
