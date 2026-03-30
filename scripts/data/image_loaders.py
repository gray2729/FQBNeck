from torch.utils.data import DataLoader
from .image_data import ImageData, TrainImageData

def create_train_loader(config, dataset_path):
    #Create dataset
    train_dataset = TrainImageData(
        root_dir=dataset_path / "Training",
        image_size=config["image_size"]
    )
    
    #Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,                 
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )

    return train_loader
    
def create_loaders(config, dataset_path):
    #Create datasets
    #train_dataset = TrainImageData(
    #    root_dir=dataset_path / "Training",
    #    image_size=config["image_size"]
    #)
    
    val_dataset = ImageData(
        root_dir=dataset_path / "Validation",
        image_size=config["image_size"]
    )

    test_dataset = ImageData(
        root_dir=dataset_path / "Testing",
        image_size=config["image_size"]
    )

    #Create dataloaders
    #train_loader = DataLoader(
    #    train_dataset,
    #    batch_size=config["batch_size"],
    #    shuffle=True,                 
    #    num_workers=config.get("num_workers", 4),
    #    pin_memory=True
    #)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,                
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )

    return val_loader, test_loader