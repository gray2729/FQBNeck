import argparse
import os
import numpy as np
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

#Global Config
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set seed
def set_seed(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
#Print metrics
def evaluate_predictions(y_true, y_probs, threshold=0.5):

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    y_preds = (y_probs > threshold).astype(int)

    acc = accuracy_score(y_true, y_preds)

    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)

    ap = average_precision_score(y_true, y_probs)

    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("Average Precision:", ap)
    print("AUC:", auc)
    
#FFT Utilities
def compute_fft(img_tensor):
    img_np = img_tensor.numpy()
    fft_features = []

    for c in range(3):
        fft = np.fft.fft2(img_np[c])
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1e-8)
        fft_features.append(magnitude)

    return np.stack(fft_features)

#Transform classes (for dataset loaders)
class FFTTransform:
    def __call__(self, img):
        img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)
        img = transforms.ToTensor()(img)
        fft_img = compute_fft(img)
        return torch.tensor(fft_img, dtype=torch.float32)
    
class RGBFFTTransform:
    def __call__(self, img):
        img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)
        rgb = transforms.ToTensor()(img)
        fft = torch.tensor(compute_fft(rgb), dtype=torch.float32)
        return torch.cat([rgb, fft], dim=0)

#Dataset loaders
def get_rgb_datasets(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train = ImageFolder(os.path.join(dataset_path, "Training"), transform=transform)
    test  = ImageFolder(os.path.join(dataset_path, "Testing"), transform=transform)
    return train, test

def get_fft_datasets(dataset_path):
    transform = FFTTransform()
    train = ImageFolder(os.path.join(dataset_path, "Training"), transform=transform)
    test  = ImageFolder(os.path.join(dataset_path, "Testing"), transform=transform)
    return train, test

def get_rgb_fft_datasets(dataset_path):
    transform = RGBFFTTransform()
    train = ImageFolder(os.path.join(dataset_path, "Training"), transform=transform)
    test  = ImageFolder(os.path.join(dataset_path, "Testing"), transform=transform)
    return train, test



#Majority baseline
def run_majority(train_dataset, test_dataset):
    print("Getting labels")
    train_labels = [label for _, label in train_dataset]
    majority_class = Counter(train_labels).most_common(1)[0][0]

    print("Predicting labels")
    test_labels = [label for _, label in test_dataset]
    probs = np.array([majority_class] * len(test_labels))

    print("Majority Class Baseline Metrics:")
    evaluate_predictions(test_labels, probs)
    
    
    
#Logistic regression baseline
def extract_cnn_features(dataset, model):

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []
    labels = []

    model.eval()

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)

            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.vstack(features), np.array(labels)

def get_feature_extractor(): #removes classifier head
    model = models.resnet50(pretrained=True)

    model.fc = nn.Identity()

    model = model.to(DEVICE)
    return model

def run_logreg(train_dataset, test_dataset):
    print("Extracting features")
    feature_model = get_feature_extractor()

    X_train, y_train = extract_cnn_features(train_dataset, feature_model)
    X_test, y_test = extract_cnn_features(test_dataset, feature_model)

    print("Fitting training")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("Predicting testing")
    probs = clf.predict_proba(X_test)[:, 1]

    print("LogReg (CNN Feature) Metrics:")
    evaluate_predictions(y_test, probs)
    
    
#ResNet model baselines
def build_resnet(input_channels=3):
    model = models.resnet50(pretrained=True)

    if input_channels != 3:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)

def train_model(model, train_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")
        
def evaluate_model(model, loader):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.cpu().numpy()

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_labels.extend(labels)
            all_probs.extend(probs)

    print("\nCNN Model Metrics:")
    evaluate_predictions(all_labels, all_probs)
    
def run_resnet(train_dataset, test_dataset, input_channels):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_resnet(input_channels=input_channels)
    print("Training model")
    train_model(model, train_loader)
    print("Testing model")
    evaluate_model(model, test_loader)
    
    
#main functions
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--model", type=str, required=True,
                        choices=[
                            "majority",
                            "logreg_rgb",
                            "logreg_fft",
                            "resnet_rgb",
                            "resnet_fft",
                            "resnet_rgb_fft"
                        ])
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()
    
    set_seed(args.seed)

    if args.model == "majority":
        print("Getting dataset")
        train, test = get_rgb_datasets(args.dataset)
        print("Starting majority")
        run_majority(train, test)

    elif args.model == "logreg_rgb":
        print("Getting dataset")
        train, test = get_rgb_datasets(args.dataset)
        print("Starting logreg")
        run_logreg(train, test)

    elif args.model == "logreg_fft":
        print("Getting dataset")
        train, test = get_fft_datasets(args.dataset)
        print("Starting logreg")
        run_logreg(train, test)

    elif args.model == "resnet_rgb":
        print("Getting dataset")
        train, test = get_rgb_datasets(args.dataset)
        print("Starting resnet")
        run_resnet(train, test, input_channels=3)

    elif args.model == "resnet_fft":
        print("Getting dataset")
        train, test = get_fft_datasets(args.dataset)
        print("Starting resnet")
        run_resnet(train, test, input_channels=3)

    elif args.model == "resnet_rgb_fft":
        print("Getting dataset")
        train, test = get_rgb_fft_datasets(args.dataset)
        print("Starting resnet")
        run_resnet(train, test, input_channels=6)


if __name__ == "__main__":
    main()