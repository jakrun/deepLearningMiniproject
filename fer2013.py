from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import math

def get_next_model_path(model_dir="models", prefix="emotion_model", suffix=".pth"):
    os.makedirs(model_dir, exist_ok=True)

    pattern = re.compile(rf"{prefix}_(\d+){suffix}")
    versions = []

    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            versions.append(int(match.group(1)))

    next_version = max(versions) + 1 if versions else 1

    return os.path.join(model_dir, f"{prefix}_{next_version}{suffix}")

def save_model(model, model_dir="models"):
    path = get_next_model_path(model_dir)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional
        self.features = nn.Sequential(
            # Block 1: 48x48 -> 24x24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            
            # Block 2: 24x24 -> 12x12
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0001

    transform = transforms.Compose([
        transforms.Grayscale(),            # ensure 1 channel
        transforms.Resize((48, 48)),       # FER standard size
        transforms.RandomHorizontalFlip(), # increases diversity of samples, mirrored images should be classified the same
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder("train", transform=transform)
    targets = [label for _, label in train_dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    print(class_counts)
    print(class_weights)
    print(class_weights.sum())
    # class_weights = class_weights / class_weights.sum() 
    print(class_weights)

    sample_weights = class_weights[torch.tensor(targets)]
    print(sample_weights)
    print(len(sample_weights))
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

    images, labels = next(iter(train_loader))

    model = EmotionCNN().to(device)

    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        total_loss=0.0
        total_samples=0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = lossfunc(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss:.4f}")

    # torch.save(model.state_dict(), "emotion_cnn")
    save_model(model)

    plt.imshow(images[0].squeeze(), cmap="gray")
    plt.title(labels[0].item())
    plt.show()

train()