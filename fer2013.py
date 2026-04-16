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
        super(EmotionCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # 48
            nn.MaxPool2d(2),

            # nn.Dropout(0.25),
            # 24
            nn.Conv2d(32, 64, 3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # 24
            nn.MaxPool2d(2),
            
            # nn.Dropout(0.25),
            # 12
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            # 12
            nn.MaxPool2d(2),
            
            # nn.Dropout(0.25),
            # 6
            nn.Flatten(),
            # 2048
            nn.Linear(6*6*128, 256),
            nn.ReLU(),
            
            nn.Dropout(0.5),
            # 256
            nn.Linear(256, 64),
            nn.ReLU(),
            # 64,
            nn.Linear(64, 7)

            # 7

        )

        # self.conv1 = nn.Conv2d(1, 6, 3)
        # self.conv2 = nn.Conv2d(6, 16, 4)

        # self.pool = nn.MaxPool2d(2, 2)

        # self.fc1 = nn.Linear(10*10*16, 256)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 10*10*16)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.network(x)
        return x


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 4
    batch_size = 32
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.Grayscale(),            # ensure 1 channel
        transforms.Resize((48, 48)),       # FER standard size
        transforms.RandomHorizontalFlip(), # increases diversity of samples, mirrored images should be classified the same
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder("1/train", transform=transform)
    targets = [label for _, label in train_dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    print(class_counts)
    print(class_weights)
    print(class_weights.sum())
    class_weights = class_weights / class_weights.sum() 
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

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # torch.save(model.state_dict(), "emotion_cnn")
    save_model(model)

    plt.imshow(images[0].squeeze(), cmap="gray")
    plt.title(labels[0].item())
    plt.show()

# train()