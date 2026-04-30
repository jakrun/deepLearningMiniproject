from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import time

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

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, device, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"] + 1

class EmotionCNN(nn.Module):
    def __init__(self, dropout=0.4):
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

            nn.MaxPool2d(2),

            # Block 3: 12x12 -> 6x6
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 7),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, device, lossfunc):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = lossfunc(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 100
    batch_size = 32
    learning_rate = 0.0001

    # Checkpointing variables for ai-lab
    use_checkpointing = False # choose if u want to use checkpoints
    MAX_RUNTIME = 19 # minutes

    transform = transforms.Compose([
        transforms.Grayscale(),            # ensure 1 channel
        transforms.Resize((48, 48)),       # FER standard size
        transforms.RandomHorizontalFlip(), # increases diversity of samples, mirrored images should be classified the same
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Make deterministic split
    val_fraction = 0.125
    n_total = len(train_dataset)
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val

    train_dataset = datasets.ImageFolder("1/train", transform=transform)
    train_subset, val_subset = random_split(train_dataset, [n_train, n_val])
    test_dataset = datasets.ImageFolder("1/test", transform=test_transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, sampler=sampler)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    images, labels = next(iter(train_loader))

    model = EmotionCNN().to(device)

    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    best_val_acc = 0.0

    if use_checkpointing and os.path.exists("checkpoint.pth"):
        start_epoch = load_checkpoint(model, optimizer, device)

    # n_total_steps = len(train_loader)
    if use_checkpointing:
        start_time = time.time()
        runtime_seconds = MAX_RUNTIME * 60



    for epoch in range(start_epoch, num_epochs):
        model.train()

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

            if use_checkpointing:
                if time.time() - start_time > runtime_seconds:
                    print("Time limit reached. Saving checkpoint...")
                    save_checkpoint(model, optimizer, epoch)
                    return

            curr_batch_size = labels.size(0)
            total_loss += loss.item() * curr_batch_size
            total_samples += curr_batch_size

        epoch_loss = total_loss / total_samples
        val_loss, val_acc = evaluate(model, val_loader, device, lossfunc)
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best.pth")
            marker = "  <- new best, saved best.pth"
        print(f"Epoch [{epoch+1}/{num_epochs}]  train_loss: {epoch_loss:.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc*100:.2f}%{marker}")
        if use_checkpointing:
            save_checkpoint(model, optimizer, epoch)

    print(f"Best val accuracy: {best_val_acc*100:.2f}%")
    # archive the best model (not the last-epoch model)
    if os.path.exists("best.pth"):
        model.load_state_dict(torch.load("best.pth", map_location=device))
    save_model(model)

    plt.imshow(images[0].cpu().squeeze(), cmap="gray")
    plt.title(labels[0].item())
    plt.show()

if __name__ == "__main__":
    train()