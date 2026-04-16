from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import math
from fer2013 import EmotionCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

transform = transforms.Compose([
    transforms.Grayscale(),          # ensure 1 channel
    transforms.Resize((48, 48)),     # FER standard size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
val_dataset = datasets.ImageFolder("1/test", transform=transform)
val_loader = DataLoader(val_dataset,  batch_size=batch_size)

path_to_model = (f"models/{os.listdir("models")[-1]}")
print(path_to_model)
model = EmotionCNN()
model.load_state_dict(torch.load(path_to_model, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_targets = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        logits = model(images)               # shape (N, C)
        preds = torch.argmax(logits, dim=1)  # predicted class indices
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())


def get_data_distribution(directory):
    emotion_dist = []
    base = f"1/{directory}"
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            emotion_dist.append(sum(1 for f in os.listdir(path)
                                        if os.path.isfile(os.path.join(path, f))
                                    ))
    dist_sum = sum(emotion_dist)
    for i in range(len(emotion_dist)):
        emotion_dist[i] = math.floor((emotion_dist[i] / dist_sum)*1000)/10
    return emotion_dist

def confusion_matrix(preds, targets, num_classes):
    class_dist = get_data_distribution('test')
    print(class_dist)
    # preds: tensor of shape (N,) with predicted class indices
    # targets: tensor of shape (N,) with true class indices
    preds = preds.cpu().numpy().ravel()
    targets = targets.cpu().numpy().ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += (1)
    #cm = cm / np.array([100, 10, 50])
    return cm


num_classes = 7
all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)
cm = confusion_matrix(all_preds, all_targets, num_classes=num_classes)

plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(range(num_classes))
plt.yticks(range(num_classes))
plt.savefig("confusion_matrix.png")
#plt.show()
