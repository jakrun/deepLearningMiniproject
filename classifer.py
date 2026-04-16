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
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    # cm = cm / np.array([100, 10, 50, ])
    return cm

num_classes = 7
all_preds = torch.cat(all_preds).cpu().numpy().ravel()
all_targets = torch.cat(all_targets).cpu().numpy().ravel()
train_dist = get_data_distribution('train')
test_dist = get_data_distribution('test')
print(f'train distribution: {train_dist}')
print(f'test distribution: {test_dist}')

# choose which model quality analysis test to check
match 2:
    # confusion matrix
    case 1:
        cm = confusion_matrix(all_preds, all_targets, num_classes=num_classes)

        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.show()
    # total accuracy + precision/recall for each class
    case 2:
        accuracy = 0
        total_samples = 0
        # TP, FP, FN
        class_accuracy = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for t, p in zip(all_targets, all_preds):
            if t == p:
                accuracy += 1
                class_accuracy[t][0] += 1
            else:
                class_accuracy[p][1] += 1
                class_accuracy[t][2] += 1
            total_samples += 1
        print(f'total accuracy: {(accuracy/total_samples):.2f}')
        print('clas | prec | reca')
        print('-----+------+-----')
        for c in range(num_classes):
            try: precision = class_accuracy[c][0]/(class_accuracy[c][0]+class_accuracy[c][1])
            except: precision = 0
            
            try: recall = class_accuracy[c][0]/(class_accuracy[c][0]+class_accuracy[c][2])
            except: recall = 0
            
            print(f'{c:>4} | {precision:.2f} | {recall:.2f}')
        
