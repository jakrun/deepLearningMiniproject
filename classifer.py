from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import torch
import os
import math
from fer2013 import EmotionCNN

emotions = [
    'ange',
    'disg',
    'fear',
    'happ',
    'neut',
    'sadn',
    'surp'
    ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

transform = transforms.Compose([
    transforms.Grayscale(),          # ensure 1 channel
    transforms.Resize((48, 48)),     # FER standard size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
val_dataset = datasets.ImageFolder("1/test", transform=transform)
val_loader = DataLoader(val_dataset,  batch_size=batch_size)

if os.path.exists("best.pth"):
    path_to_model = "best.pth"
else:
    path_to_model = os.path.join("models", sorted(os.listdir("models"))[-1])
print(path_to_model)
model = EmotionCNN()
model.load_state_dict(torch.load(path_to_model, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_targets = []
all_classes = []
all_images = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        logits = model(images)               # shape (N, C)
        preds = torch.argmax(logits, dim=1)  # predicted class indices
        probs = torch.softmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
        all_classes.append(probs.cpu())
        all_images.append(images)

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
    return cm

def game_time(images, perc, targets):
    total_samples = len(images)
    # user score, model score
    tally = [0, 0]
    image_count = 40
    for image_num in range(image_count):
        i = int(random.random()*total_samples)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.show()
        user_input = input('> ')
        if user_input not in ['0','1','2','3','4','5','6','7']:
            user_input = '0'
        user_input = int(user_input)

        model_guess = [[emotions[j], perc[i][j]] for j in range(num_classes)]
        model_guess.sort(key=lambda item: item[1], reverse=True)
        top3 = ' | '.join([f'{emo[0]} {round(emo[1]*100)}%' for emo in model_guess[:3]])
    
        print(f'user guess: {emotions[user_input]}')
        print(f'data label: {emotions[targets[i]]}')
        print(f'modl guess: {top3}')
        if user_input == targets[i]:
            tally[0] += 1
        if model_guess[0][0] == emotions[targets[i]]:
            tally[1] += 1
        print(f'image {image_num+1}/{image_count} score: {tally[0]}-{tally[1]}')
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.show()


num_classes = 7
all_preds = torch.cat(all_preds).cpu().numpy().ravel()
all_targets = torch.cat(all_targets).cpu().numpy().ravel()
all_classes  = torch.cat(all_classes ).cpu().numpy()
all_images   = torch.cat(all_images  ).cpu().numpy()
train_dist = get_data_distribution('train')
test_dist = get_data_distribution('test')
print(f'train distribution: {train_dist}')
print(f'test distribution: {test_dist}')

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# choose which model quality analysis test to check
match 2:
    # confusion matrix
    case 1:
        cm = confusion_matrix(all_preds, all_targets, num_classes=num_classes)
        # Row-normalize: each cell = fraction of true class predicted as that column.
        # Diagonal cells now read as per-class recall.
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(np.float64), row_sums, where=row_sums > 0)

        plt.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(label='Recall')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (row-normalized)')
        plt.xticks(range(num_classes), EMOTIONS, rotation=45, ha='right')
        plt.yticks(range(num_classes), EMOTIONS)
        for i in range(num_classes):
            for j in range(num_classes):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                plt.text(j, i, f'{val * 100:.0f}', ha='center', va='center', color=color, fontsize=9)
        plt.tight_layout()
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
    # game time
    case 3:
        game_time(all_images, all_classes, all_targets)
