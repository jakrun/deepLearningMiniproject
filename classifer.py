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

def confusion_matrix_percent(perc, targets, num_classes):
    if len(perc[0]) != num_classes:
        raise Exception(f'incorrect number of classes per prediction. len(perc[0]) = {len(perc[0])}')
    cm = np.zeros((num_classes, num_classes), dtype=np.float32)
    for t, p in zip(targets, perc):
        for c in range(num_classes):
            cm[t, c] += p[c]/test_dist[t]
    return cm

def precision_recall(all_preds, all_targets):
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

def game_time(images, perc, targets):
    print(f'shape of list images {np.shape(images)}')
    total_samples = len(images)
    for _ in range(20):
        i = int(random.random()*total_samples)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.show()
        user_input = input(' user guess: ')
        if user_input not in ['0','1','2','3','4','5','6','7']:
            user_input = '0'
        user_input = int(user_input)
        model_guess = [f'   {emotions[j]} {"."*(round(perc[i][j]*20))}' for j in range(num_classes)]
        model_guess[targets[i]] = ' D' + model_guess[targets[i]][2:]
        model_guess[user_input] = 'U' + model_guess[user_input][1:]
        model_guess = '\n'.join(model_guess)
        print(            f' data label: {targets[i]}')
        print(            f'model guess: \n{model_guess}')
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_classes = 7

transform = transforms.Compose([
    transforms.Grayscale(),          # ensure 1 channel
    transforms.Resize((48, 48)),     # FER standard size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
val_dataset = datasets.ImageFolder("1/test", transform=transform)
val_loader = DataLoader(val_dataset,  batch_size=batch_size)

path_to_model = "models/" + os.listdir("models")[-1]
print(f'testing model: {path_to_model}')
model = EmotionCNN()
model.load_state_dict(torch.load(path_to_model, map_location=device))
model.to(device)
model.eval()

all_images   = []
all_preds    = []
all_percents = []
all_classes  = []
all_targets  = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        all_images.append(images)
        logits = model(images)              # shape (N, C)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1) # predicted class indices
        percents = probs[ torch.arange(probs.size(0)), labels.to(device) ] # similar to preds, but instead of class indicies, it is percents of the target class
        all_preds.append(preds.cpu())
        all_percents.append(percents.cpu())
        all_classes.append(probs.cpu())
        all_targets.append(labels.cpu())

all_images   = torch.cat(all_images  ).cpu().numpy()
all_preds    = torch.cat(all_preds   ).cpu().numpy().ravel()
all_percents = torch.cat(all_percents).cpu().numpy().ravel()
all_classes  = torch.cat(all_classes ).cpu().numpy()
all_targets  = torch.cat(all_targets ).cpu().numpy().ravel()

train_dist = get_data_distribution('train')
test_dist  = get_data_distribution('test')
print(f'train distribution: {train_dist}')
print(f'test distribution: {test_dist}')

precision_recall(all_preds, all_targets)

# choose which model quality analysis test to check
match 3:
    case 1: # confusion matrix
        cm = confusion_matrix(all_preds, all_targets, num_classes=num_classes)

        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.show()
    case 2: # confusion matrix using percentages of each class
        cm = confusion_matrix_percent(all_classes, all_targets, num_classes=num_classes)

        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.show()
    case 3: # game time !
        game_time(all_images, all_classes, all_targets)
