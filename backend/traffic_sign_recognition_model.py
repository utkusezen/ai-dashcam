import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from classes.sign_classification_nn import TrafficSignCNN

IMG_SIZE = (64, 64)
EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
TRAIN_PATH = "data/Classification/images"
TEST_PATH = "data/Classification/test"
METRICS_PATH = "metrics/recognition_metrics.csv"


def extract_sign_data(base_path):
    """
    Load images and labels from folder structure.

    Folder names are interpreted as class labels.
    """

    images = []
    labels = []

    base_path = Path(base_path)

    for class_dir in tqdm(list(base_path.iterdir())):

        if not class_dir.is_dir():
            continue

        label = int(class_dir.name)

        for img_path in class_dir.iterdir():

            if not img_path.is_file():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image = image.resize(IMG_SIZE)
                image = np.array(image)

                images.append(image)
                labels.append(label)

            except Exception as e:
                print(f"Image could not be opened {img_path}: {e}")

    return np.array(images), np.array(labels)


train_x, train_y = extract_sign_data(TRAIN_PATH)
test_x, test_y = extract_sign_data(TEST_PATH)
num_classes = len(set(train_y))

train_x, train_y = shuffle(train_x, train_y, random_state=42)
test_x, test_y = shuffle(test_x, test_y, random_state=42)


class TrafficSignDataset(Dataset):
    """
    A PyTorch Dataset class to simplify access and transformations on the data
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dataset = TrafficSignDataset(train_x, train_y, transform=transform)
test_dataset = TrafficSignDataset(test_x, test_y, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignCNN(num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    sum_loss = 0.0
    num_correct = 0
    total = 0

    for batch_id, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predictions = outputs.max(1)
        total += labels.size(0)
        num_correct += predictions.eq(labels).sum().item()

    train_accuracy = 100.0 * num_correct / total
    avg_loss = sum_loss / len(train_loader)
    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

model.eval()
all_predictions = []
total = len(test_loader) * BATCH_SIZE

with torch.no_grad():
    for batch_id, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)
        outputs = model(images)
        _, predictions = outputs.max(1)
        all_predictions.extend(predictions.cpu().numpy())

conf_matrix = confusion_matrix(test_y, all_predictions)
TP = np.diag(conf_matrix)
FP = conf_matrix.sum(axis=0) - TP
FN = conf_matrix.sum(axis=1) - TP
accuracy = TP.sum() / total
precision = TP / (FP + TP + 1e-8)
recall = TP / (FN + TP + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
f1_avg = np.mean(f1)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-Score: {f1_avg:.4f}")

plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("metrics/recognition_confusion_matrix.png")

metrics = pd.DataFrame(data={"Accuracy": [accuracy], "F1-Score": [f1_avg],
                             "Precision": [np.mean(precision)], "Recall": [np.mean(recall)]},
                       columns=["Accuracy", "F1-Score", "Precision", "Recall"])
metrics.to_csv(METRICS_PATH, mode='a', header=not os.path.exists(METRICS_PATH), index=False)

torch.save(model.state_dict(), "models/new_sign_recognition_model_state.pt")
