import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

IMG_SIZE = (64, 64)
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

train_df = pd.read_csv('data/GTSRB/Train.csv', sep='\t')
test_df = pd.read_csv('data/GTSRB/Test.csv', sep='\t')

train_df = pd.DataFrame(data=[row[0].split(',') for row in train_df.values.tolist()], columns=train_df.columns.values[0].split(','))
test_df = pd.DataFrame(data=[row[0].split(',') for row in test_df.values.tolist()], columns=test_df.columns.values[0].split(','))

def extract_sign_data(df):
    """
    Extract sign data from the images
    :param df: dataframe containing attributes of the images
    :return: extracted images of signs with the corresponding labels
    """
    images = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join('data/GTSRB', row['Path'])
        label = int(row['ClassId'])

        bounding_box = {
            "upper_left_x": int(row['Roi.X1']),
            "upper_left_y": int(row['Roi.Y1']),
            "lower_right_x": int(row['Roi.X2']),
            "lower_right_y": int(row['Roi.Y2'])
        }

        image = Image.open(img_path)
        image = image.crop((bounding_box["upper_left_x"], bounding_box["upper_left_y"],
                            bounding_box["lower_right_x"], bounding_box["lower_right_y"]))
        image = image.resize(IMG_SIZE)
        image = np.array(image)

        images.append(image)
        labels.append(label)

    return images, labels

train_x, train_y = extract_sign_data(train_df)
test_x, test_y = extract_sign_data(test_df)
num_classes = len(set(train_y))

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)


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

class TrafficSignCNN(nn.Module):
    """
    A Neural Network class that uses convolutional feature extractors and fully connected classifier layers
    for traffic sign recognition
    """
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()
        self.feature_extractor_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor_layers(x)
        x = self.classifier_layers(x)
        return x



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
num_correct = 0
total = 0

with torch.no_grad():
    for batch_id, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images: torch.Tensor = images.to(device)
        labels: torch.Tensor = labels.to(device)
        outputs = model(images)
        _, predictions = outputs.max(1)
        total += labels.size(0)
        num_correct += predictions.eq(labels).sum().item()

test_accuracy = 100.0 * num_correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

