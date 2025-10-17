import os
from math import floor

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

import image_feature_extraction as ft_extr
from tqdm import tqdm

DATA_PATH = "data/bdd100k/bdd100k/bdd100k/images/10k/train"
LABELS_PATH = "data/bdd100k/manual_speed_recommendation_labels.csv"

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

def load_data_and_extract_features(path):
    """
    Loads images and extracts features with the image_feature_extraction module.
    :param path: the path to the images folder
    :return: a dataframe containing the extracted features
    """
    rows = []
    img_paths = os.listdir(path)
    for img_path in tqdm(img_paths):
        img = cv2.imread(path + "/" + img_path)

        brightness, contrast = ft_extr.compute_brightness_and_contrast(img)
        _, driveable_area = ft_extr.compute_driveable_area(img)
        (num_lanes, max_lane_len,
         angle_right, angle_left,
         vp_found, vp_offset_x, vp_offset_y) = ft_extr.compute_lane_features(img)

        num_lanes_norm = np.clip(num_lanes, 0, 6) / 6
        angle_right_norm = angle_right / 90 if angle_right is not None else 0
        angle_left_norm = angle_left / 90 if angle_left is not None else 0

        rows.append({
            "image": img_path,
            "brightness": brightness,
            "contrast": contrast,
            "driveable_area": driveable_area,
            "num_lanes": num_lanes_norm,
            "max_lane_len": max_lane_len,
            "angle_right": angle_right_norm,
            "angle_left": angle_left_norm,
            "vp_found": vp_found,
            "vp_offset_x": vp_offset_x if vp_offset_x is not None else 0,
            "vp_offset_y": vp_offset_y if vp_offset_y is not None else 0,
        })
    return pd.DataFrame(rows)

def range_to_discrete_value(range_label:str):
    """
    Maps ranges of values to discrete values. Ranges are mapped to the maximum value in the range.
    :param range_label: the range to map
    :return: maximum value in the range
    """
    s = str(range_label).strip()
    if s.startswith('<'):
        return 10.0
    if s.startswith('>'):
       return 110.0
    if '-' in s:
        return float(s[3:])
    return np.nan

labels = pd.read_csv(LABELS_PATH)
labels = labels.drop(columns=["annotator", "lead_time", "updated_at", "created_at", "annotation_id", "id"])
labels["image"] = [name[24:] for name in labels["image"]]
labels.sort_values("image", inplace=True)
y = labels["choice"].apply(range_to_discrete_value).to_numpy()
y = y.reshape(-1, 1)

data = load_data_and_extract_features(DATA_PATH)
data.sort_values("image", inplace=True)
data.drop(columns=["image"], inplace=True)
X = data.to_numpy()

class SpeedRecommendationDataset(Dataset):
    """
    A PyTorch Dataset class to simplify access and transformations on the data
    """
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]

        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(label, (np.ndarray, float, int)):
            label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        return features, label

class SpeedRecommendationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

dataset = SpeedRecommendationDataset(features=X, labels=y)
dummy_set = SpeedRecommendationDataset(features=np.zeros((1000, 10), dtype=float), labels=np.ones((1000, 1), dtype=float))
train_set, test_set = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
dummy_loader = DataLoader(dummy_set, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeedRecommendationModel().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model.train()
for epoch in range(EPOCHS):
    sum_loss = 0.0
    sum_deviance = 0.0
    num_major_deviance = 0.0
    num_overshot = 0.0
    num_undershot = 0.0
    num_correct = 0.0
    for batch in tqdm(train_loader):
        X, y = batch
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        for pred, truth in zip(output, y):
            pred = pred.item()
            truth = truth.item()
            deviance = abs(pred - truth)
            num_major_deviance += 1 if deviance > 10 else 0
            num_overshot += 1 if floor(pred) > truth else 0
            num_undershot += 1 if floor(pred) < truth else 0
            num_correct += 1 if floor(pred) == truth else 0
            sum_deviance += deviance

    avg_loss = sum_loss / (len(train_loader) * BATCH_SIZE)
    avg_deviance = sum_deviance / (len(train_loader) * BATCH_SIZE)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss}")
    print(f"Average deviance: {avg_deviance}")
    print(f"Number of major errors: {num_major_deviance}")
    print(f"Number of times overshot: {num_overshot}")
    print(f"Number of times undershot: {num_undershot}")
    print(f"Number of times correct: {num_correct}")
print("Model Training finished.")

model.eval()
sum_deviance = 0.0
num_major_deviance = 0.0
num_overshot = 0.0
num_undershot = 0.0
num_correct = 0.0
with torch.no_grad():
    for batch in tqdm(test_loader):
        X = X.to(device)
        y = y.to(device)

        output = model(X)

        for pred, truth in zip(output, y):
            pred = pred.item()
            truth = truth.item()
            deviance = abs(pred - truth)
            num_major_deviance += 1 if deviance > 10 else 0
            num_overshot += 1 if floor(pred) > truth else 0
            num_undershot += 1 if floor(pred) < truth else 0
            num_correct += 1 if floor(pred) == truth else 0
            sum_deviance += deviance

avg_deviance = sum_deviance / (len(test_loader) * BATCH_SIZE)
print(f"Total Test Data: {len(test_set)}")
print(f"Average deviance: {avg_deviance:.2f}")
print(f"Number of major errors: {num_major_deviance}")
print(f"Number of times overshot: {num_overshot}")
print(f"Number of times undershot: {num_undershot}")
print(f"Number of times correct: {num_correct}")
print("Model Evaluation finished.")


torch.save(model, "models/speed_recommendation_model.pt")
