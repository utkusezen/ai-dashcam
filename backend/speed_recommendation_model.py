import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import image_feature_extraction as ft_extr
from tqdm import tqdm

DATA_PATH = "data/bdd100k/bdd100k/bdd100k/images/10k/train"
LABELS_PATH = "data/bdd100k/manual_speed_recommendation_labels.csv"

BATCH_SIZE = 64

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

        if self.transform:
            features = self.transform(features)

        return features, label

dataset = SpeedRecommendationDataset(features=X, labels=y)
train_set, test_set = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






