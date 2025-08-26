import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

TRAIN_DATA_PATH = "data/GTSDB_TT100k/Train"
TEST_DATA_PATH = "data/GTSDB_TT100k/Test"

def collect_sign_detection_path_data(directory):
    """
    Collects image and label path data from the given directory. Does not open image data to save memory.
    Images that do not have labels are assigned None as label path.
    :param directory: directory with subdirectories of images and labels
    :return: image path list and label path list
    """
    images_path = directory + "/images/"
    labels_path = directory + "/labels/"

    path_data_images = []
    path_data_labels = []
    for image_path in tqdm(os.listdir(images_path)):
        path_data_images.append(os.path.join(images_path, image_path))
        label_path = os.path.join(labels_path, image_path.split(".")[0] + ".txt")
        if os.path.exists(label_path):
            path_data_labels.append(label_path)
        else:
            path_data_labels.append(None)

    return path_data_images, path_data_labels

def load_image_and_label_data(image_path, label_path):
    """
    Loads image and label data from given path and label data from given path.
    Images that do not have labels are assigned a dummy label.
    :param image_path: The path to the image
    :param label_path: The path to the label
    :return: image data and label data
    """
    image = Image.open(image_path).convert("RGB").copy()
    bounding_boxes = []

    if label_path:
        with open(label_path, "r") as f:
            lines = f.readlines()
            bounding_boxes.append([list(map(float, l.strip().split())) for l in lines])
    else:
        bounding_boxes = [[-1, -1, -1, -1, -1]]

    return image, bounding_boxes

class TrafficSignDataset(Dataset):
    """
    A PyTorch Dataset class to simplify access and transformations on the data.
    Uses paths to save memory and only loads data on access.
    """
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image, label = load_image_and_label_data(image_path, label_path)

        if self.transform:
            image = self.transform(image)

        return image, label


train_x, train_y = collect_sign_detection_path_data(TRAIN_DATA_PATH)
test_x, test_y = collect_sign_detection_path_data(TEST_DATA_PATH)
