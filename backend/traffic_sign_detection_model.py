import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

TRAIN_DATA_PATH = "data/GTSDB/Train"
TEST_DATA_PATH = "data/GTSDB/Test"

def load_sign_detection_data(directory):
    """
    Loads image and label data from the given directory and returns them as numpy arrays
    :param directory: directory with subdirectories of images and labels
    :return: X, y (data, labels)
    """
    images_path = directory + "/images/"
    labels_path = directory + "/labels/"
    images = []
    labels = []
    for image_file in tqdm(os.listdir(images_path)):
        image = Image.open(images_path + image_file)
        label_file = image_file.split(".")[0] + ".txt"
        label = []
        if not os.path.exists(labels_path + label_file):
            label = [-1, -1, -1, -1]
        else:
            with open(labels_path + label_file, "r") as f:
                label = f.readline().split(" ")
                label.pop(0)
        images.append(image)
        labels.append(list(map(float, label)))
    return np.array(images), np.array(labels)

train_x, train_y = load_sign_detection_data(TRAIN_DATA_PATH)
test_x, test_y = load_sign_detection_data(TEST_DATA_PATH)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)