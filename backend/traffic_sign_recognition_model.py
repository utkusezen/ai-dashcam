import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

IMG_SIZE = (64, 64)

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

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)


class TrafficSignDataset(Dataset):
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

        return image, label



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dataset = TrafficSignDataset(train_x, train_y, transform=transform)
test_dataset = TrafficSignDataset(test_x, test_y, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

