import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_SIZE = (64, 64)

train_df = pd.read_csv('data/GTSRB/Train.csv', sep='\t')
test_df = pd.read_csv('data/GTSRB/Test.csv', sep='\t')

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