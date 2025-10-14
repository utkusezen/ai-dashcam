import os
import cv2
import numpy as np
import pandas as pd
import image_feature_extraction as ft_extr
from tqdm import tqdm

DATA_PATH = "data/bdd100k/bdd100k/bdd100k/images/10k/train"
LABELS_PATH = "data/bdd100k/manual_speed_recommendation_labels.csv"

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

        rows.append({
            "image": img_path,
            "brightness": brightness,
            "contrast": contrast,
            "driveable_area": driveable_area,
            "num_lanes": num_lanes,
            "max_lane_len": max_lane_len,
            "angle_right": angle_right,
            "angle_left": angle_left,
            "vp_found": vp_found,
            "vp_offset_x": vp_offset_x,
            "vp_offset_y": vp_offset_y
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
