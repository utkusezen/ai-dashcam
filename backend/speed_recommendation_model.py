import os
import cv2
import pandas as pd
import image_feature_extraction as ft_extr
from tqdm import tqdm

DATA_PATH = "data/bdd100k/bdd100k/bdd100k/images/10k/train"
LABELS_PATH = "data/bdd100k/manual_speed_recommendation_labels.csv"

def load_data_and_extract_features(path):
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


labels = pd.read_csv(LABELS_PATH)
labels = labels.drop(columns=["annotator", "lead_time", "updated_at", "created_at", "annotation_id", "id"])
labels["image"] = [name[24:] for name in labels["image"]]

data = load_data_and_extract_features(DATA_PATH)
print(data.shape)

