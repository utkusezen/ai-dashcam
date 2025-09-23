import json
import pandas as pd
from tqdm import tqdm

TRAIN_DATA_PATH = "data/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
TEST_DATA_PATH = "data/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"

TRAIN_LABEL_PATH = ""
TEST_LABEL_PATH = ""

def load_and_filter_training_data(path):
    lanes_per_image = []
    drivable_areas_per_image = []
    with open(path, "r") as f:
        data = json.load(f)
        for d in data:
            lanes = []
            drivable_areas = []
            for entry in d["labels"]:
                if entry["category"] == "lane":
                    lanes.append((entry["attributes"]["laneDirection"], entry["attributes"]["laneStyle"] ,entry["poly2d"][0]["vertices"]))
                elif entry["category"] == "drivable area":
                    drivable_areas.append(entry["poly2d"][0]["vertices"])
            lanes_per_image.append(lanes)
            drivable_areas_per_image.append(drivable_areas)
    return lanes_per_image, drivable_areas_per_image


lanes, drivable_areas = load_and_filter_training_data(TRAIN_DATA_PATH)
print(f"lanes: {len(lanes)}, drivable areas: {len(drivable_areas)}")








