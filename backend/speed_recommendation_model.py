import json
import pandas as pd
from tqdm import tqdm

DATA_PATH = "data/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
LABELS_PATH = "data/bdd100k/manual_speed_recommendation_labels.csv"

def load_and_filter_training_data(path):
    """
    Loads data and extracts the two most important attributes: image names, lanes and driveable_areas
    :param path: the path to the json file
    :return: a pandas dataframe object containing image names, lanes and driveable areas
    """
    lanes_per_image = []
    drivable_areas_per_image = []
    image_names = []
    with open(path, "r") as f:
        data = json.load(f)

        for image_descr in data:
            image_names.append(image_descr["name"])
            lanes = []
            drivable_areas = []

            for label in image_descr["labels"]:
                if label["category"] == "lane":
                    lanes.append({
                        "direction":    label["attributes"]["laneDirection"],
                        "style":        label["attributes"]["laneStyle"] ,
                        "vertices":     label["poly2d"][0]["vertices"]
                    })

                elif label["category"] == "drivable area":
                    drivable_areas.append(label["poly2d"][0]["vertices"])

            lanes_per_image.append(lanes)
            drivable_areas_per_image.append(drivable_areas)

    filtered_data = pd.DataFrame()
    filtered_data["name"] = image_names
    filtered_data["lane"] = lanes_per_image
    filtered_data["drivable_area"] = drivable_areas_per_image
    return filtered_data

road_information_per_image = load_and_filter_training_data(DATA_PATH)

labels = pd.read_csv(LABELS_PATH)
labels = labels.drop(columns=["annotator", "lead_time", "updated_at", "created_at", "annotation_id", "id"])
labels["image"] = [name[24:] for name in labels["image"]]
annotated_road_information_per_image = road_information_per_image[road_information_per_image["name"].isin(labels["image"])]









