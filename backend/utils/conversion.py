import sys

import torch
from classes.sign_classification_nn import TrafficSignCNN
from classes.speed_recommendation_nn import SpeedRecommendationModel

def convert_detection_model(
    old_path="../models/sign_detection_model.pt",
    new_path="../models/sign_detection_state_dict.pt",
    device="cpu"
):
    model = torch.load(old_path, map_location=device)
    torch.save(model.state_dict(), new_path)

def convert_classification_model(
    old_path="../models/sign_recognition_model.pt",
    new_path="../models/sign_recognition_state_dict.pt",
    device="cpu"
):
    model = torch.load(old_path, map_location=device)
    torch.save(model.state_dict(), new_path)

def convert_speed_model(
    old_path="../models/speed_recommendation_model.pt",
    new_path="../models/speed_recommendation_state_dict.pt",
    device="cpu"
):
    model = torch.load(old_path, map_location=device)
    torch.save(model.state_dict(), new_path)


if __name__ == "__main__":
    DEVICE = "cpu"
    sys.modules["__main__"].TrafficSignCNN = TrafficSignCNN
    sys.modules["__main__"].SpeedRecommendationModel = SpeedRecommendationModel

    convert_detection_model(device=DEVICE)
    convert_classification_model(device=DEVICE)
    convert_speed_model(device=DEVICE)

    print("All models converted successfully")
