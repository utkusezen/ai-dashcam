import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from classes.sign_classification_nn import TrafficSignCNN
from classes.speed_recommendation_nn import SpeedRecommendationModel
from utils.image_transforms import resize_image_keep_ratio

MAX_IMG_SIZE = 1024
CLASSIFICATION_NUM_CLASSES = 43
MIN_SCORE = 0.6

detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
detection_model.load_state_dict(torch.load("models/sign_detection_state_dict.pt", map_location="cpu"))
detection_transform = transforms.Compose([transforms.ToTensor()])
detection_model.eval()



classification_model = TrafficSignCNN(num_classes=CLASSIFICATION_NUM_CLASSES)
classification_model.load_state_dict(torch.load("models/sign_recognition_state_dict.pt", map_location="cpu"))
classification_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
classification_model.eval()


speed_model = SpeedRecommendationModel()
speed_model.load_state_dict(torch.load("models/speed_recommendation_state_dict.pt", map_location="cpu"))
speed_model.eval()

@torch.no_grad()
def run(image: Image.Image) -> dict:
    results = {
        "signs": [],
        "recommended_speed": None
    }

    image_np = np.array(image)
    resized_np, scale = resize_image_keep_ratio(MAX_IMG_SIZE, image_np)

    tensor_input = transforms.ToTensor()(resized_np).unsqueeze(0)
    detections = detection_model(tensor_input)[0]

    boxes = detections["boxes"]
    scores = detections["scores"]

    for box, score in zip(boxes, scores):
        if score < MIN_SCORE:
            continue

        x1, y1, x2, y2 = box.tolist()
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # bounding_box = [x1, y1, x2, y2]
        # confidence = float(score)
        # TODO: draw bbox on image and save to output


        # TODO: add optional padding
        cropped_tensor = image.crop((x1, y1, x2, y2))


        cropped_tensor = classification_transform(cropped_tensor).unsqueeze(0)
        class_preds = classification_model(cropped_tensor)
        sign_class = class_preds.argmax(dim=1).item()

        results["signs"].append(sign_class)

    # speed_features = extract(...)
    # speed_tensor = torch.tensor(speed_features).unsqueeze(0)
    # speed = speed_model(speed_tensor).item()
    # results["recommended_speed"] = speed
    # TODO: Speed model

    return results