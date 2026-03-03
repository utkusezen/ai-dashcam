import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import image_feature_extraction as feature_extraction
from classes.sign_classification_nn import TrafficSignCNN
from classes.speed_recommendation_nn import SpeedRecommendationModel
from utils.draw_image import draw_bounding_boxes
from utils.image_transforms import resize_image_keep_ratio

MAX_IMG_SIZE = 1024
CLASSIFICATION_NUM_CLASSES = 43
MIN_SCORE = 0.6
BOX_PADDING = 5

detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
detection_model.load_state_dict(
    torch.load("models/sign_detection_state_dict.pt", map_location="cpu", weights_only=True)
)
detection_model.to("cpu")
detection_model.eval()



classification_model = TrafficSignCNN(num_classes=CLASSIFICATION_NUM_CLASSES)
classification_model.load_state_dict(
    torch.load("models/sign_recognition_state_dict.pt", map_location="cpu", weights_only=True)
)
classification_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
classification_model.to("cpu")
classification_model.eval()


speed_model = SpeedRecommendationModel()
speed_model.load_state_dict(
    torch.load("models/speed_recommendation_state_dict.pt", map_location="cpu", weights_only=True)
)
speed_model.to("cpu")
speed_model.eval()

@torch.no_grad()
def run(image: Image.Image) -> dict:
    results = {
        "signs": [],
        "recommended_speed": None
    }

    image_np_rgb = np.array(image)
    image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
    resized_np_bgr, scale = resize_image_keep_ratio(MAX_IMG_SIZE, image_np_bgr)
    resized_np_rgb = cv2.cvtColor(resized_np_bgr, cv2.COLOR_BGR2RGB)

    tensor_input = transforms.ToTensor()(resized_np_rgb).unsqueeze(0)
    detections = detection_model(tensor_input)[0]

    boxes = detections["boxes"]
    scores = detections["scores"]
    draw_bounding_boxes(resized_np_bgr, boxes.cpu().numpy(), scores.cpu().numpy())


    for box, score in zip(boxes, scores):
        if score < MIN_SCORE:
            continue

        x1, y1, x2, y2 = box.tolist()
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale
        x1 = max(0, x1 - BOX_PADDING)
        y1 = max(0, y1 - BOX_PADDING)
        x2 = min(image.width, x2 + BOX_PADDING)
        y2 = min(image.height, y2 + BOX_PADDING)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_tensor = classification_transform(cropped_image).unsqueeze(0)
        class_preds = classification_model(cropped_tensor)
        sign_class = class_preds.argmax(dim=1).item()

        results["signs"].append(sign_class)

    brightness, contrast = feature_extraction.compute_brightness_and_contrast(image_np_rgb)
    _, driveable_area = feature_extraction.compute_driveable_area(image_np_rgb)

    (num_lanes, max_lane_len,
     angle_right, angle_left,
     vp_found, vp_offset_x, vp_offset_y) = feature_extraction.compute_lane_features(image_np_rgb)

    num_lanes_norm = np.clip(num_lanes, 0, 6) / 6
    angle_right_norm = angle_right / 90 if angle_right is not None else 0
    angle_left_norm = angle_left / 90 if angle_left is not None else 0

    speed_features = np.array([
        brightness,
        contrast,
        driveable_area,
        num_lanes_norm,
        max_lane_len,
        angle_right_norm,
        angle_left_norm,
        float(vp_found),
        vp_offset_x if vp_offset_x is not None else 0,
        vp_offset_y if vp_offset_y is not None else 0,
    ], dtype=np.float32)

    speed_tensor = torch.tensor(speed_features).unsqueeze(0)
    speed = speed_model(speed_tensor).item()
    speed = (speed // 10) * 10
    results["recommended_speed"] = speed

    return results