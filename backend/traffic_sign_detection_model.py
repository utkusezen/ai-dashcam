import os
import random
from unittest.mock import DEFAULT

import cv2
import numpy as np
import torch
import torchvision.models.detection
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

TRAIN_DATA_PATH = "data/GTSDB_TT100k/Train"
TEST_DATA_PATH = "data/GTSDB_TT100k/Test"
OUTPUT_PATH = "output/detection/"
METRICS_PATH = "metrics/detection_metrics"
MAX_IMG_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 4
MIN_SCORE = 0.6


def clear_output_folder():
    """
    Clears the output folder
    :return: None
    """
    for file in os.listdir(OUTPUT_PATH):
        os.remove(os.path.join(OUTPUT_PATH, file))


def collect_sign_detection_path_data(directory):
    """
    Collects image and label path data from the given directory. Does not open images to save memory.
    Images that do not have labels are assigned None as label path.
    :param directory: directory with subdirectories of images and labels
    :return: image path list and label path list
    """
    images_path = directory + "/images/"
    labels_path = directory + "/labels/"

    path_data_images = []
    path_data_labels = []
    for image_path in tqdm(os.listdir(images_path), desc="Collecting image and label paths", ):
        path_data_images.append(os.path.join(images_path, image_path))
        label_path = os.path.join(labels_path, image_path.split(".")[0] + ".txt")
        if os.path.exists(label_path):
            path_data_labels.append(label_path)
        else:
            path_data_labels.append(None)

    return path_data_images, path_data_labels


def filter_empties(img_paths, label_paths, percentage=0.05, seed=123):
    """
    Filters out images with no labels according to given percentage.
    :param img_paths: image paths
    :param label_paths: label paths (None if empty)
    :param percentage: the desired percentage of images with no labels
    :param seed: seed for random filtering
    :return: img_paths, label_paths
    """
    empty = []
    not_empty = []

    for img_paths, label_paths in zip(img_paths, label_paths):
        if label_paths is None:
            empty.append((img_paths, label_paths))
        else:
            not_empty.append((img_paths, label_paths))

    random.seed(seed)
    filtered = random.sample(empty, round(len(empty) * percentage))
    filtered.extend(not_empty)
    random.shuffle(filtered)
    images, labels = zip(*filtered)
    return list(images), list(labels)


def filter_small_signs(boxes, min_size=24*24):
    """
    Remove very small signs, that may impair model training. Filters out signs with an image area less than min_size.
    :param boxes: x1, y1, x2, y2
    :param min_size: the minimum size of a sign on an image
    :return: filtered bounding boxes
    """
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w * h >= min_size:
            new_boxes.append(box)

    return new_boxes


def load_image_and_label_data(image_path, label_path):
    """
    Loads image and label data from given path and label data from given path.
    :param image_path: The path to the image
    :param label_path: The path to the label
    :return: image data and label data
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labels = []

    if label_path:
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                label = list(map(float, line.strip().split()))
                labels.append(label)
    return image, labels


def convert_bounding_box_to_coordinates(bounding_box, image_shape):
    """
    Converts bounding box coordinates (x, y, w, h) to coordinate-form and scales them to the size of the image.
    :param bounding_box: list of attributes of the bounding box, assumed to be of form (x, y, w, h)
    :param image_shape: shape of the image
    :return: coordinates of form (x1, y1, x2, y2)
    """
    x, y, w, h = bounding_box
    image_h, image_w, _ = image_shape
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1 * image_w, y1 * image_h, x2 * image_w, y2 * image_h


def resize_image_and_bounding_boxes(max_size, image, boxes):
    """
    Resizes image, so that the longest side has the maximum size, while the aspect ratio stays the same.
    Scales all points according to the new image size.
    :param max_size: The maximum size of the longest side of the resized image
    :param image: The image to resize
    :param boxes: The list of bounding box points to resize, assumed to be of form (x1, y1, x2, y2)
    :return: The resized image and points
    """
    cur_height, cur_width, _ = image.shape

    if cur_width >= cur_height:
        scale = max_size / cur_width
    else:
        scale = max_size / cur_height

    new_width = int(cur_width * scale)
    new_height = int(cur_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    new_boxes = [[x1 * scale, y1 * scale,
                  x2 * scale, y2 * scale] for x1, y1, x2, y2 in boxes]

    return resized_image, new_boxes


def custom_collate_fn(batch):
    """
    Custom collate function that orders the batch into tuples of data and labels.
    :param batch: The batch that gets reorganized. Of form [(data1, label1), (data2, label2), (...)]
    :return: tuples of data and labels of form (data1, data2, ...) (label1, label2, ...)
    """
    return tuple(zip(*batch))


def draw_bounding_boxes(image, pred_boxes, scores, true_boxes):
    """
    Draws predicted bounding boxes on an image with corresponding scores.
    Saves images as .jpg in "output/detection/"
    :param image: image to draw bounding boxes on
    :param pred_boxes: the predicted bounding boxes
    :param scores: the predicted scores
    :param true_boxes: the true bounding boxes of the dataset
    :return: None
    """

    if len(pred_boxes) == 0 and len(true_boxes) == 0:
        return

    file_path = os.path.join(OUTPUT_PATH, f"{len(os.listdir(OUTPUT_PATH))}.jpg")
    num_signs_found = len(pred_boxes)
    num_signs = len(true_boxes)
    img = image.copy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2) in true_boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    for (x1, y1, x2, y2), score in zip(pred_boxes, scores):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5),
                    font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    sidebar_width = 350
    sidebar = np.full((h, sidebar_width, 3), fill_value=255, dtype=np.uint8)
    x_text, y_text = 10, 100
    cv2.putText(sidebar, f"{num_signs_found} / {num_signs} signs found", (x_text, y_text),
                font, 1, (0, 0, 0), 3, cv2.LINE_AA)
    y_text += 100
    cv2.putText(sidebar, "Green: Prediction", (x_text, y_text),
                font, 1, (0, 200, 0), 3, cv2.LINE_AA)
    y_text += 100
    cv2.putText(sidebar, "Yellow: Ground Truth", (x_text, y_text),
                font, 1, (0, 200, 200), 3, cv2.LINE_AA)

    combined = np.concatenate((img, sidebar), axis=1)
    cv2.imwrite(file_path, combined)


class TrafficSignDataset(Dataset):
    """
    A PyTorch Dataset class to simplify access and transformations on the data.
    Uses paths to save memory and only loads data on access.
    """

    def __init__(self, image_paths, label_paths, max_size, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.max_size = max_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image, labels = load_image_and_label_data(image_path, label_path)
        boxes = [l[1:] for l in labels]
        boxes = [convert_bounding_box_to_coordinates(box, image.shape) for box in boxes]
        image, boxes = resize_image_and_bounding_boxes(self.max_size, image, boxes)
        boxes = filter_small_signs(boxes)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float)
            class_ids = torch.full((len(labels),), 1, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float)
            class_ids = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": class_ids}

        if self.transform:
            image = self.transform(image)

        return image, target

os.makedirs(OUTPUT_PATH, exist_ok=True)
clear_output_folder()

train_x, train_y = collect_sign_detection_path_data(TRAIN_DATA_PATH)
test_x, test_y = collect_sign_detection_path_data(TEST_DATA_PATH)
train_x, train_y = filter_empties(train_x, train_y)
test_x, test_y = filter_empties(test_x, test_y)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = TrafficSignDataset(train_x, train_y, MAX_IMG_SIZE, transform)
test_dataset = TrafficSignDataset(test_x, test_y, MAX_IMG_SIZE, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)

"""
for images, targets in tqdm(test_loader):
    for image, target in zip(images, targets):
        im = image.permute(1, 2, 0).numpy()
        draw_bounding_boxes(im, [], [], target["boxes"])


counts = []
for _, targets in tqdm(test_loader):
    for t in targets:
        counts.append(len(t["boxes"]))

# ZÃ¤hlen, wie oft jede Anzahl vorkommt
count_dict = Counter(counts)
print(count_dict)

# Plotten
plt.bar(count_dict.keys(), count_dict.values())
plt.xlabel("Anzahl Schilder pro Bild")
plt.ylabel("Anzahl Bilder")
plt.title("Verteilung der Verkehrsschilder im Datensatz")
plt.show()
"""


#model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler("cuda")
model = model.to(device)
model.train()
for epoch in range(EPOCHS):
    sum_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", colour="blue"):
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            losses = model(images, targets)
            loss = sum(losses.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_loss += loss.item()
    avg_loss = sum_loss / (len(train_loader) * BATCH_SIZE)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

model.eval()
found_signs = 0
total_signs = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc=f"Evaluating Model", colour="green"):
        images, targets = batch
        images = [img.to(device) for img in images]

        predictions = model(images)

        for image, p, target in zip(images, predictions, targets):
            mask_good_predictions = p["scores"] >= MIN_SCORE
            best_boxes = p["boxes"][mask_good_predictions].cpu().numpy()
            best_scores = p["scores"][mask_good_predictions].cpu().numpy()
            true_boxes = target["boxes"]

            image = image.cpu().permute(1, 2, 0).numpy()
            draw_bounding_boxes(image, best_boxes, best_scores, true_boxes)
            found_signs += len(best_boxes)
            total_signs += len(true_boxes)


print(f"Signs Detected: {found_signs} / {total_signs} signs found.")
torch.save(model, "models/sign_detection_model.pt")