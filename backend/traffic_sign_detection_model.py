import os
import torch
import torchvision.models.detection
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

TRAIN_DATA_PATH = "data/GTSDB_TT100k/Train"
TEST_DATA_PATH = "data/GTSDB_TT100k/Test"
MAX_IMG_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 4

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
    for image_path in tqdm(os.listdir(images_path)):
        path_data_images.append(os.path.join(images_path, image_path))
        label_path = os.path.join(labels_path, image_path.split(".")[0] + ".txt")
        if os.path.exists(label_path):
            path_data_labels.append(label_path)
        else:
            path_data_labels.append(None)

    return path_data_images, path_data_labels

def load_image_and_label_data(image_path, label_path):
    """
    Loads image and label data from given path and label data from given path.
    Images that do not have labels are assigned an empty label.
    :param image_path: The path to the image
    :param label_path: The path to the label
    :return: image data and label data
    """
    image = Image.open(image_path).convert("RGB").copy()
    labels = []

    if label_path:
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                label = list(map(float, line.strip().split()))
                labels.append(label)
    return image, labels

def convert_bounding_box_to_coordinates(bounding_box):
    """
    Converts bounding box coordinates (x, y, w, h) to two x, y coordinates.
    :param bounding_box: list of attributes of the bounding box, assumed to be of form (x, y, w, h)
    :return: a list of x, y coordinates of form (x1, y1, x2, y2)
    """
    x, y, w, h = bounding_box
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2

def resize_image_and_bounding_boxes(max_size, image, boxes):
    """
    Resizes image, so that the longest side has the maximum size, while the aspect ratio stays the same.
    Scales all points according to the new image size.
    :param max_size: The maximum size of the longest side of the resized image
    :param image: The image to resize
    :param boxes: The list of bounding box points to resize, assumed to be of form (x1, y1, x2, y2)
    :return: The resized image and points
    """
    cur_width, cur_height = image.size

    if cur_width >= cur_height:
        new_width = max_size
        new_height = int(max_size * cur_height / cur_width)
    else:
        new_height = max_size
        new_width = int(max_size * cur_width / cur_height)

    scale_x = new_width / cur_width
    scale_y = new_height / cur_height

    resized_image = image.resize((new_width, new_height))
    new_boxes = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in boxes]

    return resized_image, new_boxes

def custom_collate_fn(batch):
    """
    Custom collate function that orders the batch into tuples of data and labels.
    :param batch: The batch that gets reorganized. Of form [(data1, label1), (data2, label2), (...)]
    :return: tuples of data and labels of form (data1, data2, ...) (label1, label2, ...)
    """
    return tuple(zip(*batch))


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

        if labels:
            boxes = [l[1:] for l in labels]
            boxes = list(map(convert_bounding_box_to_coordinates, boxes))
            image, boxes = resize_image_and_bounding_boxes(self.max_size, image, boxes)
            boxes = torch.tensor(boxes, dtype=torch.float)
            class_ids = torch.full((len(labels),), 1, dtype=torch.int64)
        else:
            image, _ = resize_image_and_bounding_boxes(self.max_size, image, [])
            boxes = torch.zeros((0, 4), dtype=torch.float)
            class_ids = torch.full((1,), 0, dtype=torch.int64)

        target = {"boxes": boxes, "labels": class_ids}

        if self.transform:
            image = self.transform(image)

        return image, target


train_x, train_y = collect_sign_detection_path_data(TRAIN_DATA_PATH)
test_x, test_y = collect_sign_detection_path_data(TEST_DATA_PATH)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = TrafficSignDataset(train_x, train_y, MAX_IMG_SIZE, transform)
test_dataset = TrafficSignDataset(test_x, test_y, MAX_IMG_SIZE, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()
for epoch in range(EPOCHS):
    sum_loss = 0.0
    for batch in tqdm(train_loader):
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        losses = model(images, targets)
        loss = sum(l for l in losses.values())
        sum_loss += loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {sum_loss:.4f}%")
