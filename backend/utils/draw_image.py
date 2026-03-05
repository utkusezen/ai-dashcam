import os
import cv2

OUTPUT_PATH_BOUNDING_BOXES = "output/live/detection"
OUTPUT_PATH_DRIVEABLE_AREA = "output/live/driveable_area"
OUTPUT_PATH_VANISHING_POINT = "output/live/vanishing_point"

def draw_bounding_boxes(image, boxes, scores):
    """
    Draws predicted bounding boxes on an image with corresponding scores.
    Saves images as .jpg in "output/live/detection"
    :param image: image to draw bounding boxes on
    :param boxes: the predicted bounding boxes
    :param scores: the predicted scores
    :return: None
    """

    file_path = os.path.join(OUTPUT_PATH_BOUNDING_BOXES, f"{len(os.listdir(OUTPUT_PATH_BOUNDING_BOXES))}.jpg")
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5),
                    font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(file_path, img)


def draw_flood_fill(image, mask, seed_point):
    img = image.copy()
    file_path = os.path.join(OUTPUT_PATH_DRIVEABLE_AREA, f"{len(os.listdir(OUTPUT_PATH_DRIVEABLE_AREA))}.jpg")
    img[~mask] = 0
    cv2.circle(img, seed_point, 5, (0, 0, 255), -1)
    cv2.imwrite(file_path, img)


def draw_vanishing_point(image, left_line, right_line, vp):
    img = image.copy()
    file_path = os.path.join(OUTPUT_PATH_VANISHING_POINT, f"{len(os.listdir(OUTPUT_PATH_VANISHING_POINT))}.jpg")

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if vp is not None:
        cv2.circle(img, vp, 5, (0, 0, 255), -1)

    cv2.imwrite(file_path, img)