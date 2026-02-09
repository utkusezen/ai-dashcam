import os
import cv2

OUTPUT_PATH = "output/live/detection"

def draw_bounding_boxes(image, boxes, scores):
    """
    Draws predicted bounding boxes on an image with corresponding scores.
    Saves images as .jpg in "output/live/detection"
    :param image: image to draw bounding boxes on
    :param boxes: the predicted bounding boxes
    :param scores: the predicted scores
    :return: None
    """

    file_path = os.path.join(OUTPUT_PATH, f"{len(os.listdir(OUTPUT_PATH))}.jpg")
    img = image.copy()
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5),
                    font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(file_path, img)