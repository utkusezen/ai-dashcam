import cv2


def resize_image_keep_ratio(max_size, image):
    h, w, _ = image.shape

    if w >= h:
        scale = max_size / w
    else:
        scale = max_size / h

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))
    return resized, scale