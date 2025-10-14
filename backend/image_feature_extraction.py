import math
import cv2
import numpy as np

FLOOD_FLAGS = 8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
FLOOD_TOLERANCE = (20, 20, 20)

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 60
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 50

STRAIGHT_MIN_ANGLE = 20
STRAIGHT_MAX_ANGLE = 160

def __default_seed_point(w, h):
    return w // 2, int(h * 0.75)

def is_asphalt_like(pixel_bgr):
    """
    Checks whether a pixel has typical road color-properties or not
    :param pixel_bgr: the pixel to check
    :return: is the pixel asphalt like
    """
    pixel = np.uint8([[pixel_bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    return s < 100 and 50 < v < 200

def __trapeze_bottom_half(w, h):
    """
    Create a trapeze on the bottom half of the image to limit the region of interest.
    :param w: width of the image
    :param h: height of the image
    :return: a trapeze on the bottom half of the image (short side is on the top)
    """
    polygon = np.array([[
        (0, h),
        (int(0.25 * w), int(0.45 * h)),
        (int(0.75 * w), int(0.45 * h)),
        (w, h)
    ]])
    return polygon

def __compute_angle_to_horizontal(x1,y1,x2,y2):
    """
    Compute the angle between two points and the horizontal plane.
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: the angle between the line and the horizontal plane
    """
    delta_y = abs(y2 - y1)
    delta_x = abs(x2 - x1)

    angle = math.atan2(delta_y, delta_x) * 180 / math.pi
    return angle

def __find_roughly_straight_lines(lines, min_angle, max_angle):
    """
    Filter out lines that are below or above the given angle boundaries.
    :param lines: the lines to filter
    :param min_angle: the minimum angle
    :param max_angle: the maximum angle
    :return: a filtered list of lines
    """
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            angle = 90
        else:
            angle = np.degrees(np.arctan2(dy, dx))

        if min_angle < abs(angle) < max_angle:
            filtered.append(line)
    return filtered

def __line_to_abc(x1, y1, x2, y2):
    """
    Convert a line from (x1, y1, x2, y2) to (A, B, C).
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: a tuple (A, B, C)
    """
    A = y2 - y1
    B = x1 - x2
    C = A*x1 + B*y1
    return A, B, C

def __intersection(line1, line2):
    """
    Find the intersection of two lines.
    :param line1: the first line
    :param line2: the second line
    :return: the intersection point (x, y)
    """
    if line1 is None or line2 is None:
        return None
    A1, B1, C1 = __line_to_abc(*line1)
    A2, B2, C2 = __line_to_abc(*line2)
    det = A1*B2 - A2*B1
    if abs(det) < 1e-4:
        return None
    x = (B2*C1 - B1*C2) / det
    y = (A1*C2 - A2*C1) / det
    return int(x), int(y)

def __mirror_line_on_y_axis(line, img_width):
    """
    Mirror a line on the y-axis.
    :param line: the line to mirror
    :param img_width: width of the image
    :return: the mirrored line
    """
    x1, y1, x2, y2 = line
    return img_width - x1, y1, img_width - x2, y2

def __compute_vanishing_point(lines, img_width):
    """
    Find the best fitting line on each side of the image and compute the vanishing point of these two lines.
    A fitting line is chosen by the least distance to the center of the image.
    :param lines: the lines to compute
    :param img_width: the width of the image
    :return: best fitting left line, best fitting right lane, the vanishing point
    """
    center_x = img_width // 2
    line_bottoms = []
    for x1, y1, x2, y2 in lines:
        if y1 > y2:
            line_bottoms.append({"line": (x1, y1, x2, y2), "bottom_x": x1})
        else:
            line_bottoms.append({"line": (x1, y1, x2, y2), "bottom_x": x2})

    left_line = max(
        (l for l in line_bottoms if (l["bottom_x"] - center_x) <= 0),
        key=lambda l: l["bottom_x"] - center_x,
        default=None
    )

    right_line = min(
        (l for l in line_bottoms if (l["bottom_x"] - center_x) > 0),
        key=lambda l: l["bottom_x"] - center_x,
        default=None
    )

    left_line = left_line["line"] if left_line is not None else None
    right_line = right_line["line"] if right_line is not None else None

    if left_line is None and right_line is not None:
        left_line = __mirror_line_on_y_axis(right_line, img_width)
    elif right_line is None and left_line is not None:
        right_line = __mirror_line_on_y_axis(left_line, img_width)

    vp = __intersection(left_line, right_line)

    return left_line, right_line, vp

def __compute_lanes(img):
    """
    Approximate lanes on an image.
    :param img: the image
    :return: lanes found
    """
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    polygon = __trapeze_bottom_half(w, h)
    cv2.fillPoly(mask, polygon, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    lines = [line[0] for line in lines] if lines is not None else []
    return __find_roughly_straight_lines(lines, STRAIGHT_MIN_ANGLE, STRAIGHT_MAX_ANGLE)

def compute_brightness_and_contrast(img):
    """
    Compute the brightness and contrast of an image. (Normalized between 0 and 1)
    :param img: the image
    :return: brightness, contrast
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return brightness / 255.0, contrast / 255.0

def compute_driveable_area(img):
    """
    Approximate the drivable area of an image. Only accounts for the bottom half of the image.
    :param img: the image
    :return: number of driveable area pixels, percentage of drivable area
    """
    h, w = img.shape[:2]
    seed_point = __default_seed_point(w, h)
    seed_point_pixel = img[seed_point[1]][seed_point[0]]
    if not is_asphalt_like(seed_point_pixel):
        return 0, 0
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(img, mask, seed_point, (255, 255, 255), FLOOD_TOLERANCE, FLOOD_TOLERANCE, FLOOD_FLAGS)
    flood_mask = mask[1:-1, 1:-1]
    bottom_half = flood_mask[h // 2:, :]

    road_pixels = cv2.countNonZero(bottom_half)
    total_pixels = bottom_half.size
    coverage = (road_pixels / total_pixels)
    return road_pixels, coverage

def compute_lane_features(img):
    """
    Compute features of lanes found in an image.
    :param img: the image
    :return: number of lanes,
             longest lane length,
             angle of the estimated right lane,
             angle of the estimated left lane,
             boolean determined by if the vanishing point was found,
             vanishing point distance to the center on the x-axis,
             vanishing point distance to the center on the y-axis
    """
    h, w = img.shape[:2]
    lanes = __compute_lanes(img)
    num_lanes = len(lanes)
    max_lane_length = max((math.dist((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lanes), default=0)
    trapeze_w = w
    trapeze_h = h * 0.45
    max_lane_length_norm = max_lane_length / math.hypot(trapeze_w, trapeze_h)


    right_lane, left_lane, vp = __compute_vanishing_point(lanes, w)
    vp_found = 1 if vp is not None else 0
    vp_offset_x = (vp[0] / w - 0.5) * 2 if vp_found else None
    vp_offset_y = (vp[1] / h - 0.5) * 2 if vp_found else None
    right_angle = __compute_angle_to_horizontal(*right_lane) if right_lane else None
    left_angle = __compute_angle_to_horizontal(*left_lane) if left_lane else None

    return num_lanes, max_lane_length_norm, right_angle, left_angle, vp_found, vp_offset_x, vp_offset_y

