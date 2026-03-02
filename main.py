import cv2
import numpy as np

target_height = 500
target_width = 600


def resize_with_padding(image, target_height, target_width):
    h, w = image.shape[:2]

    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Preserve number of channels (3 or 4)
    channels = image.shape[2]
    canvas = np.zeros((target_height, target_width, channels), dtype=np.uint8)

    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def remove_bg(image, sat_thresh=40, val_thresh=180, kernel_size=5):
    # If image has alpha, convert to BGR first
    if image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image_bgr = image

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, val_thresh])
    upper_white = np.array([179, sat_thresh, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    foreground_mask = cv2.bitwise_not(white_mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_OPEN, kernel
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_CLOSE, kernel
    )

    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = foreground_mask

    return bgra


# ---------- MAIN ----------

img = cv2.imread('img/img-out/1_transparent.png', cv2.IMREAD_UNCHANGED)
print("Shape:", img.shape)
if img is None:
    raise ValueError("Image not found")

# Resize
img_resized = resize_with_padding(img, target_height, target_width)

# Convert to BGR before HSV (because HSV needs 3 channels)
if img_resized.shape[2] == 4:
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2BGR)
else:
    img_bgr = img_resized

hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

cv2.imshow("Original", img)
cv2.imshow("HSV", hsv_img)

cv2.waitKey(0)
cv2.destroyAllWindows()