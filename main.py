import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
INPUT_FOLDER = "img/img-in"
OUTPUT_FOLDER = "img/img-out"
TARGET_HEIGHT = 500
TARGET_WIDTH = 600

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# Resize With Padding
# =========================
def resize_with_padding(image, target_height, target_width):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


# =========================
# Remove White Background
# =========================
def remove_white_background(image, sat_thresh=40, val_thresh=180, kernel_size=5):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, val_thresh])
    upper_white = np.array([179, sat_thresh, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    foreground_mask = cv2.bitwise_not(white_mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = foreground_mask

    return bgra


# =========================
# Analyze Banana Colors
# =========================
def analyze_banana_colors(image):

    b, g, r, a = cv2.split(image)
    banana_mask = a > 0
    bgr = cv2.merge((b, g, r))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Green
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Yellow (expanded)
    lower_y1 = np.array([22, 120, 120])
    upper_y1 = np.array([35, 255, 255])
    lower_y2 = np.array([20, 60, 170])
    upper_y2 = np.array([35, 160, 255])
    lower_y3 = np.array([18, 30, 190])
    upper_y3 = np.array([35, 140, 255])
    lower_y4 = np.array([15, 80, 120])
    upper_y4 = np.array([22, 255, 255])

    # Brown
    lower_brown = np.array([0, 50, 20])
    upper_brown = np.array([18, 255, 130])

    mask_green = cv2.inRange(hsv, lower_green, upper_green) > 0

    mask_yellow = (
        (cv2.inRange(hsv, lower_y1, upper_y1) > 0) |
        (cv2.inRange(hsv, lower_y2, upper_y2) > 0) |
        (cv2.inRange(hsv, lower_y3, upper_y3) > 0) |
        (cv2.inRange(hsv, lower_y4, upper_y4) > 0)
    )

    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown) > 0

    green_pixels  = np.sum(mask_green & banana_mask)
    yellow_pixels = np.sum(mask_yellow & banana_mask)
    brown_pixels  = np.sum(mask_brown & banana_mask)

    total = green_pixels + yellow_pixels + brown_pixels

    if total == 0:
        return 0, 0, 0

    # Normalize to 100%
    return (
        green_pixels / total * 100,
        yellow_pixels / total * 100,
        brown_pixels / total * 100
    )


# =========================
# PROCESS ALL IMAGES
# =========================

green_list = []
yellow_list = []
brown_list = []
image_names = []

for filename in sorted(os.listdir(INPUT_FOLDER)):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_transparent.png")

    print(f"Processing: {filename}")

    img = cv2.imread(input_path)
    if img is None:
        continue

    img_resized = resize_with_padding(img, TARGET_HEIGHT, TARGET_WIDTH)
    img_transparent = remove_white_background(img_resized)

    green, yellow, brown = analyze_banana_colors(img_transparent)

    print(f"Green: {green:.2f}% | Yellow: {yellow:.2f}% | Brown: {brown:.2f}%")

    green_list.append(green)
    yellow_list.append(yellow)
    brown_list.append(brown)
    image_names.append(name)

    cv2.imwrite(output_path, img_transparent)


# =========================
# PLOT LINE GRAPH
# =========================
# =========================
# PLOT LINE GRAPH (Color Matched)
# =========================

# =========================
# PLOT LINE GRAPH (Thicker)
# =========================

plt.figure(figsize=(10, 6))

plt.plot(image_names, green_list,  color="#2E7D32", linewidth=4)
plt.plot(image_names, yellow_list, color="#FFD600", linewidth=4)
plt.plot(image_names, brown_list,  color="#6D4C41", linewidth=4)

plt.xlabel("Image")
plt.ylabel("Percentage (%)")
plt.title("Banana Ripeness Color Analysis")

plt.legend(["Green", "Yellow", "Brown"])
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()