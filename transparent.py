import cv2
import numpy as np

# =========================
# 1️⃣ Load Transparent PNG
# =========================
img = cv2.imread("img/img-out/1_transparent.png", cv2.IMREAD_UNCHANGED)

if img is None:
    print("Image not found")
    exit()

if img.shape[2] != 4:
    print("Image does not have alpha channel")
    exit()

# Split channels (BGRA)
b, g, r, a = cv2.split(img)

# =========================
# 2️⃣ Create Banana Mask
# =========================
banana_mask = a > 0
banana_pixels = np.sum(banana_mask)

# Merge BGR
bgr = cv2.merge((b, g, r))

# Convert to HSV
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# =========================
# 3️⃣ Define HSV Ranges
# =========================

# --- Green ---
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# --- Strong Yellow ---
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# --- Light Yellow (NEW) ---
lower_light_yellow = np.array([20, 40, 170])
upper_light_yellow = np.array([35, 120, 255])

# --- Brown ---
lower_brown = np.array([0, 30, 0])
upper_brown = np.array([25, 255, 150])

# =========================
# 4️⃣ Create Masks
# =========================
mask_green = cv2.inRange(hsv, lower_green, upper_green) > 0

mask_yellow_strong = cv2.inRange(hsv, lower_yellow, upper_yellow) > 0
mask_yellow_light  = cv2.inRange(hsv, lower_light_yellow, upper_light_yellow) > 0
mask_yellow = mask_yellow_strong | mask_yellow_light

mask_brown = cv2.inRange(hsv, lower_brown, upper_brown) > 0

# Optional: remove small noise
kernel = np.ones((3,3), np.uint8)
mask_green  = cv2.morphologyEx(mask_green.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
mask_yellow = cv2.morphologyEx(mask_yellow.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
mask_brown  = cv2.morphologyEx(mask_brown.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

# =========================
# 5️⃣ Count Pixels (Inside Banana Only)
# =========================
green_pixels  = np.sum(mask_green & banana_mask)
yellow_pixels = np.sum(mask_yellow & banana_mask)
brown_pixels  = np.sum(mask_brown & banana_mask)

# =========================
# 6️⃣ Normalize To 100%
# =========================
total_detected = green_pixels + yellow_pixels + brown_pixels

if total_detected == 0:
    print("No color detected")
    exit()

green_percent  = (green_pixels  / total_detected) * 100
yellow_percent = (yellow_pixels / total_detected) * 100
brown_percent  = (brown_pixels  / total_detected) * 100

# =========================
# 7️⃣ Print Results
# =========================
print("Total banana pixels:", banana_pixels)
print(f"Green:  {green_percent:.2f}%")
print(f"Yellow: {yellow_percent:.2f}%")
print(f"Brown:  {brown_percent:.2f}%")
print("Total:", green_percent + yellow_percent + brown_percent)

# =========================
# 8️⃣ Visualization
# =========================
result = np.zeros_like(bgr)

result[mask_green & banana_mask]  = [0, 255, 0]       # Green
result[mask_yellow & banana_mask] = [0, 255, 255]     # Yellow
result[mask_brown & banana_mask]  = [42, 42, 165]     # Brown

cv2.imshow("Original", bgr)
cv2.imshow("Color Classification", result)
cv2.waitKey(0)
cv2.destroyAllWindows()