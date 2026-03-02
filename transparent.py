import cv2
import numpy as np

# =========================
# 1️⃣ Load Transparent PNG
# =========================
img = cv2.imread("img/img-out/7_transparent.png", cv2.IMREAD_UNCHANGED)

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

# --- YELLOW RANGES (Expanded) ---

# 1️⃣ Strong yellow
lower_y1 = np.array([22, 120, 120])
upper_y1 = np.array([35, 255, 255])

# 2️⃣ Light yellow
lower_y2 = np.array([20, 60, 170])
upper_y2 = np.array([35, 160, 255])

# 3️⃣ Creamy pale yellow
lower_y3 = np.array([18, 30, 190])
upper_y3 = np.array([35, 140, 255])

# 4️⃣ Slight orange-yellow (late ripe)
lower_y4 = np.array([15, 80, 120])
upper_y4 = np.array([22, 255, 255])

# --- Brown (refined for dark spots) ---
lower_brown = np.array([0, 50, 0])
upper_brown = np.array([18, 255, 130])

# =========================
# 4️⃣ Create Masks
# =========================

mask_green = cv2.inRange(hsv, lower_green, upper_green) > 0

mask_y1 = cv2.inRange(hsv, lower_y1, upper_y1) > 0
mask_y2 = cv2.inRange(hsv, lower_y2, upper_y2) > 0
mask_y3 = cv2.inRange(hsv, lower_y3, upper_y3) > 0
mask_y4 = cv2.inRange(hsv, lower_y4, upper_y4) > 0

mask_yellow = mask_y1 | mask_y2 | mask_y3 | mask_y4

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