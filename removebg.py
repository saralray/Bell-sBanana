import cv2
import numpy as np
import os

def remove_white_background(input_path, output_path,
                            sat_thresh=40,
                            val_thresh=180,
                            kernel_size=5):

    image = cv2.imread(input_path)
    if image is None:
        print(f"Failed to load: {input_path}")
        return

    # Convert once
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White mask using OpenCV (faster than numpy boolean)
    lower_white = np.array([0, 0, val_thresh])
    upper_white = np.array([179, sat_thresh, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert mask → keep foreground
    foreground_mask = cv2.bitwise_not(white_mask)

    # Morphology clean (open + close)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    # Convert to BGRA
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Assign alpha directly
    bgra[:, :, 3] = foreground_mask

    cv2.imwrite(output_path, bgra)
    print(f"Saved: {output_path}")


# --- Batch processing ---
input_folder = "img"
for i in range(1, 9):
    remove_white_background(
        os.path.join(input_folder, f"{i}.jpg"),
        os.path.join(input_folder, f"img-out/{i}_transparent.png")
    )