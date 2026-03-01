import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("img/1.jpg")

# Convert BGR → RGB (IMPORTANT!)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Split channels
r, g, b = cv2.split(img_rgb)

# Plot histograms
plt.figure()
plt.hist(r.ravel(), bins=256, range=[0,256])
plt.hist(g.ravel(), bins=256, range=[0,256])
plt.hist(b.ravel(), bins=256, range=[0,256])

plt.title("RGB Color Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()