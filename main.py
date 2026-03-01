import cv2
target_height = 500
target_width = 600
img = cv2.imread('1.jpg')
def resize_image(image, target_height, target_width):
    h, w = image.shape[:2]
    scale = target_height / h
    new_width = int(w * scale)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    resized = resized[0:target_height, 0:target_width]
    return resized

# Show result
img = resize_image(img, target_height, target_width)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Resized", img)
cv2.imshow("HSV", hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()