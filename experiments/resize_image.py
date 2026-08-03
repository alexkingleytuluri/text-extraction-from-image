import cv2

img = cv2.imread("test.png")

resized = cv2.resize(img, (320, 240))

cv2.imwrite("resized.png", resized)

print("Image resized successfully!")