import cv2

img = cv2.imread("test.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray.png", gray)

print("Image converted successfully!")