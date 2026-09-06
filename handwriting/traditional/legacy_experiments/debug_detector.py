import easyocr
import cv2

# 1. Load EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

# 2. Load image
image_path = "word_img2.jpg"
img = cv2.imread(image_path)

# 3. Find words
print("Finding words...")
results = reader.readtext(image_path, detail=1, paragraph=False)

# 4. Draw red boxes around everything it found
for (bbox, text, conf) in results:
    (tl, tr, br, bl) = bbox
    cv2.rectangle(img, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 0, 255), 2)

# 5. Save the image
cv2.imwrite("boxes.jpg", img)
print("Saved 'boxes.jpg'! Open this file and look at where the red boxes are drawn.")