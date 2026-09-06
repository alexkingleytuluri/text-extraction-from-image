import easyocr
import cv2

# 1. Load EasyOCR (This model does BOTH detection and recognition!)
print("Loading EasyOCR AI Model...")
reader = easyocr.Reader(['en'], gpu=False)

# 2. Load the image
image_path = "word_img2.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

# 3. Read the entire image at once!
# paragraph=True groups words together into lines automatically
# detail=0 means it just returns the text, not the bounding boxes
print("Finding and reading text...\n")
results = reader.readtext(image_path, detail=0, paragraph=True)

print("==================================")
print("AI Predicted Text:")
for line in results:
    print(line)
print("==================================")