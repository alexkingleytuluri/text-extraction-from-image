import os
import cv2
import pytesseract

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "word_img2.jpg")

# Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --------------------------------------------------
# Load image
# --------------------------------------------------

if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    exit()

img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not read {IMAGE_PATH}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# Tesseract OCR
# --------------------------------------------------

print("Reading with Tesseract (Open Source)...\n")

# PSM 6 assumes a single uniform block of text
text = pytesseract.image_to_string(
    gray,
    config="--psm 6"
)

print("==================================")
print("Tesseract Output:")
print("==================================")
print(text)
