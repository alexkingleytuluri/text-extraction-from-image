import os
# CRITICAL FIX: Disable OneDNN AND the buggy PIR executor for Windows
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'

from paddleocr import PaddleOCR
import cv2

# 1. Load the General Industry OCR Model
print("Loading PaddleOCR (General Model)...")
ocr = PaddleOCR(lang='en', use_textline_orientation=True)

# 2. Load the image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "word_img2.jpg")  # The messy prescription
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

print("\nAnalyzing document with General AI...\n")

# 3. Run the OCR engine
results = ocr.predict(img)

# 4. Print the results
print("==================================")
print("AI Predicted Text:")
print("==================================")

if results and results[0] is not None:
    for res in results:
        if 'rec_texts' in res:
            for text, score in zip(res['rec_texts'], res['rec_scores']):
                if score > 0.4:
                    print(f"{text}  (Confidence: {score*100:.1f}%)")
        else:
            for line in res:
                text = line[1][0]
                score = line[1][1]
                if score > 0.4:
                    print(f"{text}  (Confidence: {score*100:.1f}%)")
else:
    print("No text found.")
print("==================================")