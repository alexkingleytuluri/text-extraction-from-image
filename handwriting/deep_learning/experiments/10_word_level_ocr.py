from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np
import torch

# Prevent Windows crash
torch.set_num_threads(1)

# 1. Load General TrOCR Model
print("Loading General TrOCR Model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

model.eval()
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# 2. Load the prescription
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "word_img2.jpg")
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Clean lines
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
no_lines = cv2.subtract(binary, detected_lines)

# 4. WORD-LEVEL GROUPING (The Magic Fix!)
# We use a smaller kernel (30, 10) to group letters into words, but NOT whole lines!
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
dilated = cv2.dilate(no_lines, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

words = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Filter out tiny noise, keep word-sized boxes
    if w > 40 and h > 20:
        words.append((x, y, w, h))

# Sort top-to-bottom, then left-to-right
words = sorted(words, key=lambda item: (item[1] // 30, item[0]))
print(f"\nFound {len(words)} individual words. Reading with TrOCR...\n")

# 5. Read each word individually
predicted_text = ""
current_y = None

for i, (x, y, w, h) in enumerate(words):
    # Add a line break if we moved to a new row
    if current_y is None:
        current_y = y
    elif abs(y - current_y) > 30:
        predicted_text += "\n"
        current_y = y
        
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    
    word_crop = img[y1:y2, x1:x2]
    word_rgb = cv2.cvtColor(word_crop, cv2.COLOR_BGR2RGB)
    
    # Ask TrOCR to read this single word
    pixel_values = processor(word_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=20, num_beams=4, early_stopping=True)
    word_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    predicted_text += word_text + " "

print("==================================")
print("AI Predicted Text (Word-by-Word):")
print("==================================")
print(predicted_text)
print("==================================")