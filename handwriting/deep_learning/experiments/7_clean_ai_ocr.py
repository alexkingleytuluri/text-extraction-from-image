import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np
import os

# Disable PyTorch threading to prevent silent crashes
torch.set_num_threads(1) 

# 1. Load the General (Zero-Shot) TrOCR Model
print("Loading General TrOCR Model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

model.eval()
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# 2. Load the image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "word_img2.jpg")  # The messy prescription
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. ADVANCED CLEANING: Remove Printed Lines
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find long horizontal lines and erase them
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
no_lines = cv2.subtract(binary, detected_lines)

# 4. Group the remaining handwriting into lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 5))
dilated = cv2.dilate(no_lines, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lines = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > 80 and h > 20:
        lines.append((x, y, w, h))
        
lines = sorted(lines, key=lambda item: item[1])
print(f"\nFound {len(lines)} clean lines. Reading with General AI...\n")

# 5. Read each clean line with TrOCR
for i, (x, y, w, h) in enumerate(lines):
    pad = 20
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    
    # Crop the clean image
    line_crop = img[y1:y2, x1:x2]
    line_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
    
    # Read the text
    pixel_values = processor(line_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=50, num_beams=4, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Line {i+1}: {generated_text}")