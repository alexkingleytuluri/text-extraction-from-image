from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np

# 1. Load the AI Model
print("Loading AI Model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 2. Load the full page image
image_path = "word_img.jpeg"
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Remove Printed Lines (Magic!)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Detect horizontal lines (long thin kernel)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Subtract the lines from the binary image, leaving only the handwriting!
no_lines = cv2.subtract(binary, detected_lines)

# 4. Group words into lines
# Dilate the remaining text horizontally to merge words into line blocks
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 5))
dilated = cv2.dilate(no_lines, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lines = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # Filter out tiny noise
    if w > 80 and h > 20:
        lines.append((x, y, w, h))
        
lines = sorted(lines, key=lambda item: item[1])
print(f"\nFound {len(lines)} lines of text. Reading with AI...\n")

# 5. Text Recognition
for i, (x, y, w, h) in enumerate(lines):
    # Add padding to the crop
    pad = 20
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    
    line_crop = img[y1:y2, x1:x2]
    
    # Convert to RGB for TrOCR
    line_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
    
    # Ask AI to read this specific line
    pixel_values = processor(line_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Line {i+1}: {generated_text}")