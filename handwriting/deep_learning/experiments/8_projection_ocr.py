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

# Dilate slightly to connect broken pen strokes within a word
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
dilated = cv2.dilate(no_lines, kernel, iterations=1)

# 4. PROJECTION PROFILING (The ultimate line separator)
# Sum the white pixels in each row
row_sums = np.sum(dilated > 0, axis=1)

# Find rows that have text (more than 10 white pixels)
threshold = 10
is_text_row = row_sums > threshold

# Find the Y-coordinates where lines start and end
line_bounds = []
in_line = False
start_y = 0

for y, has_text in enumerate(is_text_row):
    if has_text and not in_line:
        start_y = y
        in_line = True
    elif not has_text and in_line:
        end_y = y
        # Only keep if the line is tall enough to be real text (ignore tiny noise)
        if end_y - start_y > 15:
            line_bounds.append((max(0, start_y - 15), min(img.shape[0], end_y + 15)))
        in_line = False

print(f"\nFound {len(line_bounds)} lines using Projection Profiling.\n")

# 5. Read each line
for i, (y1, y2) in enumerate(line_bounds):
    # Crop the full width of this specific line
    line_crop = img[y1:y2, :]
    
    # Convert to RGB
    line_rgb = cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
    
    # Ask TrOCR to read it
    pixel_values = processor(line_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=100, num_beams=4, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Line {i+1}: {generated_text}")