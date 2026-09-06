import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import numpy as np

# 1. Load both models
print("Loading AI Detector & Recognizer...")
# EasyOCR is our "Detector" (finds the text boxes)
detector = easyocr.Reader(['en'], gpu=False) 
# TrOCR is our "Recognizer" (reads the text)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 2. Load the image
image_path = "word_img2.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

# 3. Use AI to find the text boxes
print("Finding text with AI...")
results = detector.readtext(image_path)

print(f"\nFound {len(results)} pieces of text. Reading with AI...\n")

# 4. Sort results top to bottom
results = sorted(results, key=lambda r: r[0][0][1])

# 5. Read each piece of text with TrOCR
for i, (bbox, text, conf) in enumerate(results):
    # Get coordinates from the bounding box
    (tl, tr, br, bl) = bbox
    x1 = int(min(tl[0], bl[0]))
    y1 = int(min(tl[1], tr[1]))
    x2 = int(max(tr[0], br[0]))
    y2 = int(max(bl[1], br[1]))
    
    # Add padding
    pad = 15
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
    
    # Crop the image
    crop = img[y1:y2, x1:x2]
    
    # Ignore tiny noise
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        continue
        
    # Convert to RGB for TrOCR
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Ask TrOCR to read it
    pixel_values = processor(crop_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Line {i+1}: {generated_text}")