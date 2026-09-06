import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2

# 1. Load both models
print("Loading AI Detector & TrOCR Recognizer...")
# EasyOCR is our "Detector" (finds text blocks)
detector = easyocr.Reader(['en'], gpu=False) 
# TrOCR is our "Recognizer" (reads handwriting)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 2. Load the image
image_path = "word_img2.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

# 3. Use EasyOCR to find text blocks (paragraph=True merges words together)
print("Finding text blocks with EasyOCR...")
results = detector.readtext(image_path, detail=1, paragraph=True)

# Sort results top to bottom
results = sorted(results, key=lambda r: r[0][0][1])

print(f"\nFound {len(results)} text blocks. Reading with TrOCR...\n")

# 4. Read each block with TrOCR
for i, result in enumerate(results):
    # With paragraph=True, result is a list containing [bbox, text]
    bbox = result[0]
    
    (tl, tr, br, bl) = bbox
    x1 = int(min(tl[0], bl[0]))
    y1 = int(min(tl[1], tr[1]))
    x2 = int(max(tr[0], br[0]))
    y2 = int(max(bl[1], br[1]))
    
    # Add padding to the crop
    pad = 15
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
    
    crop = img[y1:y2, x1:x2]
    
    # Ignore tiny noise
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        continue
        
    # Convert to RGB for TrOCR
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Ask TrOCR to read it
    pixel_values = processor(crop_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Block {i+1}: {generated_text}")