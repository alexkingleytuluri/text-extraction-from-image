import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2

# 1. Load both models
print("Loading AI Detector & TrOCR Recognizer...")
detector = easyocr.Reader(['en'], gpu=False) 
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 2. Load the image
image_path = "word_img2.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"ERROR: Could not read {image_path}")
    exit()

# 3. Find individual words using EasyOCR
print("Finding words with EasyOCR...")
results = detector.readtext(image_path, detail=1, paragraph=False)

# Sort results top to bottom based on the top-left corner
results = sorted(results, key=lambda r: r[0][0][1])

# 4. Group words into lines based on their Y coordinate
lines = []
current_line = []
last_y = -9999

for res in results:
    bbox, text, conf = res
    y = bbox[0][1]
    # If the word is much lower than the last one, it's a new line
    if abs(y - last_y) > 15:
        if current_line:
            lines.append(current_line)
        current_line = [res]
    else:
        current_line.append(res)
    last_y = y

if current_line:
    lines.append(current_line)

print(f"\nGrouped into {len(lines)} lines. Reading with TrOCR...\n")

# 5. Read each line with TrOCR
for i, line in enumerate(lines):
    # Sort words in this line from left to right
    line = sorted(line, key=lambda r: r[0][0][0])
    
    # Find the bounding box that covers ALL words in this line
    xs = [pt[0] for res in line for pt in res[0]]
    ys = [pt[1] for res in line for pt in res[0]]
    
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    
    # Add padding
    pad = 10
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
    
    crop = img[y1:y2, x1:x2]
    
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        continue
        
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # Ask TrOCR to read the whole line
    pixel_values = processor(crop_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"Line {i+1}: {generated_text}")