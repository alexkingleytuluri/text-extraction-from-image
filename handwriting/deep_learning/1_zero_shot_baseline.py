import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "word_img2.jpg")

MODEL_NAME = "microsoft/trocr-base-handwritten"


print("=" * 60)
print("ZERO-SHOT TrOCR BASELINE")
print("=" * 60)

print("\nLoading pre-trained TrOCR model...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

print(f"Loading image: {IMAGE_PATH}")
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not read image: {IMAGE_PATH}")
    raise SystemExit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("\nAnalyzing handwriting with pre-trained TrOCR...\n")

pixel_values = processor(
    img_rgb,
    return_tensors="pt"
).pixel_values

generated_ids = model.generate(
    pixel_values,
    max_new_tokens=64
)

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("=" * 60)
print(f"Predicted Text: {generated_text}")
print("=" * 60)