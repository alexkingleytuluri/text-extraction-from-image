import os
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "finetuned_trocr"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "finetune_data",
    "1.png"
)


print("=" * 60)
print("FINE-TUNED TrOCR — SINGLE IMAGE")
print("=" * 60)

print("\nLoading fine-tuned model...")

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

model.eval()

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id


print(f"Testing image: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not read image: {IMAGE_PATH}")
    raise SystemExit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("\nRunning OCR...\n")

pixel_values = processor(
    img_rgb,
    return_tensors="pt"
).pixel_values

generated_ids = model.generate(
    pixel_values,
    max_new_tokens=64,
    num_beams=4,
    early_stopping=True
)

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]


print("=" * 60)
print(f"Predicted Text: {generated_text}")
print("=" * 60)

print(
    "\nNote: This image is part of the fine-tuning dataset "
    "and is used only as a training-sample demonstration."
)