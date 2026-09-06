import os
import csv
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

print("Loading Processor...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Read the first line of your CSV
with open("finetune_data/labels.csv", 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    first_row = next(reader)
    file_name = first_row[0]
    text = first_row[1]

print(f"Testing first image: {file_name} with label: {text}")

# Open the image
img_path = os.path.join("finetune_data", file_name)
try:
    image = Image.open(img_path).convert("RGB")
    print(f"Image opened successfully! Size: {image.size}")
except Exception as e:
    print(f"ERROR opening image: {e}")
    exit()

# Try to process the image
try:
    print("Processing image with TrOCR Processor...")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(f"Image processed! Tensor shape: {pixel_values.shape}")
    
    print("Tokenizing text...")
    labels = processor.tokenizer(text, padding="max_length", max_length=64, return_tensors="pt").input_ids
    print(f"Text tokenized! Shape: {labels.shape}")
    
    print("\nSUCCESS! Data is ready for training.")
except Exception as e:
    print(f"ERROR processing data: {e}")