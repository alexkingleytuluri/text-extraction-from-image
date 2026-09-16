import os
import csv
import torch

torch.set_num_threads(1)

from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Adafactor
)
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "finetune_data")
LABEL_FILE = os.path.join(DATA_DIR, "labels.csv")
SAVE_PATH = os.path.join(BASE_DIR, "finetuned_trocr")

MODEL_NAME = "microsoft/trocr-base-handwritten"


class HandwritingDataset(Dataset):

    def __init__(self, data_dir, label_file, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.data = []

        with open(label_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) >= 2:
                    self.data.append((row[0], row[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name, text = self.data[idx]

        img_path = os.path.join(
            self.data_dir,
            file_name
        )

        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).input_ids

        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels.squeeze()
        }


print("=" * 70)
print("TrOCR FINE-TUNING")
print("=" * 70)

print("\nLoading TrOCR model...")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)


# Configure decoder tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id


# Freeze vision encoder
print("Freezing vision encoder...")

for param in model.encoder.parameters():
    param.requires_grad = False


# Dataset
dataset = HandwritingDataset(
    data_dir=DATA_DIR,
    label_file=LABEL_FILE,
    processor=processor
)

print(f"\nDataset samples: {len(dataset)}")

if len(dataset) == 0:
    raise RuntimeError("No training samples were found.")


dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

print(f"Batches per epoch: {len(dataloader)}")


# Training configuration
device = torch.device("cpu")
model.to(device)

epochs = 3

optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)


print(f"\nStarting training for {epochs} epochs...")
print(f"Device: {device}\n")


training_failed = False

for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    print(f"--- Epoch {epoch + 1}/{epochs} ---")

    for batch_idx, batch in enumerate(dataloader):

        try:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"Batch {batch_idx + 1}/{len(dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )

        except Exception as e:

            print("\nTRAINING FAILED")
            print(f"Error: {e}")

            training_failed = True
            break

    if training_failed:
        break

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch + 1} completed "
        f"| Average Loss: {average_loss:.4f}\n"
    )


if training_failed:

    print("=" * 70)
    print("TRAINING FAILED")
    print("The model was NOT saved as a final fine-tuned model.")
    print("=" * 70)

    raise SystemExit(1)


# Save only after successful completion
print("Training completed successfully.")
print(f"\nSaving model to:\n{SAVE_PATH}")

model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)

print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)
print(f"Model saved to: {SAVE_PATH}")