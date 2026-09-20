import os
import cv2
import textdistance

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "finetuned_trocr"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "word_img.jpeg"
)


DICTIONARY = [
    "Azithromycin",
    "Dolo",
    "650",
    "Crocin",
    "Aspirin",
    "Paracetamol",
    "Ibuprofen",
    "Telma",
    "Ecosprin",
    "gold"
]


def correct_spelling(word):

    best_match = word
    lowest_distance = float("inf")

    for dictionary_word in DICTIONARY:

        distance = textdistance.levenshtein(
            word.lower(),
            dictionary_word.lower()
        )

        if distance < lowest_distance:
            lowest_distance = distance
            best_match = dictionary_word

    max_allowed_distance = max(
        1,
        int(len(word) * 0.3)
    )

    if lowest_distance <= max_allowed_distance:
        return best_match

    return word


print("=" * 70)
print("FINE-TUNED TrOCR — FULL PAGE SHOWCASE")
print("=" * 70)

print("\nLoading fine-tuned model...")

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

model.eval()

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id


print(f"Loading image: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not read image: {IMAGE_PATH}")
    raise SystemExit(1)


gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)


# Remove long horizontal printed lines
_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (50, 1)
)

detected_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel,
    iterations=2
)

no_lines = cv2.subtract(
    binary,
    detected_lines
)


# Group nearby handwriting into lines
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (100, 5)
)

dilated = cv2.dilate(
    no_lines,
    kernel,
    iterations=1
)


contours, _ = cv2.findContours(
    dilated,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


lines = []

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    if w > 80 and h > 20:
        lines.append((x, y, w, h))


lines.sort(key=lambda item: item[1])


print(f"\nDetected lines: {len(lines)}")
print("Running OCR + optional dictionary correction...\n")


for i, (x, y, w, h) in enumerate(lines):

    pad = 20

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)

    x2 = min(
        img.shape[1],
        x + w + pad
    )

    y2 = min(
        img.shape[0],
        y + h + pad
    )

    line_crop = img[y1:y2, x1:x2]

    line_rgb = cv2.cvtColor(
        line_crop,
        cv2.COLOR_BGR2RGB
    )

    pixel_values = processor(
        line_rgb,
        return_tensors="pt"
    ).pixel_values

    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=64,
        num_beams=4,
        early_stopping=True
    )

    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    corrected_words = []

    for word in raw_text.split():
        corrected_words.append(
            correct_spelling(word)
        )

    corrected_text = " ".join(
        corrected_words
    )

    print(f"Line {i + 1}")
    print(f"  Raw OCR : {raw_text}")
    print(f"  Corrected: {corrected_text}")
    print()