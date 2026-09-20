import os

import cv2
import textdistance

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "handwriting",
    "deep_learning",
    "finetuned_trocr"
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


_processor = None
_model = None


def load_model():
    """Load the fine-tuned TrOCR model once."""

    global _processor, _model

    if _processor is None or _model is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Fine-tuned TrOCR model not found: {MODEL_PATH}"
            )

        _processor = TrOCRProcessor.from_pretrained(
            MODEL_PATH
        )

        _model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_PATH
        )

        _model.eval()

        _model.config.decoder_start_token_id = (
            _processor.tokenizer.cls_token_id
        )

        _model.config.pad_token_id = (
            _processor.tokenizer.pad_token_id
        )

        _model.config.eos_token_id = (
            _processor.tokenizer.sep_token_id
        )

    return _processor, _model


def correct_spelling(word):
    """Apply the existing dictionary-based correction."""

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


def detect_lines(img):
    """Detect handwritten text lines using OpenCV."""

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

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w > 80 and h > 20:
            lines.append(
                (x, y, w, h)
            )

    lines.sort(
        key=lambda item: item[1]
    )

    return lines


def recognize_line(
    line_crop,
    processor,
    model
):
    """Recognize one handwritten line using TrOCR."""

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

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]


def extract_text(
    image_path,
    use_dictionary=True
):
    """
    Extract handwritten text from an image.

    Args:
        image_path: Path to the input image.
        use_dictionary: Whether to apply dictionary correction.

    Returns:
        str: Extracted text with one detected line per output line.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    processor, model = load_model()

    lines = detect_lines(img)

    if not lines:
        raise ValueError(
            "No handwritten text lines detected."
        )

    extracted_lines = []

    for x, y, w, h in lines:

        pad = 20

        x1 = max(
            0,
            x - pad
        )

        y1 = max(
            0,
            y - pad
        )

        x2 = min(
            img.shape[1],
            x + w + pad
        )

        y2 = min(
            img.shape[0],
            y + h + pad
        )

        line_crop = img[
            y1:y2,
            x1:x2
        ]

        raw_text = recognize_line(
            line_crop,
            processor,
            model
        )

        if use_dictionary:

            corrected_words = []

            for word in raw_text.split():
                corrected_words.append(
                    correct_spelling(word)
                )

            final_text = " ".join(
                corrected_words
            )

        else:
            final_text = raw_text

        extracted_lines.append(
            final_text
        )

    return "\n".join(
        extracted_lines
    )