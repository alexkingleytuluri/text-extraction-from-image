import os
import base64

import requests
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY")

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "handwriting",
    "API_integration",
    "word_img2.jpg"
)

OCR_URL = "https://api.ocr.space/parse/image"


def extract_text(image_path):
    if not OCRSPACE_API_KEY:
        raise ValueError(
            "OCRSPACE_API_KEY not found in .env"
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    payload = {
        "apikey": OCRSPACE_API_KEY,
        "base64Image": (
            "data:image/jpeg;base64,"
            + encoded_image
        ),
        "language": "eng",
        "isOverlayRequired": False,
        "isTable": False,
        "scale": True,
        "OCREngine": 2
    }

    response = requests.post(
        OCR_URL,
        data=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    if result.get("IsErroredOnProcessing"):
        error_message = result.get(
            "ErrorMessage",
            "Unknown OCR.space error"
        )
        raise RuntimeError(
            f"OCR.space error: {error_message}"
        )

    parsed_results = result.get(
        "ParsedResults",
        []
    )

    text_parts = []

    for parsed_result in parsed_results:
        parsed_text = parsed_result.get(
            "ParsedText",
            ""
        ).strip()

        if parsed_text:
            text_parts.append(parsed_text)

    return "\n\n".join(text_parts)


if __name__ == "__main__":
    try:
        text = extract_text(IMAGE_PATH)

        print("\n--- OCR.space Result ---\n")
        print(text)

    except Exception as error:
        print(f"\nOCR failed: {error}")