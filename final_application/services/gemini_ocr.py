import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)


load_dotenv(ENV_PATH)


API_KEY = os.getenv("GEMINI_API_KEY")


_client = None


def get_client():
    """Create the Gemini client when first needed."""

    global _client

    if _client is None:

        if not API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found in .env"
            )

        _client = genai.Client(
            api_key=API_KEY
        )

    return _client


def extract_text(file_path):
    """
    Extract text from an image or PDF using Google Gemini.

    Args:
        file_path: Path to the input image or PDF.

    Returns:
        str: Gemini OCR transcription.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with open(file_path, "rb") as file:
        file_data = file.read()

    extension = os.path.splitext(
        file_path
    )[1].lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
    }

    mime_type = mime_types.get(extension)

    if mime_type is None:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    prompt = """
You are performing OCR on an image or PDF document.

Transcribe all visible handwritten or printed text exactly as it appears.

Rules:
- Do not guess or invent text.
- Preserve the original line breaks where possible.
- Preserve numbers, units, and punctuation.
- Return only the transcription.
"""

    client = get_client()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_text(
                text=prompt
            ),
            types.Part.from_bytes(
                data=file_data,
                mime_type=mime_type
            )
        ]
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return response.text.strip()