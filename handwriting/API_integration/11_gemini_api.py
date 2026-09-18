import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "word_img2.jpg")

# Load .env from project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# --------------------------------------------------
# API configuration
# --------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit()

client = genai.Client(api_key=API_KEY)

# --------------------------------------------------
# Load image
# --------------------------------------------------

if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    exit()

with open(IMAGE_PATH, "rb") as image_file:
    image_data = image_file.read()

# --------------------------------------------------
# OCR prompt
# --------------------------------------------------

prompt = """
You are performing handwriting OCR.

Transcribe all handwritten text in this image exactly as it appears.

Rules:
- Do not guess or invent text.
- Preserve the original line breaks.
- Preserve numbers, units, and punctuation.
- Return only the transcription.
"""

print("Sending prescription to Google Gemini AI...\n")

# --------------------------------------------------
# Gemini request
# --------------------------------------------------

try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(
                data=image_data,
                mime_type="image/jpeg"
            )
        ]
    )

    print("==================================")
    print("Google Gemini AI Output:")
    print("==================================")
    print(response.text)

except Exception as e:
    print("Gemini API Error:")
    print(e)
