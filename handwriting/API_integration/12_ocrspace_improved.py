import os
import requests
import cv2
import base64
from dotenv import load_dotenv

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "word_img2.jpg")

# Load .env from project root
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# --------------------------------------------------
# Load API key
# --------------------------------------------------

API_KEY = os.getenv("OCRSPACE_API_KEY")

if not API_KEY:
    print("ERROR: OCRSPACE_API_KEY not found in .env")
    print("OCR.space API test skipped.")
    exit()

# --------------------------------------------------
# Load image
# --------------------------------------------------

if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    exit()

img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"ERROR: Could not read {IMAGE_PATH}")
    exit()

# --------------------------------------------------
# Convert image to Base64 Data URI
# --------------------------------------------------

_, encoded_image = cv2.imencode(".jpg", img)
img_base64 = base64.b64encode(encoded_image).decode("utf-8")
img_data_uri = "data:image/jpeg;base64," + img_base64

# --------------------------------------------------
# OCR.space API
# --------------------------------------------------

API_URL = "https://api.ocr.space/parse/image"

headers = {
    "apikey": API_KEY
}

payload = {
    "base64Image": img_data_uri,
    "language": "eng",
    "isTable": False,
    "scale": True,
    "OCREngine": 2
}

print("Sending prescription to OCR.space (Engine 2)...\n")

try:
    response = requests.post(
        API_URL,
        headers=headers,
        data=payload
    )

    if response.status_code == 200:
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            print("API Error:", result.get("ErrorMessage"))
        else:
            parsed_results = result.get("ParsedResults", [])

            if parsed_results:
                parsed_text = parsed_results[0].get("ParsedText", "")

                print("==================================")
                print("OCR.space Engine 2 Output:")
                print("==================================")
                print(parsed_text)
            else:
                print("OCR.space returned no parsed text.")

    else:
        print(f"HTTP Error {response.status_code}: {response.text}")

except requests.RequestException as e:
    print("OCR.space request failed:")
    print(e)
