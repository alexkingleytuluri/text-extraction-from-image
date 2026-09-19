import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv("../.env")

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found.")

client = Mistral(api_key=api_key)

response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png"
    }
)

print("Mistral OCR connection successful.")
print()
print(response.pages[0].markdown)