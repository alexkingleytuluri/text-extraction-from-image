print("START")

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

img = Image.open("test.png")

print(img)

text = pytesseract.image_to_string(img)

print("Extracted text:")
print(text)
