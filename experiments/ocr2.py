import cv2
import pytesseract
import fitz
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


#IMAGE
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not open: {path}")
    return img

def preprocess(img):
    blocksize = 11
    c = 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.medianBlur(gray,3)
    #noise = cv2.GaussianBlur(gray,(3,3),0)
    #noise = cv2.blur(gray, (3, 3))
    #noise = cv2.bilateralFilter(gray, 9, 75, 75)
    cleaned = cv2.adaptiveThreshold(noise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c)
    cv2.imwrite("debug_cleaned.jpeg", cleaned)
    return cleaned

def ocr_image(path):
    raw = load_image(path)
    cleaned = preprocess(raw)
    text = pytesseract.image_to_string(cleaned, config = '--psm 11')
    return text

#PDF
def ocr_pdf(path):
    doc = fitz.open(path)
    full_text = ""

    for page_number, page in enumerate(doc):
        direct_text = page.get_text()

        if direct_text.strip():
            print(f"Page{page_number+1}: found digital text, using it directly")
            full_text += direct_text

        else:
            print(f"Page{page_number+1}: no text found, running OCR instead")

            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("jpeg")
            page_image = Image.open(io.BytesIO(img_bytes))

            page_text = pytesseract.image_to_string(page_image)
            full_text += page_text
    return full_text


#main flow
def run(path):
    if path.lower().endswith(".pdf"):
        return ocr_pdf(path)
    else:
        return ocr_image(path)
    
if __name__ == "__main__":
    print("Start")

    result = run("sample-local-pdf.pdf")
    #result = run("test.png")
    print("\nExtracted text:")
    print(result)