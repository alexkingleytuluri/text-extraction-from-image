import os
import cv2
import pytesseract
import pymupdf


# Tesseract installation path
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_text_from_image(image_path):
    """Extract printed text from an image using Tesseract."""

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tesseract LSTM OCR for printed text
    text = pytesseract.image_to_string(
        gray,
        config="--oem 1 --psm 6"
    )

    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF page by page."""

    document = pymupdf.open(pdf_path)
    extracted_pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            # First try extracting existing PDF text
            text = page.get_text("text").strip()

            # If the PDF is scanned/image-based, render the page
            # and run Tesseract OCR.
            if not text:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                image = cv2.imdecode(
                    __import__("numpy").frombuffer(
                        pixmap.tobytes("png"),
                        dtype="uint8"
                    ),
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                text = pytesseract.image_to_string(
                    gray,
                    config="--oem 1 --psm 6"
                ).strip()

            if text:
                extracted_pages.append(
                    f"--- Page {page_number} ---\n{text}"
                )

    finally:
        document.close()

    return "\n\n".join(extracted_pages)


def extract_text(file_path):
    """Automatically select OCR method based on file type."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    if extension in [".jpg", ".jpeg", ".png", ".webp"]:
        return extract_text_from_image(file_path)

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )