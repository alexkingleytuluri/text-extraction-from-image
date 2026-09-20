import os
import base64
from dotenv import load_dotenv
from mistralai.client import Mistral


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)


# ---------------------------------------------------------
# API configuration
# ---------------------------------------------------------

API_KEY = os.getenv("MISTRAL_API_KEY")

_client = None


def get_client():
    """Create and return the Mistral client."""

    global _client

    if _client is None:

        if not API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY not found in .env"
            )

        _client = Mistral(
            api_key=API_KEY
        )

    return _client


# ---------------------------------------------------------
# Image OCR
# ---------------------------------------------------------

def extract_text_from_image(image_path):
    """Extract text from an image using Mistral OCR."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    extension = os.path.splitext(
        image_path
    )[1].lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }

    mime_type = mime_types.get(
        extension
    )

    if mime_type is None:
        raise ValueError(
            f"Unsupported image format: {extension}"
        )

    # Read image
    with open(
        image_path,
        "rb"
    ) as image_file:

        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    # Convert image to a Base64 data URL
    data_url = (
        f"data:{mime_type};base64,{image_data}"
    )

    client = get_client()

    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": data_url
        }
    )

    pages = []

    for page in response.pages:

        if page.markdown:
            pages.append(
                page.markdown
            )

    return "\n\n".join(
        pages
    ).strip()


# ---------------------------------------------------------
# PDF OCR
# ---------------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using Mistral OCR."""

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    client = get_client()

    # Upload PDF to Mistral
    with open(
        pdf_path,
        "rb"
    ) as pdf_file:

        uploaded_file = client.files.upload(
            file={
                "file_name": os.path.basename(
                    pdf_path
                ),
                "content": pdf_file
            },
            purpose="ocr"
        )

    try:

        # Generate temporary signed URL
        signed_url = client.files.get_signed_url(
            file_id=uploaded_file.id,
            expiry=1
        )

        # Process PDF
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url
            }
        )

        pages = []

        for page in response.pages:

            if page.markdown:
                pages.append(
                    page.markdown
                )

        return "\n\n".join(
            pages
        ).strip()

    finally:

        # Delete uploaded file after processing
        try:

            client.files.delete(
                file_id=uploaded_file.id
            )

        except Exception:
            pass


# ---------------------------------------------------------
# Main OCR function
# ---------------------------------------------------------

def extract_text(file_path):
    """
    Automatically select the appropriate
    Mistral OCR method based on file type.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = os.path.splitext(
        file_path
    )[1].lower()

    # Image files
    if extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:

        return extract_text_from_image(
            file_path
        )

    # PDF files
    if extension == ".pdf":

        return extract_text_from_pdf(
            file_path
        )

    raise ValueError(
        f"Unsupported file type: {extension}"
    )