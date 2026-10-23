# Multi-Model OCR System

A Streamlit application that brings together five OCR approaches for recognizing handwritten characters, handwriting, and printed documents.

## 1. OCR Modes

1. **Traditional ML — Character Recognition:** Classifies individual handwritten characters using a traditional machine-learning pipeline.
2. **Printed Document OCR:** Extracts printed or typed text from images and PDFs using Tesseract OCR.
3. **Deep Learning — Handwriting OCR:** Uses a fine-tuned TrOCR model for handwriting recognition.
4. **Gemini AI:** Uses the Gemini API for multimodal OCR.
5. **Mistral Vision:** Uses Mistral's vision model for image-based OCR.

## 2. Features

- Single Streamlit interface for all five approaches
- Image preview and file upload
- OCR results displayed in the application
- Downloadable text output
- Input validation and error handling
- Optional dictionary correction in the deep-learning mode

## 3. Supported Inputs

Supported image formats: JPG, JPEG, PNG, and WEBP.

PDF support depends on the selected mode:
- Printed Document OCR
- Gemini AI

Mistral Vision currently supports image OCR. Traditional ML character recognition and the local deep-learning pipeline are intended for image inputs.

## 4. Project Structure

- `app.py` — Streamlit user interface and routing
- `services/traditional_ocr.py` — Traditional ML character recognition
- `services/printed_ocr.py` — Printed-document OCR
- `services/deep_learning_ocr.py` — TrOCR handwriting recognition
- `services/gemini_ocr.py` — Gemini OCR integration
- `services/mistral_ocr.py` — Mistral Vision integration
- `final_tests/` — Sample inputs for testing the five modes
- `requirements.txt` — Python dependencies
- `../tools/verify_mentor_zip.py` — Mentor ZIP verification utility
- `../tools/test_verify_mentor_zip.py` — Automated tests for the ZIP verifier

## 5. Setup

Use a supported Python environment and run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\final_application\requirements.txt
python -m streamlit run .\final_application\app.py
```

Install the Tesseract OCR executable separately and ensure it is accessible to the application.

Some modes also require locally available model files. Large trained models are not included in Git and may need to be obtained or trained separately.

## 6. API Configuration

Gemini and Mistral modes require API credentials. Configure the following variables in the `.env` file in the location expected by the service code:

```text
GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

Replace the placeholders with your own credentials. Never commit real API keys or `.env` files to GitHub.

API access, usage limits, and billing depend on the provider and account.

## 7. Testing and Evaluation

Run the ZIP verification regression tests from the repository root:

```powershell
python -m unittest tools.test_verify_mentor_zip -v
```

Verify a mentor ZIP with:

```powershell
python .\tools\verify_mentor_zip.py .\OCR_Project_Mentor.zip
```

The verifier checks archive readability, required files, and configured patterns for sensitive or unwanted files.

The Traditional ML evaluation recorded **40 correct predictions out of 62 test characters (64.52%)**. This result applies only to that evaluation set; it is not a standardized comparison of all five OCR modes.

## 8. Limitations

- Traditional ML recognizes individual characters rather than complete documents.
- Recognition quality varies with image quality, handwriting style, and preprocessing.
- PDF support varies by OCR mode.
- API-based OCR requires connectivity and valid provider credentials.
- OCR results should be reviewed for transcription errors.
- A common labeled benchmark and standardized accuracy comparison across all five approaches remain future improvements.

## 9. Demonstration Inputs

The `final_tests/` directory contains example inputs for the Traditional ML, Printed OCR, Deep Learning, Gemini AI, and Mistral Vision modes.

Use these samples to demonstrate the application and inspect the output of each approach.

## Project Status

The integrated Streamlit interface has been manually checked across its five modes in the current development environment. The ZIP verification utility has four passing automated regression tests. A fresh-machine installation and a standardized accuracy comparison across all OCR modes are separate validation tasks.