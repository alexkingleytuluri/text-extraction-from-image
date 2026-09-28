
# Final OCR Application

This directory contains the integrated Streamlit application for the OCR project.

The application provides a single interface for testing traditional machine learning, printed-document OCR, deep learning handwriting OCR, and cloud-based OCR approaches.

## OCR Modes

1. Traditional ML - Character Recognition
2. Printed Document OCR
3. Deep Learning - Handwriting OCR
4. Gemini AI
5. Mistral OCR

## Supported Input Formats

The application accepts:

- JPG
- JPEG
- PNG
- WEBP
- PDF

PDF support is available for Printed Document OCR, Gemini AI, and Mistral OCR.

Traditional ML character recognition and the local Deep Learning handwriting pipeline are intended for image inputs.

## Application Features

- File upload through the Streamlit interface
- Image preview before processing
- PDF detection and handling
- OCR output displayed in the interface
- Downloadable text output
- Error and warning handling
- Optional dictionary correction for the Deep Learning OCR mode

## Application Structure

- app.py - Streamlit user interface and application routing
- services/traditional_ocr.py - Traditional ML character recognition
- services/printed_ocr.py - Printed image and PDF OCR
- services/deep_learning_ocr.py - TrOCR handwriting recognition
- services/gemini_ocr.py - Gemini multimodal OCR
- services/mistral_ocr.py - Mistral document OCR

## Requirements

Python dependencies are listed in requirements.txt.

The local Tesseract OCR executable must also be installed for the Printed Document OCR service.

## Configuration and Model Files

API-based modes require the appropriate API credentials to be stored in the project .env file.

The .env file must never be committed to Git.

Large trained model files are stored locally and excluded from Git, including the Traditional ML model and fine-tuned Deep Learning model.

## Status

The integrated OCR application is complete and has been tested across the available OCR modes and supported input types.
