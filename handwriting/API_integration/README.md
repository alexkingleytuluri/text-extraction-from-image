
# API Integration

This directory contains the external and local OCR implementations developed during the API integration stage of the project.

## OCR Components

- Gemini API - multimodal handwriting OCR
- OCR.space - cloud OCR implementation
- Tesseract OCR - local OCR baseline
- Mistral OCR - cloud document OCR

## Gemini API

The Gemini integration uses the Google Gen AI Python SDK to perform multimodal handwriting transcription from handwritten images and documents.

The API key is loaded securely from the project-level .env file. The .env file is excluded from Git and must never be committed.

### Implementation

The Gemini script includes:

- Secure API key loading
- Portable image paths
- Image loading and transmission
- Handwriting transcription prompt
- API response handling

### Gemini Test Result

The Gemini API was successfully tested on the available handwritten prescription image.

The model returned a readable transcription containing clinical notes, laboratory values, and prescription information.

## OCR.space

OCR.space is implemented as a cloud OCR option using the OCRSPACE_API_KEY environment variable.

The implementation uses Base64 image data, OCR Engine 2, scaling support, and error handling.

No API key is stored in the repository. If OCR.space is used, the key must be placed in .env and must never be hard-coded or committed to Git.

The current project documentation does not claim a formal OCR.space accuracy evaluation.

## Tesseract OCR

Tesseract was tested as a local OCR baseline.

Configuration:

- Tesseract version: 5.5.0
- Python library: pytesseract
- Image processing: OpenCV
- Page segmentation mode: PSM 6

Tesseract successfully recognized the available printed-text test image. When tested on the handwritten prescription image, it returned an empty transcription.

## Mistral OCR

Mistral OCR was added as a cloud document OCR option using the Mistral AI API.

The implementation supports image and PDF inputs and loads the API key securely from .env.

Image and PDF integration tests were completed successfully.

## Final Status

- Gemini API: Tested successfully
- OCR.space: Secure implementation available; no formal accuracy evaluation documented
- Tesseract OCR: Tested as local baseline
- Mistral OCR: Image and PDF integration tested successfully
- API integration stage: Complete

The completed OCR project integrates the API-based approaches with the final Streamlit application.

## Security

API credentials must be stored in .env and excluded from Git. Never commit API keys, tokens, or other secrets.
