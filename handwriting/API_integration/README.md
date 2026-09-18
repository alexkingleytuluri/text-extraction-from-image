# API Integration

This directory contains the API and external OCR implementations used during the handwriting OCR project.

## Current Components

- Gemini API - multimodal handwriting OCR
- OCR.space - experimental implementation
- Tesseract OCR - local OCR baseline

## Gemini API

The Gemini integration uses the Google Gen AI Python SDK to perform multimodal handwriting transcription from prescription and handwritten images.

The API key is loaded securely from the project-level .env file.

The .env file is excluded from Git and must never be committed.

### Gemini Implementation

The Gemini script includes:

- Secure API key loading
- Portable image paths
- Image loading and transmission
- Handwriting transcription prompt
- API response handling

### Gemini Test Result

The Gemini API was successfully tested on the available handwritten prescription image.

The model returned a readable transcription containing clinical notes, laboratory values, and prescription information.

The result demonstrates that the Gemini integration can process the tested handwritten prescription image.

## Tesseract OCR

Tesseract was tested as a local OCR baseline.

### Configuration

- Tesseract version: 5.5.0
- Python library: pytesseract
- Image processing: OpenCV
- Page segmentation mode: PSM 6

### Test Results

Tesseract successfully recognized the available printed-text test image.

However, when tested on the handwritten prescription image, Tesseract returned an empty transcription.

This provides a useful baseline for comparison with the multimodal Gemini approach.

## OCR.space

OCR.space is currently kept as an experimental implementation.

No OCR.space API key is currently configured in the project environment, so the implementation has not been tested as part of the current API integration stage.

Any future OCR.space API key should be stored in .env and must not be hard-coded or committed to Git.

## Current Status

- Gemini API: Tested successfully
- Tesseract OCR: Tested as local baseline
- OCR.space: Not yet tested
- Final API integration: Pending

## Planned Work

1. Complete OCR.space evaluation if an appropriate API key is available
2. Compare the tested OCR approaches
3. Integrate the selected approach into the final application
