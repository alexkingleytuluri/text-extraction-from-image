# Final OCR Application

This directory contains the integrated Streamlit application for the OCR project.

The application provides a single interface for testing traditional machine learning, printed-document OCR, deep learning handwriting OCR, and cloud-based OCR approaches.

## OCR Modes

1. Traditional ML - Character Recognition
2. Printed Document OCR
3. Deep Learning - Handwriting OCR
4. Gemini AI
5. Mistral Vision

## Supported Input Formats

The application accepts:

* JPG
* JPEG
* PNG
* WEBP
* PDF

PDF support is available for Printed Document OCR and Gemini AI.
Mistral Vision currently supports image OCR.

Traditional ML character recognition and the local Deep Learning handwriting pipeline are intended for image inputs.

## Application Features

* File upload through the Streamlit interface
* Image preview before processing
* PDF detection and handling
* OCR output displayed in the interface
* Downloadable text output
* Error and warning handling
* Optional dictionary correction for the Deep Learning OCR mode

## Application Structure

* app.py - Streamlit user interface and application routing
* services/traditional\_ocr.py - Traditional ML character recognition
* services/printed\_ocr.py - Printed image and PDF OCR
* services/deep\_learning\_ocr.py - TrOCR handwriting recognition
* services/gemini\_ocr.py - Gemini multimodal OCR
* services/mistral\_ocr.py - Mistral document OCR

## Requirements

Python dependencies are listed in requirements.txt.

The local Tesseract OCR executable must also be installed for the Printed Document OCR service.

## Configuration and Model Files

API-based modes require the appropriate API credentials to be stored in the project .env file.

The .env file must never be committed to Git.

Large trained model files are stored locally and excluded from Git, including the Traditional ML model and fine-tuned Deep Learning model.

## Status

The integrated OCR application is complete and has been tested across the available OCR modes and supported input types.

\## Final Test Files



The `final\_tests/` directory contains the test inputs used to verify the integrated application.



\- `1\_Traditional\_ML/` — single-character test for Traditional ML character recognition

\- `2\_Printed\_OCR/` — printed image and PDF tests

\- `3\_Deep\_Learning/` — handwritten text test for the fine-tuned TrOCR pipeline

\- `4\_Gemini\_AI/` — handwritten prescription test for Gemini OCR

\- `5\_Mistral\_OCR/` — handwritten prescription test for Mistral Vision



These files are demonstration inputs used for final application testing.

### Development Note

Final application testing and validation are in progress.

### Development Note � Day 1
Project documentation and final application testing are being maintained during the examination period.

### Development Note � Day 2
Continued maintaining the OCR project repository and development streak.

### Development Note - Day 3
Reviewed project documentation and maintained the OCR development log.

### Development Note - Day 4
Continued project maintenance and repository updates.

### Development Note - Day 5
Reviewed repository progress and continued maintaining the OCR project.

### Development Note - Day 6
Verified the five OCR application modes and prepared the project for final validation.
