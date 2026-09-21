# Handwriting Text Recognition

A multi-stage OCR project that explores handwriting and document text recognition using Traditional Machine Learning, Deep Learning, cloud OCR APIs, and a final integrated Streamlit application.

The project was developed progressively from character-level classical OCR to a multi-model OCR system capable of processing images and PDF documents.

---

## Project Overview

The project consists of four major stages:

Traditional OCR
      |
      v
Deep Learning OCR
      |
      v
API Integration
      |
      v
Final Integrated OCR Application

---

## 1. Traditional OCR - Completed

The Traditional OCR stage focuses on handwritten character recognition using classical computer vision and machine learning techniques.

### Pipeline

Input Character Image
        |
        v
Image Preprocessing
        |
        v
Thresholding
        |
        v
Ink-region Detection
        |
        v
Bounding-box Cropping
        |
        v
28x28 Normalization
        |
        v
Center-of-mass Alignment
        |
        v
Data Augmentation
        |
        v
HOG Feature Extraction
        |
        v
RBF-SVM Classification
        |
        v
Character Prediction

### Recognized Classes

- Digits: 0-9
- Uppercase letters: A-Z
- Lowercase letters: a-z

Total classes: 62

### Final Model

- Feature extraction: HOG
- Orientations: 9
- Pixels per cell: (4, 4)
- Cells per block: (2, 2)
- Block normalization: L2-Hys
- Feature size: 1296
- Classifier: RBF-SVM
- C = 50.0
- gamma = 0.01

### Final External Evaluation

40 / 62 correct

Accuracy: 64.52%

Detailed documentation:

handwriting/traditional/README.md

---

## 2. Deep Learning OCR - Completed

The Deep Learning stage explores handwritten word and line recognition using TrOCR.

### Work Completed

- Zero-shot TrOCR baseline
- Custom handwritten dataset preparation
- TrOCR fine-tuning
- Single-word recognition
- Full-page handwriting recognition
- Prescription-image testing
- Dictionary-based output correction
- OCR preprocessing and segmentation experiments

### Fine-tuning Dataset

The custom dataset contains:

- 20 handwritten text labels
- 5 examples per label
- 100 total handwritten training images

The dataset contains medicine names, units, and prescription-related text.

### Model

The fine-tuned TrOCR model is stored locally because of its large size and is excluded from Git.

The final application loads the local model from:

handwriting/deep_learning/finetuned_trocr/

Detailed documentation:

handwriting/deep_learning/README.md

---

## 3. API Integration - Completed

Multiple OCR approaches were integrated and tested during this stage.

### Gemini AI

Google Gemini was integrated for multimodal OCR of handwritten and document images.

The integration:

- Loads the API key from .env
- Supports image input
- Supports PDF input
- Preserves line breaks where possible
- Preserves numbers, units, and punctuation

### OCR.space

OCR.space was integrated as an external OCR API.

The integration includes:

- Secure API-key loading through .env
- Base64 image transmission
- OCR Engine 2
- Scaling
- Error handling

### Tesseract OCR

Tesseract was tested as a local OCR baseline for printed or typed documents.

### Mistral OCR

Mistral OCR was integrated and tested for:

- Image OCR
- PDF OCR

API keys are stored locally in .env and are not committed to Git.

Detailed documentation:

handwriting/API_integration/README.md

---

## 4. Final Integrated OCR Application - Completed

The final application combines multiple OCR approaches into a single Streamlit interface.

Location:

final_application/

### Available OCR Methods

1. Traditional ML - Character Recognition
   - HOG + RBF-SVM
   - Individual handwritten character recognition

2. Printed Document OCR
   - Tesseract OCR
   - Printed or typed images and PDFs

3. Deep Learning - Handwriting OCR
   - Fine-tuned TrOCR
   - Handwritten words and lines
   - Optional dictionary correction

4. Gemini AI
   - Multimodal OCR
   - Image and PDF support

5. Mistral OCR
   - Image and PDF OCR through the Mistral OCR API

### Supported Upload Formats

- JPG
- JPEG
- PNG
- WEBP
- PDF

### Application Features

- Simple Streamlit interface
- OCR method selection
- Image preview
- PDF upload
- OCR progress indicators
- Error handling
- Extracted text display
- Download OCR result as .txt

---

## Project Structure

OCR/
|
+-- README.md
+-- .gitignore
|
+-- experiments/
|
+-- final_application/
|   +-- app.py
|   +-- requirements.txt
|   +-- services/
|       +-- traditional_ocr.py
|       +-- printed_ocr.py
|       +-- deep_learning_ocr.py
|       +-- gemini_ocr.py
|       +-- mistral_ocr.py
|
+-- handwriting/
    |
    +-- traditional/
    |   +-- data/
    |   +-- data_clean/
    |   +-- test/
    |   +-- test_clean/
    |   +-- preprocess.py
    |   +-- final_evaluation.py
    |   +-- handwriting_svm.joblib
    |   +-- requirements.txt
    |   +-- README.md
    |
    +-- deep_learning/
    |   +-- finetune_data/
    |   +-- finetuned_trocr/
    |   +-- experiments/
    |   +-- 1_zero_shot_baseline.py
    |   +-- 2_train_finetune.py
    |   +-- 3_predict_single_word.py
    |   +-- 4_full_page_showcase.py
    |   +-- 5_prescription_test.py
    |   +-- requirements.txt
    |   +-- README.md
    |
    +-- API_integration/
        +-- 9_cloud_api_ocr.py
        +-- 11_gemini_api.py
        +-- 12_ocrspace_improved.py
        +-- 13_tesseract_ocr.py
        +-- requirements.txt
        +-- README.md

---

## Installation

### 1. Clone the repository

git clone https://github.com/alexkingleytuluri/text-extraction-from-image.git

cd text-extraction-from-image

### 2. Install final application dependencies

pip install -r final_application/requirements.txt

### 3. Install Tesseract OCR

The Printed Document OCR module uses the Tesseract OCR engine.

The current Windows implementation expects:

C:\Program Files\Tesseract-OCR\tesseract.exe

If Tesseract is installed elsewhere, update the path in:

final_application/services/printed_ocr.py

### 4. Configure API keys

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
OCRSPACE_API_KEY=your_ocrspace_api_key

Do not commit .env to Git.

---

## Running the Final Application

From the project root:

streamlit run final_application/app.py

The Streamlit interface will open in the browser.

Upload an image or PDF, select an OCR method, and click Run OCR.

The extracted text can then be copied or downloaded as a text file.

---

## Important Model Files

The trained models are intentionally excluded from Git because of their size.

### Traditional ML

handwriting/traditional/handwriting_svm.joblib

### Deep Learning

handwriting/deep_learning/finetuned_trocr/

These local model files are required for the corresponding methods in the final application.

---

## Evaluation Summary

| Stage | Approach | Status |
|---|---|---|
| Traditional OCR | HOG + RBF-SVM | Completed |
| Deep Learning OCR | Fine-tuned TrOCR | Completed |
| Gemini | Multimodal API OCR | Completed |
| OCR.space | Cloud OCR API | Completed |
| Tesseract | Local printed OCR | Completed |
| Mistral OCR | Image + PDF OCR API | Completed |
| Final Application | Streamlit multi-model OCR | Completed |

Traditional OCR external evaluation:

40 / 62 = 64.52%

The different OCR approaches were developed for different recognition tasks, so their results are not directly interchangeable as a single accuracy metric.

---

## Security

API credentials are stored in .env and excluded from Git.

The repository does not contain active API keys.

Trained model files and large generated model artifacts are also excluded from Git.

---

## Future Improvements

Possible future work includes:

- Improved handwritten character recognition
- Larger and more diverse handwriting datasets
- Better word and line segmentation
- More robust prescription recognition
- Additional OCR APIs
- Model comparison on standardized datasets
- Improved PDF processing
- Better confidence estimation
- Support for more document types
- Further optimization of the final application

---

## Author

Alex Kingley

GitHub:
https://github.com/alexkingleytuluri/text-extraction-from-image
