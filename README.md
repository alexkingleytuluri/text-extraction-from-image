# Handwriting Text Recognition

A handwriting recognition project developed in multiple stages, progressing from classical computer vision and machine learning toward more advanced recognition approaches and final integration.

## Project Status

### Traditional OCR — Completed ✅

The Traditional OCR phase has been completed and documented.

The finalized classical pipeline includes:

- Image preprocessing
- Thresholding and ink-region detection
- Bounding-box cropping
- 28×28 normalization
- Center-of-mass alignment
- Data augmentation
- HOG feature extraction
- RBF-SVM classification
- Character prediction
- External evaluation
- Error analysis

The system recognizes:

- Digits: `0–9`
- Uppercase letters: `A–Z`
- Lowercase letters: `a–z`

**Final external evaluation: 40/62 correct (64.52%)**

Detailed Traditional OCR documentation is available in:

`handwriting/traditional/README.md`

## Project Roadmap

```text
Traditional OCR
      ↓
Deep Learning OCR
      ↓
Model Evaluation & Refinement
      ↓
API Integration
      ↓
Final OCR Application