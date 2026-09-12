# Traditional Handwriting OCR

A classical machine learning pipeline for handwritten character recognition using image preprocessing, Histogram of Oriented Gradients (HOG) features, and an RBF-kernel Support Vector Machine (SVM).

The system is designed to recognize **62 handwritten character classes**:

- Digits: `0–9`
- Uppercase letters: `A–Z`
- Lowercase letters: `a–z`

The Traditional OCR phase focuses on a complete classical computer-vision and machine-learning pipeline, from image preprocessing and feature extraction to model training, prediction, evaluation, and error analysis.

---

## Pipeline Overview

The finalized Traditional OCR pipeline follows these stages:

```text
Input Image
     ↓
Grayscale Conversion
     ↓
Binary Thresholding
     ↓
Ink Region Detection
     ↓
Bounding-Box Cropping
     ↓
Aspect-Ratio Preserving Resize
     ↓
28 × 28 Canvas
     ↓
Center-of-Mass Alignment
     ↓
Normalization
     ↓
Data Augmentation
     ↓
HOG Feature Extraction
     ↓
RBF-SVM Classification
     ↓
Character Prediction
     ↓
External Evaluation
     ↓
Error Analysis
> The prediction script can also be executed from the repository root using the full project-relative path.

## Repository Status

The Traditional Handwriting OCR pipeline has been finalized with its preprocessing, HOG feature extraction, SVM training, prediction, evaluation, and error-analysis components.
