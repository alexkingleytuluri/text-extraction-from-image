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

## Final Evaluation

The final Traditional OCR model was evaluated on 62 independent handwritten character samples.

- Correct predictions: 40 / 62
- Overall accuracy: 64.52%
- Digits: 3 / 10 (30.00%)
- Uppercase: 24 / 26 (92.31%)
- Lowercase: 13 / 26 (50.00%)

## Development Status

The Traditional OCR stage is complete and integrated as the character-recognition component of the final OCR application.

## Experimentation

The project investigated threshold variations, HOG configurations, data augmentation settings, SVM hyperparameters, feature fusion, shape features, center-of-mass alignment, aspect-ratio analysis, class balancing, quality analysis, prototype approaches, PCA, hierarchical approaches, margin correction, pairwise specialists, and class weighting.

The final configuration was selected after comparing these approaches against the independent external test set.

## Important Limitation

This is a single-character recognition model, not a complete handwriting document OCR engine.

The SVM receives one normalized character image and predicts one of the 62 character classes.

Whole-document handwriting recognition requires additional stages such as line detection, word segmentation, character segmentation, recognition, and text reconstruction.

The final application therefore labels this component as Traditional ML - Character Recognition.

## Final Status

- Dataset preparation: Completed
- Preprocessing: Completed
- HOG feature extraction: Completed
- Data augmentation: Completed
- SVM training: Completed
- External evaluation: Completed
- Error analysis: Completed
- Traditional OCR stage: Completed
- Integration into final application: Completed
