# Handwritten Text Recognition

This directory contains the development of a handwritten character recognition system.

## Current Phase

- Dataset preparation
- Image preprocessing
- HOG feature extraction
- SVM model training
- Character prediction and evaluation

The project is currently under active development.

## Dataset

The current training dataset includes:

- Digits: 0–9
- Uppercase letters: A–Z
- Lowercase letters: a–z
- Multiple handwriting styles per character

Total training images: **620**

The current test dataset includes:

- Digits: 0–9
- Uppercase letters: A–Z

Total test images: **36**

Lowercase test images are still being prepared.

## Project Structure

handwriting/
├── data/             # Training images
├── data_clean/       # Preprocessed training images
├── test/             # Test images
├── test_clean/       # Preprocessed test images
├── preprocess.py     # Image preprocessing
├── load_dataset.py   # Dataset loading and HOG extraction
├── train_model.py    # SVM model training
├── predict_test.py   # Test image prediction
└── README.md         # Project documentation

## Current Approach

The current version uses a classical machine learning pipeline:

1. Image preprocessing
2. Image normalization
3. HOG feature extraction
4. SVM classification
5. Test prediction
6. Accuracy evaluation

## Feature Extraction

The current model uses Histogram of Oriented Gradients (HOG) to convert
handwritten character images into numerical feature vectors.

The images are resized to **28 × 28 pixels** and normalized before HOG
features are extracted.

Current HOG configuration:

- Orientations: 9
- Pixels per cell: 4 × 4
- Cells per block: 2 × 2
- Block normalization: L2-Hys
- Feature vector size: **1296**

## Current Baseline

Using the current 36-image test set:

- Correct predictions: **27**
- Incorrect predictions: **9**
- Test accuracy: **75.00%**

## Model Training

The current version uses a Support Vector Machine (SVM) classifier
with an RBF kernel.

The SVM is trained using the HOG feature vectors extracted from
the cleaned handwriting dataset.

Current training configuration:

- Algorithm: SVM
- Kernel: RBF
- C: 5.0
- Gamma: scale
- Input: 1296-dimensional HOG feature vectors

The current training run reports an accuracy of **63.9%**.

## Next Steps

- Add lowercase test images
- Evaluate all 62 character classes
- Analyze commonly confused characters
- Improve model performance
- Add support for additional symbols
- Explore more advanced approaches after completing the classical ML version