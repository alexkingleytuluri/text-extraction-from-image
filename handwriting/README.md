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

## Current Approach

The current version uses a classical machine learning pipeline:

1. Image preprocessing
2. Image normalization
3. HOG feature extraction
4. SVM classification
5. Test prediction
6. Accuracy evaluation

## Current Baseline

Using the current 36-image test set:

- Correct predictions: **27**
- Incorrect predictions: **9**
- Test accuracy: **75.00%**

## Next Steps

- Add lowercase test images
- Evaluate all 62 character classes
- Analyze commonly confused characters
- Improve model performance