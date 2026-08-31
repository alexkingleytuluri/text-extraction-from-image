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

The current HOG + SVM model was evaluated on a 62-character test set.

- Total test samples: **62**
- Correct predictions: **40**
- Incorrect predictions: **22**
- Test accuracy: **64.52%**

The current model uses data augmentation and tuned SVM
hyperparameters.

Best parameters found:

- C: **50**
- Gamma: **0.01**

This result serves as the current baseline for future improvements.

## Prediction Results

The current model correctly recognizes most uppercase letters, while
some handwritten digits and visually similar characters are misclassified.

Current incorrect predictions:

- 0 → C
- 1 → F
- 2 → 0
- 4 → Q
- 5 → T
- 6 → 4
- D → O
- J → 2
- P → R

These results will be used later to identify commonly confused
characters and improve the model.

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

## Target Character Set

The long-term target for the current classical ML version is recognition of **62 characters**:

- 10 digits: `0–9`
- 26 uppercase letters: `A–Z`
- 26 lowercase letters: `a–z`

The training dataset currently contains all 62 character classes.

The lowercase test set is still being prepared. Once it is ready, the complete 62-character test set will be used to evaluate the model.

## Current Limitations

- The lowercase test dataset is not yet available.
- The current 75% test accuracy is based only on 36 test images.
- Some visually similar characters are still being confused.
- The current model is a baseline using HOG features and an SVM classifier.
- Model improvements will be evaluated after the complete 62-character test set is available.

## Next Steps

- Add lowercase test images
- Evaluate all 62 character classes
- Analyze commonly confused characters
- Improve model performance
- Add support for additional symbols
- Explore more advanced approaches after completing the classical ML version