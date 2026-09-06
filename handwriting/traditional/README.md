# Traditional Handwriting OCR

A classical machine learning pipeline for handwritten character recognition using image preprocessing, HOG features, and an RBF-kernel SVM.

## Current Approach

The finalized traditional ML pipeline follows these stages:

1. Image preprocessing
2. Binary image generation
3. Bounding-box cropping
4. Character resizing to a 28 × 28 canvas
5. Center-of-mass alignment
6. Image normalization
7. Data augmentation
8. HOG feature extraction
9. RBF-kernel SVM classification
10. External test evaluation
11. Error and confusion analysis

## Image Preprocessing

The preprocessing pipeline:

- Converts images to grayscale when required
- Applies binary thresholding
- Detects the complete ink region
- Crops the character to its bounding box
- Scales the character while preserving its aspect ratio
- Places it on a 28 × 28 canvas
- Aligns the character using its center of mass

Configuration:

- Threshold: `100`
- Maximum character dimension: `20`
- Canvas size: `28 × 28`
- Center-of-mass alignment: Enabled

## HOG Feature Extraction

Histogram of Oriented Gradients (HOG) is used to represent the shape and edge structure of handwritten characters.

Configuration:

- Orientations: `9`
- Pixels per cell: `4 × 4`
- Cells per block: `2 × 2`
- Block normalization: `L2-Hys`
- Feature vector size: `1296`

## Data Augmentation

The training images are augmented using:

- Original image
- Shift right
- Shift left
- Rotation

This produces **2,480 training samples** from the original **620 training images**.

## Final SVM Configuration

The final classifier uses an RBF-kernel Support Vector Machine.

Configuration:

- Algorithm: SVM
- Kernel: RBF
- C: `50.0`
- Gamma: `0.01`
- Input feature size: `1296`
- Training samples: `2480`

## Final External Evaluation

The final model was evaluated on all **62 external test images**.

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Digits | 3 | 10 | 30.00% |
| Uppercase | 24 | 26 | 92.31% |
| Lowercase | 13 | 26 | 50.00% |
| **Overall** | **40** | **62** | **64.52%** |

### Overall Result

- Correct predictions: **40/62**
- Incorrect predictions: **22/62**
- External accuracy: **64.52%**

## Error Analysis

The final evaluation identified **22 incorrect predictions**.

Errors by category:

- Digits: **7**
- Uppercase: **2**
- Lowercase: **13**

The main confusion patterns involve visually similar handwritten characters, including:

- Digit ↔ letter confusion
- Uppercase ↔ lowercase confusion
- Similar character shapes within the same category

Examples:

```text
1 → l
5 → T
7 → P
8 → e
J → 1
O → o
g → 9
l → 1
u → 4
v → 9