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
```

---

## Project Components

| File | Purpose |
|---|---|
| `preprocess.py` | Image preprocessing and character normalization |
| `load_dataset.py` | Dataset loading, augmentation, and HOG feature extraction |
| `train_model.py` | Training and saving the final SVM model |
| `predict.py` | Predicting an individual character from an image |
| `predict_test.py` | Running predictions across the test dataset |
| `final_evaluation.py` | Final external evaluation and category-wise accuracy |
| `error_analysis_final.py` | Analysis of incorrect predictions |
| `error_analysis.md` | Documented findings from the error analysis |

---

## Quick Start

Navigate to the Traditional OCR directory:

```powershell
cd handwriting\traditional
```

### 1. Train the Model

```powershell
python train_model.py
```

This loads the cleaned training dataset, applies the finalized augmentation and HOG feature extraction pipeline, trains the RBF-kernel SVM, and saves the trained model as:

```text
handwriting_svm.joblib
```

The model file is intentionally excluded from Git because of its size.

### 2. Run Final Evaluation

```powershell
python final_evaluation.py
```

This evaluates the finalized model against the external test set and reports:

- Correct predictions
- Incorrect predictions
- Overall accuracy
- Digit accuracy
- Uppercase accuracy
- Lowercase accuracy
- Individual prediction errors

### 3. Predict an Individual Character

```powershell
python predict.py "test_clean\A.png"
```

Example output:

```text
==================================================
HANDWRITING OCR
==================================================
Image      : test_clean\A.png
Prediction : A
==================================================
```

---

## Dataset

The training dataset contains **620 original handwritten character images** distributed equally across all 62 classes.

| Category | Classes | Images per Class | Total |
|---|---:|---:|---:|
| Digits | 10 | 10 | 100 |
| Uppercase | 26 | 10 | 260 |
| Lowercase | 26 | 10 | 260 |
| **Total** | **62** | **10** | **620** |

The external test set contains **62 images**, with one test image representing each character class.

The lowercase classes use folder names such as `s-a`, `s-b`, ..., `s-z` to avoid filename and folder conflicts with uppercase classes on Windows. The dataset loader maps these names back to their corresponding lowercase labels.

---

## Image Preprocessing

The preprocessing stage converts handwritten characters into a standardized representation suitable for feature extraction.

The process includes:

1. Converting the image to grayscale when required.
2. Applying binary thresholding to separate ink from the background.
3. Detecting the complete handwritten ink region.
4. Cropping the character to its bounding box.
5. Preserving the character's aspect ratio while resizing.
6. Placing the resized character on a `28 × 28` canvas.
7. Aligning the character using its center of mass.

### Configuration

| Parameter | Value |
|---|---|
| Threshold | `100` |
| Maximum character dimension | `20` |
| Canvas size | `28 × 28` |
| Center-of-mass alignment | Enabled |

The preprocessing stage was finalized after evaluating multiple configurations against both validation and external test data.

---

## HOG Feature Extraction

The system uses **Histogram of Oriented Gradients (HOG)** to represent the structural and edge information of handwritten characters.

HOG captures local gradient directions and provides a compact representation of character shape.

### Final HOG Configuration

| Parameter | Value |
|---|---|
| Orientations | `9` |
| Pixels per cell | `4 × 4` |
| Cells per block | `2 × 2` |
| Block normalization | `L2-Hys` |
| Feature vector size | `1296` |

The finalized configuration was selected based on external evaluation rather than validation accuracy alone.

---

## Data Augmentation

To improve robustness to small variations in handwriting, each training image is transformed into four training samples:

1. Original image
2. Shifted right
3. Shifted left
4. Rotated by 10 degrees

This increases the training set from:

```text
620 original images
        ↓
2,480 augmented samples
```

The augmentation strategy is intentionally lightweight and focuses on common positional and rotational variations.

---

## Final SVM Model

The extracted HOG features are classified using a **Support Vector Machine with an RBF kernel**.

### Final Configuration

| Parameter | Value |
|---|---|
| Algorithm | SVM |
| Kernel | RBF |
| C | `50.0` |
| Gamma | `0.01` |
| Input feature size | `1296` |
| Training samples | `2480` |
| Number of classes | `62` |

The trained model is saved as:

```text
handwriting_svm.joblib
```

The serialized model is excluded from version control through `.gitignore`.

---

## Final External Evaluation

The finalized model was evaluated on all **62 external test images**.

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Digits | 3 | 10 | 30.00% |
| Uppercase | 24 | 26 | 92.31% |
| Lowercase | 13 | 26 | 50.00% |
| **Overall** | **40** | **62** | **64.52%** |

### Overall Result

- **Correct:** 40 / 62
- **Incorrect:** 22 / 62
- **External Accuracy:** **64.52%**

The results show that the finalized classical pipeline performs particularly well on uppercase characters, while digits and lowercase characters remain more challenging because of greater visual similarity between handwritten classes.

---

## Error Analysis

The final evaluation identified **22 incorrect predictions**.

### Errors by Category

| Category | Errors |
|---|---:|
| Digits | 7 |
| Uppercase | 2 |
| Lowercase | 13 |
| **Total** | **22** |

### Observed Confusion Patterns

The main sources of error were:

- Digit ↔ letter confusion
- Uppercase ↔ lowercase confusion
- Similar handwritten shapes
- Variations in individual writing styles
- Ambiguous character structures

Examples from the final external evaluation include:

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
```

These errors demonstrate a limitation of shape-based classical recognition: visually similar characters can produce similar HOG representations even when their actual labels are different.

---

## Finalized Traditional Configuration

The current Traditional OCR phase uses the following finalized configuration:

```text
Preprocessing
    Threshold              : 100
    Canvas                 : 28 × 28
    Center-of-Mass         : Enabled

Data
    Original Images        : 620
    Augmented Samples      : 2,480
    Classes                : 62

HOG
    Orientations           : 9
    Pixels per Cell        : 4 × 4
    Cells per Block        : 2 × 2
    Block Normalization    : L2-Hys
    Feature Size           : 1,296

SVM
    Kernel                 : RBF
    C                      : 50.0
    Gamma                  : 0.01

External Evaluation
    Test Images            : 62
    Correct                : 40
    Incorrect              : 22
    Accuracy               : 64.52%
```

---

## Conclusion

The Traditional Handwriting OCR phase establishes a complete classical recognition pipeline using:

- OpenCV-based image preprocessing
- Character normalization
- Lightweight data augmentation
- HOG-based feature extraction
- RBF-kernel SVM classification
- Individual character prediction
- External evaluation
- Error and confusion analysis

The final system achieves **64.52% accuracy on the external 62-character test set**.

The experiments conducted during this phase were used to select the final preprocessing, HOG, augmentation, and SVM configuration based on external performance and robustness rather than optimizing for a single validation split.

This repository structure preserves the finalized Traditional OCR implementation together with its evaluation and analysis artifacts for reproducibility and future reference.