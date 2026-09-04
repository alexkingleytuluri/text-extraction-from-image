# Handwriting OCR — Error Analysis

## Current Baseline

The current classical machine learning pipeline uses:

- Image preprocessing with OpenCV
- 28×28 normalized character images
- HOG feature extraction
- RBF-kernel SVM classifier
- C = 50
- Gamma = 0.01

The fixed external test set contains 62 images covering:

- 10 digits
- 26 uppercase letters
- 26 lowercase letters

The training dataset contains 620 images, with exactly 10 images for each of the 62 classes.

## Dataset Balance

A dataset distribution check confirmed that all 62 classes contain exactly 10 training images.

Therefore, class imbalance is not currently considered a major cause of the classification errors.

## External Test Results

- Total test images: 62
- Correct predictions: 40
- Incorrect predictions: 22
- Overall accuracy: **64.52%**

### Category Accuracy

| Category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Digits | 3 | 10 | 30.00% |
| Uppercase | 24 | 26 | 92.31% |
| Lowercase | 13 | 26 | 50.00% |

The model performs strongly on uppercase characters but struggles significantly with digits and lowercase characters.

## Misclassified Characters

The current 22 errors are:

| Actual | Predicted |
|---|---|
| 0 | 3 |
| 1 | l |
| 2 | 9 |
| 5 | T |
| 6 | 4 |
| 7 | P |
| 8 | e |
| J | 1 |
| O | o |
| a | u |
| c | e |
| e | R |
| f | c |
| g | 9 |
| h | r |
| j | f |
| k | B |
| l | 1 |
| s | o |
| t | k |
| u | 4 |
| v | 9 |

## Observations

No single confusion occurs repeatedly; each of the 22 incorrect predictions occurs once.

The errors mainly involve visually similar characters across different classes, particularly:

- Digits vs letters
- Uppercase vs lowercase characters
- Similar-shaped lowercase characters
- Characters with similar stroke structures

Examples include:

- `1 → l`
- `5 → T`
- `6 → 4`
- `8 → e`
- `O → o`
- `a → u`
- `e → R`
- `g → 9`
- `k → B`
- `l → 1`
- `u → 4`
- `v → 9`

The error pattern suggests that the main limitation is character shape similarity rather than class imbalance.

## Validation Methodology Note

An earlier internal validation experiment reported 92.7% accuracy.

However, that experiment augmented the complete dataset before performing the train-validation split. This can allow augmented versions of the same original image to appear in both training and validation sets, producing an overly optimistic validation score.

Therefore, the **64.52% fixed external test result is currently the more reliable baseline**.

Future experiments will split the original images first and apply augmentation only to the training portion.

## Next Steps

The next phase will use a clean validation methodology and systematically evaluate:

1. Preprocessing variations
2. HOG feature configurations
3. SVM parameters
4. Augmentation strategies
5. Validation errors

After selecting the best classical ML pipeline, the final model will be evaluated again on the untouched 62-image external test set.