import os
import cv2
import numpy as np
from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog

# 1. Train on ALL augmented data
print("Training SVM model on augmented data...")
X, y = load_dataset("data_clean")

# Updated C=50 to match the tuned model, removed probability=True
model = SVC(kernel="rbf", C=50.0, gamma=0.01)
model.fit(X, y)

test_folder = "test_clean"

print("Predicting Test Images (SVM + HOG)...\n")

files = sorted(
    os.listdir(test_folder),
    key=lambda x: (not x[0].isdigit(), x)
)

correct = 0
total = 0

digit_correct = 0
digit_total = 0

uppercase_correct = 0
uppercase_total = 0

lowercase_correct = 0
lowercase_total = 0

for filename in files:
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    true_label = os.path.splitext(filename)[0]
    if true_label.startswith("s-"):
        true_label = true_label[2:]
        
    path = os.path.join(test_folder, filename)

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        continue

    if gray.shape != (28, 28):
        gray = cv2.resize(gray, (28, 28))

    normalized = gray / 255.0

    # Extract HOG features for the test image
    features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    flat = features.reshape(1, -1)

    prediction = model.predict(flat)[0]
    if str(prediction).startswith("s-"):
        prediction = str(prediction)[2:]

    print(
        f"File: {filename} | "
        f"True: {true_label} | "
        f"Predicted: {prediction}"
    )

    total += 1

    if prediction == true_label:
        correct += 1

# Track accuracy by character category
    if true_label.isdigit():
        digit_total += 1
        if prediction == true_label:
            digit_correct += 1

    elif true_label.isupper():
        uppercase_total += 1
        if prediction == true_label:
            uppercase_correct += 1

    elif true_label.islower():
        lowercase_total += 1
        if prediction == true_label:
            lowercase_correct += 1

accuracy = (correct / total) * 100

print("\n--- Evaluation Results ---")
print("Total test samples:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)
print(f"Test Accuracy: {accuracy:.2f}%")

print("\n--- Category Accuracy ---")

if digit_total > 0:
    print(f"Digits: {digit_correct}/{digit_total} "
          f"({digit_correct / digit_total * 100:.2f}%)")

if uppercase_total > 0:
    print(f"Uppercase: {uppercase_correct}/{uppercase_total} "
          f"({uppercase_correct / uppercase_total * 100:.2f}%)")

if lowercase_total > 0:
    print(f"Lowercase: {lowercase_correct}/{lowercase_total} "
          f"({lowercase_correct / lowercase_total * 100:.2f}%)")