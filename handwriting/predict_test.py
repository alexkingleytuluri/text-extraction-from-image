import os
import cv2
import numpy as np
from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog

# Train on ALL data
X, y = load_dataset("data_clean")

model = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True)
model.fit(X, y)

test_folder = "test_clean"

print("Predicting Test Images (SVM + HOG)...\n")

files = sorted(
    os.listdir(test_folder),
    key=lambda x: (not x[0].isdigit(), x)
)

correct = 0
total = 0

for filename in files:
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    true_label = os.path.splitext(filename)[0]
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

    print(
        f"File: {filename} | "
        f"True: {true_label} | "
        f"Predicted: {prediction}"
    )

    total += 1

    if prediction == true_label:
        correct += 1

accuracy = (correct / total) * 100

print("\n--- Evaluation Results ---")
print("Total test samples:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)
print(f"Test Accuracy: {accuracy:.2f}%")