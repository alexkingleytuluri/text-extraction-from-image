import os
import cv2
import numpy as np

from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog


# 1. Train the model
print("Training SVM model...")

X, y = load_dataset("data_clean")

model = SVC(kernel="rbf", C=50.0, gamma=0.01)
model.fit(X, y)


# 2. Evaluate test images
test_folder = "test_clean"

errors = []
correct = 0
total = 0

files = sorted(os.listdir(test_folder))


for filename in files:

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    true_label = os.path.splitext(filename)[0]

    # Convert s-a -> a
    if true_label.startswith("s-"):
        true_label = true_label[2:]

    path = os.path.join(test_folder, filename)

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        continue

    if gray.shape != (28, 28):
        gray = cv2.resize(gray, (28, 28))

    normalized = gray / 255.0

    features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    prediction = model.predict(features.reshape(1, -1))[0]

    if str(prediction).startswith("s-"):
        prediction = str(prediction)[2:]

    total += 1

    if prediction == true_label:
        correct += 1
    else:
        errors.append((true_label, prediction))


# 3. Print results
print("\n==============================")
print("ERROR ANALYSIS")
print("==============================")

print(f"Total test images: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {len(errors)}")
print(f"Accuracy: {correct / total * 100:.2f}%")

print("\n--- Misclassified Characters ---")

for true_label, prediction in errors:
    print(f"{true_label} -> {prediction}")


# 4. Count repeated confusions
confusions = {}

for true_label, prediction in errors:
    pair = (true_label, prediction)
    confusions[pair] = confusions.get(pair, 0) + 1


print("\n--- Most Common Confusions ---")

sorted_confusions = sorted(
    confusions.items(),
    key=lambda item: item[1],
    reverse=True
)

for (true_label, prediction), count in sorted_confusions:
    print(f"{true_label} -> {prediction}: {count}")