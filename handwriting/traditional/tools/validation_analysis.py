import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from skimage.feature import hog
from load_dataset import augment_image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_original_images(data_folder):
    images = []
    labels = []

    for folder_name in sorted(os.listdir(data_folder)):
        folder_path = os.path.join(data_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        if folder_name.startswith("s-"):
            label = folder_name[2:]
        else:
            label = folder_name

        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            path = os.path.join(folder_path, filename)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                continue

            if gray.shape != (28, 28):
                gray = cv2.resize(gray, (28, 28))

            images.append(gray / 255.0)
            labels.append(label)

    return np.array(images), np.array(labels)


def extract_hog(images):
    features = []

    for image in images:
        features.append(
            hog(
                image,
                orientations=9,
                pixels_per_cell=(4, 4),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )
        )

    return np.array(features)


print("=" * 60)
print("CLEAN VALIDATION ERROR ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# Load original dataset
# --------------------------------------------------

images, labels = load_original_images("data_clean")

# --------------------------------------------------
# Same split as train_model_valid.py
# --------------------------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# --------------------------------------------------
# Augment training data ONLY
# --------------------------------------------------

augmented_train = []
augmented_labels = []

for image, label in zip(X_train, y_train):
    for variation in augment_image(image):
        augmented_train.append(variation)
        augmented_labels.append(label)

X_train = np.array(augmented_train)
y_train = np.array(augmented_labels)

# --------------------------------------------------
# HOG
# --------------------------------------------------

X_train_hog = extract_hog(X_train)
X_valid_hog = extract_hog(X_valid)

# --------------------------------------------------
# Train current baseline SVM
# --------------------------------------------------

model = SVC(
    kernel="rbf",
    C=50.0,
    gamma=0.01
)

model.fit(X_train_hog, y_train)

predictions = model.predict(X_valid_hog)

# --------------------------------------------------
# Overall accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_valid, predictions)

print("\nOverall Accuracy")
print("----------------")
print(f"{accuracy * 100:.2f}%")
print(f"Correct: {np.sum(predictions == y_valid)} / {len(y_valid)}")

# --------------------------------------------------
# Category accuracy
# --------------------------------------------------

categories = {
    "Digits": set("0123456789"),
    "Uppercase": set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "Lowercase": set("abcdefghijklmnopqrstuvwxyz")
}

print("\nCategory Accuracy")
print("-----------------")

for category, class_set in categories.items():

    indices = [
        i for i, label in enumerate(y_valid)
        if label in class_set
    ]

    correct = sum(
        predictions[i] == y_valid[i]
        for i in indices
    )

    total = len(indices)

    print(
        f"{category}: "
        f"{correct}/{total} "
        f"({correct / total * 100:.2f}%)"
    )

# --------------------------------------------------
# Misclassified characters
# --------------------------------------------------

print("\nMisclassified Characters")
print("------------------------")

errors = []

for actual, predicted in zip(y_valid, predictions):

    if actual != predicted:
        errors.append((actual, predicted))

for actual, predicted in errors:
    print(f"{actual} -> {predicted}")

# --------------------------------------------------
# Confusion counts
# --------------------------------------------------

print("\nMost Common Confusions")
print("----------------------")

confusions = {}

for actual, predicted in errors:

    pair = (actual, predicted)

    confusions[pair] = confusions.get(pair, 0) + 1

for (actual, predicted), count in sorted(
    confusions.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{actual} -> {predicted}: {count}")

# --------------------------------------------------
# Per-class accuracy
# --------------------------------------------------

print("\nPer-Class Accuracy")
print("------------------")

classes = sorted(set(labels))

for character in classes:

    indices = [
        i for i, label in enumerate(y_valid)
        if label == character
    ]

    if not indices:
        continue

    correct = sum(
        predictions[i] == y_valid[i]
        for i in indices
    )

    print(
        f"{character}: "
        f"{correct}/{len(indices)} "
        f"({correct / len(indices) * 100:.0f}%)"
    )

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)