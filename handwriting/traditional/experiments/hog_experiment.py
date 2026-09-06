import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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


def extract_hog(images, orientations, pixels_per_cell, cells_per_block):
    features = []

    for image in images:
        feature = hog(
            image,
            orientations=orientations,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            block_norm="L2-Hys"
        )

        features.append(feature)

    return np.array(features)


print("=" * 60)
print("HOG FEATURE EXPERIMENT")
print("=" * 60)

# --------------------------------------------------
# Load original images
# --------------------------------------------------

print("\nLoading original dataset...")

images, labels = load_original_images("data_clean")

print("Total original images:", len(images))

# --------------------------------------------------
# Same clean split
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

print("Augmenting training data...")

augmented_train = []
augmented_labels = []

for image, label in zip(X_train, y_train):

    for variation in augment_image(image):
        augmented_train.append(variation)
        augmented_labels.append(label)

X_train = np.array(augmented_train)
y_train = np.array(augmented_labels)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_valid))

# --------------------------------------------------
# HOG configurations
# --------------------------------------------------

experiments = [
    ("1x1 Blocks", 12, 4, 1),
    ("2x2 Blocks", 12, 4, 2),
    ("3x3 Blocks", 12, 4, 3)
]

results = []

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

for name, orientations, pixels_per_cell, cells_per_block in experiments:

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(
        f"Orientations: {orientations} | "
        f"Pixels/Cell: {pixels_per_cell}x{pixels_per_cell} | "
        f"Block: {cells_per_block}x{cells_per_block}"
    )

    print("Extracting HOG...")

    X_train_hog = extract_hog(
        X_train,
        orientations,
        pixels_per_cell,
        cells_per_block
    )

    X_valid_hog = extract_hog(
        X_valid,
        orientations,
        pixels_per_cell,
        cells_per_block
    )

    print("Feature size:", X_train_hog.shape[1])

    print("Training SVM...")

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(X_train_hog, y_train)

    predictions = model.predict(X_valid_hog)

    accuracy = accuracy_score(y_valid, predictions)

    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    results.append(
        (name, X_train_hog.shape[1], accuracy * 100)
    )

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("HOG EXPERIMENT SUMMARY")
print("=" * 60)

for name, feature_size, accuracy in results:

    print(
        f"{name:<20} "
        f"Features: {feature_size:<5} "
        f"Accuracy: {accuracy:.2f}%"
    )

print("=" * 60)