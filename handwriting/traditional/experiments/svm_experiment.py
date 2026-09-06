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


def extract_hog(images):
    features = []

    for image in images:
        feature = hog(
            image,
            orientations=12,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features.append(feature)

    return np.array(features)


print("=" * 60)
print("SVM PARAMETER EXPERIMENT")
print("=" * 60)

# --------------------------------------------------
# 1. Load original images
# --------------------------------------------------

print("\nLoading original dataset...")

images, labels = load_original_images("data_clean")

print("Total original images:", len(images))

# --------------------------------------------------
# 2. Clean train/validation split
# --------------------------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nDataset split:")
print("Training originals:", len(X_train))
print("Validation originals:", len(X_valid))

# --------------------------------------------------
# 3. Augment training data ONLY
# --------------------------------------------------

print("\nAugmenting training data...")

augmented_train = []
augmented_labels = []

for image, label in zip(X_train, y_train):

    for variation in augment_image(image):
        augmented_train.append(variation)
        augmented_labels.append(label)

X_train = np.array(augmented_train)
y_train = np.array(augmented_labels)

print("Training samples after augmentation:", len(X_train))
print("Validation samples:", len(X_valid))

# --------------------------------------------------
# 4. HOG features
# --------------------------------------------------

print("\nExtracting HOG features...")

X_train_hog = extract_hog(X_train)
X_valid_hog = extract_hog(X_valid)

print("HOG feature size:", X_train_hog.shape[1])

# --------------------------------------------------
# 5. SVM parameter combinations
# --------------------------------------------------

C_values = [1, 5, 10, 25, 50, 100, 200]

gamma_values = [
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1
]

results = []

total_experiments = len(C_values) * len(gamma_values)
experiment_number = 0

# --------------------------------------------------
# 6. Run SVM experiments
# --------------------------------------------------

for C in C_values:

    for gamma in gamma_values:

        experiment_number += 1

        print(
            f"\n[{experiment_number}/{total_experiments}] "
            f"C={C}, Gamma={gamma}"
        )

        model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma
        )

        model.fit(X_train_hog, y_train)

        predictions = model.predict(X_valid_hog)

        accuracy = accuracy_score(
            y_valid,
            predictions
        )

        accuracy_percent = accuracy * 100

        print(
            f"Validation Accuracy: "
            f"{accuracy_percent:.2f}%"
        )

        results.append(
            (C, gamma, accuracy_percent)
        )

# --------------------------------------------------
# 7. Sort results
# --------------------------------------------------

results.sort(
    key=lambda result: result[2],
    reverse=True
)

# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("SVM EXPERIMENT RESULTS")
print("=" * 60)

print(
    f"{'Rank':<6}"
    f"{'C':<8}"
    f"{'Gamma':<10}"
    f"{'Accuracy':<12}"
)

print("-" * 36)

for rank, (C, gamma, accuracy) in enumerate(
    results,
    start=1
):

    print(
        f"{rank:<6}"
        f"{C:<8}"
        f"{gamma:<10}"
        f"{accuracy:.2f}%"
    )

# --------------------------------------------------
# 9. Best configuration
# --------------------------------------------------

best_C, best_gamma, best_accuracy = results[0]

print("\n" + "=" * 60)
print("BEST SVM CONFIGURATION")
print("=" * 60)

print("Best C:", best_C)
print("Best Gamma:", best_gamma)
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

print("=" * 60)