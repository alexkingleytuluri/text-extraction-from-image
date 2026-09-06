import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from skimage.feature import hog


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


def augment_image(image, strategy):
    """
    Apply one of several augmentation strategies.

    Strategy 1:
        Original only

    Strategy 2:
        Current augmentation
        Original + shift ±2 + rotation +10

    Strategy 3:
        Mild augmentation
        Original + shift ±1 + rotation ±5

    Strategy 4:
        Moderate augmentation
        Original + shift ±2 + rotation ±5 + small scaling
    """

    images = [image]

    if strategy == 1:
        return images

    if strategy == 2:

        # Shift right
        M_right = np.float32([
            [1, 0, 2],
            [0, 1, 0]
        ])

        # Shift left
        M_left = np.float32([
            [1, 0, -2],
            [0, 1, 0]
        ])

        images.append(
            cv2.warpAffine(image, M_right, (28, 28))
        )

        images.append(
            cv2.warpAffine(image, M_left, (28, 28))
        )

        # Rotate +10 degrees
        M_rot = cv2.getRotationMatrix2D(
            (14, 14),
            10,
            1
        )

        images.append(
            cv2.warpAffine(image, M_rot, (28, 28))
        )

    elif strategy == 3:

        # Shift right +1
        M_right = np.float32([
            [1, 0, 1],
            [0, 1, 0]
        ])

        # Shift left -1
        M_left = np.float32([
            [1, 0, -1],
            [0, 1, 0]
        ])

        images.append(
            cv2.warpAffine(image, M_right, (28, 28))
        )

        images.append(
            cv2.warpAffine(image, M_left, (28, 28))
        )

        # Rotate +5 degrees
        M_rot_plus = cv2.getRotationMatrix2D(
            (14, 14),
            5,
            1
        )

        # Rotate -5 degrees
        M_rot_minus = cv2.getRotationMatrix2D(
            (14, 14),
            -5,
            1
        )

        images.append(
            cv2.warpAffine(image, M_rot_plus, (28, 28))
        )

        images.append(
            cv2.warpAffine(image, M_rot_minus, (28, 28))
        )

    elif strategy == 4:

        # Shift right
        M_right = np.float32([
            [1, 0, 2],
            [0, 1, 0]
        ])

        # Shift left
        M_left = np.float32([
            [1, 0, -2],
            [0, 1, 0]
        ])

        images.append(
            cv2.warpAffine(image, M_right, (28, 28))
        )

        images.append(
            cv2.warpAffine(image, M_left, (28, 28))
        )

        # Rotate +5
        M_rot_plus = cv2.getRotationMatrix2D(
            (14, 14),
            5,
            1
        )

        # Rotate -5
        M_rot_minus = cv2.getRotationMatrix2D(
            (14, 14),
            -5,
            1
        )

        images.append(
            cv2.warpAffine(image, M_rot_plus, (28, 28))
        )

        images.append(
            cv2.warpAffine(image, M_rot_minus, (28, 28))
        )

        # Small scale increase
        enlarged = cv2.resize(
            image,
            None,
            fx=1.08,
            fy=1.08,
            interpolation=cv2.INTER_LINEAR
        )

        h, w = enlarged.shape

        start_y = max((h - 28) // 2, 0)
        start_x = max((w - 28) // 2, 0)

        scaled = enlarged[
            start_y:start_y + 28,
            start_x:start_x + 28
        ]

        if scaled.shape == (28, 28):
            images.append(scaled)

    return images


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
print("AUGMENTATION EXPERIMENT")
print("=" * 60)

# --------------------------------------------------
# Load ORIGINAL images
# --------------------------------------------------

print("\nLoading original dataset...")

images, labels = load_original_images("data_clean")

print("Total original images:", len(images))

# --------------------------------------------------
# Clean split
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
# Experiment definitions
# --------------------------------------------------

experiments = [
    ("No Augmentation", 1),
    ("Current Augmentation", 2),
    ("Mild Augmentation", 3),
    ("Moderate Augmentation", 4)
]

results = []

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

for name, strategy in experiments:

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    augmented_train = []
    augmented_labels = []

    for image, label in zip(X_train, y_train):

        variations = augment_image(
            image,
            strategy
        )

        for variation in variations:

            augmented_train.append(variation)
            augmented_labels.append(label)

    augmented_train = np.array(augmented_train)
    augmented_labels = np.array(augmented_labels)

    print(
        "Training samples:",
        len(augmented_train)
    )

    print(
        "Validation samples:",
        len(X_valid)
    )

    # --------------------------------------------------
    # HOG
    # --------------------------------------------------

    print("Extracting HOG...")

    X_train_hog = extract_hog(
        augmented_train
    )

    X_valid_hog = extract_hog(
        X_valid
    )

    print(
        "HOG feature size:",
        X_train_hog.shape[1]
    )

    # --------------------------------------------------
    # SVM
    # --------------------------------------------------

    print("Training SVM...")

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_train_hog,
        augmented_labels
    )

    predictions = model.predict(
        X_valid_hog
    )

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
        (
            name,
            len(augmented_train),
            accuracy_percent
        )
    )

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("AUGMENTATION EXPERIMENT SUMMARY")
print("=" * 60)

print(
    f"{'Strategy':<25}"
    f"{'Train Samples':<16}"
    f"{'Accuracy':<12}"
)

print("-" * 53)

for name, samples, accuracy in results:

    print(
        f"{name:<25}"
        f"{samples:<16}"
        f"{accuracy:.2f}%"
    )

print("=" * 60)

# --------------------------------------------------
# Best strategy
# --------------------------------------------------

best = max(
    results,
    key=lambda result: result[2]
)

print("\nBEST AUGMENTATION STRATEGY")
print("--------------------------")
print("Strategy:", best[0])
print("Training samples:", best[1])
print(f"Validation Accuracy: {best[2]:.2f}%")
print("=" * 60)