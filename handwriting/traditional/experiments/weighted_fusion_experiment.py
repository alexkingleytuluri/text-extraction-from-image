import os
import cv2
import numpy as np

from sklearn.svm import SVC
from skimage.feature import hog


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def augment_image(img):
    images = [img]

    # Shift Right
    M_right = np.float32([
        [1, 0, 2],
        [0, 1, 0]
    ])
    images.append(cv2.warpAffine(img, M_right, (28, 28)))

    # Shift Left
    M_left = np.float32([
        [1, 0, -2],
        [0, 1, 0]
    ])
    images.append(cv2.warpAffine(img, M_left, (28, 28)))

    # Rotate 10 degrees
    M_rot = cv2.getRotationMatrix2D(
        (14, 14),
        10,
        1
    )
    images.append(cv2.warpAffine(img, M_rot, (28, 28)))

    return images


def load_images(data_folder):
    images = []
    labels = []

    for folder_name in sorted(os.listdir(data_folder)):
        folder_path = os.path.join(
            data_folder,
            folder_name
        )

        if not os.path.isdir(folder_path):
            continue

        if folder_name.startswith("s-"):
            label = folder_name[2:]
        else:
            label = folder_name

        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            path = os.path.join(
                folder_path,
                filename
            )

            gray = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if gray is None:
                continue

            if gray.shape != (28, 28):
                gray = cv2.resize(
                    gray,
                    (28, 28)
                )

            images.append(gray / 255.0)
            labels.append(label)

    return np.array(images), np.array(labels)


def extract_hog(images):
    features = []

    for image in images:
        feature = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features.append(feature)

    return np.array(features)


def extract_combined_features(images, raw_weight):
    hog_features = extract_hog(images)

    raw_pixels = images.reshape(
        len(images),
        -1
    )

    weighted_raw = raw_pixels * raw_weight

    combined = np.concatenate(
        [hog_features, weighted_raw],
        axis=1
    )

    return combined


def create_augmented_dataset(images, labels):
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        variations = augment_image(image)

        for variation in variations:
            augmented_images.append(variation)
            augmented_labels.append(label)

    return (
        np.array(augmented_images),
        np.array(augmented_labels)
    )


def evaluate(predictions, labels):
    correct = np.sum(predictions == labels)
    total = len(labels)

    print("\n--- Evaluation Results ---")
    print("Total test samples:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)
    print(
        f"Test Accuracy: "
        f"{correct / total * 100:.2f}%"
    )

    categories = [
        ("Digits", str.isdigit),
        ("Uppercase", str.isupper),
        ("Lowercase", str.islower)
    ]

    print("\n--- Category Accuracy ---")

    for name, check_function in categories:
        indices = [
            i
            for i, label in enumerate(labels)
            if check_function(label)
        ]

        category_correct = sum(
            predictions[i] == labels[i]
            for i in indices
        )

        category_total = len(indices)

        print(
            f"{name}: "
            f"{category_correct}/{category_total} "
            f"({category_correct / category_total * 100:.2f}%)"
        )


print("=" * 65)
print("WEIGHTED HOG + RAW PIXEL EXPERIMENT")
print("=" * 65)

print("\nLoading training images...")

train_images, train_labels = load_images(
    "data_clean"
)

print(
    "Original training images:",
    len(train_images)
)

print("\nApplying augmentation...")

train_images, train_labels = create_augmented_dataset(
    train_images,
    train_labels
)

print(
    "Training images after augmentation:",
    len(train_images)
)

print("\nLoading external test images...")

test_images = []
test_labels = []
test_filenames = []

for filename in sorted(
    os.listdir("test_clean"),
    key=lambda x: (not x[0].isdigit(), x)
):

    if not filename.lower().endswith(IMAGE_EXTENSIONS):
        continue

    path = os.path.join(
        "test_clean",
        filename
    )

    gray = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        continue

    if gray.shape != (28, 28):
        gray = cv2.resize(
            gray,
            (28, 28)
        )

    label = os.path.splitext(filename)[0]

    if label.startswith("s-"):
        label = label[2:]

    test_images.append(gray / 255.0)
    test_labels.append(label)
    test_filenames.append(filename)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(
    "External test images:",
    len(test_images)
)


# ------------------------------------------------------------
# Test different raw-pixel weights
# ------------------------------------------------------------

weights = [
    0.25,
    0.50,
    0.75
]

results = {}

for raw_weight in weights:

    print("\n" + "-" * 65)
    print(
        f"RAW PIXEL WEIGHT = {raw_weight}"
    )
    print("-" * 65)

    print("Extracting training features...")

    X_train = extract_combined_features(
        train_images,
        raw_weight
    )

    print(
        "Training feature size:",
        X_train.shape[1]
    )

    print("Extracting test features...")

    X_test = extract_combined_features(
        test_images,
        raw_weight
    )

    print(
        "Test feature size:",
        X_test.shape[1]
    )

    print("Training SVM...")

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_train,
        train_labels
    )

    predictions = model.predict(X_test)

    correct = np.sum(
        predictions == test_labels
    )

    accuracy = (
        correct / len(test_labels)
    ) * 100

    results[raw_weight] = accuracy

    print(
        f"Accuracy: {accuracy:.2f}%"
    )


print("\n" + "=" * 65)
print("WEIGHTED FUSION SUMMARY")
print("=" * 65)

print(
    f"{'Raw Weight':<15}"
    f"{'Accuracy':<15}"
)

print("-" * 30)

for weight, accuracy in results.items():
    print(
        f"{weight:<15}"
        f"{accuracy:.2f}%"
    )

best_weight = max(
    results,
    key=results.get
)

print("\nBest raw-pixel weight:", best_weight)
print(
    f"Best accuracy: "
    f"{results[best_weight]:.2f}%"
)

print("\nBaseline HOG-only accuracy: 64.52%")

print("=" * 65)