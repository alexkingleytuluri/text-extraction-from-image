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

            normalized = gray / 255.0

            images.append(normalized)
            labels.append(label)

    return np.array(images), np.array(labels)


def extract_features(images):
    features = []

    for image in images:

        # HOG features
        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        # Raw 28x28 pixel features
        raw_pixels = image.flatten()

        # Combine HOG + raw pixels
        combined = np.concatenate([
            hog_features,
            raw_pixels
        ])

        features.append(combined)

    return np.array(features)


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


def evaluate_category(
    predictions,
    labels,
    category
):
    total = 0
    correct = 0

    for prediction, true_label in zip(
        predictions,
        labels
    ):
        if category == "digit" and true_label.isdigit():
            total += 1
            if prediction == true_label:
                correct += 1

        elif category == "uppercase" and true_label.isupper():
            total += 1
            if prediction == true_label:
                correct += 1

        elif category == "lowercase" and true_label.islower():
            total += 1
            if prediction == true_label:
                correct += 1

    if total > 0:
        return correct, total

    return 0, 0


print("=" * 65)
print("HOG + RAW PIXEL FEATURE FUSION EXPERIMENT")
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

print("\nExtracting HOG + raw pixel features...")

X_train = extract_features(
    train_images
)

print(
    "Feature vector size:",
    X_train.shape[1]
)

print("\nTraining SVM...")

model = SVC(
    kernel="rbf",
    C=50.0,
    gamma=0.01
)

model.fit(
    X_train,
    train_labels
)

print("Training complete.")

print("\nLoading external test images...")

test_images = []
test_labels = []

for filename in sorted(
    os.listdir("test_clean"),
    key=lambda x: (not x[0].isdigit(), x)
):

    if not filename.lower().endswith(
        IMAGE_EXTENSIONS
    ):
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

    normalized = gray / 255.0

    true_label = os.path.splitext(
        filename
    )[0]

    if true_label.startswith("s-"):
        true_label = true_label[2:]

    test_images.append(normalized)
    test_labels.append(true_label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(
    "External test images:",
    len(test_images)
)

print("\nExtracting test features...")

X_test = extract_features(
    test_images
)

print(
    "Test feature vector size:",
    X_test.shape[1]
)

print("\nPredicting...\n")

predictions = model.predict(X_test)

correct = 0

for filename, true_label, prediction in zip(
    sorted(
        os.listdir("test_clean"),
        key=lambda x: (not x[0].isdigit(), x)
    ),
    test_labels,
    predictions
):

    if not filename.lower().endswith(
        IMAGE_EXTENSIONS
    ):
        continue

    print(
        f"File: {filename} | "
        f"True: {true_label} | "
        f"Predicted: {prediction}"
    )

    if prediction == true_label:
        correct += 1

total = len(test_labels)

print("\n--- Evaluation Results ---")

print(
    "Total test samples:",
    total
)

print(
    "Correct predictions:",
    correct
)

print(
    "Incorrect predictions:",
    total - correct
)

print(
    f"Test Accuracy: "
    f"{correct / total * 100:.2f}%"
)

print("\n--- Category Accuracy ---")

for category in [
    "digit",
    "uppercase",
    "lowercase"
]:

    correct_category, total_category = evaluate_category(
        predictions,
        test_labels,
        category
    )

    if total_category > 0:
        print(
            f"{category.capitalize()}: "
            f"{correct_category}/{total_category} "
            f"({correct_category / total_category * 100:.2f}%)"
        )

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETE")
print("=" * 65)