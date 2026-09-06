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

        # Convert s-a, s-b, etc. back to a, b, etc.
        if folder_name.startswith("s-"):
            label = folder_name[2:]
        else:
            label = folder_name

        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(folder_path, filename)

            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                continue

            if gray.shape != (28, 28):
                gray = cv2.resize(gray, (28, 28))

            normalized = gray / 255.0

            images.append(normalized)
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


print("=" * 50)
print("CLEAN VALIDATION EXPERIMENT")
print("=" * 50)

# --------------------------------------------------
# 1. Load ORIGINAL images
# --------------------------------------------------

print("\nLoading ORIGINAL images...")

images, labels = load_original_images("data_clean")

print("Total original images:", len(images))

# --------------------------------------------------
# 2. Split ORIGINAL images first
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
# 3. Augment TRAINING images only
# --------------------------------------------------

print("\nAugmenting training images...")

augmented_train = []
augmented_labels = []

for image, label in zip(X_train, y_train):

    variations = augment_image(image)

    for variation in variations:
        augmented_train.append(variation)
        augmented_labels.append(label)

X_train = np.array(augmented_train)
y_train = np.array(augmented_labels)

print("Training samples after augmentation:", len(X_train))
print("Validation samples:", len(X_valid))

# --------------------------------------------------
# 4. Extract HOG features
# --------------------------------------------------

print("\nExtracting HOG features...")

X_train_hog = extract_hog(X_train)
X_valid_hog = extract_hog(X_valid)

print("HOG feature size:", X_train_hog.shape[1])

# --------------------------------------------------
# 5. Train current SVM
# --------------------------------------------------

print("\nTraining SVM...")

model = SVC(
    kernel="rbf",
    C=50.0,
    gamma=0.01
)

model.fit(X_train_hog, y_train)

# --------------------------------------------------
# 6. Evaluate validation set
# --------------------------------------------------

predictions = model.predict(X_valid_hog)

accuracy = accuracy_score(y_valid, predictions)

print("\n" + "=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)

print("Validation samples:", len(y_valid))
print("Correct predictions:", np.sum(predictions == y_valid))
print("Incorrect predictions:", np.sum(predictions != y_valid))

print(f"Validation Accuracy: {accuracy * 100:.2f}%")