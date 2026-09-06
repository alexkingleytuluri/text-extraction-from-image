import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from skimage.feature import hog


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ============================================================
# PREPROCESSING
# ============================================================

def process_image(img, threshold):
    """
    Preprocess one image using the selected threshold.

    Steps:
    1. Convert to grayscale
    2. Threshold
    3. Find ink bounding box
    4. Crop
    5. Resize to maximum dimension 20
    6. Place on 28x28 canvas
    7. Center using center of mass
    """

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Threshold
    _, binary = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Find all ink pixels
    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    # Bounding box
    x, y, w, h = cv2.boundingRect(coords)

    if w == 0 or h == 0:
        return None

    cropped = binary[y:y + h, x:x + w]

    # Resize while maintaining aspect ratio
    max_dim = max(w, h)

    if max_dim == 0:
        return None

    scale = 20.0 / max_dim

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2

    # Safety check
    if (
        x_off < 0
        or y_off < 0
        or x_off + new_w > 28
        or y_off + new_h > 28
    ):
        return None

    canvas[
        y_off:y_off + new_h,
        x_off:x_off + new_w
    ] = resized

    # ========================================================
    # CENTER OF MASS ALIGNMENT
    # ========================================================

    cy, cx = np.where(canvas > 0)

    if len(cx) > 0 and len(cy) > 0:

        shift_x = int(
            np.round(14 - np.mean(cx))
        )

        shift_y = int(
            np.round(14 - np.mean(cy))
        )

        M = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        canvas = cv2.warpAffine(
            canvas,
            M,
            (28, 28)
        )

    return canvas / 255.0


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

def load_original_images(data_folder, threshold):

    images = []
    labels = []

    for folder_name in sorted(os.listdir(data_folder)):

        folder_path = os.path.join(
            data_folder,
            folder_name
        )

        if not os.path.isdir(folder_path):
            continue

        # Convert s-a -> a
        if folder_name.startswith("s-"):
            label = folder_name[2:]
        else:
            label = folder_name

        for filename in sorted(
            os.listdir(folder_path)
        ):

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            image_path = os.path.join(
                folder_path,
                filename
            )

            img = cv2.imread(image_path)

            if img is None:
                continue

            processed = process_image(
                img,
                threshold
            )

            if processed is None:
                continue

            images.append(processed)
            labels.append(label)

    return (
        np.array(images),
        np.array(labels)
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_image(image):

    images = [image]

    # --------------------------------------------------------
    # Shift Right
    # --------------------------------------------------------

    M_right = np.float32([
        [1, 0, 2],
        [0, 1, 0]
    ])

    images.append(
        cv2.warpAffine(
            image,
            M_right,
            (28, 28)
        )
    )

    # --------------------------------------------------------
    # Shift Left
    # --------------------------------------------------------

    M_left = np.float32([
        [1, 0, -2],
        [0, 1, 0]
    ])

    images.append(
        cv2.warpAffine(
            image,
            M_left,
            (28, 28)
        )
    )

    # --------------------------------------------------------
    # Rotate +10 degrees
    # --------------------------------------------------------

    M_rot = cv2.getRotationMatrix2D(
        (14, 14),
        10,
        1
    )

    images.append(
        cv2.warpAffine(
            image,
            M_rot,
            (28, 28)
        )
    )

    return images


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

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


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("=" * 60)
print("PREPROCESSING THRESHOLD EXPERIMENT")
print("=" * 60)

# Thresholds we want to compare
thresholds = [
    70,
    85,
    100,
    120,
    140
]

results = []


# ============================================================
# TEST EACH THRESHOLD
# ============================================================

for threshold in thresholds:

    print("\n" + "-" * 60)
    print(f"THRESHOLD = {threshold}")
    print("-" * 60)

    # --------------------------------------------------------
    # Process original images
    # --------------------------------------------------------

    print("\nProcessing original images...")

    images, labels = load_original_images(
        "data",
        threshold
    )

    print(
        "Processed images:",
        len(images)
    )

    # --------------------------------------------------------
    # Check dataset integrity
    # --------------------------------------------------------

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    class_counts = dict(
        zip(unique_labels, counts)
    )

    affected_classes = {
        label: count
        for label, count in class_counts.items()
        if count != 10
    }

    # We expect exactly 620 images:
    # 62 classes × 10 images
    if (
        len(images) != 620
        or len(affected_classes) > 0
    ):

        print(
            "\nWARNING: This threshold changed "
            "the dataset!"
        )

        print(
            "Expected images: 620"
        )

        print(
            "Actual images:",
            len(images)
        )

        if affected_classes:

            print(
                "\nAffected classes:"
            )

            for label, count in sorted(
                affected_classes.items()
            ):

                print(
                    f"  {label}: {count} images"
                )

        print(
            "\nSkipping this threshold "
            "because the dataset is incomplete."
        )

        results.append(
            (
                threshold,
                len(images),
                None
            )
        )

        continue

    print(
        "Dataset integrity: OK "
        "(620 images, 10 per class)"
    )

    # --------------------------------------------------------
    # Clean train/validation split
    # --------------------------------------------------------

    print(
        "\nSplitting original images..."
    )

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            images,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    )

    print(
        "Training originals:",
        len(X_train)
    )

    print(
        "Validation originals:",
        len(X_valid)
    )

    # --------------------------------------------------------
    # Augment TRAINING data only
    # --------------------------------------------------------

    print(
        "\nApplying augmentation "
        "to training data..."
    )

    augmented_train = []
    augmented_labels = []

    for image, label in zip(
        X_train,
        y_train
    ):

        variations = augment_image(
            image
        )

        for variation in variations:

            augmented_train.append(
                variation
            )

            augmented_labels.append(
                label
            )

    X_train = np.array(
        augmented_train
    )

    y_train = np.array(
        augmented_labels
    )

    print(
        "Training after augmentation:",
        len(X_train)
    )

    # --------------------------------------------------------
    # HOG
    # --------------------------------------------------------

    print("\nExtracting HOG features...")

    X_train_hog = extract_hog(
        X_train
    )

    X_valid_hog = extract_hog(
        X_valid
    )

    print(
        "HOG feature size:",
        X_train_hog.shape[1]
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    print(
        "\nTraining SVM..."
    )

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_train_hog,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "Running validation..."
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
        f"\nValidation Accuracy: "
        f"{accuracy_percent:.2f}%"
    )

    results.append(
        (
            threshold,
            len(images),
            accuracy_percent
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD EXPERIMENT SUMMARY")
print("=" * 60)

print(
    f"{'Threshold':<12}"
    f"{'Images':<12}"
    f"{'Accuracy':<12}"
)

print("-" * 36)

for (
    threshold,
    image_count,
    accuracy
) in results:

    if accuracy is None:

        print(
            f"{threshold:<12}"
            f"{image_count:<12}"
            f"{'SKIPPED':<12}"
        )

    else:

        print(
            f"{threshold:<12}"
            f"{image_count:<12}"
            f"{accuracy:.2f}%"
        )

print("=" * 60)


# ============================================================
# FIND BEST VALID THRESHOLD
# ============================================================

valid_results = [
    result
    for result in results
    if result[2] is not None
]


if valid_results:

    best = max(
        valid_results,
        key=lambda result: result[2]
    )

    print("\nBEST THRESHOLD")
    print("-" * 20)

    print(
        "Threshold:",
        best[0]
    )

    print(
        "Processed images:",
        best[1]
    )

    print(
        f"Validation Accuracy: "
        f"{best[2]:.2f}%"
    )

else:

    print(
        "\nNo valid thresholds "
        "preserved all 620 images."
    )

print("=" * 60)