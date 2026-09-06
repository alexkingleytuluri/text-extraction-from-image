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

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    if w == 0 or h == 0:
        return None

    cropped = binary[y:y + h, x:x + w]

    max_dim = max(w, h)

    scale = 20.0 / max_dim

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2

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

    # Center of mass alignment
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
# LOAD DATASET
# ============================================================

def load_dataset_images(data_folder, threshold):

    images = []
    labels = []

    for folder_name in sorted(
        os.listdir(data_folder)
    ):

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

        for filename in sorted(
            os.listdir(folder_path)
        ):

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            path = os.path.join(
                folder_path,
                filename
            )

            img = cv2.imread(path)

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
# AUGMENTATION
# ============================================================

def augment_image(image):

    images = [image]

    # Shift right
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

    # Shift left
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

    # Rotate +10 degrees
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
# HOG
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
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    images,
    labels,
    random_state
):

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            images,
            labels,
            test_size=0.2,
            random_state=random_state,
            stratify=labels
        )
    )

    # Augment training data only
    augmented_train = []
    augmented_labels = []

    for image, label in zip(
        X_train,
        y_train
    ):

        for variation in augment_image(image):

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

    # HOG
    X_train_hog = extract_hog(
        X_train
    )

    X_valid_hog = extract_hog(
        X_valid
    )

    # SVM
    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_train_hog,
        y_train
    )

    predictions = model.predict(
        X_valid_hog
    )

    return accuracy_score(
        y_valid,
        predictions
    ) * 100


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("MULTI-SPLIT PREPROCESSING ROBUSTNESS EXPERIMENT")
print("=" * 65)

thresholds = [100, 140]

random_states = [
    42,
    7,
    21,
    99,
    123
]

all_results = {}


for threshold in thresholds:

    print("\n" + "-" * 65)
    print(f"THRESHOLD = {threshold}")
    print("-" * 65)

    print("Processing dataset...")

    images, labels = load_dataset_images(
        "data",
        threshold
    )

    print(
        "Processed images:",
        len(images)
    )

    if len(images) != 620:

        print(
            "WARNING: Dataset is incomplete."
        )

        print(
            "Skipping this threshold."
        )

        continue

    threshold_results = []

    for random_state in random_states:

        print(
            f"\nRandom state: {random_state}"
        )

        accuracy = run_experiment(
            images,
            labels,
            random_state
        )

        threshold_results.append(
            accuracy
        )

        print(
            f"Accuracy: {accuracy:.2f}%"
        )

    all_results[threshold] = (
        threshold_results
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL ROBUSTNESS SUMMARY")
print("=" * 65)

print(
    f"{'Threshold':<15}"
    f"{'Split 42':<12}"
    f"{'Split 7':<12}"
    f"{'Split 21':<12}"
    f"{'Split 99':<12}"
    f"{'Split 123':<12}"
    f"{'Average':<12}"
)

print("-" * 75)

for threshold, results in all_results.items():

    average = np.mean(results)

    print(
        f"{threshold:<15}"
        f"{results[0]:<12.2f}"
        f"{results[1]:<12.2f}"
        f"{results[2]:<12.2f}"
        f"{results[3]:<12.2f}"
        f"{results[4]:<12.2f}"
        f"{average:<12.2f}"
    )

print("=" * 65)


# ============================================================
# BEST THRESHOLD
# ============================================================

if all_results:

    averages = {
        threshold: np.mean(results)
        for threshold, results
        in all_results.items()
    }

    best_threshold = max(
        averages,
        key=averages.get
    )

    print(
        "\nMost robust threshold:",
        best_threshold
    )

    print(
        f"Average validation accuracy: "
        f"{averages[best_threshold]:.2f}%"
    )

print("=" * 65)