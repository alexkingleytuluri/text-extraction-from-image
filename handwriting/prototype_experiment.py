import os
import cv2
import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from skimage.feature import hog


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def process_image(img):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(
        gray,
        100,
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

    canvas[
        y_off:y_off + new_h,
        x_off:x_off + new_w
    ] = resized

    # Center-of-mass alignment
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


def extract_hog(img):

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


def load_training_data():

    images = []
    labels = []
    filenames = []

    for folder_name in sorted(
        os.listdir("data")
    ):

        folder_path = os.path.join(
            "data",
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

            processed = process_image(img)

            if processed is None:
                continue

            images.append(processed)
            labels.append(label)
            filenames.append(
                os.path.join(
                    folder_name,
                    filename
                )
            )

    return (
        np.array(images),
        np.array(labels),
        filenames
    )


def augment_image(img):

    images = [img]

    # Shift right
    M_right = np.float32([
        [1, 0, 2],
        [0, 1, 0]
    ])

    images.append(
        cv2.warpAffine(
            img,
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
            img,
            M_left,
            (28, 28)
        )
    )

    # Rotate +10
    M_rot = cv2.getRotationMatrix2D(
        (14, 14),
        10,
        1
    )

    images.append(
        cv2.warpAffine(
            img,
            M_rot,
            (28, 28)
        )
    )

    return images


def create_augmented_dataset(
    images,
    labels
):

    augmented_images = []
    augmented_labels = []

    for image, label in zip(
        images,
        labels
    ):

        for variation in augment_image(
            image
        ):

            augmented_images.append(
                variation
            )

            augmented_labels.append(
                label
            )

    return (
        np.array(augmented_images),
        np.array(augmented_labels)
    )


def load_external_test():

    images = []
    labels = []
    filenames = []

    for filename in sorted(
        os.listdir("test"),
        key=lambda x: (
            not x[0].isdigit(),
            x
        )
    ):

        if not filename.lower().endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        path = os.path.join(
            "test",
            filename
        )

        img = cv2.imread(path)

        if img is None:
            continue

        processed = process_image(img)

        if processed is None:
            continue

        label = os.path.splitext(
            filename
        )[0]

        if label.startswith("s-"):
            label = label[2:]

        images.append(processed)
        labels.append(label)
        filenames.append(filename)

    return (
        np.array(images),
        np.array(labels),
        filenames
    )


# ==================================================
# SVM
# ==================================================

def run_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    filenames
):

    print("\nTraining SVM...")

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = (
        np.mean(predictions == y_test)
        * 100
    )

    print(
        f"SVM accuracy: {accuracy:.2f}%"
    )

    return accuracy, predictions


# ==================================================
# PROTOTYPE CLASSIFIER
# ==================================================

def build_prototypes(
    X,
    y
):

    prototypes = {}
    
    for label in sorted(
        np.unique(y)
    ):

        class_features = X[
            y == label
        ]

        prototypes[label] = np.mean(
            class_features,
            axis=0
        )

    return prototypes


def predict_prototype(
    X,
    prototypes
):

    labels = sorted(
        prototypes.keys()
    )

    prototype_matrix = np.array([
        prototypes[label]
        for label in labels
    ])

    distances = pairwise_distances(
        X,
        prototype_matrix,
        metric="euclidean"
    )

    nearest = np.argmin(
        distances,
        axis=1
    )

    return np.array([
        labels[index]
        for index in nearest
    ])


# ==================================================
# MEDOID CLASSIFIER
# ==================================================

def build_medoids(
    X,
    y
):

    medoids = {}

    for label in sorted(
        np.unique(y)
    ):

        class_indices = np.where(
            y == label
        )[0]

        class_features = X[
            class_indices
        ]

        class_distances = pairwise_distances(
            class_features,
            class_features,
            metric="euclidean"
        )

        mean_distances = np.mean(
            class_distances,
            axis=1
        )

        medoid_position = np.argmin(
            mean_distances
        )

        medoid_index = class_indices[
            medoid_position
        ]

        medoids[label] = X[
            medoid_index
        ]

    return medoids


def predict_medoid(
    X,
    medoids
):

    labels = sorted(
        medoids.keys()
    )

    medoid_matrix = np.array([
        medoids[label]
        for label in labels
    ])

    distances = pairwise_distances(
        X,
        medoid_matrix,
        metric="euclidean"
    )

    nearest = np.argmin(
        distances,
        axis=1
    )

    return np.array([
        labels[index]
        for index in nearest
    ])


# ==================================================
# MAIN
# ==================================================

print("=" * 70)
print("PROTOTYPE / MEDOID CLASSIFICATION EXPERIMENT")
print("=" * 70)


print("\nLoading training data...")

train_images, train_labels, train_filenames = (
    load_training_data()
)

print(
    "Original training images:",
    len(train_images)
)


print("\nApplying current augmentation...")

aug_images, aug_labels = (
    create_augmented_dataset(
        train_images,
        train_labels
    )
)

print(
    "Training samples after augmentation:",
    len(aug_images)
)


print("\nExtracting HOG features...")

X_aug = np.array([
    extract_hog(image)
    for image in aug_images
])

print(
    "Training feature size:",
    X_aug.shape[1]
)


print("\nLoading external test...")

test_images, y_test, filenames = (
    load_external_test()
)

X_test = np.array([
    extract_hog(image)
    for image in test_images
])

print(
    "External test images:",
    len(X_test)
)


# ==================================================
# 1. SVM BASELINE
# ==================================================

svm_accuracy, svm_predictions = run_svm(
    X_aug,
    aug_labels,
    X_test,
    y_test,
    filenames
)


# ==================================================
# 2. PROTOTYPES
# ==================================================

print("\nBuilding class prototypes...")

prototypes = build_prototypes(
    X_aug,
    aug_labels
)

prototype_predictions = predict_prototype(
    X_test,
    prototypes
)

prototype_accuracy = (
    np.mean(
        prototype_predictions == y_test
    )
    * 100
)

print(
    f"Prototype accuracy: "
    f"{prototype_accuracy:.2f}%"
)


# ==================================================
# 3. MEDOIDS
# ==================================================

print("\nBuilding class medoids...")

medoids = build_medoids(
    X_aug,
    aug_labels
)

medoid_predictions = predict_medoid(
    X_test,
    medoids
)

medoid_accuracy = (
    np.mean(
        medoid_predictions == y_test
    )
    * 100
)

print(
    f"Medoid accuracy: "
    f"{medoid_accuracy:.2f}%"
)


# ==================================================
# COMPARISON
# ==================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"SVM:          {svm_accuracy:.2f}%"
)

print(
    f"Prototype:    {prototype_accuracy:.2f}%"
)

print(
    f"Medoid:       {medoid_accuracy:.2f}%"
)


# ==================================================
# MISCLASSIFICATION COUNTS
# ==================================================

print("\n" + "=" * 70)
print("ERROR COUNTS")
print("=" * 70)

print(
    "SVM errors:",
    np.sum(svm_predictions != y_test)
)

print(
    "Prototype errors:",
    np.sum(
        prototype_predictions != y_test
    )
)

print(
    "Medoid errors:",
    np.sum(
        medoid_predictions != y_test
    )
)


# ==================================================
# PROTOTYPE ERRORS
# ==================================================

print("\n" + "=" * 70)
print("PROTOTYPE MISCLASSIFICATIONS")
print("=" * 70)

for filename, true_label, prediction in zip(
    filenames,
    y_test,
    prototype_predictions
):

    if true_label != prediction:

        print(
            f"{filename}: "
            f"{true_label} -> {prediction}"
        )


# ==================================================
# MEDOID ERRORS
# ==================================================

print("\n" + "=" * 70)
print("MEDOID MISCLASSIFICATIONS")
print("=" * 70)

for filename, true_label, prediction in zip(
    filenames,
    y_test,
    medoid_predictions
):

    if true_label != prediction:

        print(
            f"{filename}: "
            f"{true_label} -> {prediction}"
        )


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)