import os
import cv2
import numpy as np

from sklearn.svm import SVC
from skimage.feature import hog


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ============================================================
# PREPROCESSING
# ============================================================

def process_image(img):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )
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

    cropped = binary[
        y:y + h,
        x:x + w
    ]

    max_dim = max(w, h)

    scale = 20.0 / max_dim

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

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
    cy, cx = np.where(
        canvas > 0
    )

    if len(cx) > 0 and len(cy) > 0:

        shift_x = int(
            np.round(
                14 - np.mean(cx)
            )
        )

        shift_y = int(
            np.round(
                14 - np.mean(cy)
            )
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
# HOG
# ============================================================

def extract_hog(img):

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


# ============================================================
# LOAD ORIGINAL TRAINING DATA
# ============================================================

def load_training_images():

    images = []
    labels = []

    for folder_name in sorted(
        os.listdir("data")
    ):

        folder_path = os.path.join(
            "data",
            folder_name
        )

        if not os.path.isdir(
            folder_path
        ):
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
                img
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
# CURRENT AUGMENTATION
# ============================================================

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

    # Rotate +10 degrees
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

    result_images = []
    result_labels = []

    for image, label in zip(
        images,
        labels
    ):

        for variation in augment_image(
            image
        ):

            result_images.append(
                variation
            )

            result_labels.append(
                label
            )

    return (
        np.array(result_images),
        np.array(result_labels)
    )


# ============================================================
# LOAD EXTERNAL TEST
# ============================================================

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

        processed = process_image(
            img
        )

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


# ============================================================
# GROUP IDENTIFICATION
# ============================================================

def get_group(label):

    if label.isdigit():
        return "digit"

    if label.isupper():
        return "uppercase"

    return "lowercase"


# ============================================================
# TRAIN SPECIALIST MODELS
# ============================================================

def train_specialist_models(
    X_train,
    y_train
):

    models = {}

    groups = [
        "digit",
        "uppercase",
        "lowercase"
    ]

    for group in groups:

        indices = np.array([
            get_group(label) == group
            for label in y_train
        ])

        group_X = X_train[
            indices
        ]

        group_y = y_train[
            indices
        ]

        print(
            f"\nTraining {group} specialist..."
        )

        print(
            "Samples:",
            len(group_X)
        )

        print(
            "Classes:",
            len(np.unique(group_y))
        )

        model = SVC(
            kernel="rbf",
            C=50.0,
            gamma=0.01
        )

        model.fit(
            group_X,
            group_y
        )

        models[group] = model

    return models


# ============================================================
# STAGE 1 GROUP CLASSIFIER
# ============================================================

def train_group_classifier(
    X_train,
    y_train
):

    group_labels = np.array([
        get_group(label)
        for label in y_train
    ])

    print(
        "\nTraining Stage-1 group classifier..."
    )

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_train,
        group_labels
    )

    return model


# ============================================================
# HIERARCHICAL PREDICTION
# ============================================================

def hierarchical_predict(
    X_test,
    group_model,
    specialist_models
):

    predicted_groups = (
        group_model.predict(
            X_test
        )
    )

    predictions = []

    for image, group in zip(
        X_test,
        predicted_groups
    ):

        model = specialist_models[
            group
        ]

        prediction = model.predict(
            image.reshape(1, -1)
        )[0]

        predictions.append(
            prediction
        )

    return np.array(
        predictions
    ), predicted_groups


# ============================================================
# GROUP ACCURACY
# ============================================================

def calculate_group_accuracy(
    y_true,
    predicted_groups
):

    true_groups = np.array([
        get_group(label)
        for label in y_true
    ])

    return (
        np.mean(
            true_groups == predicted_groups
        )
        * 100
    )


# ============================================================
# CHARACTER GROUP RESULTS
# ============================================================

def print_group_results(
    y_true,
    predictions
):

    print(
        "\nGroup-wise character accuracy:"
    )

    for group in [
        "digit",
        "uppercase",
        "lowercase"
    ]:

        indices = np.array([
            get_group(label) == group
            for label in y_true
        ])

        accuracy = (
            np.mean(
                predictions[indices]
                == y_true[indices]
            )
            * 100
        )

        correct = np.sum(
            predictions[indices]
            == y_true[indices]
        )

        total = np.sum(indices)

        print(
            f"{group:<10}: "
            f"{correct}/{total} "
            f"({accuracy:.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("HIERARCHICAL CLASSIFICATION EXPERIMENT")
print("=" * 70)


# ------------------------------------------------------------
# LOAD TRAINING
# ------------------------------------------------------------

print(
    "\nLoading training images..."
)

train_images, train_labels = (
    load_training_images()
)

print(
    "Original training images:",
    len(train_images)
)


# ------------------------------------------------------------
# AUGMENT
# ------------------------------------------------------------

print(
    "\nApplying current augmentation..."
)

train_images, train_labels = (
    create_augmented_dataset(
        train_images,
        train_labels
    )
)

print(
    "Training samples after augmentation:",
    len(train_images)
)


# ------------------------------------------------------------
# HOG
# ------------------------------------------------------------

print(
    "\nExtracting HOG features..."
)

X_train = np.array([
    extract_hog(image)
    for image in train_images
])

print(
    "HOG feature size:",
    X_train.shape[1]
)


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

print(
    "\nLoading external test..."
)

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


# ============================================================
# BASELINE 62-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("BASELINE 62-CLASS SVM")
print("=" * 70)

baseline_model = SVC(
    kernel="rbf",
    C=50.0,
    gamma=0.01
)

baseline_model.fit(
    X_train,
    train_labels
)

baseline_predictions = (
    baseline_model.predict(
        X_test
    )
)

baseline_accuracy = (
    np.mean(
        baseline_predictions == y_test
    )
    * 100
)

print(
    f"Baseline accuracy: "
    f"{baseline_accuracy:.2f}%"
)

print_group_results(
    y_test,
    baseline_predictions
)


# ============================================================
# HIERARCHICAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL MODEL")
print("=" * 70)


group_model = train_group_classifier(
    X_train,
    train_labels
)

specialist_models = train_specialist_models(
    X_train,
    train_labels
)


hierarchical_predictions, predicted_groups = (
    hierarchical_predict(
        X_test,
        group_model,
        specialist_models
    )
)


# ============================================================
# STAGE 1 ACCURACY
# ============================================================

group_accuracy = calculate_group_accuracy(
    y_test,
    predicted_groups
)

print(
    "\nStage-1 group accuracy:",
    f"{group_accuracy:.2f}%"
)


# ============================================================
# FINAL CHARACTER ACCURACY
# ============================================================

hierarchical_accuracy = (
    np.mean(
        hierarchical_predictions
        == y_test
    )
    * 100
)

print(
    "\nHierarchical character accuracy:",
    f"{hierarchical_accuracy:.2f}%"
)

print_group_results(
    y_test,
    hierarchical_predictions
)


# ============================================================
# ERRORS
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL MISCLASSIFICATIONS")
print("=" * 70)

for filename, true_label, prediction, group in zip(
    filenames,
    y_test,
    hierarchical_predictions,
    predicted_groups
):

    if true_label != prediction:

        print(
            f"{filename}: "
            f"{true_label} -> {prediction} "
            f"[group={group}]"
        )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"62-class SVM:       "
    f"{baseline_accuracy:.2f}%"
)

print(
    f"Hierarchical SVM:   "
    f"{hierarchical_accuracy:.2f}%"
)

difference = (
    hierarchical_accuracy
    - baseline_accuracy
)

print(
    f"Difference:         "
    f"{difference:+.2f} percentage points"
)

print("=" * 70)


if hierarchical_accuracy > baseline_accuracy:

    print(
        "\nResult: Hierarchical classification "
        "improves external accuracy."
    )

elif hierarchical_accuracy < baseline_accuracy:

    print(
        "\nResult: Hierarchical classification "
        "hurts external accuracy."
    )

else:

    print(
        "\nResult: Hierarchical classification "
        "produces the same accuracy."
    )

print("=" * 70)