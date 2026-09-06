import os
import cv2
import numpy as np

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


def augment_image(img):

    images = [img]

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


def extract_hog_features(img):

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


def extract_shape_features(img):

    features = []

    # --------------------------------------------------
    # 1. Overall ink density
    # --------------------------------------------------

    density = np.mean(img > 0)
    features.append(density)

    # --------------------------------------------------
    # 2. Bounding-box width and height
    # --------------------------------------------------

    coords = np.argwhere(img > 0)

    if len(coords) > 0:

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        width = x_max - x_min + 1
        height = y_max - y_min + 1

    else:

        width = 0
        height = 0

    features.append(width / 28.0)
    features.append(height / 28.0)

    # Aspect ratio
    if height > 0:
        aspect_ratio = width / height
    else:
        aspect_ratio = 0

    features.append(aspect_ratio)

    # --------------------------------------------------
    # 3. Horizontal projection
    # --------------------------------------------------

    horizontal = np.sum(img > 0, axis=1) / 28.0

    features.extend(horizontal)

    # --------------------------------------------------
    # 4. Vertical projection
    # --------------------------------------------------

    vertical = np.sum(img > 0, axis=0) / 28.0

    features.extend(vertical)

    # --------------------------------------------------
    # 5. Center of mass
    # --------------------------------------------------

    if len(coords) > 0:

        center_y = np.mean(coords[:, 0]) / 28.0
        center_x = np.mean(coords[:, 1]) / 28.0

    else:

        center_y = 0
        center_x = 0

    features.append(center_x)
    features.append(center_y)

    return np.array(features, dtype=np.float32)


def extract_combined_features(images):

    features = []

    for img in images:

        hog_features = extract_hog_features(img)

        shape_features = extract_shape_features(img)

        combined = np.concatenate([
            hog_features,
            shape_features
        ])

        features.append(combined)

    return np.array(features)


def load_original_images(data_folder):

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

            processed = process_image(img)

            if processed is None:
                continue

            images.append(processed)
            labels.append(label)

    return (
        np.array(images),
        np.array(labels)
    )


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

        for variation in augment_image(image):

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


def run_experiment():

    print("\nProcessing training images...")

    train_images, train_labels = (
        load_original_images("data")
    )

    print(
        "Original training images:",
        len(train_images)
    )

    print("Applying augmentation...")

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

    print("\nExtracting combined features...")

    X_train = extract_combined_features(
        train_images
    )

    print(
        "Combined feature size:",
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

    print("\nLoading external test...")

    test_images, y_test, filenames = (
        load_external_test()
    )

    print(
        "External test images:",
        len(test_images)
    )

    X_test = extract_combined_features(
        test_images
    )

    predictions = model.predict(
        X_test
    )

    correct = np.sum(
        predictions == y_test
    )

    total = len(y_test)

    accuracy = (
        correct / total
    ) * 100

    print(
        f"\nHOG + shape features accuracy: "
        f"{accuracy:.2f}%"
    )

    print("\nMisclassifications:")

    for filename, true_label, prediction in zip(
        filenames,
        y_test,
        predictions
    ):

        if true_label != prediction:

            print(
                f"{filename}: "
                f"{true_label} -> {prediction}"
            )

    return accuracy


print("=" * 65)
print("HOG + SHAPE FEATURE EXPERIMENT")
print("=" * 65)

accuracy = run_experiment()

print("\n" + "=" * 65)
print("RESULT")
print("=" * 65)

print(
    f"Current HOG baseline:     64.52%"
)

print(
    f"HOG + shape features:     {accuracy:.2f}%"
)

print(
    f"Difference:                "
    f"{accuracy - 64.52:+.2f} percentage points"
)

if accuracy > 64.52:

    print(
        "\nResult: Shape features improve "
        "external accuracy."
    )

elif accuracy < 64.52:

    print(
        "\nResult: Shape features hurt "
        "external accuracy."
    )

else:

    print(
        "\nResult: No change in external accuracy."
    )

print("=" * 65)