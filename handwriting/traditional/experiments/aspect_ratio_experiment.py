import os
import cv2
import numpy as np

from sklearn.svm import SVC
from skimage.feature import hog


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def process_image(img, mode):
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

    # --------------------------------------------------
    # MODE 1: CURRENT BASELINE
    # Largest dimension = 20
    # --------------------------------------------------
    if mode == "baseline":

        max_dim = max(w, h)

        scale = 20.0 / max_dim

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

    # --------------------------------------------------
    # MODE 2: FIXED HEIGHT
    # Height = 20
    # Width keeps original aspect ratio
    # --------------------------------------------------
    elif mode == "fixed_height":

        scale = 20.0 / h

        new_h = 20
        new_w = max(1, int(w * scale))

        # Prevent width from exceeding canvas
        if new_w > 20:
            new_w = 20

    # --------------------------------------------------
    # MODE 3: FIXED WIDTH
    # Width = 20
    # Height keeps original aspect ratio
    # --------------------------------------------------
    elif mode == "fixed_width":

        scale = 20.0 / w

        new_w = 20
        new_h = max(1, int(h * scale))

        # Prevent height from exceeding canvas
        if new_h > 20:
            new_h = 20

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

    # Keep center-of-mass alignment
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


def load_original_images(data_folder, mode):

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
                mode
            )

            if processed is None:
                continue

            images.append(processed)
            labels.append(label)

    return (
        np.array(images),
        np.array(labels)
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

    # Rotate 10 degrees
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


def load_external_test(mode):

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
            img,
            mode
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


def run_experiment(mode):

    print("\n" + "=" * 60)
    print("MODE:", mode)
    print("=" * 60)

    print("\nProcessing training images...")

    train_images, train_labels = (
        load_original_images(
            "data",
            mode
        )
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

    print("Extracting HOG features...")

    X_train = extract_hog(
        train_images
    )

    print(
        "Training feature size:",
        X_train.shape[1]
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

    print("Loading external test images...")

    test_images, y_test, filenames = (
        load_external_test(
            mode
        )
    )

    print(
        "External test images:",
        len(test_images)
    )

    X_test = extract_hog(
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
        f"Accuracy: {accuracy:.2f}%"
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
print("ASPECT-RATIO NORMALIZATION EXPERIMENT")
print("=" * 65)

baseline = run_experiment(
    "baseline"
)

fixed_height = run_experiment(
    "fixed_height"
)

fixed_width = run_experiment(
    "fixed_width"
)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

print(
    f"Current baseline:  {baseline:.2f}%"
)

print(
    f"Fixed height:      {fixed_height:.2f}%"
)

print(
    f"Fixed width:       {fixed_width:.2f}%"
)

print("=" * 65)