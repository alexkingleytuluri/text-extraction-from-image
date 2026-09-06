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


# --------------------------------------------------
# AUGMENTATION VARIANTS
# --------------------------------------------------

def augment_image(img, mode):

    images = [img]

    # Current baseline augmentation
    if mode == "baseline":

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

    # --------------------------------------------------
    # THICKER STROKES
    # --------------------------------------------------
    elif mode == "thicker":

        kernel = np.ones(
            (2, 2),
            dtype=np.uint8
        )

        thicker = cv2.dilate(
            img,
            kernel,
            iterations=1
        )

        images.append(thicker)

    # --------------------------------------------------
    # THINNER STROKES
    # --------------------------------------------------
    elif mode == "thinner":

        kernel = np.ones(
            (2, 2),
            dtype=np.uint8
        )

        thinner = cv2.erode(
            img,
            kernel,
            iterations=1
        )

        images.append(thinner)

    # --------------------------------------------------
    # BLUR
    # --------------------------------------------------
    elif mode == "blur":

        blurred = cv2.GaussianBlur(
            img,
            (3, 3),
            0
        )

        images.append(blurred)

    # --------------------------------------------------
    # SMALL ROTATIONS
    # --------------------------------------------------
    elif mode == "small_rotation":

        M_plus = cv2.getRotationMatrix2D(
            (14, 14),
            5,
            1
        )

        M_minus = cv2.getRotationMatrix2D(
            (14, 14),
            -5,
            1
        )

        images.append(
            cv2.warpAffine(
                img,
                M_plus,
                (28, 28)
            )
        )

        images.append(
            cv2.warpAffine(
                img,
                M_minus,
                (28, 28)
            )
        )

    # --------------------------------------------------
    # SMALL TRANSLATIONS
    # --------------------------------------------------
    elif mode == "small_shift":

        M_right = np.float32([
            [1, 0, 1],
            [0, 1, 0]
        ])

        M_left = np.float32([
            [1, 0, -1],
            [0, 1, 0]
        ])

        M_down = np.float32([
            [1, 0, 0],
            [0, 1, 1]
        ])

        M_up = np.float32([
            [1, 0, 0],
            [0, 1, -1]
        ])

        images.append(
            cv2.warpAffine(
                img,
                M_right,
                (28, 28)
            )
        )

        images.append(
            cv2.warpAffine(
                img,
                M_left,
                (28, 28)
            )
        )

        images.append(
            cv2.warpAffine(
                img,
                M_down,
                (28, 28)
            )
        )

        images.append(
            cv2.warpAffine(
                img,
                M_up,
                (28, 28)
            )
        )

    return images


def create_augmented_dataset(
    images,
    labels,
    mode
):

    augmented_images = []
    augmented_labels = []

    for image, label in zip(
        images,
        labels
    ):

        for variation in augment_image(
            image,
            mode
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


def run_experiment(mode):

    print("\n" + "=" * 65)
    print("AUGMENTATION MODE:", mode)
    print("=" * 65)

    train_images, train_labels = (
        load_original_images("data")
    )

    print(
        "Original training images:",
        len(train_images)
    )

    train_images, train_labels = (
        create_augmented_dataset(
            train_images,
            train_labels,
            mode
        )
    )

    print(
        "Training samples after augmentation:",
        len(train_images)
    )

    X_train = extract_hog(
        train_images
    )

    print(
        "HOG feature size:",
        X_train.shape[1]
    )

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    print("Training SVM...")

    model.fit(
        X_train,
        train_labels
    )

    test_images, y_test, filenames = (
        load_external_test()
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
        f"External accuracy: {accuracy:.2f}%"
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
print("HANDWRITING STYLE AUGMENTATION EXPERIMENT")
print("=" * 65)

baseline = run_experiment(
    "baseline"
)

thicker = run_experiment(
    "thicker"
)

thinner = run_experiment(
    "thinner"
)

blur = run_experiment(
    "blur"
)

small_rotation = run_experiment(
    "small_rotation"
)

small_shift = run_experiment(
    "small_shift"
)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)

print(
    f"Current baseline:       {baseline:.2f}%"
)

print(
    f"Thicker strokes:        {thicker:.2f}%"
)

print(
    f"Thinner strokes:        {thinner:.2f}%"
)

print(
    f"Blur:                    {blur:.2f}%"
)

print(
    f"Small rotations:        {small_rotation:.2f}%"
)

print(
    f"Small translations:     {small_shift:.2f}%"
)

print("=" * 65)