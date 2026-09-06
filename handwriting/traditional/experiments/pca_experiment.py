import os
import cv2
import numpy as np

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.feature import hog


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


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


def extract_hog(img):

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


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


def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    components
):

    if components is None:

        X_train_final = X_train
        X_test_final = X_test

        actual_components = "None"

    else:

        pca = PCA(
            n_components=components,
            whiten=False,
            random_state=42
        )

        X_train_final = pca.fit_transform(
            X_train
        )

        X_test_final = pca.transform(
            X_test
        )

        actual_components = (
            X_train_final.shape[1]
        )

    print(
        f"\nTesting PCA components: "
        f"{actual_components}"
    )

    print(
        "Final feature size:",
        X_train_final.shape[1]
    )

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_train_final,
        y_train
    )

    predictions = model.predict(
        X_test_final
    )

    accuracy = (
        np.mean(
            predictions == y_test
        ) * 100
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    return accuracy


print("=" * 70)
print("PCA + HOG EXPERIMENT")
print("=" * 70)


print("\nLoading training images...")

train_images, train_labels = (
    load_training_images()
)

print(
    "Original training images:",
    len(train_images)
)


print("\nApplying current augmentation...")

train_images, train_labels = (
    create_augmented_dataset(
        train_images,
        train_labels
    )
)

print(
    "Training samples:",
    len(train_images)
)


print("\nExtracting HOG features...")

X_train = np.array([
    extract_hog(image)
    for image in train_images
])

print(
    "Original HOG dimensions:",
    X_train.shape[1]
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
    len(test_images)
)


results = {}

components_list = [
    None,
    300,
    200,
    100,
    50
]

for components in components_list:

    accuracy = run_experiment(
        X_train,
        train_labels,
        X_test,
        y_test,
        components
    )

    results[components] = accuracy


print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

for components, accuracy in results.items():

    if components is None:
        name = "No PCA"
    else:
        name = f"PCA {components}"

    print(
        f"{name:<15}: "
        f"{accuracy:.2f}%"
    )


best_components = max(
    results,
    key=results.get
)

best_accuracy = results[
    best_components
]

baseline = results[None]

print("\n" + "=" * 70)

if best_components is None:

    print(
        "Result: Original HOG remains best."
    )

else:

    print(
        f"Best PCA setting: "
        f"{best_components} components"
    )

    print(
        f"Best accuracy: "
        f"{best_accuracy:.2f}%"
    )

print(
    f"Baseline: "
    f"{baseline:.2f}%"
)

print(
    f"Improvement: "
    f"{best_accuracy - baseline:+.2f} pp"
)

print("=" * 70)