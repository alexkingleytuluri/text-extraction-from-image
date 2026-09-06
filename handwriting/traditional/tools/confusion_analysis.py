import os
import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import hog


def extract_hog(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


def augment_image(img):
    images = [img]

    M_right = np.float32([
        [1, 0, 2],
        [0, 1, 0]
    ])
    images.append(
        cv2.warpAffine(img, M_right, (28, 28))
    )

    M_left = np.float32([
        [1, 0, -2],
        [0, 1, 0]
    ])
    images.append(
        cv2.warpAffine(img, M_left, (28, 28))
    )

    M_rot = cv2.getRotationMatrix2D(
        (14, 14),
        10,
        1
    )
    images.append(
        cv2.warpAffine(img, M_rot, (28, 28))
    )

    return images


def load_training_data():
    images = []
    labels = []

    for folder_name in sorted(os.listdir("data_clean")):

        folder_path = os.path.join(
            "data_clean",
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
                (".png", ".jpg", ".jpeg")
            ):
                continue

            path = os.path.join(
                folder_path,
                filename
            )

            image = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = image / 255.0

            for variation in augment_image(image):

                images.append(
                    extract_hog(variation)
                )

                labels.append(label)

    return np.array(images), np.array(labels)


def load_test_data():
    images = []
    labels = []

    for filename in sorted(
        os.listdir("test_clean"),
        key=lambda x: (not x[0].isdigit(), x)
    ):

        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        path = os.path.join(
            "test_clean",
            filename
        )

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        image = image / 255.0

        label = os.path.splitext(filename)[0]

        if label.startswith("s-"):
            label = label[2:]

        images.append(
            extract_hog(image)
        )

        labels.append(label)

    return np.array(images), np.array(labels)


print("=" * 65)
print("BASELINE CONFUSION ANALYSIS")
print("=" * 65)

print("\nLoading training data...")

X_train, y_train = load_training_data()

print(
    "Training samples:",
    len(X_train)
)

print("Feature size:", X_train.shape[1])

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

print("Training complete.")

print("\nLoading external test data...")

X_test, y_test = load_test_data()

print(
    "Test samples:",
    len(X_test)
)

predictions = model.predict(X_test)

print("\n" + "=" * 65)
print("MISCLASSIFICATIONS")
print("=" * 65)

for true_label, prediction in zip(
    y_test,
    predictions
):

    if true_label != prediction:
        print(
            f"{true_label} -> {prediction}"
        )


print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y_test,
        predictions,
        labels=sorted(np.unique(y_test)),
        zero_division=0
    )
)


print("\n" + "=" * 65)
print("CONFUSION PAIRS")
print("=" * 65)

confusions = {}

for true_label, prediction in zip(
    y_test,
    predictions
):

    if true_label != prediction:

        pair = (
            true_label,
            prediction
        )

        confusions[pair] = (
            confusions.get(pair, 0) + 1
        )


for (true_label, prediction), count in sorted(
    confusions.items(),
    key=lambda item: item[1],
    reverse=True
):

    print(
        f"{true_label} -> {prediction}: "
        f"{count}"
    )


print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)