import os
import cv2
import numpy as np

from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog


# ============================================================
# FINAL HANDWRITING OCR EVALUATION
# ============================================================

print("=" * 70)
print("FINAL HANDWRITING OCR EVALUATION")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

C_VALUE = 50.0
GAMMA_VALUE = 0.01

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)

TEST_FOLDER = "test_clean"


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

X_train, y_train = load_dataset("data_clean")

y_train = np.array([
    str(label)
    for label in y_train
])

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Feature size: {X_train.shape[1]}"
)

print(
    f"Classes: {len(np.unique(y_train))}"
)


# ============================================================
# LOAD EXTERNAL TEST DATA
# ============================================================

print("\nLoading external test data...")

X_test = []
y_test = []
filenames = []


files = sorted(
    os.listdir(TEST_FOLDER),
    key=lambda x: (not x[0].isdigit(), x)
)


for filename in files:

    if not filename.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        continue

    true_label = os.path.splitext(filename)[0]

    # Convert s-a -> a
    if true_label.startswith("s-"):
        true_label = true_label[2:]

    image_path = os.path.join(
        TEST_FOLDER,
        filename
    )

    gray = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        continue

    if gray.shape != (28, 28):
        gray = cv2.resize(
            gray,
            (28, 28)
        )

    normalized = gray / 255.0

    features = hog(
        normalized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys"
    )

    X_test.append(features)
    y_test.append(true_label)
    filenames.append(filename)


X_test = np.array(X_test)
y_test = np.array(y_test)


print(
    f"Test samples: {len(X_test)}"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\nTraining final SVM model...")

model = SVC(
    kernel="rbf",
    C=C_VALUE,
    gamma=GAMMA_VALUE
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning final predictions...")

predictions = model.predict(
    X_test
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

correct = np.sum(
    predictions == y_test
)

total = len(y_test)

accuracy = (
    correct / total
) * 100


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Correct predictions : {correct}/{total}"
)

print(
    f"Incorrect predictions: {total - correct}/{total}"
)

print(
    f"Overall accuracy     : {accuracy:.2f}%"
)


# ============================================================
# CATEGORY ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY RESULTS")
print("=" * 70)


categories = [
    ("digits", str.isdigit),
    ("uppercase", str.isupper),
    ("lowercase", str.islower)
]


category_results = {}


for category_name, checker in categories:

    mask = np.array([
        checker(label)
        for label in y_test
    ])

    category_correct = np.sum(
        predictions[mask] == y_test[mask]
    )

    category_total = np.sum(mask)

    category_accuracy = (
        category_correct
        / category_total
    ) * 100

    category_results[category_name] = (
        category_correct,
        category_total,
        category_accuracy
    )

    print(
        f"{category_name:<10}: "
        f"{category_correct}/"
        f"{category_total} "
        f"({category_accuracy:.2f}%)"
    )


# ============================================================
# COMPLETE PREDICTION TABLE
# ============================================================

print("\n" + "=" * 70)
print("ALL PREDICTIONS")
print("=" * 70)

print(
    f"{'FILE':<20}"
    f"{'TRUE':<10}"
    f"{'PREDICTED':<12}"
    f"{'STATUS'}"
)

print("-" * 70)


for filename, true_label, prediction in zip(
    filenames,
    y_test,
    predictions
):

    status = (
        "CORRECT"
        if true_label == prediction
        else "ERROR"
    )

    print(
        f"{filename:<20}"
        f"{true_label:<10}"
        f"{prediction:<12}"
        f"{status}"
    )


# ============================================================
# ERROR SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)


errors = []

for filename, true_label, prediction in zip(
    filenames,
    y_test,
    predictions
):

    if true_label != prediction:

        errors.append(
            (
                filename,
                true_label,
                prediction
            )
        )


print(
    f"Total errors: {len(errors)}"
)


for filename, true_label, prediction in errors:

    print(
        f"{filename}: "
        f"{true_label} -> {prediction}"
    )


# ============================================================
# ERROR COUNT BY CATEGORY
# ============================================================

digit_errors = 0
uppercase_errors = 0
lowercase_errors = 0


for _, true_label, _ in errors:

    if true_label.isdigit():

        digit_errors += 1

    elif true_label.isupper():

        uppercase_errors += 1

    elif true_label.islower():

        lowercase_errors += 1


print("\n" + "=" * 70)
print("ERRORS BY CATEGORY")
print("=" * 70)

print(
    f"Digits    : {digit_errors}"
)

print(
    f"Uppercase : {uppercase_errors}"
)

print(
    f"Lowercase : {lowercase_errors}"
)


# ============================================================
# CONFUSION PAIRS
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION PAIRS")
print("=" * 70)


confusions = {}


for _, true_label, prediction in errors:

    pair = (
        true_label,
        prediction
    )

    confusions[pair] = (
        confusions.get(pair, 0)
        + 1
    )


sorted_confusions = sorted(
    confusions.items(),
    key=lambda item: item[1],
    reverse=True
)


for (true_label, prediction), count in sorted_confusions:

    print(
        f"{true_label} -> "
        f"{prediction}: "
        f"{count}"
    )


# ============================================================
# FINAL CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL CONFIGURATION")
print("=" * 70)

print(
    f"Threshold           : 100"
)

print(
    f"Max dimension       : 20"
)

print(
    f"Canvas size         : 28 x 28"
)

print(
    f"Center alignment    : Enabled"
)

print(
    f"HOG orientations    : {HOG_ORIENTATIONS}"
)

print(
    f"Pixels per cell     : "
    f"{HOG_PIXELS_PER_CELL}"
)

print(
    f"Cells per block     : "
    f"{HOG_CELLS_PER_BLOCK}"
)

print(
    f"SVM kernel           : RBF"
)

print(
    f"SVM C                : {C_VALUE}"
)

print(
    f"SVM gamma            : {GAMMA_VALUE}"
)

print(
    f"Augmented samples    : {len(X_train)}"
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

print(
    f"Final external accuracy: "
    f"{accuracy:.2f}% "
    f"({correct}/{total})"
)

print(
    "Classical handwriting OCR evaluation complete."
)

print("=" * 70)