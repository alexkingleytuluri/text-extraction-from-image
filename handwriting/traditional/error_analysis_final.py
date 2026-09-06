import os
import cv2
import numpy as np

from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog


# ============================================================
# FINAL ERROR ANALYSIS
# ============================================================

print("=" * 70)
print("FINAL HANDWRITING OCR ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# FINAL MODEL CONFIGURATION
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


# ============================================================
# LOAD TEST DATA
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

print("\nTraining final SVM...")

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
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# COLLECT ERRORS
# ============================================================

errors = []

for filename, true_label, prediction in zip(
    filenames,
    y_test,
    predictions
):

    if true_label != prediction:

        errors.append({
            "filename": filename,
            "true": true_label,
            "predicted": prediction
        })


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

total = len(y_test)
correct = total - len(errors)

print(
    f"Total test samples : {total}"
)

print(
    f"Correct            : {correct}"
)

print(
    f"Errors             : {len(errors)}"
)

print(
    f"Accuracy           : "
    f"{(correct / total) * 100:.2f}%"
)


# ============================================================
# ERRORS BY CATEGORY
# ============================================================

digit_errors = []
uppercase_errors = []
lowercase_errors = []

for error in errors:

    label = error["true"]

    if label.isdigit():

        digit_errors.append(error)

    elif label.isupper():

        uppercase_errors.append(error)

    elif label.islower():

        lowercase_errors.append(error)


print("\n" + "=" * 70)
print("ERRORS BY CATEGORY")
print("=" * 70)

print(
    f"Digits    : "
    f"{len(digit_errors)}"
)

print(
    f"Uppercase : "
    f"{len(uppercase_errors)}"
)

print(
    f"Lowercase : "
    f"{len(lowercase_errors)}"
)


# ============================================================
# PRINT CATEGORY ERRORS
# ============================================================

def print_category_errors(
    title,
    category_errors
):

    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    if len(category_errors) == 0:

        print("No errors.")

        return

    for error in category_errors:

        print(
            f"{error['filename']:<15} "
            f"{error['true']} -> "
            f"{error['predicted']}"
        )


print_category_errors(
    "DIGIT ERRORS",
    digit_errors
)

print_category_errors(
    "UPPERCASE ERRORS",
    uppercase_errors
)

print_category_errors(
    "LOWERCASE ERRORS",
    lowercase_errors
)


# ============================================================
# PREDICTED CLASS ERROR COUNTS
# ============================================================

predicted_error_counts = {}

for error in errors:

    predicted = error["predicted"]

    predicted_error_counts[predicted] = (
        predicted_error_counts.get(
            predicted,
            0
        ) + 1
    )


print("\n" + "=" * 70)
print("MOST COMMON INCORRECT PREDICTIONS")
print("=" * 70)

sorted_predictions = sorted(
    predicted_error_counts.items(),
    key=lambda item: item[1],
    reverse=True
)

for predicted, count in sorted_predictions:

    print(
        f"{predicted}: "
        f"{count} error(s)"
    )


# ============================================================
# TRUE CLASS ERROR COUNTS
# ============================================================

true_error_counts = {}

for error in errors:

    true_label = error["true"]

    true_error_counts[true_label] = (
        true_error_counts.get(
            true_label,
            0
        ) + 1
    )


print("\n" + "=" * 70)
print("CLASSES MOST OFTEN MISCLASSIFIED")
print("=" * 70)

sorted_true_errors = sorted(
    true_error_counts.items(),
    key=lambda item: item[1],
    reverse=True
)

for true_label, count in sorted_true_errors:

    print(
        f"{true_label}: "
        f"{count} error(s)"
    )


# ============================================================
# CONFUSION PAIRS
# ============================================================

confusion_pairs = {}

for error in errors:

    pair = (
        error["true"],
        error["predicted"]
    )

    confusion_pairs[pair] = (
        confusion_pairs.get(
            pair,
            0
        ) + 1
    )


print("\n" + "=" * 70)
print("CONFUSION PAIRS")
print("=" * 70)

sorted_pairs = sorted(
    confusion_pairs.items(),
    key=lambda item: item[1],
    reverse=True
)

for (
    (true_label, predicted),
    count
) in sorted_pairs:

    print(
        f"{true_label} -> "
        f"{predicted}: "
        f"{count}"
    )


# ============================================================
# DIGIT VS LETTER CONFUSIONS
# ============================================================

digit_letter_confusions = []

for error in errors:

    true_label = error["true"]
    predicted = error["predicted"]

    if (
        true_label.isdigit()
        and predicted.isalpha()
    ):

        digit_letter_confusions.append(
            error
        )

    elif (
        true_label.isalpha()
        and predicted.isdigit()
    ):

        digit_letter_confusions.append(
            error
        )


print("\n" + "=" * 70)
print("DIGIT ↔ LETTER CONFUSIONS")
print("=" * 70)

print(
    f"Total: "
    f"{len(digit_letter_confusions)}"
)

for error in digit_letter_confusions:

    print(
        f"{error['filename']}: "
        f"{error['true']} -> "
        f"{error['predicted']}"
    )


# ============================================================
# CASE CONFUSIONS
# ============================================================

case_confusions = []

for error in errors:

    true_label = error["true"]
    predicted = error["predicted"]

    if (
        true_label.isalpha()
        and predicted.isalpha()
        and true_label.lower()
        == predicted.lower()
        and true_label != predicted
    ):

        case_confusions.append(
            error
        )


print("\n" + "=" * 70)
print("UPPERCASE ↔ LOWERCASE CONFUSIONS")
print("=" * 70)

print(
    f"Total: "
    f"{len(case_confusions)}"
)

for error in case_confusions:

    print(
        f"{error['filename']}: "
        f"{error['true']} -> "
        f"{error['predicted']}"
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print(
    "1. Uppercase characters are the strongest category."
)

print(
    "2. Digits and lowercase characters account "
    "for most recognition errors."
)

print(
    "3. Several mistakes occur between visually "
    "similar handwritten characters."
)

print(
    "4. Digit-letter and uppercase-lowercase "
    "confusions indicate overlap in handwriting shapes."
)

print(
    "5. The external test contains handwriting "
    "variation that is not fully represented by "
    "the limited training samples."
)

print(
    "6. The final classical pipeline achieves "
    f"{(correct / total) * 100:.2f}% external accuracy."
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL CONFIGURATION")
print("=" * 70)

print("Threshold           : 100")
print("Max dimension       : 20")
print("Canvas size         : 28 x 28")
print("Center alignment    : Enabled")
print(
    f"HOG orientations    : "
    f"{HOG_ORIENTATIONS}"
)
print(
    f"Pixels per cell     : "
    f"{HOG_PIXELS_PER_CELL}"
)
print(
    f"Cells per block     : "
    f"{HOG_CELLS_PER_BLOCK}"
)
print("SVM kernel          : RBF")
print(
    f"SVM C               : "
    f"{C_VALUE}"
)
print(
    f"SVM gamma           : "
    f"{GAMMA_VALUE}"
)
print(
    f"Training samples    : "
    f"{len(X_train)}"
)
print(
    f"Test samples        : "
    f"{len(X_test)}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINAL ERROR ANALYSIS COMPLETE")
print("=" * 70)