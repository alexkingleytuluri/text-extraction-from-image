import numpy as np

from load_dataset import load_dataset
from sklearn.svm import SVC


# ============================================================
# CONFIGURATION
# ============================================================

CONFUSION_PAIRS = [
    ("1", "i"),
    ("1", "l"),
    ("i", "j"),
    ("0", "3"),
    ("2", "9"),
    ("4", "6"),
    ("8", "9"),
    ("8", "e"),
    ("g", "9"),
    ("g", "q"),
    ("c", "e"),
    ("h", "r"),
    ("j", "f"),
    ("k", "B"),
    ("s", "o"),
    ("t", "k"),
    ("u", "v"),
]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("PAIRWISE CONFUSION SPECIALIST EXPERIMENT")
print("=" * 70)

print("\nLoading training data...")

X_train, y_train = load_dataset("data_clean")

print("Training samples:", len(X_train))
print("Feature size:", X_train.shape[1])


# ============================================================
# TRAIN BASELINE
# ============================================================

print("\nTraining baseline 62-class SVM...")

baseline_model = SVC(
    kernel="rbf",
    C=50.0,
    gamma=0.01
)

baseline_model.fit(
    X_train,
    y_train
)

print("Baseline training complete.")


# ============================================================
# TRAIN PAIRWISE SPECIALISTS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING PAIRWISE SPECIALISTS")
print("=" * 70)

specialists = {}

for class_a, class_b in CONFUSION_PAIRS:

    mask = np.isin(
        y_train,
        [class_a, class_b]
    )

    X_pair = X_train[mask]
    y_pair = y_train[mask]

    print(
        f"\n{class_a} vs {class_b}"
    )

    print(
        f"Samples: {len(X_pair)}"
    )

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    model.fit(
        X_pair,
        y_pair
    )

    specialists[
        (class_a, class_b)
    ] = model


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING EXTERNAL TEST")
print("=" * 70)

# Reuse the same test-loading logic as predict_test.py
import os
import cv2
from skimage.feature import hog


test_folder = "test_clean"

X_test = []
y_test = []
filenames = []

files = sorted(
    os.listdir(test_folder),
    key=lambda x: (not x[0].isdigit(), x)
)

for filename in files:

    if not filename.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        continue

    true_label = os.path.splitext(
        filename
    )[0]

    if true_label.startswith("s-"):
        true_label = true_label[2:]

    path = os.path.join(
        test_folder,
        filename
    )

    gray = cv2.imread(
        path,
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
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    X_test.append(features)
    y_test.append(true_label)
    filenames.append(filename)


X_test = np.array(X_test)
y_test = np.array(y_test)

print(
    "Test samples:",
    len(X_test)
)


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

baseline_predictions = (
    baseline_model.predict(
        X_test
    )
)


# ============================================================
# PAIRWISE CORRECTION
# ============================================================

final_predictions = []
corrections = []

for i, (features, baseline_prediction) in enumerate(
    zip(X_test, baseline_predictions)
):

    candidate_predictions = []

    for pair, specialist in specialists.items():

        class_a, class_b = pair

        if baseline_prediction not in pair:
            continue

        specialist_prediction = specialist.predict(
            features.reshape(1, -1)
        )[0]

        candidate_predictions.append(
            (
                pair,
                specialist_prediction
            )
        )

    # --------------------------------------------------------
    # No specialist applies
    # --------------------------------------------------------

    if not candidate_predictions:

        final_prediction = baseline_prediction

        reason = "No specialist"

    else:

        specialist_values = [
            prediction
            for pair, prediction
            in candidate_predictions
        ]

        unique_predictions = set(
            specialist_values
        )

        # ----------------------------------------------------
        # Conservative rule:
        # Only change prediction if ALL applicable
        # specialists agree.
        # ----------------------------------------------------

        if len(unique_predictions) == 1:

            specialist_prediction = (
                specialist_values[0]
            )

            if (
                specialist_prediction
                != baseline_prediction
            ):

                final_prediction = (
                    specialist_prediction
                )

                reason = (
                    "Specialist correction"
                )

                corrections.append({
                    "filename": filenames[i],
                    "true": y_test[i],
                    "old": baseline_prediction,
                    "new": specialist_prediction,
                    "pairs": candidate_predictions
                })

            else:

                final_prediction = (
                    baseline_prediction
                )

                reason = "Specialists agree"

        else:

            final_prediction = (
                baseline_prediction
            )

            reason = (
                "Specialists disagree"
            )

    final_predictions.append(
        final_prediction
    )


final_predictions = np.array(
    final_predictions
)


# ============================================================
# BASELINE ACCURACY
# ============================================================

baseline_correct = np.sum(
    baseline_predictions == y_test
)

baseline_accuracy = (
    baseline_correct
    / len(y_test)
) * 100


# ============================================================
# PAIRWISE ACCURACY
# ============================================================

final_correct = np.sum(
    final_predictions == y_test
)

final_accuracy = (
    final_correct
    / len(y_test)
) * 100


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"Baseline SVM:          "
    f"{baseline_accuracy:.2f}% "
    f"({baseline_correct}/{len(y_test)})"
)

print(
    f"Pairwise specialists:  "
    f"{final_accuracy:.2f}% "
    f"({final_correct}/{len(y_test)})"
)

difference = (
    final_accuracy
    - baseline_accuracy
)

print(
    f"Difference:            "
    f"{difference:+.2f} percentage points"
)


# ============================================================
# CORRECTIONS
# ============================================================

print("\n" + "=" * 70)
print("SPECIALIST CORRECTIONS")
print("=" * 70)

if corrections:

    for correction in corrections:

        print(
            f"{correction['filename']}: "
            f"{correction['old']} -> "
            f"{correction['new']} "
            f"| true={correction['true']}"
        )

else:

    print("No predictions were changed.")


# ============================================================
# CORRECTION QUALITY
# ============================================================

print("\n" + "=" * 70)
print("CORRECTION QUALITY")
print("=" * 70)

helped = 0
hurt = 0
neutral = 0

for correction in corrections:

    true_label = correction["true"]
    old_prediction = correction["old"]
    new_prediction = correction["new"]

    old_correct = (
        old_prediction == true_label
    )

    new_correct = (
        new_prediction == true_label
    )

    if not old_correct and new_correct:

        helped += 1

        print(
            f"HELPED: {correction['filename']} "
            f"{old_prediction} -> {new_prediction}"
        )

    elif old_correct and not new_correct:

        hurt += 1

        print(
            f"HURT:   {correction['filename']} "
            f"{old_prediction} -> {new_prediction}"
        )

    else:

        neutral += 1


print(
    f"\nCorrections that helped: {helped}"
)

print(
    f"Corrections that hurt:   {hurt}"
)

print(
    f"Neutral corrections:     {neutral}"
)


# ============================================================
# FINAL ERRORS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MISCLASSIFICATIONS")
print("=" * 70)

for filename, true_label, prediction in zip(
    filenames,
    y_test,
    final_predictions
):

    if true_label != prediction:

        print(
            f"{filename}: "
            f"{true_label} -> {prediction}"
        )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)