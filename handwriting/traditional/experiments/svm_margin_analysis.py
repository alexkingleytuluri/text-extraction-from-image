import os
import cv2
import numpy as np

from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog


# ============================================================
# TRAIN USING EXACT CURRENT PIPELINE
# ============================================================

print("=" * 70)
print("SVM MARGIN / CONFIDENCE ANALYSIS")
print("=" * 70)

print("\nLoading training data...")

X_train, y_train = load_dataset("data_clean")

print("Training samples:", len(X_train))
print("Feature size:", X_train.shape[1])


print("\nTraining SVM...")

model = SVC(
    kernel="rbf",
    C=50.0,
    gamma=0.01,
    decision_function_shape="ovr"
)

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# LOAD TEST DATA
# ============================================================

test_folder = "test_clean"

X_test = []
y_test = []
filenames = []


print("\nLoading external test data...")

files = sorted(
    os.listdir(test_folder),
    key=lambda x: (not x[0].isdigit(), x)
)

for filename in files:

    if not filename.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        continue

    true_label = os.path.splitext(filename)[0]

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
# PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

decision_scores = model.decision_function(X_test)

classes = model.classes_


# ============================================================
# VERIFY BASELINE
# ============================================================

correct = np.sum(
    predictions == y_test
)

total = len(y_test)

accuracy = (
    correct / total
) * 100


print("\n" + "=" * 70)
print("BASELINE VERIFICATION")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print(
    f"Correct: {correct}/{total}"
)


# ============================================================
# TOP-2 ANALYSIS
# ============================================================

results = []

for i in range(len(X_test)):

    scores = decision_scores[i]

    order = np.argsort(
        scores
    )[::-1]

    top1_index = order[0]
    top2_index = order[1]

    top1 = classes[top1_index]
    top2 = classes[top2_index]

    top1_score = scores[top1_index]
    top2_score = scores[top2_index]

    margin = (
        top1_score
        - top2_score
    )

    results.append({
        "filename": filenames[i],
        "true": y_test[i],
        "prediction": predictions[i],
        "top1": top1,
        "top2": top2,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
        "correct": predictions[i] == y_test[i]
    })


# ============================================================
# ALL SAMPLES SORTED BY MARGIN
# ============================================================

results_sorted = sorted(
    results,
    key=lambda x: x["margin"]
)


print("\n" + "=" * 70)
print("ALL TEST SAMPLES — LOWEST MARGIN FIRST")
print("=" * 70)

print(
    "\nFile         True  Pred  Competitor  "
    "Top1Score  Top2Score  Margin  Status"
)

print("-" * 90)

for r in results_sorted:

    status = (
        "CORRECT"
        if r["correct"]
        else "ERROR"
    )

    print(
        f"{r['filename']:<12} "
        f"{r['true']:<5} "
        f"{r['prediction']:<5} "
        f"{r['top2']:<11} "
        f"{r['top1_score']:>9.3f}  "
        f"{r['top2_score']:>9.3f}  "
        f"{r['margin']:>6.3f}  "
        f"{status}"
    )


# ============================================================
# MARGIN STATISTICS
# ============================================================

errors = [
    r for r in results
    if not r["correct"]
]

correct_predictions = [
    r for r in results
    if r["correct"]
]


print("\n" + "=" * 70)
print("MARGIN STATISTICS")
print("=" * 70)

print(
    f"Correct predictions: "
    f"{len(correct_predictions)}"
)

print(
    f"Incorrect predictions: "
    f"{len(errors)}"
)


if correct_predictions:

    correct_margins = [
        r["margin"]
        for r in correct_predictions
    ]

    print(
        f"\nCorrect average margin: "
        f"{np.mean(correct_margins):.3f}"
    )

    print(
        f"Correct minimum margin: "
        f"{np.min(correct_margins):.3f}"
    )

    print(
        f"Correct maximum margin: "
        f"{np.max(correct_margins):.3f}"
    )


if errors:

    error_margins = [
        r["margin"]
        for r in errors
    ]

    print(
        f"\nError average margin: "
        f"{np.mean(error_margins):.3f}"
    )

    print(
        f"Error minimum margin: "
        f"{np.min(error_margins):.3f}"
    )

    print(
        f"Error maximum margin: "
        f"{np.max(error_margins):.3f}"
    )


# ============================================================
# ERROR DETAILS
# ============================================================

print("\n" + "=" * 70)
print("ERRORS WITH THEIR COMPETING CLASS")
print("=" * 70)

for r in results_sorted:

    if not r["correct"]:

        print(
            f"{r['filename']}: "
            f"{r['true']} -> {r['prediction']} "
            f"| competitor={r['top2']} "
            f"| margin={r['margin']:.3f}"
        )


# ============================================================
# HOW MANY ERRORS ARE LOW-MARGIN?
# ============================================================

print("\n" + "=" * 70)
print("LOW-MARGIN ERROR ANALYSIS")
print("=" * 70)

thresholds = [
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00
]

for threshold in thresholds:

    low_margin_errors = [
        r for r in errors
        if r["margin"] <= threshold
    ]

    low_margin_correct = [
        r for r in correct_predictions
        if r["margin"] <= threshold
    ]

    total_low_margin = (
        len(low_margin_errors)
        + len(low_margin_correct)
    )

    print(
        f"\nMargin <= {threshold:.2f}"
    )

    print(
        f"  Errors:  "
        f"{len(low_margin_errors)}/{len(errors)}"
    )

    print(
        f"  Correct: "
        f"{len(low_margin_correct)}/{len(correct_predictions)}"
    )

    if total_low_margin > 0:

        error_rate = (
            len(low_margin_errors)
            / total_low_margin
        ) * 100

        print(
            f"  Error rate among these: "
            f"{error_rate:.2f}%"
        )


# ============================================================
# CONFUSION PAIRS
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION PAIRS")
print("=" * 70)

pairs = {}

for r in errors:

    pair = (
        r["true"],
        r["prediction"]
    )

    pairs[pair] = (
        pairs.get(pair, 0) + 1
    )


for pair, count in sorted(
    pairs.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{pair[0]} -> {pair[1]} : {count}"
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)