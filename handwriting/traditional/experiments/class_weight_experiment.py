import os
import cv2
import numpy as np

from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog


# ============================================================
# CONFIGURATION
# ============================================================

DIFFICULT_CLASSES = {
    "0", "1", "2", "5", "6", "7", "8",
    "a", "c", "e", "f", "g", "h", "j",
    "k", "l", "s", "t", "u", "v"
}


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("CLASS WEIGHTING EXPERIMENT")
print("=" * 70)

print("\nLoading training data...")

X_train, y_train = load_dataset("data_clean")

# Normalize all labels to strings
y_train = np.array([
    str(label)
    for label in y_train
])

print("Training samples:", len(X_train))
print("Feature size:", X_train.shape[1])


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading external test data...")

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

    true_label = os.path.splitext(filename)[0]

    # Convert s-a -> a
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
# CREATE SAMPLE WEIGHTS
# ============================================================

def create_sample_weights(labels, multiplier):
    """
    Give difficult classes extra importance.
    Normal classes receive weight 1.0.
    """

    weights = np.ones(
        len(labels),
        dtype=float
    )

    for i, label in enumerate(labels):

        if label in DIFFICULT_CLASSES:
            weights[i] = multiplier

    return weights


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    strategy_name,
    sample_weights=None
):

    print("\n" + "=" * 70)
    print(
        f"STRATEGY: {strategy_name}"
    )
    print("=" * 70)

    model = SVC(
        kernel="rbf",
        C=50.0,
        gamma=0.01
    )

    print("Training...")

    if sample_weights is None:

        model.fit(
            X_train,
            y_train
        )

    else:

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )

    predictions = model.predict(
        X_test
    )

    correct = np.sum(
        predictions == y_test
    )

    accuracy = (
        correct / len(y_test)
    ) * 100

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print(
        f"Correct: "
        f"{correct}/{len(y_test)}"
    )

    print("\nCategory accuracy:")

    for category in [
        "digits",
        "uppercase",
        "lowercase"
    ]:

        if category == "digits":

            mask = np.array([
                label.isdigit()
                for label in y_test
            ])

        elif category == "uppercase":

            mask = np.array([
                label.isupper()
                for label in y_test
            ])

        else:

            mask = np.array([
                label.islower()
                for label in y_test
            ])

        category_correct = np.sum(
            predictions[mask]
            == y_test[mask]
        )

        category_total = np.sum(mask)

        category_accuracy = (
            category_correct
            / category_total
        ) * 100

        print(
            f"{category:<10}: "
            f"{category_correct}/"
            f"{category_total} "
            f"({category_accuracy:.2f}%)"
        )

    return accuracy, predictions


# ============================================================
# RUN ALL STRATEGIES
# ============================================================

results = {}


# ------------------------------------------------------------
# 1. BASELINE
# ------------------------------------------------------------

results["baseline"] = run_experiment(
    "baseline"
)


# ------------------------------------------------------------
# 2. BALANCED
# ------------------------------------------------------------

# Every class already has exactly 10 original images,
# so balanced weighting is expected to behave like baseline.

balanced_weights = np.ones(
    len(y_train),
    dtype=float
)

results["balanced"] = run_experiment(
    "balanced",
    balanced_weights
)


# ------------------------------------------------------------
# 3. DIFFICULT CLASSES × 1.5
# ------------------------------------------------------------

weights_1_5 = create_sample_weights(
    y_train,
    1.5
)

results["difficult_x1.5"] = run_experiment(
    "difficult_x1.5",
    weights_1_5
)


# ------------------------------------------------------------
# 4. DIFFICULT CLASSES × 2
# ------------------------------------------------------------

weights_2 = create_sample_weights(
    y_train,
    2.0
)

results["difficult_x2"] = run_experiment(
    "difficult_x2",
    weights_2
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

baseline_accuracy = results[
    "baseline"
][0]

for strategy_name, (
    accuracy,
    predictions
) in results.items():

    difference = (
        accuracy
        - baseline_accuracy
    )

    print(
        f"{strategy_name:<20} "
        f"{accuracy:>6.2f}% "
        f"({difference:+.2f} pp)"
    )


# ============================================================
# BEST STRATEGY
# ============================================================

best_strategy = max(
    results,
    key=lambda name: results[name][0]
)

best_accuracy = results[
    best_strategy
][0]

print("\n" + "=" * 70)
print("BEST STRATEGY")
print("=" * 70)

print(
    f"{best_strategy}: "
    f"{best_accuracy:.2f}%"
)

print(
    f"Baseline: "
    f"{baseline_accuracy:.2f}%"
)

print(
    f"Improvement: "
    f"{best_accuracy - baseline_accuracy:+.2f} pp"
)


# ============================================================
# CHANGED PREDICTIONS
# ============================================================

if best_strategy != "baseline":

    best_predictions = results[
        best_strategy
    ][1]

    baseline_predictions = results[
        "baseline"
    ][1]

    print("\n" + "=" * 70)
    print("CHANGED PREDICTIONS")
    print("=" * 70)

    changed = 0
    helped = 0
    hurt = 0

    for (
        filename,
        true_label,
        old,
        new
    ) in zip(
        filenames,
        y_test,
        baseline_predictions,
        best_predictions
    ):

        if old != new:

            changed += 1

            if (
                new == true_label
                and old != true_label
            ):

                status = "HELPED"
                helped += 1

            elif (
                old == true_label
                and new != true_label
            ):

                status = "HURT"
                hurt += 1

            else:

                status = (
                    "NO CHANGE IN CORRECTNESS"
                )

            print(
                f"{filename}: "
                f"{true_label} | "
                f"{old} -> {new} | "
                f"{status}"
            )

    print(
        f"\nTotal changed predictions: "
        f"{changed}"
    )

    print(
        f"Helped: {helped}"
    )

    print(
        f"Hurt: {hurt}"
    )

else:

    print(
        "\nBest strategy is the baseline."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)