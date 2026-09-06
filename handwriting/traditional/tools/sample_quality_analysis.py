import os
import cv2
import numpy as np

from sklearn.metrics import pairwise_distances
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


def extract_hog(img):

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


def load_training_data():

    images = []
    labels = []
    filenames = []

    for folder_name in sorted(
        os.listdir("data")
    ):

        folder_path = os.path.join(
            "data",
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

            feature = extract_hog(processed)

            images.append(processed)
            labels.append(label)
            filenames.append(
                os.path.join(
                    folder_name,
                    filename
                )
            )

    return (
        np.array(images),
        np.array(labels),
        filenames
    )


print("=" * 70)
print("TRAINING SAMPLE QUALITY ANALYSIS")
print("=" * 70)

images, labels, filenames = load_training_data()

print("\nTraining images loaded:", len(images))
print("HOG feature size:", len(extract_hog(images[0])))

print("\nCalculating pairwise HOG distances...")

features = np.array([
    extract_hog(image)
    for image in images
])

distances = pairwise_distances(
    features,
    metric="euclidean"
)

# Ignore self-distance
np.fill_diagonal(
    distances,
    np.inf
)


# --------------------------------------------------
# 1. WITHIN-CLASS CONSISTENCY
# --------------------------------------------------

print("\n" + "=" * 70)
print("1. WITHIN-CLASS CONSISTENCY")
print("=" * 70)

class_results = []

for label in sorted(
    np.unique(labels)
):

    indices = np.where(
        labels == label
    )[0]

    class_distances = distances[
        np.ix_(indices, indices)
    ]

    # Ignore diagonal
    np.fill_diagonal(
        class_distances,
        np.nan
    )

    mean_distance = np.nanmean(
        class_distances
    )

    class_results.append(
        (label, mean_distance)
    )

print("\nClasses with MOST variation:")
print("(Higher distance = training samples look less alike)\n")

for label, distance in sorted(
    class_results,
    key=lambda x: x[1],
    reverse=True
)[:15]:

    print(
        f"{label:>2} : "
        f"{distance:.4f}"
    )


print("\nClasses with LEAST variation:")
print("(Lower distance = training samples look more alike)\n")

for label, distance in sorted(
    class_results,
    key=lambda x: x[1]
)[:15]:

    print(
        f"{label:>2} : "
        f"{distance:.4f}"
    )


# --------------------------------------------------
# 2. HARDEST INDIVIDUAL TRAINING SAMPLES
# --------------------------------------------------

print("\n" + "=" * 70)
print("2. HARDEST INDIVIDUAL TRAINING SAMPLES")
print("=" * 70)

sample_scores = []

for i in range(len(images)):

    same_class_indices = np.where(
        labels == labels[i]
    )[0]

    same_class_indices = (
        same_class_indices[
            same_class_indices != i
        ]
    )

    if len(same_class_indices) == 0:
        continue

    nearest_same_class = np.min(
        distances[
            i,
            same_class_indices
        ]
    )

    sample_scores.append(
        (
            nearest_same_class,
            labels[i],
            filenames[i]
        )
    )

print(
    "\nSamples whose nearest same-class example "
    "is FAR away:"
)

for distance, label, filename in sorted(
    sample_scores,
    reverse=True
)[:30]:

    print(
        f"{label:>2} | "
        f"{distance:.4f} | "
        f"{filename}"
    )


# --------------------------------------------------
# 3. CROSS-CLASS CONFUSION POTENTIAL
# --------------------------------------------------

print("\n" + "=" * 70)
print("3. CROSS-CLASS NEAREST NEIGHBORS")
print("=" * 70)

cross_class_results = []

for i in range(len(images)):

    different_class_indices = np.where(
        labels != labels[i]
    )[0]

    nearest_index = (
        different_class_indices[
            np.argmin(
                distances[
                    i,
                    different_class_indices
                ]
            )
        ]
    )

    cross_distance = distances[
        i,
        nearest_index
    ]

    cross_class_results.append(
        (
            cross_distance,
            labels[i],
            labels[nearest_index],
            filenames[i],
            filenames[nearest_index]
        )
    )

print(
    "\nMost visually similar samples "
    "from DIFFERENT classes:"
)

for (
    distance,
    true_label,
    other_label,
    filename,
    other_filename
) in sorted(
    cross_class_results
)[:40]:

    print(
        f"{true_label:>2} ~ {other_label:<2} | "
        f"{distance:.4f} | "
        f"{filename} <-> {other_filename}"
    )


# --------------------------------------------------
# 4. PROBLEMATIC CLASS ANALYSIS
# --------------------------------------------------

problematic_classes = [
    "0", "1", "2", "5", "6", "7", "8",
    "a", "c", "e", "f", "g", "h",
    "j", "k", "l", "s", "t", "u", "v"
]

print("\n" + "=" * 70)
print("4. KNOWN PROBLEMATIC CLASSES")
print("=" * 70)

print(
    "\nThese are classes that appeared in "
    "our external-test errors.\n"
)

for label in problematic_classes:

    if label not in labels:
        continue

    indices = np.where(
        labels == label
    )[0]

    class_distances = distances[
        np.ix_(indices, indices)
    ]

    np.fill_diagonal(
        class_distances,
        np.nan
    )

    mean_distance = np.nanmean(
        class_distances
    )

    max_distance = np.nanmax(
        class_distances
    )

    min_distance = np.nanmin(
        class_distances
    )

    print(
        f"{label:>2} | "
        f"mean={mean_distance:.4f} | "
        f"min={min_distance:.4f} | "
        f"max={max_distance:.4f}"
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)