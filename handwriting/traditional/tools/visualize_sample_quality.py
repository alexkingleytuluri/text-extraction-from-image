import os
import cv2
import numpy as np

from skimage.feature import hog
from sklearn.metrics import pairwise_distances


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

    return canvas


def extract_hog(img):

    return hog(
        img / 255.0,
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


def make_contact_sheet(
    items,
    output_path,
    title,
    columns=5
):

    if not items:
        return

    cell_width = 180
    cell_height = 180

    rows = int(
        np.ceil(
            len(items) / columns
        )
    )

    sheet = np.ones(
        (
            rows * cell_height,
            columns * cell_width,
            3
        ),
        dtype=np.uint8
    ) * 255

    for index, item in enumerate(items):

        image = item["image"]

        image = cv2.resize(
            image,
            (112, 112),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert binary image to display format
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

        # Invert so ink appears dark
        image = 255 - image

        x = (
            index % columns
        ) * cell_width

        y = (
            index // columns
        ) * cell_height

        sheet[
            y + 10:y + 122,
            x + 34:x + 146
        ] = image

        lines = item["text"].split("\n")

        for line_index, line in enumerate(lines):

            cv2.putText(
                sheet,
                line,
                (
                    x + 5,
                    y + 142 + line_index * 17
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        sheet
    )

    print(
        "Saved:",
        output_path
    )


print("=" * 70)
print("VISUAL SAMPLE QUALITY ANALYSIS")
print("=" * 70)


images, labels, filenames = (
    load_training_data()
)

print(
    "\nTraining images loaded:",
    len(images)
)


features = np.array([
    extract_hog(image)
    for image in images
])

print(
    "Calculating HOG distances..."
)

distances = pairwise_distances(
    features,
    metric="euclidean"
)

np.fill_diagonal(
    distances,
    np.inf
)


# ==================================================
# 1. HARDEST SAME-CLASS SAMPLES
# ==================================================

same_class_candidates = []

for i in range(len(images)):

    same_indices = np.where(
        labels == labels[i]
    )[0]

    same_indices = (
        same_indices[
            same_indices != i
        ]
    )

    if len(same_indices) == 0:
        continue

    nearest_distance = np.min(
        distances[
            i,
            same_indices
        ]
    )

    same_class_candidates.append(
        (
            nearest_distance,
            i
        )
    )


same_class_candidates.sort(
    reverse=True
)


hardest_items = []

for distance, index in same_class_candidates[:30]:

    hardest_items.append(
        {
            "image": images[index],
            "text": (
                f"Class: {labels[index]}\n"
                f"Distance: {distance:.2f}\n"
                f"{filenames[index]}"
            )
        }
    )


# ==================================================
# 2. CROSS-CLASS LOOKALIKES
# ==================================================

cross_candidates = []

for i in range(len(images)):

    different_indices = np.where(
        labels != labels[i]
    )[0]

    nearest_index = (
        different_indices[
            np.argmin(
                distances[
                    i,
                    different_indices
                ]
            )
        ]
    )

    cross_distance = distances[
        i,
        nearest_index
    ]

    cross_candidates.append(
        (
            cross_distance,
            i,
            nearest_index
        )
    )


cross_candidates.sort()


cross_items = []

for distance, index, other_index in (
    cross_candidates[:30]
):

    cross_items.append(
        {
            "image": images[index],
            "text": (
                f"{labels[index]} ~ "
                f"{labels[other_index]}\n"
                f"Distance: {distance:.2f}\n"
                f"{filenames[index]}"
            )
        }
    )


# ==================================================
# 3. SPECIFIC PROBLEMATIC CLASSES
# ==================================================

problematic_classes = [
    "0",
    "1",
    "2",
    "5",
    "6",
    "7",
    "8",
    "a",
    "c",
    "e",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "s",
    "t",
    "u",
    "v"
]


problem_items = []

for label in problematic_classes:

    indices = np.where(
        labels == label
    )[0]

    # Calculate average distance
    # to other samples of same class

    scores = []

    for index in indices:

        other_indices = (
            indices[
                indices != index
            ]
        )

        if len(other_indices) == 0:
            continue

        score = np.mean(
            distances[
                index,
                other_indices
            ]
        )

        scores.append(
            (
                score,
                index
            )
        )

    scores.sort(
        reverse=True
    )

    # Take the hardest sample
    if scores:

        score, index = scores[0]

        problem_items.append(
            {
                "image": images[index],
                "text": (
                    f"Class: {label}\n"
                    f"Mean distance: {score:.2f}\n"
                    f"{filenames[index]}"
                )
            }
        )


# ==================================================
# SAVE
# ==================================================

os.makedirs(
    "sample_quality",
    exist_ok=True
)

make_contact_sheet(
    hardest_items,
    "sample_quality/hardest_same_class.png",
    "Hardest same-class samples"
)

make_contact_sheet(
    cross_items,
    "sample_quality/cross_class_confusions.png",
    "Cross-class lookalikes"
)

make_contact_sheet(
    problem_items,
    "sample_quality/problematic_classes.png",
    "Problematic classes"
)


print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "\nOpen these three files:"
)

print(
    "sample_quality/hardest_same_class.png"
)

print(
    "sample_quality/cross_class_confusions.png"
)

print(
    "sample_quality/problematic_classes.png"
)