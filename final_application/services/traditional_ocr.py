import os
import importlib.util

import cv2
import joblib
from skimage.feature import hog


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "handwriting",
    "traditional",
    "handwriting_svm.joblib"
)


PREPROCESS_PATH = os.path.join(
    BASE_DIR,
    "handwriting",
    "traditional",
    "preprocess.py"
)


# --------------------------------------------------
# Load the existing preprocessing module
# --------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "traditional_preprocess",
    PREPROCESS_PATH
)

preprocess_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    preprocess_module
)

process_image = preprocess_module.process_image


# --------------------------------------------------
# Image preparation
# --------------------------------------------------

def prepare_image(image):
    """
    Prepare an image for HOG feature extraction.

    28x28 images are treated as already-preprocessed.
    Other image sizes use the finalized preprocessing pipeline.
    """

    if image.shape == (28, 28):
        return image

    return process_image(image)


# --------------------------------------------------
# HOG feature extraction
# --------------------------------------------------

def extract_features(image):
    """Extract the finalized HOG features."""

    normalized = image / 255.0

    features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)


# --------------------------------------------------
# Character prediction
# --------------------------------------------------

def predict_character(image_path):
    """
    Predict a handwritten character from an image.

    Returns:
        str: Predicted character.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Traditional OCR model not found: {MODEL_PATH}"
        )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    prepared = prepare_image(image)

    if prepared is None:
        raise ValueError(
            "No handwritten character detected."
        )

    features = extract_features(prepared)

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(features)[0]

    return str(prediction)