import os
import sys

import cv2
import joblib
import numpy as np
from skimage.feature import hog

from preprocess import process_image


MODEL_PATH = "handwriting_svm.joblib"


def prepare_image(image):
    """
    Prepare an image for HOG extraction.

    28x28 images are treated as already-preprocessed images.
    Other image sizes go through the finalized preprocessing pipeline.
    """

    if image.shape == (28, 28):
        return image

    return process_image(image)


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


def predict_character(image_path):

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found: {MODEL_PATH}")
        return

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    prepared = prepare_image(image)

    if prepared is None:
        print("Error: No handwritten character detected.")
        return

    features = extract_features(prepared)

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(features)[0]

    print("=" * 50)
    print("HANDWRITING OCR")
    print("=" * 50)
    print(f"Image      : {image_path}")
    print(f"Prediction : {prediction}")
    print("=" * 50)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python predict.py <image_path>")
        sys.exit(1)

    predict_character(sys.argv[1])