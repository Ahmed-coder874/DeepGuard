"""
preprocessing.py
================
Image preprocessing for the DeepGuard prototype.

This module turns raw uploaded bytes into a clean, model-ready NumPy array.
It performs only the classical image-processing steps that every image
classifier needs:

    1. Decode the file into an image.
    2. Convert the colour order from OpenCV's BGR to the usual RGB.
    3. Resize the image to 224 x 224 pixels.
    4. Normalise pixel values from 0-255 to the range 0-1.
    5. Add a batch dimension so the array has the shape a CNN expects.

IMPORTANT: This module only prepares the display array (values 0-1). It does
NOT run a model. The trained network is fed the separate normalised
(1, 3, 224, 224) tensor built by ``src.inference.preprocess_for_inference``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# Every image is standardised to this size before it could be given to a model.
# OpenCV expects (width, height).
TARGET_SIZE: Tuple[int, int] = (224, 224)


class PreprocessingError(Exception):
    """Raised when an image cannot be preprocessed."""


# ---------------------------------------------------------------------------
# Individual steps (each one is small and easy to test on its own)
# ---------------------------------------------------------------------------
def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw file bytes into an image array.

    OpenCV loads images in BGR colour order by default, so the returned array
    is in BGR order and has shape (height, width, 3).
    """
    if not image_bytes:
        raise PreprocessingError("No image data was provided.")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise PreprocessingError("OpenCV could not decode this image.")

    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from BGR colour order (OpenCV) to RGB colour order."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = TARGET_SIZE,
) -> np.ndarray:
    """Resize an image to the given (width, height) using high-quality area interpolation."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale pixel values from the 0-255 range to the 0-1 range (float32)."""
    return image.astype(np.float32) / 255.0


def prepare_model_input(normalized: np.ndarray) -> np.ndarray:
    """
    Add a batch dimension to a normalised image.

    A convolutional neural network normally expects a batch of images, so a
    single (224, 224, 3) image becomes (1, 224, 224, 3). This array is the
    display form (values 0-1); the trained model receives the normalised
    (1, 3, 224, 224) tensor from ``src.inference.preprocess_for_inference``.
    """
    return np.expand_dims(normalized, axis=0)


# ---------------------------------------------------------------------------
# Main entry point used by the app
# ---------------------------------------------------------------------------
def preprocess_image(
    image_bytes: bytes,
    size: Tuple[int, int] = TARGET_SIZE,
) -> Dict[str, object]:
    """
    Run the full preprocessing pipeline on raw image bytes.

    Returns a dictionary containing the intermediate results, which is used for
    display. The trained model is fed the normalised tensor built by
    ``src.inference.preprocess_for_inference`` (not this display array).
    """
    try:
        bgr = decode_image(image_bytes)
        original_height, original_width = bgr.shape[:2]

        rgb = bgr_to_rgb(bgr)
        resized = resize_image(rgb, size)
        normalized = normalize_image(resized)
        model_input = prepare_model_input(normalized)

    except PreprocessingError:
        # Already a friendly error - pass it straight through.
        raise
    except cv2.error as error:
        raise PreprocessingError(
            f"OpenCV reported an error while processing the image: {error}"
        )
    except Exception as error:  # Last-resort safety net so the app never crashes.
        raise PreprocessingError(
            f"Unexpected error while preprocessing the image: {error}"
        )

    return {
        "original_bgr": bgr,
        "original_size": (original_width, original_height),
        "rgb": rgb,
        "resized_rgb": resized,
        "normalized": normalized,
        "model_input": model_input,
        "processed_size": (size[0], size[1]),
        "value_range": (float(normalized.min()), float(normalized.max())),
        "model_input_shape": tuple(model_input.shape),
    }
