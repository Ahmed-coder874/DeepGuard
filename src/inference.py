"""
inference.py
============
Stage 6 inference module for DeepGuard.

This module provides the functions that WILL run a trained EfficientNet-B0
checkpoint on a single image and return a real/fake prediction with model
confidence.

IMPORTANT - HONESTY RULES
-------------------------
1. A prediction is only produced by a REAL trained checkpoint produced by
   ``train_model.py``. There is no random result, no placeholder probability
   and no dummy classifier anywhere in this module.
2. If any requirement is missing, the module raises
   ``InferenceNotAvailableError`` with a clear, user-friendly message:
       * PyTorch is not installed in the running environment, OR
       * no trained checkpoint file exists, OR
       * the checkpoint is corrupted, OR
       * the checkpoint does not match the expected architecture.
3. The module never claims that a deepfake prediction is "certain". The value
   returned is called **model confidence** and is exactly what the model
   produced.

Preprocessing here must match the training pipeline exactly:
    224 x 224 via random-free evaluation transforms (resize + centre-crop),
    tensor conversion and ImageNet mean/std normalisation. The output tensor
    has shape (1, 3, 224, 224).

Class mapping (as in ``config.LABEL_TO_INDEX``):
    real = 0
    fake = 1

The module does not import PyTorch at module level, so the application can
keep working even in an environment without PyTorch (such as ``venv``). Every
function that needs PyTorch raises a friendly error when it is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make the project root importable so that "import config" always works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import model_interface, training  # noqa: E402

# PyTorch availability is shared with the training module, which already
# guards the import.
TORCH_AVAILABLE = training.TORCH_AVAILABLE


class InferenceNotAvailableError(Exception):
    """Raised when a real prediction cannot be produced."""


def _torch_explanation() -> str:
    return (
        "PyTorch is not installed in the environment running this application. "
        "Install the application dependencies (requirements.txt, which includes "
        "the CPU build of torch and torchvision) so that the trained model can "
        "be loaded. The separate training environment (venv-train) is not "
        "required to run the Streamlit application."
    )


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def inference_available(
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
) -> bool:
    """
    Return True only when a REAL prediction is possible:
        - PyTorch is importable, AND
        - a trained checkpoint file exists on disk.
    """
    if not TORCH_AVAILABLE:
        return False
    return model_interface.is_model_available(checkpoint_dir, filename)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_trained_model(
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
    device=None,
):
    """
    Load the real trained model and its checkpoint metadata.

    Returns a tuple ``(model, checkpoint)``. Raises
    ``InferenceNotAvailableError`` when PyTorch is missing, when no checkpoint
    exists, or when the checkpoint is corrupted or incompatible.

    The architecture is rebuilt with the code used during training (`from
    src.training import build_model`) and never downloads weights: parameters
    come exclusively from the checkpoint.
    """
    if not TORCH_AVAILABLE:
        raise InferenceNotAvailableError(_torch_explanation())

    import torch  # noqa: E402

    if not model_interface.is_model_available(checkpoint_dir, filename):
        path = model_interface.get_checkpoint_path(checkpoint_dir, filename)
        raise InferenceNotAvailableError(
            f"No trained checkpoint was found at '{path}'.\n\n"
            "Place the model file at:\n"
            f"    <project root>{os.sep}models{os.sep}{config.CHECKPOINT_FILENAME}\n\n"
            "The Streamlit application resolves this path relative to the "
            "project folder, so a fresh `git clone` needs no extra "
            "configuration. Only the application environment (`venv`) is "
            "required to load it; the training environment (`venv-train`) is "
            "not needed to run the app."
        )

    path = model_interface.get_checkpoint_path(checkpoint_dir, filename)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001
        raise InferenceNotAvailableError(
            "The checkpoint file could not be read. It may be corrupted or "
            "produced by an incompatible version of the tools."
        ) from error

    if "model_state_dict" not in checkpoint:
        raise InferenceNotAvailableError(
            "The checkpoint does not contain model weights. It cannot be used "
            "for prediction."
        )

    model_name = str(checkpoint.get("model_name", config.MODEL_NAME))
    model = training.build_model(model_name=model_name, pretrained=False)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as error:  # noqa: BLE001
        raise InferenceNotAvailableError(
            "The checkpoint is incompatible with the expected model "
            "architecture and cannot be loaded."
        ) from error

    if device is None:
        device = training.get_device()
    model.to(device)
    model.eval()

    return model, checkpoint


# ---------------------------------------------------------------------------
# Inference preprocessing
# ---------------------------------------------------------------------------
def preprocess_for_inference(
    image,  # PIL.Image or RGB ndarray of shape (H, W, 3)
    device=None,
):
    """
    Convert one RGB image into the exact model input.

    The prepared tensor has shape (1, 3, 224, 224) with ImageNet mean/std
    normalisation and lives on the given device (CPU by default).

    NOTE: this is different from the display array produced by Stage 1
    preprocessing (which stays in 0-1 for display). The trained model receives
    this normalised tensor instead.
    """
    if not TORCH_AVAILABLE:
        raise InferenceNotAvailableError(_torch_explanation())

    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    if not isinstance(image, Image.Image):
        raise ValueError(
            "preprocess_for_inference expects a PIL Image or an (H, W, 3) "
            "RGB NumPy array."
        )

    image = image.convert("RGB")
    transform = training.get_eval_transforms()
    tensor = transform(image).unsqueeze(0)  # -> (1, 3, 224, 224)

    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_image(
    image,  # PIL.Image or RGB ndarray of shape (H, W, 3)
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
    device=None,
) -> dict:
    """
    Run real inference on one image and return actual model output.

    Returns a dictionary with:
        prediction_index      - 0 (real) or 1 (fake), exactly as mapped
        prediction_class      - "real" or "fake"
        confidence            - probability of the predicted class (0..1)
        probabilities         - {"real": p_real, "fake": p_fake}, p_real+p_fake=1

    No value is rounded before the internal calculation. Values are returned
    at full precision; rounding (if any) is only a display concern.

    Raises ``InferenceNotAvailableError`` if a real prediction is impossible.
    """
    if not TORCH_AVAILABLE:
        raise InferenceNotAvailableError(_torch_explanation())

    import torch  # noqa: E402
    from torch.nn import functional as F  # noqa: E402

    model, _checkpoint = load_trained_model(checkpoint_dir, filename, device)
    device = next(model.parameters()).device
    tensor = preprocess_for_inference(image, device=device)

    with torch.no_grad():
        logits = model(tensor)  # (1, num_classes)
    probabilities = F.softmax(logits, dim=1)[0]  # (num_classes,)

    confidence, prediction_index = probabilities.max(dim=0)
    prediction_index = int(prediction_index.item())
    confidence = float(confidence.item())

    return {
        "prediction_index": prediction_index,
        "prediction_class": config.INDEX_TO_LABEL[prediction_index],
        "confidence": confidence,
        "probabilities": {
            name: float(probabilities[config.LABEL_TO_INDEX[name]].item())
            for name in config.CLASS_NAMES
        },
        "label_to_index": dict(config.LABEL_TO_INDEX),
        "class_names": list(config.CLASS_NAMES),
        "checkpoint_path": str(model_interface.get_checkpoint_path(checkpoint_dir, filename)),
    }