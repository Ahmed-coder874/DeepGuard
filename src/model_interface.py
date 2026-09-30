"""
model_interface.py
==================
Future-facing model contract for DeepGuard.

This module describes how a *trained* model will be located and loaded. It is
deliberately written so that it can be used by the Streamlit app **before** any
model exists.

What it does
------------
    - Reports whether a trained checkpoint file is present on disk.
    - Describes the selected architecture and expected checkpoint location.
    - Refuses to load a model that does not exist, with a clear message.

What it does NOT do
-------------------
    - It does NOT contain a model.
    - It does NOT produce predictions.
    - It does NOT return random or placeholder probabilities.

No fake model and no fake inference are provided anywhere in this file. Real
inference is implemented in ``src/inference.py`` and is wired into the
Streamlit application (see ``app.py``). This module only reports whether a
checkpoint is present and where it lives.

The module avoids importing PyTorch at import time, so the Streamlit
application keeps working even though PyTorch is only installed in the
separate ``venv-train`` training environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Make the project root importable so that "import config" always works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

FRAMEWORK_NAME = "PyTorch (torch + torchvision)"


class ModelNotAvailableError(Exception):
    """Raised when inference is requested but no trained model exists."""


class InferenceNotImplementedError(Exception):
    """Raised when a checkpoint exists but inference is not wired up yet."""


# ---------------------------------------------------------------------------
# Checkpoint location
# ---------------------------------------------------------------------------
def get_checkpoint_path(
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
) -> Path:
    """Return the path where the best trained checkpoint is (or will be) saved."""
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else config.CHECKPOINT_DIR
    name = filename if filename is not None else config.CHECKPOINT_FILENAME
    return directory / name


def checkpoint_exists(
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
) -> bool:
    """Return True only if a real checkpoint file is present on disk."""
    return get_checkpoint_path(checkpoint_dir, filename).is_file()


def is_model_available(
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
) -> bool:
    """Return True if a trained checkpoint file exists."""
    return checkpoint_exists(checkpoint_dir, filename)


# ---------------------------------------------------------------------------
# Status reporting (used by the Streamlit app)
# ---------------------------------------------------------------------------
def get_model_status(
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
) -> dict:
    """
    Return a truthful description of the model situation.

    ``available`` is True only when a real checkpoint file exists. Nothing in
    the returned dictionary is a prediction or a performance metric.
    """
    path = get_checkpoint_path(checkpoint_dir, filename)
    exists = path.is_file()

    if exists:
        message = (
            "A trained checkpoint file was found. Inference is implemented "
            "(see `src/inference.py`) and the Streamlit application runs "
            "predictions from this checkpoint."
        )
    else:
        message = (
            "Model not available. No trained checkpoint exists yet, so no "
            "prediction can be produced."
        )

    return {
        "architecture": config.MODEL_NAME,
        "alternative_architecture": config.ALT_MODEL_NAME,
        "framework": FRAMEWORK_NAME,
        "num_classes": config.NUM_CLASSES,
        "class_names": list(config.CLASS_NAMES),
        "label_to_index": dict(config.LABEL_TO_INDEX),
        "checkpoint_path": str(path),
        "checkpoint_exists": exists,
        "available": exists,
        "message": message,
    }


def format_status_message(status: Optional[dict] = None) -> str:
    """Return a short, human-readable status line for display."""
    status = status if status is not None else get_model_status()
    return str(status.get("message", "Model not available."))


# ---------------------------------------------------------------------------
# Loading / inference (not implemented yet, on purpose)
# ---------------------------------------------------------------------------
def load_model(
    checkpoint_dir: Optional[Path | str] = None,
    filename: Optional[str] = None,
):
    """
    Load the trained model.

    Raises ``ModelNotAvailableError`` when no checkpoint exists. By design this
    module never loads a model itself: real loading lives in
    ``src.inference.load_trained_model`` (used by the Streamlit app). This
    function never returns a model, so it can never be mistaken for a real
    inference path.
    """
    if not is_model_available(checkpoint_dir, filename):
        raise ModelNotAvailableError(
            "Model not available: no trained checkpoint exists at "
            f"'{get_checkpoint_path(checkpoint_dir, filename)}'. "
            "Train the model first (see train_model.py)."
        )

    raise InferenceNotImplementedError(
        "Loading is deliberately not implemented in this module. Real loading "
        "and inference are provided by src.inference.load_trained_model."
    )


def predict(*_args, **_kwargs):
    """
    Placeholder that never returns a value.

    This function intentionally raises instead of returning any value, so that
    it can never be mistaken for a real prediction. Real single-image
    prediction is provided by ``src.inference.predict_image``.
    """
    raise ModelNotAvailableError(
        "Prediction is not available through this module. DeepGuard runs real "
        "inference via src.inference.predict_image (see app.py)."
    )
