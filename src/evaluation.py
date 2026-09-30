"""
evaluation.py
=============
Stage 5 evaluation pipeline for DeepGuard.

This module evaluates a *trained* checkpoint on the held-out **test** split.
The test split is never used during training, so it gives an honest estimate
of performance.

It computes the standard classification metrics required by the project:

    - accuracy
    - precision, recall and F1-score (per class and macro-averaged)
    - the confusion matrix

IMPORTANT
    This module does not invent anything. If there is no trained checkpoint,
    or no test images, it refuses to run and raises a clear error. It never
    returns placeholder metrics. Every number it reports is computed from a
    real model evaluated on real images.

    The metric helper functions (``compute_confusion_matrix`` and
    ``compute_metrics``) are plain Python and can be used even when PyTorch is
    not installed. Only ``load_checkpoint``, ``load_trained_model`` and
    ``evaluate`` need PyTorch.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Make the project root importable so that "import config" always works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import dataset as dataset_utils  # noqa: E402
from src import training  # noqa: E402


class EvaluationNotAvailableError(Exception):
    """Raised when evaluation cannot run because a requirement is missing."""


# ---------------------------------------------------------------------------
# Pure metric functions (no PyTorch required)
# ---------------------------------------------------------------------------
def compute_confusion_matrix(
    labels: Sequence[int],
    predictions: Sequence[int],
    num_classes: int = config.NUM_CLASSES,
) -> List[List[int]]:
    """
    Build a confusion matrix.

    Rows are the actual class, columns are the predicted class, so
    ``matrix[actual][predicted]`` is the number of images counted there.
    """
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for actual, predicted in zip(labels, predictions):
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            matrix[actual][predicted] += 1
    return matrix


def compute_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str] = config.CLASS_NAMES,
) -> Dict[str, object]:
    """
    Compute accuracy, per-class precision/recall/F1 and macro averages.

    These are standard definitions:
        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)
        F1        = 2 * precision * recall / (precision + recall)

    A metric is reported as 0.0 when its denominator is zero (for example a
    class that never appears in the test set). No value is estimated.
    """
    class_names = list(class_names)
    num_classes = len(class_names)
    matrix = compute_confusion_matrix(labels, predictions, num_classes)

    total = len(labels)
    correct = sum(matrix[i][i] for i in range(num_classes))
    accuracy = correct / total if total > 0 else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(class_names):
        true_positive = matrix[index][index]
        false_positive = sum(
            matrix[row][index] for row in range(num_classes) if row != index
        )
        false_negative = sum(
            matrix[index][col] for col in range(num_classes) if col != index
        )
        support = true_positive + false_negative

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_precision = (
        sum(values["precision"] for values in per_class.values()) / num_classes
        if num_classes > 0
        else 0.0
    )
    macro_recall = (
        sum(values["recall"] for values in per_class.values()) / num_classes
        if num_classes > 0
        else 0.0
    )
    macro_f1 = (
        sum(values["f1"] for values in per_class.values()) / num_classes
        if num_classes > 0
        else 0.0
    )

    positive_name = config.POSITIVE_CLASS
    positive = per_class.get(positive_name, {})

    return {
        "num_samples": total,
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "class_names": class_names,
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "positive_class": positive_name,
        "positive": {
            "precision": positive.get("precision", 0.0),
            "recall": positive.get("recall", 0.0),
            "f1": positive.get("f1", 0.0),
            "support": positive.get("support", 0),
        },
    }


# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------
def check_evaluation_ready(
    processed_dir: Path | str = config.PROCESSED_DIR,
    checkpoint_path: Optional[Path | str] = None,
) -> Dict[str, object]:
    """
    Report whether a real evaluation can run.

    Evaluation needs BOTH a trained checkpoint AND images in both classes of
    the test split. Anything less means it is not ready.
    """
    path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else config.get_checkpoint_path()
    )
    processed_dir = Path(processed_dir)

    test_counts = {
        name: dataset_utils.count_image_files(processed_dir / "test" / name)
        for name in config.CLASS_NAMES
    }
    test_ok = all(count > 0 for count in test_counts.values())
    checkpoint_exists = path.is_file()

    return {
        "processed_dir": str(processed_dir),
        "checkpoint_path": str(path),
        "checkpoint_exists": checkpoint_exists,
        "test_counts": test_counts,
        "test_ok": test_ok,
        "ready": checkpoint_exists and test_ok,
    }


# ---------------------------------------------------------------------------
# Loading (requires PyTorch)
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_path: Optional[Path | str] = None) -> dict:
    """Load a trained checkpoint dictionary from disk."""
    training.require_torch()
    import torch  # noqa: E402

    path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else config.get_checkpoint_path()
    )
    if not path.is_file():
        raise EvaluationNotAvailableError(
            f"No trained checkpoint was found at '{path}'. "
            "Train the model first with train_model.py."
        )

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - report a clear message instead
        raise EvaluationNotAvailableError(
            f"The checkpoint at '{path}' could not be loaded ({error})."
        ) from error


def load_trained_model(checkpoint: dict, device=None):
    """Rebuild the architecture and load the trained weights into it."""
    training.require_torch()

    model_name = str(checkpoint.get("model_name", config.MODEL_NAME))
    model = training.build_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=False,  # weights come from the checkpoint, not the internet
    )

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise EvaluationNotAvailableError(
            "The checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate(
    processed_dir: Path | str = config.PROCESSED_DIR,
    checkpoint_path: Optional[Path | str] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    device=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Evaluate the trained checkpoint on the test split.

    Raises ``EvaluationNotAvailableError`` if the checkpoint or the test split
    is missing. Returns a report containing only values computed by this run.
    """
    training.require_torch()
    import torch  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    ready = check_evaluation_ready(processed_dir, checkpoint_path)
    if not ready["ready"]:
        raise EvaluationNotAvailableError(
            "Evaluation cannot run: a trained checkpoint and test images in "
            "both classes are required."
        )

    path = Path(str(ready["checkpoint_path"]))
    checkpoint = load_checkpoint(path)

    if device is None:
        device = training.get_device()

    model = load_trained_model(checkpoint, device=device)

    test_dataset = training.DeepGuardDataset(
        Path(processed_dir) / "test", training.get_eval_transforms()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_labels, all_predictions)

    if verbose:
        print(f"Evaluated {metrics['num_samples']} test image(s).")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(path),
        "processed_dir": str(processed_dir),
        "device": str(device),
        "model_name": checkpoint.get("model_name", config.MODEL_NAME),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "checkpoint_validation_accuracy": checkpoint.get("validation_accuracy"),
        "torch_version": str(torch.__version__),
        "metrics": metrics,
    }
    return report


def save_report(
    report: dict,
    reports_dir: Path | str = config.REPORTS_DIR,
) -> Path:
    """Write an evaluation report to JSON and return the path."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = config.get_evaluation_report_path(reports_dir)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return path
