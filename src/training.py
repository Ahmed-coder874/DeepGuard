"""
training.py
===========
Stage 4 training pipeline for DeepGuard.

This module contains the reusable pieces used to fine-tune an image
classifier (EfficientNet-B0 by default) on the dataset prepared in Stage 2:

    - reproducible seeding
    - a Dataset that reads data/processed/<split>/<class>
    - modest training augmentation and deterministic evaluation transforms
    - model construction with a two-class head (real = 0, fake = 1)
    - one training epoch and one validation epoch
    - a training loop with early stopping and best-checkpoint saving

IMPORTANT
    This module does not train anything by itself. Training only happens when
    ``train_model.py`` is run on a real prepared dataset. If no dataset is
    present, nothing is trained and no checkpoint is created.

    No performance metric produced here is the project's final result. The
    test split is reserved for Stage 5 evaluation and is never used here.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make the project root importable so that "import config" always works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import dataset as dataset_utils  # noqa: E402

from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
# PyTorch is installed only in the separate "venv-train" environment. The
# import is guarded so that this module can still be imported (and its
# non-torch helpers tested) when PyTorch is not available.
try:  # pragma: no cover - depends on the environment
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    TORCH_AVAILABLE = False

# When PyTorch is absent, fall back to a plain base class so that this module
# can still be imported (the application environment has no PyTorch).
_DatasetBase = Dataset if TORCH_AVAILABLE else object


def require_torch() -> None:
    """Raise a friendly error if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available in this Python environment. "
            "Create the training environment and install requirements-train.txt."
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed Python, NumPy and PyTorch so a run is as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # Not expected on this machine.
            torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the compute device. CPU is expected because there is no GPU."""
    require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DeepGuardDataset(_DatasetBase):
    """
    Reads one split of the processed dataset.

    The class folders are ``real`` and ``fake``. The label mapping is applied
    explicitly from ``config.LABEL_TO_INDEX`` (real = 0, fake = 1) instead of
    relying on TorchVision's ``ImageFolder``, which sorts folder names
    alphabetically and would assign fake = 0.
    """

    def __init__(self, split_dir: Path | str, transform=None) -> None:
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for class_name in config.CLASS_NAMES:
            label = config.LABEL_TO_INDEX[class_name]
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and dataset_utils.is_supported_image(path):
                    self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label

    def class_counts(self) -> Dict[str, int]:
        """Return the number of images per class name."""
        counts = {name: 0 for name in config.CLASS_NAMES}
        for _path, label in self.samples:
            counts[config.INDEX_TO_LABEL[label]] += 1
        return counts


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transforms():
    """
    Modest augmentation for the training split only.

    Kept deliberately light: heavy colour/blur/noise/erasing transforms can
    destroy the subtle forensic traces that distinguish real from fake images.
    """
    require_torch()
    size = config.IMAGE_SIZE[0]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def get_eval_transforms():
    """Deterministic transforms for the validation and test splits."""
    require_torch()
    size = config.IMAGE_SIZE[0]
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset availability
# ---------------------------------------------------------------------------
def check_dataset_ready(processed_dir: Path | str = config.PROCESSED_DIR) -> Dict[str, object]:
    """
    Inspect the processed dataset and report what is really on disk.

    Training requires images in BOTH classes of BOTH the training and the
    validation splits. The test split is not required for training.
    """
    processed_dir = Path(processed_dir)

    counts: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        counts[split_name] = {
            class_name: dataset_utils.count_image_files(
                processed_dir / split_name / class_name
            )
            for class_name in config.CLASS_NAMES
        }

    train_ok = all(counts["train"][name] > 0 for name in config.CLASS_NAMES)
    validation_ok = all(
        counts["validation"][name] > 0 for name in config.CLASS_NAMES
    )

    return {
        "processed_dir": str(processed_dir),
        "exists": processed_dir.is_dir(),
        "counts": counts,
        "train_ok": train_ok,
        "validation_ok": validation_ok,
        "ready": train_ok and validation_ok,
    }


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(
    processed_dir: Path | str = config.PROCESSED_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
):
    """Create the training and validation DataLoaders."""
    require_torch()
    processed_dir = Path(processed_dir)

    train_dataset = DeepGuardDataset(processed_dir / "train", get_train_transforms())
    validation_dataset = DeepGuardDataset(
        processed_dir / "validation", get_eval_transforms()
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, validation_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(
    model_name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.USE_PRETRAINED,
):
    """
    Build the classifier with a two-class head.

    Supported names: "efficientnet_b0" (primary) and "resnet50" (alternative).
    The output has ``num_classes`` logits ordered as real (0), fake (1).
    """
    require_torch()
    name = str(model_name).lower().replace("-", "_")

    if name in ("efficientnet_b0", "efficientnetb0"):
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif name in ("resnet50", "resnet_50"):
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: "
            f"'{config.MODEL_NAME}', '{config.ALT_MODEL_NAME}'."
        )

    return model


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    """Run one training epoch. Returns (mean loss, accuracy)."""
    require_torch()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    total = max(total, 1)
    return running_loss / total, correct / total


def validate_one_epoch(model, loader, criterion, device) -> Tuple[float, float]:
    """Run one validation epoch (no gradient updates). Returns (mean loss, accuracy)."""
    require_torch()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    total = max(total, 1)
    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: Path | str,
    model,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    model_name: str,
    seed: int,
) -> Path:
    """Save a real checkpoint dictionary and return the path."""
    require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_classes": config.NUM_CLASSES,
        "class_names": list(config.CLASS_NAMES),
        "label_to_index": dict(config.LABEL_TO_INDEX),
        "image_size": list(config.IMAGE_SIZE),
        "imagenet_mean": list(config.IMAGENET_MEAN),
        "imagenet_std": list(config.IMAGENET_STD),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "seed": seed,
        "torch_version": str(torch.__version__),
    }
    torch.save(checkpoint, path)
    return path


def train(
    processed_dir: Path | str = config.PROCESSED_DIR,
    model_name: str = config.MODEL_NAME,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
    pretrained: bool = config.USE_PRETRAINED,
    checkpoint_path: Optional[Path | str] = None,
    device=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Fine-tune the model and save the best checkpoint (lowest validation loss).

    Raises RuntimeError if the dataset is not ready. Returns a history
    dictionary containing only values produced by this run.
    """
    require_torch()

    status = check_dataset_ready(processed_dir)
    if not status["ready"]:
        raise RuntimeError(
            "The prepared dataset is not ready for training. Images are "
            "required in both classes of the train and validation splits."
        )

    set_seed(seed)
    device = device if device is not None else get_device()
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else config.get_checkpoint_path()
    )

    train_loader, validation_loader = build_dataloaders(
        processed_dir=processed_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = build_model(model_name=model_name, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = validate_one_epoch(
            model, validation_loader, criterion, device
        )
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(current_lr)

        if verbose:
            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                model_name=model_name,
                seed=seed,
            )
        else:
            epochs_without_improvement += 1

        scheduler.step(validation_loss)

        if epochs_without_improvement >= patience:
            stopped_early = True
            if verbose:
                print(
                    f"Early stopping: validation loss did not improve for "
                    f"{patience} consecutive epoch(s)."
                )
                print()
            break

    return {
        "model_name": model_name,
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history["train_loss"]),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stopped_early": stopped_early,
        "checkpoint_path": str(checkpoint_path),
        "train_images": len(train_loader.dataset),
        "validation_images": len(validation_loader.dataset),
        "seed": seed,
    }
