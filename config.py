"""
config.py
=========
Central configuration for the DeepGuard prototype.

Everything that a beginner may want to change lives here in one place:
folder locations, image size, the random seed, the train/validation/test
ratios and the maximum number of images used per class during prototype
training.

IMPORTANT
    The raw dataset is NEVER stored inside this repository automatically.
    You download it yourself and place it in the folders shown below.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

# Raw dataset layout (this is where YOU put the downloaded dataset):
#
#   data/raw/real/   <- genuine images
#   data/raw/fake/   <- manipulated / AI-generated images
#
RAW_REAL_DIR = RAW_DIR / "real"
RAW_FAKE_DIR = RAW_DIR / "fake"

# Processed dataset layout (created automatically by prepare_dataset.py):
#
#   data/processed/train/real,       data/processed/train/fake
#   data/processed/validation/real,  data/processed/validation/fake
#   data/processed/test/real,        data/processed/test/fake
PROCESSED_TRAIN_DIR = PROCESSED_DIR / "train"
PROCESSED_VALIDATION_DIR = PROCESSED_DIR / "validation"
PROCESSED_TEST_DIR = PROCESSED_DIR / "test"

SPLIT_NAMES = ("train", "validation", "test")
CLASS_NAMES = ("real", "fake")

# ---------------------------------------------------------------------------
# Image settings (these must match src/preprocessing.py)
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# Splitting settings
# ---------------------------------------------------------------------------
# A fixed seed makes the split reproducible: running the script twice gives
# exactly the same train/validation/test assignment.
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Prototype limit: only this many images per class are copied into the
# processed dataset. This keeps training fast on a normal student laptop.
# Set it to None to use every available image.
MAX_IMAGES_PER_CLASS = 500

# ---------------------------------------------------------------------------
# Dataset summary file (written by prepare_dataset.py, read by app.py)
# ---------------------------------------------------------------------------
SUMMARY_FILENAME = "dataset_summary.json"

# ---------------------------------------------------------------------------
# Model and training settings (Stage 3 selection / Stage 4 training)
# ---------------------------------------------------------------------------
# These are PLANNED, CONFIGURABLE settings. They have NOT been experimentally
# optimised, and no model has been trained with them yet. They are kept here so
# that the training code does not hard-code values in many different files.
#
# The architecture choice is documented in docs/model_architecture_research.md
# and the planned training design in docs/training_plan.md.
#
# Label order is FIXED and must be used consistently by training and inference:
#   real -> 0
#   fake -> 1
LABEL_TO_INDEX = {"real": 0, "fake": 1}
INDEX_TO_LABEL = {index: name for name, index in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(CLASS_NAMES)

MODEL_NAME = "efficientnet_b0"   # primary architecture (Stage 3 selection)
ALT_MODEL_NAME = "resnet50"      # documented alternative
USE_PRETRAINED = True            # use ImageNet transfer-learning weights

# Training hyperparameters (planned defaults, not claimed to be optimal).
BATCH_SIZE = 16                  # conservative for CPU-only training
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3      # epochs without validation-loss improvement
NUM_WORKERS = 0                  # Windows + CPU: keep at 0 for safety

# ImageNet normalisation statistics, required by the pretrained weights.
# Stage 1 preprocessing normalises to 0-1 for display; training and future
# inference must both apply these statistics for consistency.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Where the best checkpoint is saved. The folder stays empty until a real
# training run produces a checkpoint; no placeholder file is ever created.
CHECKPOINT_DIR = PROJECT_ROOT / "models"
CHECKPOINT_FILENAME = "deepguard_efficientnet_b0.pt"

# ---------------------------------------------------------------------------
# Evaluation settings (Stage 5)
# ---------------------------------------------------------------------------
# Evaluation is performed ONLY on the test split, which is never used during
# training. The "fake" class (index 1) is treated as the positive class,
# because detecting manipulated content is the purpose of the project.
POSITIVE_CLASS = "fake"

# Where the evaluation report is written after a real evaluation run. The
# folder is created only when evaluation actually happens; no report is
# written and no metric is invented when the model or dataset is missing.
REPORTS_DIR = PROJECT_ROOT / "reports"
EVALUATION_REPORT_FILENAME = "evaluation_report.json"


# ---------------------------------------------------------------------------
# FaceForensics++ source data + frame extraction (Stage 2 experimental)
# ---------------------------------------------------------------------------
# The downloaded FaceForensics++ release is kept intact under:
#   data/faceforensics/original_sequences/youtube/c23/videos/*.mp4     (real)
#   data/faceforensics/manipulated_sequences/Deepfakes/c23/videos/*.mp4 (fake)
FACEFORENSICS_DIR = DATA_DIR / "faceforensics"
FFPP_COMPRESSION = "c23"

# Video folders inside the FaceForensics++ download.
FFPP_ORIGINAL_VIDEOS_DIR = (
    FACEFORENSICS_DIR / "original_sequences" / "youtube" / FFPP_COMPRESSION / "videos"
)
FFPP_MANIPULATED_VIDEOS_DIR = (
    FACEFORENSICS_DIR / "manipulated_sequences" / "Deepfakes" / FFPP_COMPRESSION / "videos"
)

# Deterministic frame sampling: how many evenly-spaced frames to extract per
# video for the prototype. 0 extracts every frame.
FFPP_FRAMES_PER_VIDEO = 30

# Provenance manifest written by extract_dataset_frames.py.
FFPP_MANIFEST_FILENAME = "frames_provenance.json"


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------
def get_summary_path(processed_dir: Path | str = PROCESSED_DIR) -> Path:
    """Return the path of the dataset summary JSON file."""
    return Path(processed_dir) / SUMMARY_FILENAME


def get_checkpoint_path(
    checkpoint_dir: Path | str = CHECKPOINT_DIR,
    filename: str = CHECKPOINT_FILENAME,
) -> Path:
    """Return the path where the best trained checkpoint is (or will be) saved."""
    return Path(checkpoint_dir) / filename


def get_evaluation_report_path(
    reports_dir: Path | str = REPORTS_DIR,
    filename: str = EVALUATION_REPORT_FILENAME,
) -> Path:
    """Return the path where the Stage 5 evaluation report is (or will be) saved."""
    return Path(reports_dir) / filename


def get_ffpp_manifest_path(raw_dir: Path | str = RAW_DIR) -> Path:
    """Return the path of the FaceForensics++ provenance manifest."""
    return Path(raw_dir) / FFPP_MANIFEST_FILENAME


def get_split_class_dir(
    processed_dir: Path | str,
    split_name: str,
    class_name: str,
) -> Path:
    """Return the folder that holds one split/class, e.g. processed/train/real."""
    return Path(processed_dir) / split_name / class_name


def validate_ratios(
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> None:
    """Raise ValueError if the three ratios are negative or do not add up to 1."""
    for name, value in (
        ("train", train_ratio),
        ("validation", validation_ratio),
        ("test", test_ratio),
    ):
        if value < 0:
            raise ValueError(f"The {name} ratio cannot be negative (got {value}).")

    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "The train, validation and test ratios must add up to 1.0 "
            f"(they currently add up to {total:.3f})."
        )
