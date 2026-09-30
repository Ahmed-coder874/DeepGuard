"""
train_model.py
==============
Command-line script that trains the DeepGuard image classifier.

What it does
------------
1. Checks whether a prepared dataset exists in data/processed.
2. If it does, fine-tunes the selected architecture (EfficientNet-B0 by
   default) on the train split and validates on the validation split.
3. Prints the real training/validation values for every epoch.
4. Saves the best checkpoint (lowest validation loss) into models/.

If no prepared dataset is available, the script stops cleanly with a clear
message. It does NOT invent data, does NOT train on nothing, and does NOT
create a placeholder model file.

Typical use
-----------
    python train_model.py
    python train_model.py --epochs 5 --batch-size 8
    python train_model.py --model resnet50
    python train_model.py --no-pretrained      # offline / quick test
    python train_model.py --status             # only report dataset readiness

IMPORTANT
    - The test split is reserved for Stage 5 evaluation and is never used here.
    - No accuracy reported by this script is the project's final result.
    - PyTorch must be installed (see requirements-train.txt, venv-train).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is importable when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import training  # noqa: E402

LINE = "=" * 62


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the DeepGuard real/fake image classifier.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=config.PROCESSED_DIR,
        help="Folder that contains the train/validation/test splits.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=(
            "Architecture to train: "
            f"'{config.MODEL_NAME}' (default) or '{config.ALT_MODEL_NAME}'."
        ),
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument(
        "--patience", type=int, default=config.EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use ImageNet weights (offline or quick test).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Where to save the best checkpoint (default: models/).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only report whether the dataset is ready for training, then exit.",
    )
    return parser


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def print_dataset_status(status: dict) -> None:
    counts = status["counts"]
    print("Dataset readiness")
    print(f"  Processed folder: {status['processed_dir']}")
    print(f"  Found on disk   : {status['exists']}")
    for split_name in ("train", "validation", "test"):
        split_counts = counts[split_name]
        print(
            f"  {split_name:>10}: "
            f"{split_counts['real']} real / {split_counts['fake']} fake"
        )
    print(f"  Ready to train  : {status['ready']}")
    print()


def print_no_dataset_help(processed_dir: Path) -> None:
    print(LINE)
    print("No prepared dataset is currently available, so Stage 4 training")
    print("cannot be executed yet.")
    print()
    print("Training needs this structure to contain images:")
    print(f"    {processed_dir / 'train' / 'real'}")
    print(f"    {processed_dir / 'train' / 'fake'}")
    print(f"    {processed_dir / 'validation' / 'real'}")
    print(f"    {processed_dir / 'validation' / 'fake'}")
    print()
    print("What to do next:")
    print("    1. Read docs/dataset_research.md and choose a dataset.")
    print("    2. Put genuine images in data/raw/real and manipulated images")
    print("       in data/raw/fake.")
    print("    3. Run:  python prepare_dataset.py")
    print("    4. Then run this script again:  python train_model.py")
    print()
    print("No training was performed and no model file was created.")
    print(LINE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(LINE)
    print("DeepGuard - Model training (Stage 4)")
    print(LINE)
    print(f"Architecture : {args.model}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed         : {args.seed}")
    print(f"Pretrained   : {not args.no_pretrained}")
    print(LINE)
    print()

    # --- Dataset readiness -----------------------------------------------
    status = training.check_dataset_ready(args.processed_dir)
    print_dataset_status(status)

    if args.status:
        print(LINE)
        return 0 if status["ready"] else 1

    if not status["ready"]:
        print_no_dataset_help(Path(args.processed_dir))
        return 1

    # --- PyTorch availability --------------------------------------------
    if not training.TORCH_AVAILABLE:
        print("[!] PyTorch is not installed in this Python environment.")
        print("    Create the training environment and install the packages:")
        print("        python -m venv venv-train")
        print("        venv-train\\Scripts\\activate")
        print("        pip install -r requirements-train.txt")
        print(LINE)
        return 2

    import torch  # noqa: E402  (only imported once it is known to exist)

    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # --- Train ------------------------------------------------------------
    try:
        result = training.train(
            processed_dir=args.processed_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            pretrained=not args.no_pretrained,
            checkpoint_path=args.checkpoint,
            verbose=True,
        )
    except RuntimeError as error:
        print(f"[!] Training could not run: {error}")
        print(LINE)
        return 1

    # --- Report -----------------------------------------------------------
    print(LINE)
    print("Training finished")
    print(f"  Device            : {result['device']}")
    print(f"  Training images   : {result['train_images']}")
    print(f"  Validation images : {result['validation_images']}")
    print(f"  Epochs completed  : {result['epochs_completed']} of {result['epochs_requested']}")
    print(f"  Best epoch        : {result['best_epoch']}")
    print(f"  Best validation loss: {result['best_validation_loss']:.4f}")
    print(f"  Early stopping    : {result['stopped_early']}")
    print(f"  Checkpoint saved  : {result['checkpoint_path']}")
    print()
    print("These are training/validation values only. The test set is reserved")
    print("for Stage 5 evaluation.")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
