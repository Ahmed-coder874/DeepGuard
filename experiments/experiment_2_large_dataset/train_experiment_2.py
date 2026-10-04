"""
train_experiment_2.py - Experiment 2 trainer with opt-in resume
==============================================================
Trains the Experiment 2 classifier (EfficientNet-B0) on the 6,000-image dataset
produced by ``build_dataset.py``.

Why this file exists instead of ``train_model.py``
--------------------------------------------------
``src/training.py`` and ``train_model.py`` belong to Experiment 1 and are NOT
modified in any way. Experiment 1 is frozen: its dataset, its checkpoint
(``models/deepguard_efficientnet_b0.pt``) and its results stay exactly as they
are, and its code path is untouched.

This script reuses the *building blocks* of ``src/training.py`` (seed handling,
device selection, DataLoaders, model construction, train/validate epoch loops,
checkpoint writer) but runs its own outer loop so that it can save and restore
an optimiser state. That is the only reason it exists.

Resume behaviour
----------------
Resume is OPT-IN. Without ``--resume`` this script starts a fresh run and
behaves exactly like Experiment 1's trainer. With ``--resume`` it looks for a
training-state file and, if it finds a usable one, continues from the next
epoch instead of starting over.

Why this matters here: Experiment 2 trains on CPU only and is expected to take
roughly 7-10 hours. ``src/training.py`` has no resume support, so a crash, a
sleep or a laptop shutdown would throw the whole run away. The state file is
written after every epoch with an atomic replace, so an interrupted write can
never corrupt it.

Safety rules enforced by this script
------------------------------------
1. It refuses to write to Experiment 1's checkpoint path.
2. It refuses to resume against a different dataset (it fingerprints the
   selected images and compares it with the one stored in the state file).
3. It never touches Experiment 1's ``reports/`` directory.

Typical use
-----------
    venv-train\\Scripts\\python.exe \
        experiments\\experiment_2_large_dataset\\train_experiment_2.py

    # after a crash, continue instead of restarting
    venv-train\\Scripts\\python.exe \
        experiments\\experiment_2_large_dataset\\train_experiment_2.py --resume

IMPORTANT
    The test split is never read here. Stage-5 evaluation (Experiment 2's own
    report folder) happens separately via evaluate_model.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import training  # noqa: E402

LINE = "=" * 62

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PROCESSED_DIR = SCRIPT_DIR / "processed"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "experiment_2_deepguard_efficientnet_b0.pt"
DEFAULT_STATE_PATH = PROJECT_ROOT / "models" / "experiment_2_training_state.pt"
DEFAULT_RUN_LOG = SCRIPT_DIR / "train_run_log.txt"
MANIFEST_FILENAME = "dataset_manifest.csv"

STATE_VERSION = 1

# Hyperparameters are pinned to the values Experiment 1 actually used (see
# reports/stage6/train_run_log.txt) so that the dataset is the only variable.
# NOTE: config.BATCH_SIZE is 16, but Experiment 1 was trained with batch 8.
EXP1_BATCH_SIZE = 8
EXP1_LEARNING_RATE = 3e-4
EXP1_WEIGHT_DECAY = 1e-4
EXP1_EPOCHS = 10
EXP1_PATIENCE = 3
EXP1_SEED = 42


class Tee:
    """Write everything printed to stdout and to a log file at the same time."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    def write(self, text: str) -> None:
        self._handle.write(text)
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


_ACTIVE_TEE: Optional[Tee] = None


def log(message: str = "", tee: Optional[Tee] = None) -> None:
    """Print to stdout and mirror the line into the run log.

    ``tee`` may still be passed explicitly. When it is omitted the run log
    created by ``main`` is used instead, so per-epoch output is captured
    without every call site having to thread the handle through. Previously the
    Tee was never passed to any call, which left train_run_log.txt holding only
    the startup header.
    """
    print(message)
    target = tee if tee is not None else _ACTIVE_TEE
    if target is not None:
        target.write(message + "\n")


# ---------------------------------------------------------------------------
# Dataset fingerprint
# ---------------------------------------------------------------------------
def dataset_fingerprint(processed_dir: Path) -> Optional[str]:
    """
    Hash the selected images listed in the manifest.

    Used to detect that the dataset changed between two runs, which would make
    a resumed training run meaningless. Returns None when no manifest exists.
    """
    manifest_path = processed_dir.parent / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    digest = hashlib.sha256()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            digest.update(
                f"{row['split']}/{row['label']}/{row['md5']}\n".encode("utf-8")
            )
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Atomic state persistence
# ---------------------------------------------------------------------------
def atomic_torch_save(payload: dict, path: Path) -> Path:
    """
    Save a checkpoint atomically.

    The payload is written to a temporary file in the same directory and then
    moved into place with ``os.replace``. If the process dies mid-write, the
    previous good state file survives untouched.
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_state(path: Path) -> Optional[dict]:
    import torch

    if not path.is_file():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"[!] Could not read the training state ({error}); starting fresh.")
        return None
    if not isinstance(state, dict) or state.get("state_version") != STATE_VERSION:
        print("[!] Training state has an incompatible format; starting fresh.")
        return None
    return state


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Experiment 2 classifier (resume-capable).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Folder containing the train/validation/test splits.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Where to save the best checkpoint. Refuses to overwrite Exp1's.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Where to save the resume state (written every epoch).",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=DEFAULT_RUN_LOG,
        help="Text log of the run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EXP1_EPOCHS,
        help=f"Maximum epochs (Experiment 1 used {EXP1_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=EXP1_BATCH_SIZE,
        help=f"Batch size (Experiment 1 used {EXP1_BATCH_SIZE}).",
    )
    parser.add_argument("--lr", type=float, default=EXP1_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=EXP1_WEIGHT_DECAY)
    parser.add_argument(
        "--patience", type=int, default=EXP1_PATIENCE,
        help="Early-stopping patience on validation loss.",
    )
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=EXP1_SEED)
    parser.add_argument(
        "--model", type=str, default=config.MODEL_NAME,
        help="Architecture (default: efficientnet_b0, same as Experiment 1).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip the ImageNet weights. Experiment 1 used them, so skipping "
             "them makes the two runs incomparable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the saved training state if one exists. Without "
             "this flag the run always starts fresh (default behaviour).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only report whether the dataset is ready, then exit.",
    )
    return parser


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------
def assert_not_experiment_1(*paths: Path) -> None:
    """Refuse to write anywhere that belongs to the frozen Experiment 1."""
    frozen = config.get_checkpoint_path().resolve()
    for path in paths:
        if path.resolve() == frozen:
            raise SystemExit(
                f"[!] Refusing to write to Experiment 1's checkpoint:\n"
                f"    {frozen}\n"
                "    That file is the frozen official result of Experiment 1 "
                "and must never be modified."
            )
    frozen_reports = config.REPORTS_DIR.resolve()
    for path in paths:
        if frozen_reports in path.resolve().parents:
            raise SystemExit(
                f"[!] Refusing to write inside Experiment 1's reports folder:\n"
                f"    {frozen_reports}\n"
                "    Experiment 2 reports go to their own directory."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    processed_dir = args.processed_dir.expanduser().resolve()

    log(LINE)
    log("DeepGuard - Experiment 2 training (Stage 4)")
    log(LINE)
    log(f"Architecture : {args.model}")
    log(f"Epochs       : {args.epochs}")
    log(f"Batch size   : {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Seed         : {args.seed}")
    log(f"Pretrained   : {not args.no_pretrained}")
    log(f"Resume       : {args.resume}")
    log(LINE)

    if not training.TORCH_AVAILABLE:
        log("[!] PyTorch is not installed in this Python environment.")
        log("    Use the training environment:")
        log("    venv-train\\Scripts\\activate")
        log(LINE)
        return 2

    try:
        assert_not_experiment_1(
            args.checkpoint.expanduser(), args.state_path.expanduser()
        )
    except SystemExit as error:
        log(str(error))
        log(LINE)
        return 1

    status = training.check_dataset_ready(processed_dir)
    log("Dataset readiness")
    log(f"  Processed folder: {processed_dir}")
    log(f"  Found on disk   : {status['exists']}")
    for split_name in ("train", "validation", "test"):
        counts = status["counts"][split_name]
        log(
            f"  {split_name:>10}: {counts['real']} real / {counts['fake']} fake"
        )
    log(f"  Ready to train  : {status['ready']}")
    log()

    if args.status:
        log(LINE)
        return 0 if status["ready"] else 1

    if not status["ready"]:
        log("[!] The Experiment 2 dataset is not ready for training.")
        log("    Build it first:")
        log("    python experiments/experiment_2_large_dataset/build_dataset.py")
        log(LINE)
        return 1

    global _ACTIVE_TEE
    tee = Tee(args.run_log)
    _ACTIVE_TEE = tee
    tee.write(f"\n{'-' * 62}\n")
    tee.write(
        f"Run started: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
        f"\n{'-' * 62}\n"
    )

    try:
        checkpoint_path = args.checkpoint.expanduser().resolve()
        state_path = args.state_path.expanduser().resolve()

        training.set_seed(args.seed)
        device = training.get_device()

        log(f"Torch version: {training.torch.__version__}")
        log(f"CUDA available: {training.torch.cuda.is_available()}")
        log(f"Device       : {device}")
        log()

        train_loader, validation_loader = training.build_dataloaders(
            processed_dir=processed_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        log(f"Training images  : {len(train_loader.dataset)}")
        log(f"Validation images: {len(validation_loader.dataset)}")
        log()

        model = training.build_model(
            model_name=args.model, pretrained=not args.no_pretrained
        ).to(device)
        criterion = training.nn.CrossEntropyLoss()
        optimizer = training.torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = training.torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )

        fingerprint = dataset_fingerprint(processed_dir)

        start_epoch = 1
        best_validation_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_early = False
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "learning_rate": [],
        }

        if args.resume:
            state = load_state(state_path)
            if state is None:
                log("[i] No usable training state found. Starting a fresh run.")
            else:
                if state.get("dataset_fingerprint") != fingerprint:
                    log(
                        "[!] The dataset changed since the state file was "
                        "written.\n    Refusing to resume onto a different "
                        "dataset. Delete the state file and start fresh:\n"
                        f"      {state_path}"
                    )
                    log(LINE)
                    return 1
                model.load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                scheduler.load_state_dict(state["scheduler_state_dict"])
                start_epoch = int(state["next_epoch"])
                best_validation_loss = float(state["best_validation_loss"])
                best_epoch = int(state["best_epoch"])
                epochs_without_improvement = int(
                    state["epochs_without_improvement"]
                )
                history = state["history"]
                completed = len(history["train_loss"])
                log(
                    f"[i] Resumed from epoch {completed}: continuing at "
                    f"epoch {start_epoch}/{args.epochs}."
                )
                log(f"    Best so far   : epoch {best_epoch}, "
                    f"validation loss {best_validation_loss:.4f}")
                log(f"    Patience left : {max(0, args.patience - epochs_without_improvement)}")
        else:
            log("[i] Resume not requested: starting a fresh run from epoch 1.")

        if start_epoch > args.epochs:
            log("[i] All requested epochs are already complete.")
            log(LINE)
            return 0

        log()
        run_started = time.time()

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_started = time.time()
            train_loss, train_accuracy = training.train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            validation_loss, validation_accuracy = training.validate_one_epoch(
                model, validation_loader, criterion, device
            )
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed_minutes = (time.time() - epoch_started) / 60.0

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
            history["learning_rate"].append(current_lr)

            log(f"Epoch {epoch}/{args.epochs}  ({elapsed_minutes:.1f} min)")
            log(f"Train Loss: {train_loss:.4f}")
            log(f"Train Accuracy: {train_accuracy:.4f}")
            log(f"Validation Loss: {validation_loss:.4f}")
            log(f"Validation Accuracy: {validation_accuracy:.4f}")
            log(f"Learning rate: {current_lr:.2e}")
            log()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                training.save_checkpoint(
                    checkpoint_path,
                    model,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    validation_accuracy=validation_accuracy,
                    model_name=args.model,
                    seed=args.seed,
                )
                log(f"[+] New best checkpoint saved: {checkpoint_path}")
            else:
                epochs_without_improvement += 1

            scheduler.step(validation_loss)

            atomic_torch_save(
                {
                    "state_version": STATE_VERSION,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "next_epoch": epoch + 1,
                    "best_validation_loss": best_validation_loss,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "history": history,
                    "seed": args.seed,
                    "dataset_fingerprint": fingerprint,
                    "model_name": args.model,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "patience": args.patience,
                    "epochs_requested": args.epochs,
                    "saved_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                },
                state_path,
            )

            if epochs_without_improvement >= args.patience:
                stopped_early = True
                log(
                    f"Early stopping: validation loss did not improve for "
                    f"{args.patience} consecutive epoch(s)."
                )
                log()
                break

        total_minutes = (time.time() - run_started) / 60.0
        completed = len(history["train_loss"])

        log(LINE)
        log("Training finished")
        log(LINE)
        log(f"  Epochs completed this run : {completed}")
        log(f"  Wall-clock this run      : {total_minutes:.1f} min")
        log(f"  Best epoch               : {best_epoch}")
        log(f"  Best validation loss     : {best_validation_loss:.4f}")
        log(f"  Best validation accuracy : "
            f"{history['validation_accuracy'][best_epoch - 1]:.4f}"
            if best_epoch > 0
            else "  Best validation accuracy : n/a")
        log(f"  Early stopped            : {stopped_early}")
        log(f"  Checkpoint               : {checkpoint_path}")
        log(f"  Resume state             : {state_path}")
        log(LINE)

        summary_path = SCRIPT_DIR / "training_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "experiment": "experiment_2_large_dataset",
                    "model_name": args.model,
                    "device": str(device),
                    "epochs_requested": args.epochs,
                    "epochs_completed": completed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "stopped_early": stopped_early,
                    "checkpoint_path": str(checkpoint_path),
                    "train_images": len(train_loader.dataset),
                    "validation_images": len(validation_loader.dataset),
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "patience": args.patience,
                    "seed": args.seed,
                    "pretrained": not args.no_pretrained,
                    "dataset_fingerprint": fingerprint,
                    "wall_clock_minutes_this_run": round(total_minutes, 2),
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        log(f"  Summary : {summary_path}")
        log(LINE)
        log("Next step: evaluate the test split. Nothing above is a final result.")
        log(LINE)
        return 0
    finally:
        _ACTIVE_TEE = None
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())