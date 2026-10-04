"""
evaluate_model.py
=================
Command-line script that evaluates the trained DeepGuard model (Stage 5).

What it does
------------
1. Checks that a trained checkpoint exists in models/.
2. Checks that the test split contains images in both classes.
3. Runs the trained model over the test split (which was never used during
   training).
4. Reports accuracy, precision, recall, F1-score and the confusion matrix.
5. Saves the report as JSON in reports/.

If the checkpoint or the test split is missing, the script stops cleanly with
a clear message. It does NOT invent metrics.

Typical use
-----------
    python evaluate_model.py
    python evaluate_model.py --status          # only report readiness
    python evaluate_model.py --checkpoint models/my_model.pt
    python evaluate_model.py --batch-size 8
    python evaluate_model.py --reports-dir experiments/experiment_2_large_dataset/reports

IMPORTANT
    Run this script inside the training environment (venv-train), because it
    needs PyTorch. The test split is used ONLY here, never during training.
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
from src import evaluation  # noqa: E402

LINE = "=" * 62


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained DeepGuard classifier on the test split.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=config.PROCESSED_DIR,
        help="Folder that contains the train/validation/test splits.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate (default: models/).",
    )
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=config.REPORTS_DIR,
        help=(
            "Where to write evaluation_report.json. The default is "
            "reports/, which holds Experiment 1's frozen official result. "
            "Point this elsewhere (for example "
            "experiments/experiment_2_large_dataset/reports) so a second "
            "experiment cannot overwrite it."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only report whether evaluation can run, then exit.",
    )
    return parser


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def print_readiness(ready: dict) -> None:
    counts = ready["test_counts"]
    print("Evaluation readiness")
    print(f"  Checkpoint      : {ready['checkpoint_path']}")
    print(f"  Checkpoint found: {ready['checkpoint_exists']}")
    print(
        f"  Test split      : {counts['real']} real / {counts['fake']} fake"
    )
    print(f"  Ready           : {ready['ready']}")
    print()


def print_no_model_help(ready: dict) -> None:
    print(LINE)
    print("Evaluation cannot run yet.")
    print()
    if not ready["checkpoint_exists"]:
        print("[!] No trained checkpoint was found.")
        print(f"    Expected at: {ready['checkpoint_path']}")
        print("    Train the model first:  python train_model.py")
    if not ready["test_ok"]:
        print("[!] The test split does not contain images in both classes.")
        print(f"    Expected under: {Path(ready['processed_dir']) / 'test'}")
        print("    Prepare the dataset first:  python prepare_dataset.py")
    print()
    print("No evaluation was performed and no report was created.")
    print(LINE)


def print_metrics(report: dict) -> None:
    metrics = report["metrics"]
    class_names = metrics["class_names"]
    matrix = metrics["confusion_matrix"]
    positive = metrics["positive"]

    print(LINE)
    print("Evaluation results (test split)")
    print(LINE)
    print(f"  Model            : {report['model_name']}")
    print(f"  Checkpoint       : {report['checkpoint_path']}")
    print(f"  Device           : {report['device']}")
    print(f"  Test images      : {metrics['num_samples']}")
    print()
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(
        f"  Precision ({metrics['positive_class']}) : "
        f"{positive['precision']:.4f}"
    )
    print(
        f"  Recall ({metrics['positive_class']})    : "
        f"{positive['recall']:.4f}"
    )
    print(f"  F1 ({metrics['positive_class']})        : {positive['f1']:.4f}")
    print(f"  Macro precision  : {metrics['macro']['precision']:.4f}")
    print(f"  Macro recall     : {metrics['macro']['recall']:.4f}")
    print(f"  Macro F1         : {metrics['macro']['f1']:.4f}")
    print()
    print("  Per-class metrics:")
    for name in class_names:
        values = metrics["per_class"][name]
        print(
            f"    {name:>5}: precision={values['precision']:.4f}  "
            f"recall={values['recall']:.4f}  f1={values['f1']:.4f}  "
            f"support={values['support']}"
        )
    print()
    print("  Confusion matrix (rows = actual, columns = predicted):")
    header = "        " + "".join(f"{name:>10}" for name in class_names)
    print(header)
    for index, name in enumerate(class_names):
        row = "".join(f"{value:>10}" for value in matrix[index])
        print(f"    {name:>4}{row}")
    print(LINE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(LINE)
    print("DeepGuard - Model evaluation (Stage 5)")
    print(LINE)

    ready = evaluation.check_evaluation_ready(args.processed_dir, args.checkpoint)
    print_readiness(ready)

    if args.status:
        print(LINE)
        return 0 if ready["ready"] else 1

    if not ready["ready"]:
        print_no_model_help(ready)
        return 1

    from src import training  # noqa: E402

    if not training.TORCH_AVAILABLE:
        print("[!] PyTorch is not installed in this Python environment.")
        print("    Use the training environment:  venv-train\\Scripts\\activate")
        print(LINE)
        return 2

    try:
        report = evaluation.evaluate(
            processed_dir=args.processed_dir,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            verbose=True,
        )
    except evaluation.EvaluationNotAvailableError as error:
        print(f"[!] Evaluation could not run: {error}")
        print(LINE)
        return 1

    report_path = evaluation.save_report(report, reports_dir=args.reports_dir)
    print_metrics(report)
    print(f"[i] Report written to: {report_path}")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
