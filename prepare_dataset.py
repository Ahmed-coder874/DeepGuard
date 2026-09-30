"""
prepare_dataset.py
==================
Command-line script that prepares the DeepGuard dataset.

What it does
------------
1. Checks that the raw dataset folders exist.
2. Scans the "real" and "fake" folders.
3. Checks every file and reports corrupted / unsupported files.
4. Creates a deterministic train / validation / test split.
5. Copies the files into the processed dataset structure.
6. Prints a final summary and saves it as JSON for the Streamlit app.

It NEVER deletes or modifies your original files.

Typical use
-----------
    python prepare_dataset.py                  # prepare with the defaults
    python prepare_dataset.py --dry-run        # only show what would happen
    python prepare_dataset.py --stats-only     # only print a report
    python prepare_dataset.py --max-per-class 200
    python prepare_dataset.py --all-images

IMPORTANT
    The AI/ML detection model is NOT part of this script. No model is
    trained here and no prediction is produced.
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
from src import dataset  # noqa: E402

LINE = "=" * 62


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the DeepGuard train/validation/test dataset.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=config.RAW_DIR,
        help="Folder that contains the 'real' and 'fake' sub-folders.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=config.PROCESSED_DIR,
        help="Folder where the processed dataset will be created.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=config.MAX_IMAGES_PER_CLASS,
        help="Maximum number of images per class (small prototype subset).",
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Use every image instead of a limited prototype subset.",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--train-ratio", type=float, default=config.TRAIN_RATIO)
    parser.add_argument(
        "--val-ratio", type=float, default=config.VALIDATION_RATIO
    )
    parser.add_argument("--test-ratio", type=float, default=config.TEST_RATIO)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without copying any files.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print a dataset report and exit.",
    )
    parser.add_argument(
        "--delete-invalid",
        action="store_true",
        help=(
            "Delete corrupted/unsupported files from the RAW folders. "
            "Off by default; originals are never removed unless you ask."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the processed dataset from scratch.",
    )
    parser.add_argument(
        "--group-aligned",
        action="store_true",
        help=(
            "Use group-aligned splitting across the real and fake classes: every "
            "identity component (same sub-folder name in both classes) stays "
            "inside exactly one split. Required to prevent identity leakage on "
            "datasets produced by extract_dataset_frames.py."
        ),
    )
    parser.add_argument(
        "--audit-leakage",
        action="store_true",
        help="Force the group-aligned split mode on, then run the pre/post-copy "
        "leakage audits and print their reports.",
    )
    return parser


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def print_missing_dataset_help(real_dir: Path, fake_dir: Path) -> None:
    print(LINE)
    print("[!] Raw dataset not found.")
    print()
    print("DeepGuard expects the downloaded dataset to be arranged like this:")
    print()
    print("    data/raw/real/   <- genuine images")
    print("    data/raw/fake/   <- manipulated / AI-generated images")
    print()
    print("The exact folders on this computer are:")
    print(f"    {real_dir}")
    print(f"    {fake_dir}")
    print()
    print("What to do next:")
    print("    1. Read docs/dataset_research.md and choose a dataset.")
    print("    2. Download it (most datasets need a short access request).")
    print("    3. Put the genuine images in the 'real' folder.")
    print("    4. Put the manipulated images in the 'fake' folder.")
    print("    5. Run this script again:  python prepare_dataset.py")
    print()
    print("No files were changed.")
    print(LINE)


def print_scan_report(label: str, scan: dataset.ScanResult) -> None:
    print(f"  {label}: {scan.directory}")
    print(f"    valid images      : {scan.valid_count}")
    print(f"    corrupted         : {scan.invalid_count}")
    print(f"    unsupported files : {scan.unsupported_count}")

    problem_lines = dataset.report_invalid_files(scan)
    if problem_lines:
        print("    problem files:")
        for line in problem_lines[:10]:
            print(line)
        if len(problem_lines) > 10:
            print(f"    ... and {len(problem_lines) - 10} more")


def print_stats_report(summary: dict | None) -> None:
    print("Dataset status")
    if summary is None:
        print("  No prepared dataset was found.")
        print("  Run  python prepare_dataset.py  after placing the raw images.")
        return

    totals = summary.get("totals", {})
    print(f"  Real images: {totals.get('real', 0)}")
    print(f"  Fake images: {totals.get('fake', 0)}")
    print(f"  Training images: {totals.get('train', 0)}")
    print(f"  Validation images: {totals.get('validation', 0)}")
    print(f"  Test images: {totals.get('test', 0)}")
    print(f"  Invalid files: {totals.get('invalid', 0)}")
    print(f"  Unsupported files: {totals.get('unsupported', 0)}")
    print(f"  Prepared at: {summary.get('generated_at', 'unknown')}")


def print_audit_report(label: str, audit: dict) -> None:
    print(f"  {label}: {'OK - no identity appears in more than one split' if audit['ok'] else 'FAILED'}")
    for split_name, count in audit.get("split_counts", {}).items():
        print(f"    {split_name}: {count} group(s)")
    if not audit["ok"]:
        for key, splits in audit.get("leaked_groups", {}).items():
            print(f"    LEAKED group {key} -> {splits}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    raw_real_dir = args.raw_dir / "real"
    raw_fake_dir = args.raw_dir / "fake"
    max_per_class = None if args.all_images else args.max_per_class

    group_aligned = args.group_aligned or args.audit_leakage

    print(LINE)
    print("DeepGuard - Dataset preparation")
    print(LINE)

    # --- Statistics only --------------------------------------------------
    if args.stats_only:
        print_stats_report(dataset.load_summary(args.processed_dir))
        print(LINE)
        return 0

    # --- Check the raw folders exist -------------------------------------
    if not raw_real_dir.is_dir() or not raw_fake_dir.is_dir():
        print_missing_dataset_help(raw_real_dir, raw_fake_dir)
        existing_summary = dataset.load_summary(args.processed_dir)
        if existing_summary is not None:
            print()
            print_stats_report(existing_summary)
            print(LINE)
        return 1

    # --- Validate the ratios ---------------------------------------------
    try:
        config.validate_ratios(
            args.train_ratio, args.val_ratio, args.test_ratio
        )
    except ValueError as error:
        print(f"[!] Invalid split ratios: {error}")
        print(LINE)
        return 2

    # --- Scan the raw dataset --------------------------------------------
    print("Scanning raw dataset (this may take a moment)...")
    print()
    real_scan, fake_scan = dataset.scan_raw_dataset(raw_real_dir, raw_fake_dir)

    print("Raw dataset")
    print_scan_report("real", real_scan)
    print_scan_report("fake", fake_scan)
    print()

    if args.delete_invalid:
        removed = dataset.delete_invalid_files(
            real_scan.invalid_files + fake_scan.invalid_files,
            allow_delete=True,
        )
        print(f"[i] Deleted {removed} corrupted file(s) from the raw folders.")
        print()

    if real_scan.valid_count == 0 and fake_scan.valid_count == 0:
        print("[!] No valid images were found in the raw folders.")
        print("    Please check that the images are JPG or PNG and are readable.")
        print(LINE)
        return 1

    if real_scan.valid_count == 0 or fake_scan.valid_count == 0:
        print("[!] One of the classes has no valid images.")
        print("    A real/fake classifier needs images in BOTH classes.")
        print(LINE)
        return 1

    # --- Create the split -------------------------------------------------
    print(
        f"Creating splits (seed={args.seed}, "
        f"ratios={args.train_ratio}/{args.val_ratio}/{args.test_ratio}, "
        f"max per class={max_per_class}, group-aligned={group_aligned})..."
    )
    splits = dataset.create_splits(
        real_files=real_scan.valid_files,
        fake_files=fake_scan.valid_files,
        real_root=raw_real_dir,
        fake_root=raw_fake_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_per_class=max_per_class,
        aligned_groups=group_aligned,
    )

    if group_aligned:
        print("Leakage audit (identity groups across real/fake):")
        print_audit_report(
            "pre-copy split audit",
            dataset.audit_split_leakage(
                splits, real_root=raw_real_dir, fake_root=raw_fake_dir
            ),
        )
    print()

    planned = dataset.split_counts(splits)
    print(
        "  planned -> "
        f"train: {planned['train']['real']} real / {planned['train']['fake']} fake, "
        f"validation: {planned['validation']['real']} real / "
        f"{planned['validation']['fake']} fake, "
        f"test: {planned['test']['real']} real / {planned['test']['fake']} fake"
    )
    print()

    # --- Dry run ----------------------------------------------------------
    if args.dry_run:
        print("[i] Dry run: no files were copied.")
        summary = dataset.build_summary(
            real_scan=real_scan,
            fake_scan=fake_scan,
            split_statistics=planned,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_per_class=max_per_class,
            copied=False,
        )
        print()
        print_stats_report(summary)
        print(LINE)
        return 0

    # --- Copy the files ---------------------------------------------------
    if args.overwrite:
        print("[i] --overwrite given: clearing the old processed folders.")
        dataset.reset_processed_splits(args.processed_dir)

    print("Copying files into the processed dataset...")
    dataset.copy_splits_to_processed(
        splits, processed_dir=args.processed_dir, overwrite=args.overwrite
    )

    if group_aligned:
        audit = dataset.audit_processed_leakage(args.processed_dir)
        if audit["ok"]:
            print("Leakage audit (processed folder names):")
            for split_name, count in audit["image_counts"].items():
                comps = ", ".join(audit["components_per_split"].get(split_name, []))
                print(f"    {split_name}: {count} image(s) from [{comps}]")
            print("    Result: OK - no identity component spans multiple splits")
        else:
            print("Leakage audit (processed folder names):")
            print("    Result: FAILED - identity leakage detected")
            for comp, splits_found in audit["leaked_components"].items():
                print(f"    LEAKED component {comp} -> {splits_found}")

    processed_stats = dataset.get_processed_statistics(args.processed_dir)
    summary = dataset.build_summary(
        real_scan=real_scan,
        fake_scan=fake_scan,
        split_statistics=processed_stats,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_per_class=max_per_class,
        copied=True,
    )
    summary_path = dataset.write_summary(summary, processed_dir=args.processed_dir)

    print()
    print(LINE)
    print("Dataset status")
    print(f"  Real images: {summary['totals']['real']}")
    print(f"  Fake images: {summary['totals']['fake']}")
    print(f"  Training images: {summary['totals']['train']}")
    print(f"  Validation images: {summary['totals']['validation']}")
    print(f"  Test images: {summary['totals']['test']}")
    print(f"  Invalid files: {summary['totals']['invalid']}")
    print(f"  Unsupported files: {summary['totals']['unsupported']}")
    print()
    print(f"[i] Processed dataset: {Path(args.processed_dir)}")
    print(f"[i] Summary written to: {summary_path}")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
