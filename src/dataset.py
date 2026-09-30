"""
dataset.py
==========
Dataset discovery, validation and preparation helpers for DeepGuard.

This module is used by two other files:

    prepare_dataset.py   - the command-line script that builds the
                           train/validation/test folders.
    app.py               - the Streamlit interface, which shows a
                           "Dataset Status" section.

The functions are written to be reusable later by the model-training code.

IMPORTANT
    These functions only organise and check image files. They do NOT train
    a model and they do NOT detect deepfakes.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

# Make the project root importable so that "import config" always works,
# no matter where the script is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    """The result of scanning one folder (for example data/raw/real)."""

    directory: Path
    exists: bool
    valid_files: List[Path] = field(default_factory=list)
    invalid_files: List[Path] = field(default_factory=list)
    unsupported_files: List[Path] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return len(self.valid_files)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_files)

    @property
    def unsupported_count(self) -> int:
        return len(self.unsupported_files)

    @property
    def total_count(self) -> int:
        return self.valid_count + self.invalid_count + self.unsupported_count


# ---------------------------------------------------------------------------
# Discovering and checking files
# ---------------------------------------------------------------------------
def is_supported_image(filename: str | Path) -> bool:
    """Return True if the file name ends with a supported image extension."""
    return Path(filename).suffix.lower() in config.SUPPORTED_EXTENSIONS


def find_images(directory: Path | str, recursive: bool = True) -> List[Path]:
    """
    Return a sorted list of files inside a folder that have a supported
    image extension. Missing folders return an empty list.
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        return []

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    files = [path for path in iterator if path.is_file() and is_supported_image(path)]
    return sorted(files)


def check_image_readable(path: Path | str) -> Tuple[bool, str]:
    """
    Try to open an image file.

    Returns (True, "") when the file is a readable image.
    Returns (False, reason) when it is corrupted or not an image at all.
    """
    try:
        with Image.open(path) as image:
            image.load()  # Actually decode the pixels; catches truncated files.
        return True, ""
    except UnidentifiedImageError:
        return False, "not a recognised image format"
    except (OSError, ValueError, SyntaxError) as error:
        return False, f"unreadable image ({error})"
    except Exception as error:  # Last-resort safety net.
        return False, f"unexpected error ({error})"


def scan_class_folder(directory: Path | str) -> ScanResult:
    """
    Scan one class folder (real or fake).

    Every file is placed in exactly one of three groups:
        valid_files       - supported extension AND readable image
        invalid_files     - supported extension BUT corrupted / unreadable
        unsupported_files - extension that this project does not support
    """
    directory = Path(directory)
    result = ScanResult(directory=directory, exists=directory.exists())

    if not result.exists or not directory.is_dir():
        return result

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if not is_supported_image(path):
            result.unsupported_files.append(path)
            continue

        readable, _reason = check_image_readable(path)
        if readable:
            result.valid_files.append(path)
        else:
            result.invalid_files.append(path)

    return result


def scan_raw_dataset(
    real_dir: Path | str = config.RAW_REAL_DIR,
    fake_dir: Path | str = config.RAW_FAKE_DIR,
) -> Tuple[ScanResult, ScanResult]:
    """Scan both the real and the fake raw folders."""
    return scan_class_folder(real_dir), scan_class_folder(fake_dir)


def count_image_files(directory: Path | str) -> int:
    """
    Quickly count image files by extension (no pixel decoding).

    This is used by the Streamlit app so that showing the number of raw
    images stays fast even when the raw dataset is large.
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() and is_supported_image(path))


def report_invalid_files(scan: ScanResult) -> List[str]:
    """Return a list of short, human-readable lines about problem files."""
    lines: List[str] = []
    for path in scan.invalid_files:
        lines.append(f"  [corrupted]   {path}")
    for path in scan.unsupported_files:
        lines.append(f"  [unsupported] {path}")
    return lines


def delete_invalid_files(files: Iterable[Path | str], allow_delete: bool = False) -> int:
    """
    Delete problem files ONLY when the caller explicitly allows it.

    This is deliberately guarded so that a beginner cannot delete their
    original dataset by accident. By default it does nothing and returns 0.
    """
    if not allow_delete:
        return 0

    deleted = 0
    for path in files:
        path = Path(path)
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Splitting into train / validation / test
# ---------------------------------------------------------------------------
def group_key(path: Path | str, class_root: Path | str) -> str:
    """
    Return a "group" name for an image.

    Images that belong together (for example all frames extracted from the
    same video, stored in the same sub-folder) get the SAME group name.
    Keeping a group inside a single split prevents an obvious form of data
    leakage, where nearly identical frames appear in both training and test.

    If the images sit directly in the class folder, each image is its own
    group.
    """
    path = Path(path)
    class_root = Path(class_root)

    try:
        relative = path.relative_to(class_root)
    except ValueError:
        # The file is not inside the class folder; fall back to the file name.
        return path.stem

    if len(relative.parts) > 1:
        return relative.parts[0]  # top-level sub-folder = one group

    return path.stem


def limit_files(files: List[Path], max_count: Optional[int], seed: int) -> List[Path]:
    """
    Randomly keep at most ``max_count`` files (deterministic for a given seed).

    ``max_count = None`` keeps every file. This is how the prototype uses a
    manageable subset of a large dataset.
    """
    if max_count is None or max_count <= 0 or len(files) <= max_count:
        return list(files)

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:max_count])


def split_class_files(
    files: List[Path],
    class_root: Path | str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    """
    Split one class (real or fake) into train / validation / test.

    Groups are kept intact and the assignment is deterministic for a given
    seed, so the same input always produces the same split.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        groups[group_key(path, class_root)].append(path)

    # Sorting first makes the starting order reproducible; shuffling with the
    # seed then makes the result random but repeatable.
    keys = sorted(groups.keys())
    random.Random(seed).shuffle(keys)

    total = len(files)
    train_target = int(round(total * train_ratio))
    validation_target = int(round(total * validation_ratio))

    result: Dict[str, List[Path]] = {"train": [], "validation": [], "test": []}

    for key in keys:
        group_files = groups[key]
        if len(result["train"]) < train_target:
            result["train"].extend(group_files)
        elif len(result["validation"]) < validation_target:
            result["validation"].extend(group_files)
        else:
            result["test"].extend(group_files)

    return result


def split_grouped_files(
    all_file_groups: Dict[str, Dict[str, List[Path]]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Split files that are grouped by a SHARED key across both classes.

    ``all_file_groups[key]["real"]`` and ``all_file_groups[key]["fake"]`` hold
    every file (of the respective class) that belongs to the same group.  A
    group is treated as atomic and is assigned to exactly ONE split, so the
    same identity/component never appears in more than one split.  This is
    the leakage-prevention guarantee: frames of the same identity stay inside
    a single split even though they exist in BOTH the real and the fake class.

    Assignment is deterministic for a given seed: keys are sorted and shuffled
    with the seed, then every key is placed into the split that is currently
    least "full" relative to its per-class target.  Ties break in favour of
    train, then validation, then test.

    Non-empty guarantee
        When at least one key per ``SPLIT_NAMES`` entry exists, every split is
        guaranteed to receive at least one group.  Without this, a dataset with
        a handful of large, atomic identity components (for example the 5
        components of this 20-video experiment) can exhaust the available keys
        on train+validation and leave the test split empty.  The guarantee uses
        only the current state, so the assignment stays deterministic and the
        identity-leakage property is unchanged.
    """
    keys = sorted(all_file_groups.keys())
    random.Random(seed).shuffle(keys)

    real_total = sum(len(g.get("real", [])) for g in all_file_groups.values())
    fake_total = sum(len(g.get("fake", [])) for g in all_file_groups.values())

    targets = {
        "train": {
            "real": int(round(real_total * train_ratio)),
            "fake": int(round(fake_total * train_ratio)),
        },
        "validation": {
            "real": int(round(real_total * validation_ratio)),
            "fake": int(round(fake_total * validation_ratio)),
        },
        "test": {
            "real": real_total
            - int(round(real_total * train_ratio))
            - int(round(real_total * validation_ratio)),
            "fake": fake_total
            - int(round(fake_total * train_ratio))
            - int(round(fake_total * validation_ratio)),
        },
    }

    counts = {
        split_name: {"real": 0, "fake": 0} for split_name in config.SPLIT_NAMES
    }
    result: Dict[str, Dict[str, List[Path]]] = {
        split_name: {"real": [], "fake": []} for split_name in config.SPLIT_NAMES
    }

    def fullness(split_name: str, real_add: int, fake_add: int) -> float:
        target_real = max(targets[split_name]["real"], 1)
        target_fake = max(targets[split_name]["fake"], 1)
        return (counts[split_name]["real"] + real_add) / target_real + (
            counts[split_name]["fake"] + fake_add
        ) / target_fake

    for position, key in enumerate(keys):
        real_files = all_file_groups[key].get("real", [])
        fake_files = all_file_groups[key].get("fake", [])
        remaining_after = len(keys) - position - 1
        empty_splits = [
            s
            for s in config.SPLIT_NAMES
            if not counts[s]["real"] and not counts[s]["fake"]
        ]
        candidates = config.SPLIT_NAMES
        if empty_splits and remaining_after < len(empty_splits):
            # Only the still-empty splits can each receive a group from the
            # remaining keys, so this group must go to one of them to keep
            # every split populated.
            candidates = empty_splits
        chosen = min(
            candidates,
            key=lambda s: (
                fullness(s, len(real_files), len(fake_files)),
                config.SPLIT_NAMES.index(s),  # tie-break: train < validation < test
            ),
        )
        result[chosen]["real"].extend(real_files)
        result[chosen]["fake"].extend(fake_files)
        counts[chosen]["real"] += len(real_files)
        counts[chosen]["fake"] += len(fake_files)

    return result


def group_files_by_key(
    files: List[Path],
    class_root: Path | str,
    class_name: str,
) -> Dict[str, Dict[str, List[Path]]]:
    """Group files by ``group_key`` and return a ``key -> {class: [paths]}`` map."""
    grouped: Dict[str, Dict[str, List[Path]]] = {}
    for path in files:
        key = group_key(path, class_root)
        grouped.setdefault(key, {class_name: []})[class_name].append(path)
    return grouped


def create_splits(
    real_files: List[Path],
    fake_files: List[Path],
    real_root: Path | str = config.RAW_REAL_DIR,
    fake_root: Path | str = config.RAW_FAKE_DIR,
    train_ratio: float = config.TRAIN_RATIO,
    validation_ratio: float = config.VALIDATION_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed: int = config.RANDOM_SEED,
    max_per_class: Optional[int] = config.MAX_IMAGES_PER_CLASS,
    aligned_groups: bool = False,
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Build the complete train/validation/test split for both classes.

    Returns a nested dictionary:
        splits["train"]["real"] -> [Path, Path, ...]
        splits["train"]["fake"] -> [...]
        ... and the same for "validation" and "test".

    Leakage prevention (``aligned_groups=True``)
        When frames of one identity are stored in the same-named sub-folder of
        BOTH ``data/raw/real`` and ``data/raw/fake`` (as produced by
        ``extract_dataset_frames.py``), the group is assigned to exactly one
        split across the two classes.  This guarantees that every frame of an
        identity component stays inside a single split, which the independent
        per-class split (``aligned_groups=False``) cannot provide.
    """
    config.validate_ratios(train_ratio, validation_ratio, test_ratio)

    real_used = limit_files(real_files, max_per_class, seed)
    fake_used = limit_files(fake_files, max_per_class, seed + 1)

    if aligned_groups:
        real_groups = group_files_by_key(real_used, real_root, "real")
        fake_groups = group_files_by_key(fake_used, fake_root, "fake")
        merged: Dict[str, Dict[str, List[Path]]] = {}
        for key, entry in real_groups.items():
            merged.setdefault(key, {})["real"] = entry["real"]
        for key, entry in fake_groups.items():
            merged.setdefault(key, {})["fake"] = entry["fake"]
        return split_grouped_files(
            merged, train_ratio, validation_ratio, test_ratio, seed
        )

    real_split = split_class_files(
        real_used, real_root, train_ratio, validation_ratio, test_ratio, seed
    )
    fake_split = split_class_files(
        fake_used, fake_root, train_ratio, validation_ratio, test_ratio, seed + 1
    )

    return {
        "train": {"real": real_split["train"], "fake": fake_split["train"]},
        "validation": {
            "real": real_split["validation"],
            "fake": fake_split["validation"],
        },
        "test": {"real": real_split["test"], "fake": fake_split["test"]},
    }


def all_split_paths(splits: Dict[str, Dict[str, List[Path]]]) -> List[Path]:
    """Return every path in a split dictionary as one flat list."""
    paths: List[Path] = []
    for split_name in config.SPLIT_NAMES:
        for class_name in config.CLASS_NAMES:
            paths.extend(splits.get(split_name, {}).get(class_name, []))
    return paths


def split_counts(splits: Dict[str, Dict[str, List[Path]]]) -> Dict[str, Dict[str, int]]:
    """Return the number of images in every split and class."""
    return {
        split_name: {
            class_name: len(splits.get(split_name, {}).get(class_name, []))
            for class_name in config.CLASS_NAMES
        }
        for split_name in config.SPLIT_NAMES
    }


# ---------------------------------------------------------------------------
# Copying files into the processed dataset
# ---------------------------------------------------------------------------
def reset_processed_splits(processed_dir: Path | str = config.PROCESSED_DIR) -> None:
    """
    Delete ONLY the processed train/validation/test folders.

    The raw dataset is never touched. This is used when the user asks to
    rebuild the processed dataset from scratch (--overwrite).
    """
    processed_dir = Path(processed_dir)
    for split_name in config.SPLIT_NAMES:
        split_dir = processed_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)


def copy_splits_to_processed(
    splits: Dict[str, Dict[str, List[Path]]],
    processed_dir: Path | str = config.PROCESSED_DIR,
    overwrite: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Copy the split files into data/processed/<split>/<class>/.

    Files are renamed with a numeric prefix so that two source images with
    the same name never overwrite each other. Original files are never
    modified or deleted.
    """
    processed_dir = Path(processed_dir)
    counts: Dict[str, Dict[str, int]] = {
        split_name: {class_name: 0 for class_name in config.CLASS_NAMES}
        for split_name in config.SPLIT_NAMES
    }

    for split_name in config.SPLIT_NAMES:
        for class_name in config.CLASS_NAMES:
            destination = config.get_split_class_dir(processed_dir, split_name, class_name)
            destination.mkdir(parents=True, exist_ok=True)

            for index, source in enumerate(splits.get(split_name, {}).get(class_name, [])):
                target = destination / f"{index:06d}_{Path(source).name}"
                if not target.exists() or overwrite:
                    shutil.copy2(source, target)
                counts[split_name][class_name] += 1

    return counts


def get_processed_statistics(
    processed_dir: Path | str = config.PROCESSED_DIR,
) -> Dict[str, Dict[str, int]]:
    """Count the image files that currently exist in the processed dataset."""
    processed_dir = Path(processed_dir)
    stats: Dict[str, Dict[str, int]] = {}

    for split_name in config.SPLIT_NAMES:
        stats[split_name] = {}
        for class_name in config.CLASS_NAMES:
            folder = config.get_split_class_dir(processed_dir, split_name, class_name)
            stats[split_name][class_name] = count_image_files(folder)

    return stats


def processed_dataset_exists(processed_dir: Path | str = config.PROCESSED_DIR) -> bool:
    """Return True if the processed dataset has at least one image."""
    stats = get_processed_statistics(processed_dir)
    return any(count > 0 for split in stats.values() for count in split.values())


# ---------------------------------------------------------------------------
# Summary file (written by prepare_dataset.py, read by the Streamlit app)
# ---------------------------------------------------------------------------
def build_summary(
    real_scan: ScanResult,
    fake_scan: ScanResult,
    split_statistics: Dict[str, Dict[str, int]],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    max_per_class: Optional[int],
    copied: bool,
) -> dict:
    """Build a plain dictionary describing the dataset. Easy to save as JSON."""
    train_total = sum(split_statistics["train"].values())
    validation_total = sum(split_statistics["validation"].values())
    test_total = sum(split_statistics["test"].values())

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "max_per_class": max_per_class,
        "copied": copied,
        "raw": {
            "real": {
                "total_files": real_scan.total_count,
                "valid": real_scan.valid_count,
                "invalid": real_scan.invalid_count,
                "unsupported": real_scan.unsupported_count,
            },
            "fake": {
                "total_files": fake_scan.total_count,
                "valid": fake_scan.valid_count,
                "invalid": fake_scan.invalid_count,
                "unsupported": fake_scan.unsupported_count,
            },
        },
        "splits": {
            "train": {
                "real": split_statistics["train"]["real"],
                "fake": split_statistics["train"]["fake"],
                "total": train_total,
            },
            "validation": {
                "real": split_statistics["validation"]["real"],
                "fake": split_statistics["validation"]["fake"],
                "total": validation_total,
            },
            "test": {
                "real": split_statistics["test"]["real"],
                "fake": split_statistics["test"]["fake"],
                "total": test_total,
            },
        },
        "totals": {
            "real": real_scan.valid_count,
            "fake": fake_scan.valid_count,
            "train": train_total,
            "validation": validation_total,
            "test": test_total,
            "invalid": real_scan.invalid_count + fake_scan.invalid_count,
            "unsupported": real_scan.unsupported_count + fake_scan.unsupported_count,
        },
    }


def write_summary(summary: dict, processed_dir: Path | str = config.PROCESSED_DIR) -> Path:
    """Save the summary dictionary as JSON and return the file path."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.get_summary_path(processed_dir)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary_path


def load_summary(processed_dir: Path | str = config.PROCESSED_DIR) -> Optional[dict]:
    """Load the summary JSON. Returns None if it does not exist or is broken."""
    summary_path = config.get_summary_path(processed_dir)
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def get_dataset_status(
    processed_dir: Path | str = config.PROCESSED_DIR,
    raw_real_dir: Path | str = config.RAW_REAL_DIR,
    raw_fake_dir: Path | str = config.RAW_FAKE_DIR,
) -> dict:
    """
    Collect everything the Streamlit app needs to show the Dataset Status
    section, using only real values read from disk.
    """
    summary = load_summary(processed_dir)

    return {
        "summary": summary,
        "processed_exists": processed_dataset_exists(processed_dir),
        "raw_real_dir": str(raw_real_dir),
        "raw_fake_dir": str(raw_fake_dir),
        "raw_real_count": count_image_files(raw_real_dir),
        "raw_fake_count": count_image_files(raw_fake_dir),
    }


# ---------------------------------------------------------------------------
# Leakage audit (train/validation/test must never share an identity group)
# ---------------------------------------------------------------------------
def _split_group_keys(splits, real_root, fake_root) -> Dict[str, set]:
    """
    Return {split_name: set of group keys seen in that split}, merged over
    both classes.

    Group keys are derived from the raw folder structure (one sub-folder per
    identity component), so this works for any properly grouped raw dataset.
    """
    roots = {"real": Path(real_root), "fake": Path(fake_root)}
    keys_per_split: Dict[str, set] = {name: set() for name in config.SPLIT_NAMES}
    for split_name, classes in splits.items():
        for class_name in config.CLASS_NAMES:
            for path in classes.get(class_name, []):
                keys_per_split[split_name].add(group_key(path, roots[class_name]))
    return keys_per_split


def audit_split_leakage(
    splits,
    real_root: Path | str = config.RAW_REAL_DIR,
    fake_root: Path | str = config.RAW_FAKE_DIR,
) -> Dict[str, object]:
    """
    Check that no group key appears in more than one split.

    Group keys are derived from the raw folder structure (one sub-folder per
    identity component).  Returns a report dict used both by the CLI and by
    tests; the group-aligned splitter guarantees no leakage, so this function
    is a verification.
    """
    keys_per_split = _split_group_keys(splits, real_root, fake_root)
    component_to_splits: Dict[str, List[str]] = {}
    for split_name in config.SPLIT_NAMES:
        for key in keys_per_split[split_name]:
            component_to_splits.setdefault(key, []).append(split_name)

    leaked = {
        key: sorted(splits)
        for key, splits in sorted(component_to_splits.items())
        if len(splits) > 1
    }

    return {
        "ok": len(leaked) == 0,
        "leaked_groups": leaked,
        "group_count": len(component_to_splits),
        "split_counts": {
            name: len(keys_per_split[name]) for name in config.SPLIT_NAMES
        },
    }


def parse_component_id(processed_filename: str) -> str:
    """
    Recover the identity component from a copied processed filename.

    Processed files are named ``000000_<component>__<video>__frame_000012.png``
    where the component is the group sub-folder name produced by the
    extraction step.  The numeric copy-prefix is stripped, then the text
    before the first ``__`` is the component id.
    """
    name = Path(processed_filename).name
    stem = name[:-len(Path(name).suffix)]
    first_part = stem.split("__")[0]
    return first_part.lstrip("0123456789").lstrip("_")


def audit_processed_leakage(processed_dir: Path | str = config.PROCESSED_DIR) -> Dict[str, object]:
    """
    Post-copy leakage audit on the prepared processed dataset.

    Re-derives each image's identity component from its (copied) file name and
    verifies that no component appears in more than one of train / validation
    / test.  Also reports the components that were seen in each split, which
    is the ground truth used to detect cross-split identity leakage even
    though the processed folders are flat.
    """
    processed_dir = Path(processed_dir)
    seen: Dict[str, set] = {split_name: set() for split_name in config.SPLIT_NAMES}
    image_counts: Dict[str, int] = {split_name: 0 for split_name in config.SPLIT_NAMES}

    for split_name in config.SPLIT_NAMES:
        for class_name in config.CLASS_NAMES:
            folder = config.get_split_class_dir(processed_dir, split_name, class_name)
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.is_file() and is_supported_image(path):
                    seen[split_name].add(parse_component_id(path.name))
                    image_counts[split_name] += 1

    components_to_splits: Dict[str, List[str]] = {}
    for split_name, comps in seen.items():
        for comp in comps:
            components_to_splits.setdefault(comp, []).append(split_name)

    leaked = {
        comp: sorted(splits)
        for comp, splits in sorted(components_to_splits.items())
        if len(splits) > 1
    }

    return {
        "ok": len(leaked) == 0,
        "leaked_components": leaked,
        "components_per_split": {
            split: sorted(comps) for split, comps in sorted(seen.items())
        },
        "component_split_counts": {
            comp: len(splits) for comp, splits in sorted(components_to_splits.items())
        },
        "image_counts": image_counts,
    }
