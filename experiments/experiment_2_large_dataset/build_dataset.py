"""
build_dataset.py - Experiment 2 dataset builder
===============================================
Builds the Experiment 2 dataset: 3,000 real + 3,000 fake images sampled from the
Kaggle dataset "140k Real and Fake Faces" (xhlulu), split 70/15/15 with a
deterministic seed of 42.

This script belongs to Experiment 2 only. It never reads from or writes to
Experiment 1's dataset (``data/raw``, ``data/processed``), checkpoint
(``models/deepguard_efficientnet_b0.pt``) or results (``reports/``).

Source dataset
--------------
    Name    : 140k Real and Fake Faces (Kaggle, xhlulu)
    URL     : https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    Size    : ~4 GB (version 2)
    Content : 70,000 real faces (NVIDIA FFHQ / Flickr photographs) and
              70,000 fake faces (StyleGAN-generated, from "1 Million Fake
              Faces"). All images resized to 256 px.
    License : "Other (specified in description)". The underlying FFHQ images are
              released by NVIDIA under CC BY-NC-SA 4.0 (non-commercial,
              attribution required). No image may be redistributed, so this
              repository commits only the manifest, never the pixels.

    ***  IMPORTANT CAVEAT  ***
    Experiment 1's "fake" class is FaceForensics++ face-swap manipulation.
    Experiment 2's "fake" class is StyleGAN synthesis. Experiment 2 is
    therefore NOT a controlled dataset-size experiment: both the dataset size
    AND the fake-image distribution change. This must be stated in any report.

Expected source layout (auto-detected; ``--source-dir`` may point at any
parent of the pool root)
------------------------------------------------------------------------
Both releases of the dataset are supported:

    Layout A (flat)                  Layout B (nested)
    <pool root>/train_real/          <pool root>/train/real/
    <pool root>/train_fake/          <pool root>/train/fake/
    <pool root>/valid_real/          <pool root>/valid/real/
    <pool root>/valid_fake/          <pool root>/valid/fake/
    <pool root>/test_real/           <pool root>/test/real/
    <pool root>/test_fake/           <pool root>/test/fake/

The real-world example looks like this::

    <somewhere>/real_vs_fake/          <- --source-dir can be any parent
        real-vs-fake/
            train/real/  train/fake/
            valid/real/  valid/fake/
            test/real/   test/fake/

The dataset is normally kept OUTSIDE this repository, because it is large
(~4 GB) and its licence does not permit redistribution. Point ``--source-dir``
at wherever you extracted it; nothing here depends on a fixed location.

Method
------
1.  **Official-split respecting.** The publisher already provides train /
    valid / test folders. Experiment 2 never crosses those boundaries: an
    Experiment 2 test image can only come from ``test_*``, a validation image
    only from ``valid_*``, a training image only from ``train_*``.
2.  **Exact-duplicate removal (MD5).** Byte-identical images are collapsed to
    one. If an identical image appears in two different official splits, that
    is a dataset defect: the copy in the highest-priority split (test >
    validation > train) is kept and the others are dropped. If an identical
    image appears as BOTH real and fake, every copy is dropped, because the
    label is contradictory.
3.  **Near-duplicate removal (pHash).** A 64-bit DCT perceptual hash is computed
    for every surviving image. Two images whose hashes differ by at most
    ``--phash-threshold`` bits (default 3) are treated as near-duplicates and
    the later one is dropped. Candidate pairs are found with multi-index
    hashing (the hash is cut into ``threshold + 1`` disjoint bands), which is
    exact for Hamming distances up to ``threshold`` - no false negatives.
4.  **Deterministic subsampling.** After de-duplication each (split, class)
    pool is sampled down to its quota with a per-pool derived seed, so the
    selection is reproducible and stable.
5.  **Audit.** Counts, class balance, cross-split hash overlap, cross-class
    hash overlap and cross-split minimum pHash distance are verified and
    written to ``dataset_report.json``.

Outputs (all under ``--output-dir``)
------------------------------------
    processed/{train,validation,test}/{real,fake}/*.jpg
    dataset_manifest.csv     one row per selected image, with md5 + phash
    dataset_report.json      counts, duplicate statistics, audit results

Typical use
-----------
    python experiments/experiment_2_large_dataset/build_dataset.py \
        --source-dir "D:/datasets/real_vs_fake"

    python experiments/experiment_2_large_dataset/build_dataset.py --validate-only

    python experiments/experiment_2_large_dataset/build_dataset.py --dry-run

``--source-dir`` may point at any parent of the pool root; the pool root is
located automatically. If it is omitted, ``<this folder>/source`` is used, so
the dataset may also sit inside the project if you prefer. Keeping it outside
is recommended: the images are ~4 GB and the licence forbids redistribution.

Needs only numpy + Pillow (no PyTorch). Run it with the app environment:
    venv\\Scripts\\python.exe experiments\\experiment_2_large_dataset\\build_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Make the project root importable when the script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LINE = "=" * 62

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "source"

MANIFEST_FILENAME = "dataset_manifest.csv"
REPORT_FILENAME = "dataset_report.json"
VALIDATION_FILENAME = "dataset_validation.json"

SPLITS: Tuple[str, ...] = ("train", "validation", "test")
CLASSES: Tuple[str, ...] = ("real", "fake")

SPLIT_FOLDER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train": ("train", "training"),
    "validation": ("valid", "validation", "val"),
    "test": ("test", "testing"),
}

# When an identical image exists in several official splits, the copy in the
# highest-priority split is kept so that evaluation material is protected.
SPLIT_PRIORITY: Dict[str, int] = {"test": 0, "validation": 1, "train": 2}

SOURCE_DATASET = "Kaggle: xhlulu/140k-real-and-fake-faces (v2)"
SOURCE_URL = "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces"
SOURCE_LICENSE = (
    "Kaggle: 'Other (specified in description)'. Underlying FFHQ photographs "
    "are CC BY-NC-SA 4.0 (NVIDIA) - non-commercial use, attribution required. "
    "No images are redistributed by this repository."
)

DISTRIBUTION_CAVEAT = (
    "Experiment 1 fake class = FaceForensics++ face-swap manipulation. "
    "Experiment 2 fake class = StyleGAN synthesis. Dataset size AND fake-image "
    "distribution both change, so Experiment 2 is NOT a controlled "
    "dataset-size experiment and must never be described as one."
)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VALID_PHASH_THRESHOLDS = (0, 1, 3, 7, 15, 31, 63)
DEFAULT_PHASH_THRESHOLD = 3

MANIFEST_FIELDS = [
    "image_id",
    "label",
    "split",
    "official_split",
    "source_file",
    "bytes",
    "md5",
    "phash_hex",
]


class SourceNotFoundError(RuntimeError):
    """Raised when the downloaded Kaggle dataset cannot be located."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base_seed: int, *parts: str) -> int:
    """Stable per-pool seed so tweaking one split never reshuffles the others."""
    material = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:12], 16)


def seeded_sample(items: Sequence[Path], count: int, seed: int) -> List[Path]:
    """Deterministic sample of ``count`` items (sorted for a stable result)."""
    ordered = sorted(items, key=lambda p: str(p).lower())
    if count >= len(ordered):
        return list(ordered)
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, count), key=lambda p: str(p).lower())


# ---------------------------------------------------------------------------
# Perceptual hashing (pHash)
# ---------------------------------------------------------------------------
_DCT_CACHE: Dict[int, np.ndarray] = {}


def _dct_matrix(size: int) -> np.ndarray:
    matrix = _DCT_CACHE.get(size)
    if matrix is None:
        n = np.arange(size)
        k = n.reshape(-1, 1)
        matrix = np.cos(np.pi * (2 * n + 1) * k / (2 * size)) * np.sqrt(2.0 / size)
        matrix[0, :] *= np.sqrt(0.5)
        matrix = matrix.astype(np.float64)
        _DCT_CACHE[size] = matrix
    return matrix


def phash_file(path: Path) -> Optional[int]:
    """
    Return a 64-bit DCT perceptual hash, or None if the image cannot be read.

    Standard construction: grayscale -> 32x32 -> 2D DCT-II -> keep the top-left
    8x8 block -> drop the DC term -> threshold at the median of the remaining
    63 coefficients.
    """
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        block = np.asarray(gray, dtype=np.float64)
    except Exception:
        return None

    dct = _dct_matrix(32)
    coefficients = dct @ block @ dct.T
    low = coefficients[:8, :8].flatten()[1:]  # 63 values, DC excluded

    median = np.median(low)
    value = 0
    for bit in (low > median).astype(np.uint8):
        value = (value << 1) | int(bit)
    return value


def phash_worker(path_str: str) -> Tuple[str, Optional[int]]:
    """Module-level worker so it can be pickled by ProcessPoolExecutor."""
    return path_str, phash_file(Path(path_str))


def hamming_distance(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def band_keys(value: int, num_bands: int, band_bits: int) -> List[int]:
    """Cut a 64-bit hash into disjoint bands, most significant first."""
    mask = (1 << band_bits) - 1
    shift = 64 - band_bits
    keys = []
    for _ in range(num_bands):
        keys.append((int(value) >> shift) & mask)
        shift -= band_bits
    return keys


def compute_phashes(
    paths: Sequence[Path], jobs: int, progress_every: int = 5000
) -> Tuple[Dict[Path, int], List[Path]]:
    """
    Compute pHash for every path. Returns (hashes, unreadable_paths).

    Uses a process pool when ``jobs > 1`` and falls back to a serial loop if the
    pool cannot be created (common on locked-down Windows installs).
    """
    hashes: Dict[Path, int] = {}
    unreadable: List[Path] = []
    total = len(paths)
    started = time.time()

    def absorb(result: Tuple[str, Optional[int]]) -> None:
        path = Path(result[0])
        if result[1] is None:
            unreadable.append(path)
        else:
            hashes[path] = result[1]

    if jobs > 1 and total > 1:
        from concurrent.futures import ProcessPoolExecutor

        try:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                for index, result in enumerate(
                    pool.map(phash_worker, [str(p) for p in paths], chunksize=256),
                    start=1,
                ):
                    absorb(result)
                    if progress_every and index % progress_every == 0:
                        print(f"  pHash {index}/{total} ...")
        except Exception as error:  # pragma: no cover - environment dependent
            print(f"[!] Parallel pHash failed ({error}); falling back to serial.")
            return compute_phashes(paths, jobs=1, progress_every=progress_every)
    else:
        for index, path in enumerate(paths, start=1):
            absorb((str(path), phash_file(path)))
            if progress_every and index % progress_every == 0:
                print(f"  pHash {index}/{total} ...")

    elapsed = max(time.time() - started, 1e-6)
    rate = len(paths) / elapsed
    print(
        f"  pHash computed for {len(hashes)} images in {elapsed:.1f}s "
        f"({rate:.0f} images/s)"
    )
    return hashes, unreadable


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """One usable source image."""

    path: Path
    split: str
    class_name: str
    size_bytes: int = 0
    md5: str = ""
    phash: int = 0


def resolve_split_dirs(root: Path) -> Optional[Dict[str, Dict[str, Path]]]:
    """
    Map {split: {class: folder}} for a candidate pool root, or None if incomplete.

    Two on-disk layouts are supported, because different releases of the Kaggle
    dataset ship different shapes:

        Layout A (flat)          Layout B (nested)
        root/train_real/         root/train/real/
        root/train_fake/         root/train/fake/
        root/valid_real/         root/valid/real/
        root/valid_fake/         root/valid/fake/
        root/test_real/          root/test/real/
        root/test_fake/          root/test/fake/

    Split folder aliases are honoured in both layouts (valid_ / val_ /
    validation_). A layout is only accepted when all six folders exist, so a
    partially-extracted download is rejected rather than silently used.
    """
    resolved: Dict[str, Dict[str, Path]] = {}

    for split, prefixes in SPLIT_FOLDER_ALIASES.items():
        chosen: Dict[str, Path] = {}
        for prefix in prefixes:
            flat = {
                class_name: root / f"{prefix}_{class_name}"
                for class_name in CLASSES
            }
            if all(folder.is_dir() for folder in flat.values()):
                chosen = flat
                break
            nested = {
                class_name: root / prefix / class_name for class_name in CLASSES
            }
            if all(folder.is_dir() for folder in nested.values()):
                chosen = nested
                break
        if len(chosen) != len(CLASSES):
            return None
        resolved[split] = chosen

    return resolved


def find_pool_root(source_dir: Path) -> Path:
    """Locate the folder that holds the six split x class folders."""
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise SourceNotFoundError(
            f"Source folder does not exist:\n  {source_dir}\n\n"
            "Download '140k Real and Fake Faces' from Kaggle, extract it, then "
            "point --source-dir at the extracted folder."
        )

    roots: List[Path] = [source_dir]
    for pattern in ("*", "*/*", "*/*/*"):
        roots.extend(sorted(p for p in source_dir.glob(pattern) if p.is_dir()))

    for root in roots:
        if resolve_split_dirs(root) is not None:
            return root

    raise SourceNotFoundError(
        "Could not find the six split x class folders.\n"
        f"Looked under: {source_dir}\n\nSupported layouts:\n"
        "  Layout A:  <root>/train_real/  <root>/train_fake/  ...\n"
        "  Layout B:  <root>/train/real/  <root>/train/fake/  ...\n\n"
        "Re-run with an explicit --source-dir if the dataset is elsewhere."
    )


def collect_candidates(
    resolved: Dict[str, Dict[str, Path]]
) -> List[Candidate]:
    """Scan the six resolved folders into a flat, deterministic candidate list."""
    candidates: List[Candidate] = []
    mismatched_labels = 0

    for split in SPLITS:
        for class_name in CLASSES:
            folder = resolved[split][class_name]
            files = [
                p
                for p in sorted(folder.iterdir(), key=lambda q: q.name.lower())
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            ]
            for path in files:
                # Some releases name files Real_00001.jpg / Fake_00001.jpg. If
                # the name carries a class, verify it agrees with the folder.
                stem_lower = path.stem.lower()
                if stem_lower.startswith("real") and class_name != "real":
                    mismatched_labels += 1
                if stem_lower.startswith("fake") and class_name != "fake":
                    mismatched_labels += 1
                candidates.append(
                    Candidate(
                        path=path,
                        split=split,
                        class_name=class_name,
                        size_bytes=path.stat().st_size,
                    )
                )

    if not candidates:
        raise SourceNotFoundError(
            "The six split x class folders were found but contain no usable "
            "images (.jpg/.jpeg/.png/.bmp/.webp)."
        )
    if mismatched_labels:
        print(
            f"[!] Warning: {mismatched_labels} file name(s) disagree with their "
            "folder class. The folder was treated as authoritative."
        )
    return candidates


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------
@dataclass
class DedupeStats:
    scanned: int = 0
    unreadable: int = 0
    exact_removed_same_class_same_split: int = 0
    exact_removed_cross_split: int = 0
    exact_removed_cross_class: int = 0
    near_removed: int = 0

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "unreadable": self.unreadable,
            "exact_duplicates_removed_same_class_same_split":
                self.exact_removed_same_class_same_split,
            "exact_duplicates_removed_cross_split": self.exact_removed_cross_split,
            "exact_duplicates_removed_cross_class": self.exact_removed_cross_class,
            "near_duplicates_removed": self.near_removed,
            "removed_total": (
                self.exact_removed_same_class_same_split
                + self.exact_removed_cross_split
                + self.exact_removed_cross_class
                + self.near_removed
            ),
        }


def hash_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Compute MD5 for every candidate (byte-level identity)."""
    total = len(candidates)
    started = time.time()
    for index, candidate in enumerate(candidates, start=1):
        candidate.md5 = md5_file(candidate.path)
        if index % 20000 == 0:
            print(f"  MD5 {index}/{total} ...")
    print(
        f"  MD5 computed for {total} images in {time.time() - started:.1f}s"
    )
    return list(candidates)


def remove_exact_duplicates(
    candidates: Sequence[Candidate], stats: DedupeStats
) -> List[Candidate]:
    """
    Collapse byte-identical images.

    - Same class + same split: keep one, drop the rest.
    - Same class + different splits: keep the copy in the highest-priority
      split (test > validation > train) so evaluation material survives and drop
      the others. This also removes any leakage the source dataset introduced.
    - Different classes: the labels contradict each other, so drop every copy.
    """
    groups: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.md5, []).append(candidate)

    kept: List[Candidate] = []
    for digest in sorted(groups):
        group = groups[digest]

        if len({c.class_name for c in group}) > 1:
            stats.exact_removed_cross_class += len(group)
            continue

        ordered = sorted(
            group, key=lambda c: (SPLIT_PRIORITY[c.split], str(c.path).lower())
        )
        kept.append(ordered[0])
        for extra in ordered[1:]:
            if extra.split == ordered[0].split:
                stats.exact_removed_same_class_same_split += 1
            else:
                stats.exact_removed_cross_split += 1

    return kept


def remove_near_duplicates(
    candidates: Sequence[Candidate], threshold: int, stats: DedupeStats
) -> List[Candidate]:
    """
    Drop perceptual near-duplicates (pHash Hamming distance <= threshold).

    Candidates are found with multi-index hashing: cutting the 64-bit hash into
    ``threshold + 1`` disjoint bands guarantees that any pair within
    ``threshold`` bits shares at least one identical band (pigeonhole), so this
    has no false negatives. Each band bucket stores the hashes already kept, and
    a real Hamming check confirms the match.

    Items are processed test -> validation -> train so that evaluation material
    wins any contest against training material.
    """
    if threshold <= 0:
        return list(candidates)

    num_bands = threshold + 1
    if 64 % num_bands != 0:
        raise ValueError(
            "--phash-threshold must be one of "
            f"{list(VALID_PHASH_THRESHOLDS)} so that the 64-bit hash divides "
            f"into threshold+1 equal bands (got {threshold})."
        )
    band_bits = 64 // num_bands

    ordered = sorted(
        candidates, key=lambda c: (SPLIT_PRIORITY[c.split], str(c.path).lower())
    )

    buckets: Dict[int, List[int]] = {}
    kept: List[Candidate] = []

    for candidate in ordered:
        keys = band_keys(candidate.phash, num_bands, band_bits)
        if any(
            hamming_distance(existing, candidate.phash) <= threshold
            for key in keys
            for existing in buckets.get(key, [])
        ):
            stats.near_removed += 1
            continue

        kept.append(candidate)
        for key in keys:
            buckets.setdefault(key, []).append(candidate.phash)

    return kept


# ---------------------------------------------------------------------------
# Manifest + report writing
# ---------------------------------------------------------------------------
def write_manifest(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
_POPCOUNT_LUT = np.frombuffer(
    bytes(bin(value).count("1") for value in range(256)), dtype=np.uint8
)


def _hamming_matrix(
    a: np.ndarray, b: np.ndarray, chunk: int = 200_000
) -> np.ndarray:
    """Chunked pairwise Hamming distances between two uint64 hash arrays."""
    out = np.empty((a.size, b.size), dtype=np.uint8)
    for start in range(0, a.size, chunk):
        stop = min(start + chunk, a.size)
        block = a[start:stop, None] ^ b[None, :]
        byte_view = block.view(np.uint8).reshape(*block.shape, 8)
        out[start:stop] = _POPCOUNT_LUT[byte_view].sum(axis=-1)
    return out


def audit_selection(
    rows: Sequence[dict],
    quotas: Dict[str, int],
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> dict:
    """
    Verify the selection is leakage-free and balanced.

    Checks performed:
      * exact expected counts and perfect class balance per split
      * no MD5 appears in more than one split (exact-duplicate leakage)
      * no MD5 appears under both classes (contradictory labels)
      * each Experiment 2 split maps to exactly one official source split
      * minimum cross-split pHash distance exceeds the near-duplicate cut-off
    """
    checks: Dict[str, bool] = {}
    counts: Dict[str, Dict[str, int]] = {
        split: {"real": 0, "fake": 0} for split in SPLITS
    }

    for row in rows:
        counts[row["split"]][row["label"]] += 1

    counts_ok = True
    for split in SPLITS:
        for class_name in CLASSES:
            if counts[split][class_name] != quotas[split]:
                counts_ok = False
        counts[split]["total"] = counts[split]["real"] + counts[split]["fake"]
    checks["exact_counts_match_quota"] = counts_ok
    checks["classes_balanced_in_every_split"] = all(
        counts[split]["real"] == counts[split]["fake"] for split in SPLITS
    )

    md5_to_splits: Dict[str, set] = {}
    md5_to_classes: Dict[str, set] = {}
    for row in rows:
        md5_to_splits.setdefault(row["md5"], set()).add(row["split"])
        md5_to_classes.setdefault(row["md5"], set()).add(row["label"])

    checks["no_exact_duplicate_across_splits"] = all(
        len(value) == 1 for value in md5_to_splits.values()
    )
    checks["no_image_labelled_both_real_and_fake"] = all(
        len(value) == 1 for value in md5_to_classes.values()
    )
    checks["experiment_split_maps_to_one_official_split"] = all(
        {row["official_split"] for row in rows if row["split"] == split} == {split}
        for split in SPLITS
    )

    phash_by_split = {
        split: np.array(
            [int(row["phash_hex"], 16) for row in rows if row["split"] == split],
            dtype=np.uint64,
        )
        for split in SPLITS
    }

    min_distances: Dict[str, int] = {}
    near_ok = True
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        if phash_by_split[left].size == 0 or phash_by_split[right].size == 0:
            min_distances[f"{left}|{right}"] = -1
            continue
        minimum = int(_hamming_matrix(phash_by_split[left], phash_by_split[right]).min())
        min_distances[f"{left}|{right}"] = minimum
        if minimum <= phash_threshold:
            near_ok = False
    checks["no_near_duplicate_across_splits"] = near_ok

    return {
        "counts": counts,
        "checks": checks,
        "min_cross_split_phash_distance": min_distances,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Experiment 2 dataset (3000 real + 3000 fake).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Folder containing (or a parent of) real_vs_fake/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where processed/, the manifest and the report are written.",
    )
    parser.add_argument("--per-class-train", type=int, default=2100)
    parser.add_argument("--per-class-val", type=int, default=450)
    parser.add_argument("--per-class-test", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=DEFAULT_PHASH_THRESHOLD,
        help="Max pHash Hamming distance for two images to count as "
             "near-duplicates.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Processes used for pHash. Use 1 if multiprocessing misbehaves.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be selected without writing anything.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing processed/ folder before rebuilding.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-audit an existing build from its manifest and exit.",
    )
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.expanduser().resolve()
    processed_dir = output_dir / "processed"
    quotas = {
        "train": args.per_class_train,
        "validation": args.per_class_val,
        "test": args.per_class_test,
    }
    expected_total = 2 * sum(quotas.values())

    print(LINE)
    print("DeepGuard - Experiment 2 dataset build")
    print(LINE)
    print(f"  Source folder : {args.source_dir}")
    print(f"  Output folder : {output_dir}")
    print(
        f"  Target        : {expected_total} images "
        f"({quotas['train']}/{quotas['validation']}/{quotas['test']} per class)"
    )
    print(f"  Seed          : {args.seed}")
    print(f"  pHash cut-off : {args.phash_threshold} bits")
    print()

    pool_root = find_pool_root(args.source_dir)
    resolved = resolve_split_dirs(pool_root)
    if resolved is None:  # pragma: no cover - find_pool_root guarantees this
        raise SourceNotFoundError(f"Incomplete dataset layout under {pool_root}.")
    print(f"  Pool root     : {pool_root}")
    print("  Detected layout:")
    for split in SPLITS:
        print(
            f"    {split:<11}: "
            + ", ".join(str(resolved[split][c]) for c in CLASSES)
        )
    print()

    print("Step 1/5  Scanning source folders")
    candidates = collect_candidates(resolved)
    stats = DedupeStats(scanned=len(candidates))
    pool_sizes: Dict[str, Dict[str, int]] = {
        split: {class_name: 0 for class_name in CLASSES} for split in SPLITS
    }
    for candidate in candidates:
        pool_sizes[candidate.split][candidate.class_name] += 1
    print(f"  Found {len(candidates)} candidate images")
    for split in SPLITS:
        print(
            f"    official {split:<10}: {pool_sizes[split]['real']} real / "
            f"{pool_sizes[split]['fake']} fake"
        )
    print()

    print("Step 2/5  Exact-duplicate detection (MD5)")
    candidates = hash_candidates(candidates)
    candidates = remove_exact_duplicates(candidates, stats)
    print(
        f"  Removed {stats.exact_removed_cross_class} image(s) with conflicting "
        "real/fake labels"
    )
    print(f"  Removed {stats.exact_removed_cross_split} cross-split duplicate(s)")
    print(
        f"  Removed {stats.exact_removed_same_class_same_split} same-split "
        "duplicate(s)"
    )
    print(f"  Remaining: {len(candidates)}")
    print()

    print("Step 3/5  Near-duplicate detection (pHash)")
    phashes, unreadable = compute_phashes(
        [c.path for c in candidates], jobs=args.jobs
    )
    stats.unreadable = len(unreadable)
    usable: List[Candidate] = []
    for candidate in candidates:
        value = phashes.get(candidate.path)
        if value is None:
            continue
        candidate.phash = value
        usable.append(candidate)
    usable = remove_near_duplicates(usable, args.phash_threshold, stats)
    print(f"  Dropped {len(unreadable)} unreadable image(s)")
    print(f"  Removed {stats.near_removed} near-duplicate(s)")
    print(f"  Remaining: {len(usable)}")
    print()

    print("Step 4/5  Deterministic subsampling")
    by_pool: Dict[Tuple[str, str], List[Candidate]] = {
        (split, class_name): []
        for split in SPLITS
        for class_name in CLASSES
    }
    for candidate in usable:
        by_pool[(candidate.split, candidate.class_name)].append(candidate)

    selected: Dict[Tuple[str, str], List[Candidate]] = {}
    pool_after_dedupe: Dict[str, Dict[str, int]] = {
        split: {} for split in SPLITS
    }
    for split in SPLITS:
        for class_name in CLASSES:
            pool = by_pool[(split, class_name)]
            quota = quotas[split]
            pool_after_dedupe[split][class_name] = len(pool)
            if len(pool) < quota:
                raise RuntimeError(
                    f"Not enough images for official split '{split}' / class "
                    f"'{class_name}': need {quota}, have {len(pool)} after "
                    "de-duplication.\n"
                    "Lower --per-class-* or add more source data."
                )
            selected[(split, class_name)] = seeded_sample(
                pool, quota, derived_seed(args.seed, split, class_name)
            )
            print(
                f"  {split:<11} {class_name:<5}: {len(pool)} available -> "
                f"{quota} selected"
            )
    print()

    rows: List[dict] = []
    total_bytes = 0
    for split in SPLITS:
        for class_name in CLASSES:
            for candidate in selected[(split, class_name)]:
                suffix = candidate.path.suffix.lower()
                total_bytes += candidate.size_bytes
                rows.append(
                    {
                        "image_id": (
                            f"{split}_{class_name}_{candidate.path.stem}{suffix}"
                        ),
                        "label": class_name,
                        "split": split,
                        "official_split": candidate.split,
                        "source_file": candidate.path.relative_to(
                            pool_root
                        ).as_posix(),
                        "bytes": candidate.size_bytes,
                        "md5": candidate.md5,
                        "phash_hex": f"{candidate.phash:016x}",
                    }
                )

    print("Step 5/5  Writing dataset")
    if args.dry_run:
        print("  --dry-run: nothing was written to disk.")
    else:
        if processed_dir.exists():
            if not args.overwrite:
                raise RuntimeError(
                    f"{processed_dir} already exists.\n"
                    "Re-run with --overwrite to rebuild it from scratch."
                )
            shutil.rmtree(processed_dir)
        for row in rows:
            destination = processed_dir / row["split"] / row["label"]
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pool_root / row["source_file"], destination / row["image_id"])
        print(f"  Copied {len(rows)} images to {processed_dir}")
        print(f"  Manifest : {output_dir / MANIFEST_FILENAME}")
        print(f"  Report   : {output_dir / REPORT_FILENAME}")

    audit = audit_selection(rows, quotas, phash_threshold=args.phash_threshold)
    payload = {
        "experiment": "experiment_2_large_dataset",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "source": {
            "dataset": SOURCE_DATASET,
            "url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "pool_root": str(pool_root),
            "official_split_sizes": pool_sizes,
        },
        "distribution_caveat": DISTRIBUTION_CAVEAT,
        "configuration": {
            "seed": args.seed,
            "per_class_quotas": quotas,
            "expected_total": expected_total,
            "phash_threshold": args.phash_threshold,
            "phash_jobs": args.jobs,
            "split_policy": (
                "Experiment 2 splits inherit the publisher's official "
                "train/valid/test folders; no image ever crosses that boundary."
            ),
        },
        "pool_sizes_after_dedupe": pool_after_dedupe,
        "deduplication": stats.as_dict(),
        "selected": {
            "total": len(rows),
            "bytes": total_bytes,
            "human_bytes": human_bytes(total_bytes),
            "counts": audit["counts"],
        },
        "min_cross_split_phash_distance": audit["min_cross_split_phash_distance"],
        "audit": audit["checks"],
        "audit_passed": audit["passed"],
    }
    if not args.dry_run:
        write_manifest(output_dir / MANIFEST_FILENAME, rows)
        write_json(output_dir / REPORT_FILENAME, payload)

    print()
    print(LINE)
    print("Build summary")
    print(LINE)
    for split in SPLITS:
        counts = audit["counts"][split]
        print(
            f"  {split:<11}: {counts['real']} real / {counts['fake']} fake "
            f"({counts['total']} total)"
        )
    print(f"  {'TOTAL':<11}: {len(rows)} images, {human_bytes(total_bytes)}")
    print()
    for name, check in audit["checks"].items():
        print(f"  [{'PASS' if check else 'FAIL'}] {name}")
    print(LINE)
    if audit["passed"]:
        print("Dataset build OK.")
    else:
        print("AUDIT FAILED - do not train on this dataset.")
    print(LINE)
    return payload


# ---------------------------------------------------------------------------
# Validation of an existing build
# ---------------------------------------------------------------------------
def validate_only(args: argparse.Namespace) -> int:
    """
    Re-audit an existing build without touching the source dataset.

    Reads the manifest, re-hashes every selected file on disk, recomputes the
    pHashes and re-runs the full audit. This detects corruption, tampering and
    accidental copying between splits after the build. The result is written to
    ``dataset_validation.json`` (the build report is left intact).
    """
    output_dir = args.output_dir.expanduser().resolve()
    processed_dir = output_dir / "processed"
    manifest_path = output_dir / MANIFEST_FILENAME

    print(LINE)
    print("DeepGuard - Experiment 2 dataset validation")
    print(LINE)

    if not manifest_path.is_file():
        print(f"[!] No manifest found: {manifest_path}")
        print("    Build the dataset first:")
        print(
            "    python experiments/experiment_2_large_dataset/build_dataset.py"
        )
        print(LINE)
        return 1

    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"  Manifest  : {manifest_path}")
    print(f"  Rows      : {len(rows)}")
    print(f"  Processed : {processed_dir}")
    print()

    missing: List[str] = []
    corrupted: List[str] = []
    paths: List[Path] = []
    present = 0
    for index, row in enumerate(rows, start=1):
        path = processed_dir / row["split"] / row["label"] / row["image_id"]
        if not path.is_file():
            missing.append(row["image_id"])
            continue
        present += 1
        paths.append(path)
        if md5_file(path) != row["md5"]:
            corrupted.append(row["image_id"])
        if index % 2000 == 0:
            print(f"  Re-hashed {index}/{len(rows)} ...")
    print(f"  Missing files  : {len(missing)}")
    print(f"  MD5 mismatches : {len(corrupted)}")
    print()

    print("  Recomputing pHash for the selected images ...")
    hashes, unreadable = compute_phashes(paths, jobs=args.jobs, progress_every=1000)

    final_rows: List[dict] = []
    for row in rows:
        path = processed_dir / row["split"] / row["label"] / row["image_id"]
        value = hashes.get(path)
        if value is None:
            continue
        final_rows.append({**row, "phash_hex": f"{value:016x}"})

    quotas = {
        "train": args.per_class_train,
        "validation": args.per_class_val,
        "test": args.per_class_test,
    }
    audit = audit_selection(
        final_rows, quotas, phash_threshold=args.phash_threshold
    )

    # Any image on disk that the manifest does not mention. These must be
    # reported, because silently dropping them would hide a planted leak.
    expected_paths = {
        (processed_dir / row["split"] / row["label"] / row["image_id"]).resolve()
        for row in rows
    }
    on_disk_paths: List[Path] = []
    for split in SPLITS:
        for class_name in CLASSES:
            folder = processed_dir / split / class_name
            if not folder.is_dir():
                continue
            on_disk_paths.extend(
                p
                for p in sorted(folder.iterdir(), key=lambda q: q.name.lower())
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
    extra_paths = [
        p for p in on_disk_paths if p.resolve() not in expected_paths
    ]
    on_disk = len(on_disk_paths)

    leaked_extra_names: List[str] = []
    if extra_paths:
        print(
            f"  [i] {len(extra_paths)} unlisted file(s) found - hashing them to "
            "prove they do not duplicate manifest images ..."
        )
        extra_md5 = {md5_file(p) for p in extra_paths}
        extra_hashes, _ = compute_phashes(
            extra_paths, jobs=args.jobs, progress_every=0
        )
        manifest_md5 = {row["md5"] for row in final_rows}
        manifest_phash = np.array(
            [int(row["phash_hex"], 16) for row in final_rows], dtype=np.uint64
        )
        for path in extra_paths:
            duplicate_of_manifest = md5_file(path) in manifest_md5
            value = extra_hashes.get(path)
            if not duplicate_of_manifest and value is not None and manifest_phash.size:
                distance = _hamming_matrix(
                    np.array([value], dtype=np.uint64), manifest_phash
                )
                duplicate_of_manifest = bool(
                    (distance.min() <= args.phash_threshold)
                )
            if duplicate_of_manifest:
                leaked_extra_names.append(path.name)

    checks = dict(audit["checks"])
    checks["no_missing_files"] = not missing
    checks["no_md5_mismatch"] = not corrupted
    checks["no_unreadable_files"] = not unreadable
    checks["no_unexpected_extra_files"] = not extra_paths
    checks["unlisted_files_do_not_duplicate_manifest_images"] = (
        not leaked_extra_names
    )

    print()
    print(LINE)
    print("Validation summary")
    print(LINE)
    for split in SPLITS:
        counts = audit["counts"][split]
        print(
            f"  {split:<11}: {counts['real']} real / {counts['fake']} fake "
            f"({counts['total']} total)"
        )
    print(f"  {'TOTAL':<11}: {on_disk} image files found on disk")
    print()
    print("  Cross-split minimum pHash distance:")
    for pair, value in audit["min_cross_split_phash_distance"].items():
        print(f"    {pair:<22}: {value} bits")
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print()

    passed = all(checks.values())
    write_json(
        output_dir / VALIDATION_FILENAME,
        {
            "experiment": "experiment_2_large_dataset",
            "validated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "rows_in_manifest": len(rows),
            "files_present": present,
            "missing_files": missing[:50],
            "md5_mismatches": corrupted[:50],
            "unreadable_files": [str(p) for p in unreadable[:50]],
            "unlisted_files": [p.name for p in extra_paths[:50]],
            "unlisted_files_duplicating_manifest": leaked_extra_names[:50],
            "counts": audit["counts"],
            "min_cross_split_phash_distance": audit[
                "min_cross_split_phash_distance"
            ],
            "checks": checks,
            "passed": passed,
        },
    )
    print(f"  Written: {output_dir / VALIDATION_FILENAME}")
    print()
    print(LINE)
    print(
        "VALIDATION PASSED - the dataset is ready for training."
        if passed
        else "VALIDATION FAILED - do not train on this dataset."
    )
    print(LINE)
    return 0 if passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            return validate_only(args)
        payload = build(args)
    except (SourceNotFoundError, RuntimeError, ValueError) as error:
        print()
        print(f"[!] {error}")
        print(LINE)
        return 1
    return 0 if payload["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())