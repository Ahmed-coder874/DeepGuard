"""
Tests for src/dataset.py.

Run them from the project folder with:

    python -m unittest discover -s tests -v

These tests use small images created on the fly in a temporary folder, so
they never touch your real dataset.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Make the project importable when the tests are run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import dataset  # noqa: E402


def make_image(path: Path, size: tuple[int, int] = (64, 64), colour: int = 120) -> None:
    """Create a small valid PNG image at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), colour, dtype="uint8")
    Image.fromarray(array).save(path)


class DatasetTestCase(unittest.TestCase):
    """Base class that sets up a temporary raw dataset."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_test_"))
        self.real_dir = self.tmp / "raw" / "real"
        self.fake_dir = self.tmp / "raw" / "fake"
        self.processed_dir = self.tmp / "processed"
        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class ScanTests(DatasetTestCase):
    def test_valid_real_image(self) -> None:
        make_image(self.real_dir / "real_1.png")
        scan = dataset.scan_class_folder(self.real_dir)
        self.assertTrue(scan.exists)
        self.assertEqual(scan.valid_count, 1)
        self.assertEqual(scan.invalid_count, 0)
        self.assertEqual(scan.unsupported_count, 0)

    def test_valid_fake_image(self) -> None:
        make_image(self.fake_dir / "fake_1.png")
        scan = dataset.scan_class_folder(self.fake_dir)
        self.assertEqual(scan.valid_count, 1)

    def test_corrupt_image_is_reported(self) -> None:
        corrupt = self.real_dir / "broken.jpg"
        corrupt.write_bytes(b"this is not a real jpeg file")
        scan = dataset.scan_class_folder(self.real_dir)
        self.assertEqual(scan.valid_count, 0)
        self.assertEqual(scan.invalid_count, 1)
        self.assertEqual(scan.unsupported_count, 0)

    def test_unsupported_file_is_reported(self) -> None:
        (self.real_dir / "notes.txt").write_text("hello", encoding="utf-8")
        scan = dataset.scan_class_folder(self.real_dir)
        self.assertEqual(scan.valid_count, 0)
        self.assertEqual(scan.unsupported_count, 1)

    def test_empty_directory(self) -> None:
        scan = dataset.scan_class_folder(self.real_dir)
        self.assertTrue(scan.exists)
        self.assertEqual(scan.total_count, 0)

    def test_missing_directory(self) -> None:
        missing = self.tmp / "does_not_exist"
        scan = dataset.scan_class_folder(missing)
        self.assertFalse(scan.exists)
        self.assertEqual(scan.total_count, 0)

    def test_find_images_is_recursive_and_sorted(self) -> None:
        make_image(self.real_dir / "b" / "two.png")
        make_image(self.real_dir / "a" / "one.png")
        found = dataset.find_images(self.real_dir)
        self.assertEqual(len(found), 2)
        self.assertEqual([p.name for p in found], ["one.png", "two.png"])

    def test_delete_invalid_is_off_by_default(self) -> None:
        corrupt = self.real_dir / "broken.jpg"
        corrupt.write_bytes(b"not an image")
        deleted = dataset.delete_invalid_files([corrupt])
        self.assertEqual(deleted, 0)
        self.assertTrue(corrupt.exists())


class SplitTests(DatasetTestCase):
    def _make_flat_files(self, folder: Path, count: int, prefix: str) -> None:
        for index in range(count):
            make_image(folder / f"{prefix}_{index:03d}.png")

    def test_split_creates_all_three_splits(self) -> None:
        self._make_flat_files(self.real_dir, 20, "real")
        self._make_flat_files(self.fake_dir, 20, "fake")

        splits = dataset.create_splits(
            real_files=dataset.find_images(self.real_dir),
            fake_files=dataset.find_images(self.fake_dir),
            real_root=self.real_dir,
            fake_root=self.fake_dir,
            max_per_class=None,
        )

        for split_name in config.SPLIT_NAMES:
            self.assertIn(split_name, splits)
            self.assertGreater(len(splits[split_name]["real"]), 0)
            self.assertGreater(len(splits[split_name]["fake"]), 0)

    def test_class_counts_are_preserved(self) -> None:
        self._make_flat_files(self.real_dir, 20, "real")
        self._make_flat_files(self.fake_dir, 20, "fake")

        splits = dataset.create_splits(
            real_files=dataset.find_images(self.real_dir),
            fake_files=dataset.find_images(self.fake_dir),
            real_root=self.real_dir,
            fake_root=self.fake_dir,
            max_per_class=None,
        )

        total_real = sum(splits[s]["real"].__len__() for s in config.SPLIT_NAMES)
        total_fake = sum(splits[s]["fake"].__len__() for s in config.SPLIT_NAMES)
        self.assertEqual(total_real, 20)
        self.assertEqual(total_fake, 20)

    def test_deterministic_splitting_same_seed(self) -> None:
        self._make_flat_files(self.real_dir, 20, "real")
        self._make_flat_files(self.fake_dir, 20, "fake")
        real = dataset.find_images(self.real_dir)
        fake = dataset.find_images(self.fake_dir)

        first = dataset.create_splits(real, fake, self.real_dir, self.fake_dir, seed=7)
        second = dataset.create_splits(real, fake, self.real_dir, self.fake_dir, seed=7)

        self.assertEqual(dataset.all_split_paths(first), dataset.all_split_paths(second))

    def test_different_seed_changes_split(self) -> None:
        self._make_flat_files(self.real_dir, 40, "real")
        self._make_flat_files(self.fake_dir, 40, "fake")
        real = dataset.find_images(self.real_dir)
        fake = dataset.find_images(self.fake_dir)

        first = dataset.create_splits(real, fake, self.real_dir, self.fake_dir, seed=1)
        second = dataset.create_splits(real, fake, self.real_dir, self.fake_dir, seed=2)

        self.assertNotEqual(
            dataset.all_split_paths(first), dataset.all_split_paths(second)
        )

    def test_no_duplicate_paths_between_splits(self) -> None:
        self._make_flat_files(self.real_dir, 20, "real")
        self._make_flat_files(self.fake_dir, 20, "fake")

        splits = dataset.create_splits(
            dataset.find_images(self.real_dir),
            dataset.find_images(self.fake_dir),
            self.real_dir,
            self.fake_dir,
            max_per_class=None,
        )

        all_paths = dataset.all_split_paths(splits)
        self.assertEqual(len(all_paths), len(set(all_paths)))
        self.assertEqual(len(all_paths), 40)

    def test_groups_are_kept_together(self) -> None:
        # Two "videos" (sub-folders) with several frames each.
        for group in ("videoA", "videoB"):
            for frame in range(5):
                make_image(self.real_dir / group / f"frame_{frame}.png")
        for group in ("videoC", "videoD"):
            for frame in range(5):
                make_image(self.fake_dir / group / f"frame_{frame}.png")

        splits = dataset.create_splits(
            dataset.find_images(self.real_dir),
            dataset.find_images(self.fake_dir),
            self.real_dir,
            self.fake_dir,
            max_per_class=None,
        )

        for split_name in config.SPLIT_NAMES:
            for class_name in config.CLASS_NAMES:
                groups = {
                    dataset.group_key(path, self.real_dir if class_name == "real" else self.fake_dir)
                    for path in splits[split_name][class_name]
                }
                # Every frame in a split must come from a distinct video group.
                self.assertEqual(
                    len(groups), len(splits[split_name][class_name]) // 5
                )

    def test_limit_files_is_deterministic(self) -> None:
        self._make_flat_files(self.real_dir, 30, "real")
        files = dataset.find_images(self.real_dir)
        first = dataset.limit_files(files, 10, seed=3)
        second = dataset.limit_files(files, 10, seed=3)
        self.assertEqual(len(first), 10)
        self.assertEqual(first, second)

    def test_max_per_class_limits_split(self) -> None:
        self._make_flat_files(self.real_dir, 30, "real")
        self._make_flat_files(self.fake_dir, 30, "fake")

        splits = dataset.create_splits(
            dataset.find_images(self.real_dir),
            dataset.find_images(self.fake_dir),
            self.real_dir,
            self.fake_dir,
            max_per_class=10,
        )
        self.assertEqual(len(dataset.all_split_paths(splits)), 20)


class CopyAndSummaryTests(DatasetTestCase):
    def test_copy_splits_creates_files(self) -> None:
        for index in range(10):
            make_image(self.real_dir / f"real_{index}.png")
            make_image(self.fake_dir / f"fake_{index}.png")

        splits = dataset.create_splits(
            dataset.find_images(self.real_dir),
            dataset.find_images(self.fake_dir),
            self.real_dir,
            self.fake_dir,
            max_per_class=None,
        )
        dataset.copy_splits_to_processed(splits, self.processed_dir)

        stats = dataset.get_processed_statistics(self.processed_dir)
        total = sum(stats[s][c] for s in config.SPLIT_NAMES for c in config.CLASS_NAMES)
        self.assertEqual(total, 20)
        self.assertTrue(dataset.processed_dataset_exists(self.processed_dir))

        # The original files must still be there.
        self.assertEqual(len(dataset.find_images(self.real_dir)), 10)

    def test_summary_round_trip(self) -> None:
        make_image(self.real_dir / "real_1.png")
        make_image(self.fake_dir / "fake_1.png")
        real_scan = dataset.scan_class_folder(self.real_dir)
        fake_scan = dataset.scan_class_folder(self.fake_dir)

        splits = dataset.create_splits(
            real_scan.valid_files,
            fake_scan.valid_files,
            self.real_dir,
            self.fake_dir,
            max_per_class=None,
        )
        dataset.copy_splits_to_processed(splits, self.processed_dir)
        stats = dataset.get_processed_statistics(self.processed_dir)

        summary = dataset.build_summary(
            real_scan, fake_scan, stats, seed=42,
            train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15,
            max_per_class=None, copied=True,
        )
        dataset.write_summary(summary, self.processed_dir)
        loaded = dataset.load_summary(self.processed_dir)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["totals"]["real"], 1)
        self.assertEqual(loaded["totals"]["fake"], 1)

    def test_load_summary_missing_returns_none(self) -> None:
        self.assertIsNone(dataset.load_summary(self.processed_dir))

    def test_get_dataset_status_without_dataset(self) -> None:
        status = dataset.get_dataset_status(
            processed_dir=self.processed_dir,
            raw_real_dir=self.tmp / "no_real",
            raw_fake_dir=self.tmp / "no_fake",
        )
        self.assertIsNone(status["summary"])
        self.assertFalse(status["processed_exists"])
        self.assertEqual(status["raw_real_count"], 0)


class ConfigTests(unittest.TestCase):
    def test_default_ratios_add_up_to_one(self) -> None:
        config.validate_ratios(0.7, 0.15, 0.15)  # should not raise

    def test_invalid_ratios_raise(self) -> None:
        with self.assertRaises(ValueError):
            config.validate_ratios(0.7, 0.7, 0.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
