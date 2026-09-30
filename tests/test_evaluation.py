"""
Tests for src/evaluation.py.

The metric functions are plain Python and are always tested. The tests that
need PyTorch are skipped automatically when PyTorch is not installed.

Run them from the project folder with:

    python -m unittest tests.test_evaluation -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import evaluation, training  # noqa: E402


def _write_image(path: Path, colour=(120, 120, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)


def _build_tiny_dataset(root: Path) -> None:
    for split_name in ("train", "validation", "test"):
        for class_name in config.CLASS_NAMES:
            for index in range(3):
                _write_image(
                    root / split_name / class_name / f"{class_name}_{index}.png"
                )


class MetricTests(unittest.TestCase):
    def test_confusion_matrix_layout(self) -> None:
        matrix = evaluation.compute_confusion_matrix(
            labels=[0, 0, 1, 1], predictions=[0, 1, 1, 1]
        )
        # rows = actual, columns = predicted
        self.assertEqual(matrix, [[1, 1], [0, 2]])

    def test_metrics_known_values(self) -> None:
        metrics = evaluation.compute_metrics(
            labels=[0, 0, 1, 1], predictions=[0, 1, 1, 1]
        )
        self.assertEqual(metrics["num_samples"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=6)
        self.assertAlmostEqual(metrics["per_class"]["real"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["real"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["fake"]["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["per_class"]["fake"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["macro"]["recall"], 0.75, places=6)

    def test_positive_class_is_fake(self) -> None:
        metrics = evaluation.compute_metrics(labels=[1, 1], predictions=[1, 1])
        self.assertEqual(metrics["positive_class"], "fake")
        self.assertAlmostEqual(metrics["positive"]["recall"], 1.0)

    def test_empty_input_is_safe(self) -> None:
        metrics = evaluation.compute_metrics(labels=[], predictions=[])
        self.assertEqual(metrics["num_samples"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["confusion_matrix"], [[0, 0], [0, 0]])

    def test_no_division_by_zero_when_class_absent(self) -> None:
        metrics = evaluation.compute_metrics(labels=[0, 0], predictions=[0, 0])
        self.assertEqual(metrics["per_class"]["fake"]["support"], 0)
        self.assertEqual(metrics["per_class"]["fake"]["precision"], 0.0)
        self.assertEqual(metrics["per_class"]["fake"]["recall"], 0.0)
        self.assertEqual(metrics["per_class"]["fake"]["f1"], 0.0)


class ReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_eval_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_not_ready_without_checkpoint_or_dataset(self) -> None:
        ready = evaluation.check_evaluation_ready(
            self.tmp / "processed", self.tmp / "models" / "missing.pt"
        )
        self.assertFalse(ready["ready"])
        self.assertFalse(ready["checkpoint_exists"])
        self.assertFalse(ready["test_ok"])

    def test_ready_with_checkpoint_and_test_images(self) -> None:
        processed = self.tmp / "processed"
        _build_tiny_dataset(processed)
        checkpoint = self.tmp / "models" / "model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"placeholder")

        ready = evaluation.check_evaluation_ready(processed, checkpoint)
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["test_counts"]["real"], 3)
        self.assertEqual(ready["test_counts"]["fake"], 3)


@unittest.skipUnless(training.TORCH_AVAILABLE, "PyTorch is not installed")
class TorchEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_evaltorch_"))
        self.processed = self.tmp / "processed"
        _build_tiny_dataset(self.processed)
        self.checkpoint = self.tmp / "models" / "tiny.pt"

        training.train(
            processed_dir=self.processed,
            epochs=1,
            batch_size=2,
            num_workers=0,
            seed=0,
            pretrained=False,
            checkpoint_path=self.checkpoint,
            verbose=False,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_evaluate_returns_real_metrics(self) -> None:
        report = evaluation.evaluate(
            processed_dir=self.processed,
            checkpoint_path=self.checkpoint,
            batch_size=2,
            num_workers=0,
            verbose=False,
        )
        metrics = report["metrics"]
        self.assertEqual(metrics["num_samples"], 6)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]), config.NUM_CLASSES)

    def test_load_checkpoint_missing_raises(self) -> None:
        with self.assertRaises(evaluation.EvaluationNotAvailableError):
            evaluation.load_checkpoint(self.tmp / "does_not_exist.pt")

    def test_evaluate_without_checkpoint_raises(self) -> None:
        with self.assertRaises(evaluation.EvaluationNotAvailableError):
            evaluation.evaluate(
                processed_dir=self.processed,
                checkpoint_path=self.tmp / "missing.pt",
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
