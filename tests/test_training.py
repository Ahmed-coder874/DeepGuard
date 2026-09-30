"""
Tests for src/training.py.

The dataset-readiness helpers are always tested. The tests that need PyTorch
are skipped automatically when PyTorch is not installed (for example in the
Streamlit application environment, where only "venv-train" has torch).

Run them from the project folder with:

    python -m unittest tests.test_training -v
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
from src import training  # noqa: E402


def _write_image(path: Path, colour=(120, 120, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)


def _build_tiny_dataset(root: Path) -> None:
    for split_name in ("train", "validation", "test"):
        for class_name in config.CLASS_NAMES:
            for index in range(2):
                _write_image(
                    root / split_name / class_name / f"{class_name}_{index}.png"
                )


class DatasetReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_train_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_dataset_is_not_ready(self) -> None:
        status = training.check_dataset_ready(self.tmp / "processed")
        self.assertFalse(status["ready"])
        self.assertFalse(status["train_ok"])
        self.assertFalse(status["validation_ok"])

    def test_ready_dataset_is_detected(self) -> None:
        processed = self.tmp / "processed"
        _build_tiny_dataset(processed)

        status = training.check_dataset_ready(processed)
        self.assertTrue(status["ready"])
        self.assertTrue(status["train_ok"])
        self.assertTrue(status["validation_ok"])
        self.assertEqual(status["counts"]["train"]["real"], 2)
        self.assertEqual(status["counts"]["train"]["fake"], 2)

    def test_one_empty_class_is_not_ready(self) -> None:
        processed = self.tmp / "processed"
        _build_tiny_dataset(processed)
        shutil.rmtree(processed / "train" / "fake")

        status = training.check_dataset_ready(processed)
        self.assertFalse(status["ready"])

    def test_torch_available_flag_is_boolean(self) -> None:
        self.assertIsInstance(training.TORCH_AVAILABLE, bool)


@unittest.skipUnless(training.TORCH_AVAILABLE, "PyTorch is not installed")
class TorchTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_torch_"))
        self.processed = self.tmp / "processed"
        _build_tiny_dataset(self.processed)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dataset_uses_configured_label_mapping(self) -> None:
        dataset = training.DeepGuardDataset(
            self.processed / "train", training.get_eval_transforms()
        )
        self.assertEqual(len(dataset), 4)

        labels = {config.INDEX_TO_LABEL[label] for _path, label in dataset.samples}
        self.assertEqual(labels, set(config.CLASS_NAMES))

        counts = dataset.class_counts()
        self.assertEqual(counts["real"], 2)
        self.assertEqual(counts["fake"], 2)

    def test_build_model_has_two_outputs(self) -> None:
        model = training.build_model(pretrained=False)
        # EfficientNet-B0: classifier[1] is the final Linear layer.
        self.assertEqual(model.classifier[1].out_features, config.NUM_CLASSES)

    def test_unknown_model_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            training.build_model(model_name="not_a_model", pretrained=False)

    def test_one_epoch_runs_and_returns_finite_values(self) -> None:
        import torch
        from torch import nn

        training.set_seed(0)
        train_loader, validation_loader = training.build_dataloaders(
            self.processed, batch_size=2, num_workers=0
        )
        model = training.build_model(pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_loss, train_accuracy = training.train_one_epoch(
            model, train_loader, criterion, optimizer, torch.device("cpu")
        )
        validation_loss, validation_accuracy = training.validate_one_epoch(
            model, validation_loader, criterion, torch.device("cpu")
        )

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(validation_loss, 0.0)
        self.assertGreaterEqual(train_accuracy, 0.0)
        self.assertLessEqual(train_accuracy, 1.0)
        self.assertGreaterEqual(validation_accuracy, 0.0)
        self.assertLessEqual(validation_accuracy, 1.0)

    def test_train_writes_a_checkpoint_and_history(self) -> None:
        checkpoint = self.tmp / "models" / "test.pt"
        result = training.train(
            processed_dir=self.processed,
            epochs=1,
            batch_size=2,
            num_workers=0,
            seed=0,
            pretrained=False,
            checkpoint_path=checkpoint,
            verbose=False,
        )

        self.assertTrue(checkpoint.is_file())
        self.assertEqual(result["epochs_completed"], 1)
        self.assertEqual(len(result["history"]["train_loss"]), 1)

    def test_train_refuses_without_dataset(self) -> None:
        with self.assertRaises(RuntimeError):
            training.train(
                processed_dir=self.tmp / "does_not_exist",
                epochs=1,
                pretrained=False,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
