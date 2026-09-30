"""
Tests for src/inference.py.

These tests confirm two things:

1. The inference module is HONEST: it refuses to produce a prediction when
   PyTorch is missing, when no real checkpoint exists, or when the checkpoint
   is corrupted/incompatible. These tests run even without PyTorch.
2. When a real (tiny, fixture) checkpoint IS available in the PyTorch
   environment, the module actually loads it and returns a genuine output with
   the correct shapes and class mapping. These tests are skipped when PyTorch
   is not installed.

The tiny checkpoint used in the torch tests is produced inside a temporary
folder by real training code. It is a test fixture, NOT a project result; its
predictions are never reported as project performance.

Run them from the project folder with:

    python -m unittest tests.test_inference -v
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
from src import inference, training  # noqa: E402


def _write_image(path: Path, colour) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)


def _build_tiny_dataset(root: Path) -> None:
    # real images are black, fake images are white, so a trained fixture model
    # is not forced to be degenerate.
    for split_name in ("train", "validation", "test"):
        for index in range(3):
            _write_image(
                root / split_name / "real" / f"real_{index}.png", (0, 0, 0)
            )
            _write_image(
                root / split_name / "fake" / f"fake_{index}.png", (255, 255, 255)
            )


class HonestFailureTests(unittest.TestCase):
    """Inference must refuse rather than fake a result."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_inf_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inference_not_available_without_checkpoint(self) -> None:
        self.assertFalse(inference.inference_available(self.tmp / "models"))

    def test_load_raises_when_checkpoint_missing(self) -> None:
        with self.assertRaises(inference.InferenceNotAvailableError):
            inference.load_trained_model(self.tmp / "models")

    def test_predict_raises_when_checkpoint_missing(self) -> None:
        image = Image.new("RGB", (64, 64), (128, 128, 128))
        with self.assertRaises(inference.InferenceNotAvailableError):
            inference.predict_image(image, checkpoint_dir=self.tmp / "models")

    def test_corrupt_checkpoint_file_is_rejected(self) -> None:
        checkpoint_dir = self.tmp / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = inference.model_interface.get_checkpoint_path(checkpoint_dir)
        path.write_bytes(b"this is not a torch checkpoint at all")

        with self.assertRaises(inference.InferenceNotAvailableError):
            inference.load_trained_model(checkpoint_dir)

    def test_availability_check_never_claims_without_torch(self) -> None:
        # In ANY environment (with or without PyTorch) an empty checkpoint
        # folder must never report inference as available.
        self.assertFalse(inference.inference_available(self.tmp / "nothing"))


@unittest.skipIf(inference.TORCH_AVAILABLE, "requires an environment without PyTorch")
class NoTorchEnvironmentTests(unittest.TestCase):
    """Only meaningful when PyTorch is absent (the application venv)."""

    def test_inference_requires_pytorch(self) -> None:
        image = Image.new("RGB", (64, 64), (128, 128, 128))
        with self.assertRaises(inference.InferenceNotAvailableError) as context:
            inference.preprocess_for_inference(image)
        self.assertIn("PyTorch", str(context.exception))


@unittest.skipUnless(inference.TORCH_AVAILABLE, "PyTorch is not installed")
class TorchInferenceTests(unittest.TestCase):
    """Real loading and prediction using a tiny fixture checkpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="deepguard_inftorch_"))
        cls.processed = cls.tmp / "processed"
        _build_tiny_dataset(cls.processed)
        cls.checkpoint_dir = cls.tmp / "models"
        cls.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cls.checkpoint_path = cls.checkpoint_dir / config.CHECKPOINT_FILENAME

        training.train(
            processed_dir=cls.processed,
            epochs=1,
            batch_size=2,
            num_workers=0,
            seed=0,
            pretrained=False,
            checkpoint_path=cls.checkpoint_path,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_inference_available_with_real_checkpoint(self) -> None:
        self.assertTrue(inference.inference_available(self.checkpoint_dir))

    def test_load_trained_model_returns_two_output_heads(self) -> None:
        model, checkpoint = inference.load_trained_model(self.checkpoint_dir)
        self.assertEqual(model.classifier[1].out_features, config.NUM_CLASSES)
        self.assertEqual(checkpoint["model_name"], config.MODEL_NAME)

    def test_preprocess_returns_batch_tensor(self) -> None:
        image = Image.new("RGB", (64, 64), (200, 100, 50))
        tensor = inference.preprocess_for_inference(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertEqual(str(tensor.dtype).startswith("torch.float"), True)

    def test_preprocess_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            inference.preprocess_for_inference(["not", "an", "image"])

    def test_predict_returns_genuine_output(self) -> None:
        image = Image.new("RGB", (64, 64), (0, 0, 0))
        result = inference.predict_image(image, checkpoint_dir=self.checkpoint_dir)

        self.assertIn(result["prediction_class"], config.CLASS_NAMES)
        self.assertEqual(
            config.LABEL_TO_INDEX[result["prediction_class"]],
            result["prediction_index"],
        )
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_predict_probabilities_sum_to_one(self) -> None:
        image = Image.new("RGB", (64, 64), (255, 255, 255))
        result = inference.predict_image(image, checkpoint_dir=self.checkpoint_dir)

        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=6)
        for name in config.CLASS_NAMES:
            probability = result["probabilities"][name]
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)