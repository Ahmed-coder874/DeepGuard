"""
Tests for src/model_interface.py.

These tests confirm that the model interface is truthful: it must report the
model as unavailable while no checkpoint exists, and it must never return a
prediction or a random probability.

Run them from the project folder with:

    python -m unittest tests.test_model_interface -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import model_interface  # noqa: E402


class ModelInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_model_"))
        self.checkpoint_dir = self.tmp / "models"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reports_unavailable_when_no_checkpoint(self) -> None:
        self.assertFalse(model_interface.is_model_available(self.checkpoint_dir))
        self.assertFalse(model_interface.checkpoint_exists(self.checkpoint_dir))

    def test_status_reports_not_available(self) -> None:
        status = model_interface.get_model_status(self.checkpoint_dir)
        self.assertFalse(status["available"])
        self.assertFalse(status["checkpoint_exists"])
        self.assertIn("not available", status["message"].lower())

    def test_status_uses_configured_architecture(self) -> None:
        status = model_interface.get_model_status(self.checkpoint_dir)
        self.assertEqual(status["architecture"], config.MODEL_NAME)
        self.assertEqual(status["alternative_architecture"], config.ALT_MODEL_NAME)
        self.assertEqual(status["num_classes"], config.NUM_CLASSES)
        self.assertEqual(status["class_names"], list(config.CLASS_NAMES))
        self.assertEqual(status["label_to_index"], dict(config.LABEL_TO_INDEX))

    def test_checkpoint_path_is_inside_checkpoint_dir(self) -> None:
        path = model_interface.get_checkpoint_path(self.checkpoint_dir)
        self.assertEqual(path.parent, self.checkpoint_dir)
        self.assertEqual(path.name, config.CHECKPOINT_FILENAME)

    def test_load_model_raises_when_no_checkpoint(self) -> None:
        with self.assertRaises(model_interface.ModelNotAvailableError):
            model_interface.load_model(self.checkpoint_dir)

    def test_predict_never_returns_a_value(self) -> None:
        # A real prediction must not be possible yet.
        with self.assertRaises(model_interface.ModelNotAvailableError):
            model_interface.predict("anything")

    def test_existing_file_changes_availability_flag_only(self) -> None:
        # A dummy file must flip "available" but must NOT enable inference.
        dummy = model_interface.get_checkpoint_path(self.checkpoint_dir)
        dummy.write_bytes(b"not a real checkpoint")

        status = model_interface.get_model_status(self.checkpoint_dir)
        self.assertTrue(status["available"])

        with self.assertRaises(model_interface.InferenceNotImplementedError):
            model_interface.load_model(self.checkpoint_dir)

    def test_format_status_message_returns_text(self) -> None:
        message = model_interface.format_status_message(
            model_interface.get_model_status(self.checkpoint_dir)
        )
        self.assertIsInstance(message, str)
        self.assertGreater(len(message), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
