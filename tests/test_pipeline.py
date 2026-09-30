"""
Tests for src/validation.py and src/preprocessing.py (Stage 1 functionality).

Run them from the project folder with:

    python -m unittest tests.test_pipeline -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import preprocessing, validation  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="deepguard_pipeline_"))

        Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype="uint8")
        ).save(cls.tmp / "sample.jpg", "JPEG")
        Image.fromarray(
            np.random.randint(0, 255, (300, 200, 3), dtype="uint8")
        ).save(cls.tmp / "sample.png", "PNG")
        Image.fromarray(
            np.random.randint(0, 255, (10, 10, 3), dtype="uint8")
        ).save(cls.tmp / "tiny.png", "PNG")
        (cls.tmp / "notes.txt").write_text("hello", encoding="utf-8")
        (cls.tmp / "corrupt.jpg").write_bytes(b"not a real jpeg")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _read(self, name: str) -> bytes:
        return (self.tmp / name).read_bytes()

    def test_valid_jpg(self) -> None:
        info = validation.validate_uploaded_file(
            "sample.jpg", "image/jpeg", self._read("sample.jpg")
        )
        self.assertEqual(info.image_format, "JPEG")
        self.assertEqual((info.width, info.height), (640, 480))
        self.assertEqual(info.channels, 3)

    def test_valid_png(self) -> None:
        info = validation.validate_uploaded_file(
            "sample.png", "image/png", self._read("sample.png")
        )
        self.assertEqual(info.image_format, "PNG")

    def test_tiny_image_rejected(self) -> None:
        with self.assertRaises(validation.ImageValidationError):
            validation.validate_uploaded_file(
                "tiny.png", "image/png", self._read("tiny.png")
            )

    def test_unsupported_extension_rejected(self) -> None:
        with self.assertRaises(validation.ImageValidationError):
            validation.validate_uploaded_file(
                "notes.txt", "text/plain", self._read("notes.txt")
            )

    def test_corrupt_image_rejected(self) -> None:
        with self.assertRaises(validation.ImageValidationError):
            validation.validate_uploaded_file(
                "corrupt.jpg", "image/jpeg", self._read("corrupt.jpg")
            )

    def test_empty_file_rejected(self) -> None:
        with self.assertRaises(validation.ImageValidationError):
            validation.validate_uploaded_file("empty.png", "image/png", b"")

    def test_preprocessing_produces_model_input(self) -> None:
        result = preprocessing.preprocess_image(self._read("sample.jpg"))

        self.assertEqual(result["original_size"], (640, 480))
        self.assertEqual(result["processed_size"], (224, 224))
        self.assertEqual(result["resized_rgb"].shape, (224, 224, 3))
        self.assertEqual(result["resized_rgb"].dtype, np.uint8)
        self.assertEqual(result["normalized"].dtype, np.float32)
        self.assertGreaterEqual(result["value_range"][0], 0.0)
        self.assertLessEqual(result["value_range"][1], 1.0)
        self.assertEqual(result["model_input_shape"], (1, 224, 224, 3))

    def test_preprocessing_rejects_corrupt_bytes(self) -> None:
        with self.assertRaises(preprocessing.PreprocessingError):
            preprocessing.preprocess_image(b"definitely not an image")


if __name__ == "__main__":
    unittest.main(verbosity=2)
