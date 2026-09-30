"""
Streamlit end-to-end tests for app.py.

These tests actually run the Streamlit application with Streamlit's official
testing helper (AppTest) and simulate uploading files and clicking buttons.
They are the same checks used in Stage 1, now kept inside the project so they
can always be re-run.

Run them from the project folder with:

    python -m unittest tests.test_app -v

The app now runs the real trained model, so the Analyze flow is expected to
produce a genuine prediction block (class + confidence). No fake or placeholder
prediction is ever asserted.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_PATH = str(PROJECT_ROOT / "app.py")


def _texts(app_test: AppTest) -> str:
    """Collect all visible text from the app so we can search it."""
    parts = []
    for attribute in (
        "markdown", "success", "warning", "error", "info", "caption",
        "subheader", "title", "text",
    ):
        for element in getattr(app_test, attribute, []):
            value = getattr(element, "value", "")
            if value:
                parts.append(str(value))
    return "\n".join(parts)


class AppEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="deepguard_app_"))

        # Valid JPG
        Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype="uint8")
        ).save(cls.tmp / "sample.jpg", "JPEG")

        # Valid PNG
        Image.fromarray(
            np.random.randint(0, 255, (300, 200, 3), dtype="uint8")
        ).save(cls.tmp / "sample.png", "PNG")

        # Too small
        Image.fromarray(
            np.random.randint(0, 255, (10, 10, 3), dtype="uint8")
        ).save(cls.tmp / "tiny.png", "PNG")

        # Unsupported extension
        (cls.tmp / "notes.txt").write_text("hello", encoding="utf-8")

        # Corrupted file pretending to be a JPG
        (cls.tmp / "corrupt.jpg").write_bytes(b"this is not a real jpeg")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _read(self, name: str) -> bytes:
        return (self.tmp / name).read_bytes()

    def _new_app(self) -> AppTest:
        app = AppTest.from_file(APP_PATH, default_timeout=60)
        app.run()
        return app

    # -- Test 1: valid JPG upload -----------------------------------------
    def test_valid_jpg_upload_shows_original_image(self) -> None:
        app = self._new_app()
        app.file_uploader[0].upload(
            "sample.jpg", self._read("sample.jpg"), "image/jpeg"
        ).run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertTrue(any("successfully loaded" in s.value for s in app.success))
        self.assertIn("Original Image", _texts(app))
        self.assertIn("sample.jpg", _texts(app))
        self.assertTrue(any(b.label == "Analyze Image" for b in app.button))

    # -- Test 2: Analyze button runs preprocessing -------------------------
    def test_analyze_button_runs_preprocessing(self) -> None:
        app = self._new_app()
        app.file_uploader[0].upload(
            "sample.jpg", self._read("sample.jpg"), "image/jpeg"
        ).run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertTrue(any("Preprocessing completed" in s.value for s in app.success))
        self.assertIn("Preprocessed Image", _texts(app))
        self.assertIn("224 x 224", _texts(app))
        self.assertIn("0 - 1", _texts(app))
        self.assertIn("(1, 224, 224, 3)", _texts(app))
        # The Analyze flow must now produce a real prediction block.
        self.assertIn("Prediction:", _texts(app))
        self.assertIn("Model confidence:", _texts(app))
        self.assertIn("EfficientNet-B0", _texts(app))
        self.assertNotIn("Not available in the current prototype", _texts(app))

    # -- Test 3: unsupported file -----------------------------------------
    def test_unsupported_file_is_rejected(self) -> None:
        app = self._new_app()
        app.file_uploader[0].upload(
            "notes.txt", self._read("notes.txt"), "text/plain"
        ).run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertTrue(
            any("Unsupported file type" in e.value for e in app.error),
            [e.value for e in app.error],
        )
        self.assertFalse(any(b.label == "Analyze Image" for b in app.button))

    # -- Test 4: corrupted image ------------------------------------------
    def test_corrupted_image_is_rejected(self) -> None:
        app = self._new_app()
        app.file_uploader[0].upload(
            "corrupt.jpg", self._read("corrupt.jpg"), "image/jpeg"
        ).run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertTrue(
            any("not a readable image" in e.value for e in app.error),
            [e.value for e in app.error],
        )

    # -- Test 5: too small image ------------------------------------------
    def test_tiny_image_is_rejected(self) -> None:
        app = self._new_app()
        app.file_uploader[0].upload(
            "tiny.png", self._read("tiny.png"), "image/png"
        ).run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertTrue(
            any("very small" in e.value for e in app.error),
            [e.value for e in app.error],
        )

    # -- Test 6: valid PNG upload and analyze -----------------------------
    def test_valid_png_upload_and_analyze(self) -> None:
        app = self._new_app()
        app.file_uploader[0].upload(
            "sample.png", self._read("sample.png"), "image/png"
        ).run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertTrue(any("Preprocessing completed" in s.value for s in app.success))
        self.assertIn("PNG", _texts(app))

    # -- Dataset status section (Stage 2) ---------------------------------
    def test_dataset_status_section_present(self) -> None:
        app = self._new_app()
        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertIn("Dataset Status", _texts(app))


if __name__ == "__main__":
    unittest.main(verbosity=2)
