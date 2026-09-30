"""
Tests for src/extraction.py (FaceForensics++ frame extraction) and for the
group/identity-aligned splitting and leakage audit in src/dataset.py.

Run from the project folder with:

    python -m unittest discover -s tests -v

Frame-extraction tests use tiny synthetic videos created with OpenCV in a
temporary folder; they never touch the real downloaded dataset.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make the project importable when the tests are run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import dataset  # noqa: E402
from src import extraction  # noqa: E402


def make_short_video(path: Path, frames: int = 50) -> int:
    """Create a small synthetic video and return the number of encoded frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps=25,
        frameSize=(64, 64),
    )
    try:
        for index in range(frames):
            frame = np.full((64, 64, 3), index * 5 % 256, dtype="uint8")
            writer.write(frame)
    finally:
        writer.release()
    return frames


class ExtractionUnitTests(unittest.TestCase):
    def test_parse_original_name(self) -> None:
        self.assertEqual(
            extraction.parse_video_name("183"), ("original", "183", None)
        )

    def test_parse_manipulated_name(self) -> None:
        self.assertEqual(
            extraction.parse_video_name("183_253"), ("manipulated", "183", "253")
        )

    def test_parse_invalid_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            extraction.parse_video_name("not_a_video")

    def test_select_frame_indices_all_when_under_limit(self) -> None:
        self.assertEqual(extraction.select_frame_indices(10, 30), list(range(10)))

    def test_select_frame_indices_deterministic(self) -> None:
        first = extraction.select_frame_indices(100, 30)
        second = extraction.select_frame_indices(100, 30)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 99)

    def test_select_frame_indices_empty_video(self) -> None:
        self.assertEqual(extraction.select_frame_indices(0, 30), [])

    def test_component_key_is_sorted_deduped(self) -> None:
        ids = ["253", "183", "253"]
        self.assertTrue(extraction._component_key(ids).endswith("_253"))
        self.assertEqual(len(set(extraction._component_key(ids).split("_"))), 2)


class IdentityComponentTests(unittest.TestCase):
    def test_bidirectional_manipulation_builds_component(self) -> None:
        videos = [
            extraction.FfppVideo(
                path=Path("fakes/183_253.mp4"), kind="manipulated",
                method="Deepfakes", sequence_id="183", source_id="253", label="fake",
            ),
            extraction.FfppVideo(
                path=Path("fakes/253_183.mp4"), kind="manipulated",
                method="Deepfakes", sequence_id="253", source_id="183", label="fake",
            ),
            extraction.FfppVideo(
                path=Path("originals/183.mp4"), kind="original",
                method=None, sequence_id="183", source_id=None, label="real",
            ),
        ]
        videos, components = extraction.assign_components(videos)
        self.assertEqual(len(components), 1)
        comp = next(iter(components))
        self.assertEqual(set(components[comp]), {"183", "253"})
        for video in videos:
            self.assertEqual(video.component_id, comp)

    def test_two_independent_pairs_are_two_components(self) -> None:
        videos = [
            extraction.FfppVideo(
                path=Path("fakes/183_253.mp4"), kind="manipulated",
                method="Deepfakes", sequence_id="183", source_id="253", label="fake",
            ),
            extraction.FfppVideo(
                path=Path("fakes/585_599.mp4"), kind="manipulated",
                method="Deepfakes", sequence_id="585", source_id="599", label="fake",
            ),
        ]
        videos, components = extraction.assign_components(videos)
        self.assertEqual(len(components), 2)


class ExtractionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_extract_"))
        self.original_videos = self.tmp / "original" / "videos"
        self.fake_videos = self.tmp / "fake" / "videos"
        self.raw_dir = self.tmp / "raw"
        self.raw_real = self.raw_dir / "real"
        self.raw_fake = self.raw_dir / "fake"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extract_all_writes_grouped_frames_and_manifest(self) -> None:
        video_a = self.tmp / "orig_a.mp4"
        video_b = self.tmp / "orig_b.mp4"
        fake_ab = self.tmp / "fake_ab.mp4"
        make_short_video(video_a)
        make_short_video(video_b)
        make_short_video(fake_ab)

        videos = [
            extraction.FfppVideo(path=video_a, kind="original", method=None,
                                 sequence_id="A", source_id=None, label="real",
                                 component_id="A_B"),
            extraction.FfppVideo(path=video_b, kind="original", method=None,
                                 sequence_id="B", source_id=None, label="real",
                                 component_id="A_B"),
            extraction.FfppVideo(path=fake_ab, kind="manipulated", method="Deepfakes",
                                 sequence_id="A", source_id="B", label="fake",
                                 component_id="A_B"),
        ]
        count = 0
        for video in videos:
            class_dir = self.raw_real if video.label == "real" else self.raw_fake
            out = class_dir / (video.component_id or "")
            result = extraction.extract_video_frames(video, out, max_frames=15)
            self.assertEqual(result.skipped_existing, 0)
            self.assertGreater(result.extracted, 0)
            count += result.extracted

        # 3 videos x up to 15 frames -> between 15 and 45 files.
        total_real = len(list(self.raw_real.rglob("*.png")))
        total_fake = len(list(self.raw_fake.rglob("*.png")))
        self.assertGreater(total_real, 0)
        self.assertGreater(total_fake, 0)
        self.assertEqual(total_real + total_fake, count)

        # Every fake frame must sit under the SAME component folder name as real.
        fake_folders = {p.parent.name for p in self.raw_fake.rglob("*.png")}
        real_folders = {p.parent.name for p in self.raw_real.rglob("*.png")}
        self.assertEqual(fake_folders, {"A_B"})
        self.assertEqual(real_folders, {"A_B"})

    def test_write_manifest_round_trip(self) -> None:
        from datetime import datetime

        report = extraction.ExtractionReport()
        report.videos = [
            extraction.FfppVideo(path=Path("x/183.mp4"), kind="original",
                                 method=None, sequence_id="183", source_id=None,
                                 label="real", component_id="183"),
        ]
        manifest = extraction.build_manifest(report, generated_at=datetime.now().isoformat())
        path = extraction.write_manifest(manifest, raw_dir=self.raw_dir)
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        self.assertEqual(loaded["videos"][0]["name"], "183.mp4")


class GroupAlignedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="deepguard_aligned_"))
        self.real_dir = self.tmp / "raw" / "real"
        self.fake_dir = self.tmp / "raw" / "fake"
        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_grouped(self) -> None:
        for comp in ("183_253", "469_481"):
            for frame in range(10):
                img = np.full((16, 16, 3), 120, dtype="uint8")
                from PIL import Image

                real_comp = self.real_dir / comp
                fake_comp = self.fake_dir / comp
                real_comp.mkdir(parents=True, exist_ok=True)
                fake_comp.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(real_comp / f"f{frame}.png")
                Image.fromarray(img).save(fake_comp / f"f{frame}.png")

    def test_group_aligned_keeps_shared_key_in_one_split(self) -> None:
        self._make_grouped()
        real = dataset.find_images(self.real_dir)
        fake = dataset.find_images(self.fake_dir)

        splits = dataset.create_splits(
            real, fake, self.real_dir, self.fake_dir,
            max_per_class=None, aligned_groups=True,
        )

        # Every real/fake frame with the same component folder must be in the
        # SAME split.
        for comp in ("183_253", "469_481"):
            comp_real_splits = {
                s for s in config.SPLIT_NAMES
                for p in splits[s]["real"] if dataset.group_key(p, self.real_dir) == comp
            }
            comp_fake_splits = {
                s for s in config.SPLIT_NAMES
                for p in splits[s]["fake"] if dataset.group_key(p, self.fake_dir) == comp
            }
            self.assertEqual(len(comp_real_splits), 1)
            self.assertEqual(comp_real_splits, comp_fake_splits)

    def test_group_aligned_fills_every_split_when_enough_components(self) -> None:
        # Three equal identity components must populate train, validation AND
        # test (regression: the old greedy loop could leave test empty when
        # only a few large atomic groups existed).
        for comp in ("1_2", "3_4", "5_6"):
            for frame in range(10):
                real_comp = self.real_dir / comp
                fake_comp = self.fake_dir / comp
                real_comp.mkdir(parents=True, exist_ok=True)
                fake_comp.mkdir(parents=True, exist_ok=True)
                img = np.full((16, 16, 3), 90, dtype="uint8")
                from PIL import Image

                Image.fromarray(img).save(real_comp / f"f{frame}.png")
                Image.fromarray(img).save(fake_comp / f"f{frame}.png")

        splits = dataset.create_splits(
            dataset.find_images(self.real_dir),
            dataset.find_images(self.fake_dir),
            self.real_dir,
            self.fake_dir,
            max_per_class=None,
            aligned_groups=True,
        )
        for split_name in config.SPLIT_NAMES:
            self.assertGreater(len(splits[split_name]["real"]), 0, split_name)
            self.assertGreater(len(splits[split_name]["fake"]), 0, split_name)
        audit = dataset.audit_split_leakage(splits, self.real_dir, self.fake_dir)
        self.assertTrue(audit["ok"])

    def test_audit_split_leakage_reports_ok(self) -> None:
        self._make_grouped()
        splits = dataset.create_splits(
            dataset.find_images(self.real_dir),
            dataset.find_images(self.fake_dir),
            self.real_dir,
            self.fake_dir,
            max_per_class=None,
            aligned_groups=True,
        )
        audit = dataset.audit_split_leakage(splits, self.real_dir, self.fake_dir)
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["leaked_groups"], {})
        self.assertEqual(audit["group_count"], 2)

    def test_parse_component_id(self) -> None:
        self.assertEqual(dataset.parse_component_id("000000_183_253__f__frame_000001.png"), "183_253")
        self.assertEqual(dataset.parse_component_id("000000_183__f__frame_000001.png"), "183")


if __name__ == "__main__":
    unittest.main(verbosity=2)