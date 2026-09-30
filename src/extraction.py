"""
extraction.py
=============
Frame extraction from the downloaded FaceForensics++ videos.

This module turns the official FaceForensics++ MP4 videos into the image
samples that the existing DeepGuard image pipeline expects:

    data/raw/real/<component>/<frame>.png     (original_sequences/youtube)
    data/raw/fake/<component>/<frame>.png     (manipulated_sequences/Deepfakes)

Key properties
--------------
* Labels come ONLY from the official FaceForensics++ directory structure:
  ``original_sequences/youtube`` -> real, ``manipulated_sequences/Deepfakes``
  -> fake.  Nothing is inferred from arbitrary file content.
* Identity-group-aware leakage prevention.  Every extracted frame belongs to
  an *identity component*: the connected component of the target/source graph
  formed by the manipulated videos.  For example ``183_253.mp4`` and
  ``253_183.mp4`` create edges between identities 183 and 253, so both frames
  of the original videos 183.mp4 / 253.mp4 AND both manipulated videos belong
  to the same group.  The SAME component id is used as the folder name in
  ``data/raw/real`` and ``data/raw/fake``, which lets the splitter in
  ``src/dataset.py`` keep one component inside exactly one split.
* Deterministic sampling.  Frame indices are chosen as EVENLY-SPACED
  positions and are the same every run (no randomness is involved).
* Provenance.  Frame file names embed the component id and the source video
  name, e.g. ``183_253__183_253__frame_000012.png``, and a JSON manifest
  records the full source path, label, identities and frame index.
* Nothing synthetic is ever created: only frames that were really decoded
  from the downloaded videos are written.

Technique
---------
OpenCV is used because it is already a project dependency (``app venv``) and
its bundled FFmpeg backend decodes the FaceForensics++ h264 MP4 files without
any extra installation.  Frames are written as PNG via Pillow (also already a
dependency) so the pixel order is unambiguous (RGB).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2  # OpenCV ships its own FFmpeg decoding backend.
    from PIL import Image
except ImportError:  # pragma: no cover - only used to decide availability.
    cv2 = None
    Image = None

# Make the project importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


class ExtractionError(Exception):
    """Raised when a video cannot be opened or a frame cannot be decoded."""


@dataclass
class FfppVideo:
    """One downloaded FaceForensics++ video, described by its official path."""

    path: Path
    kind: str                      # "original" or "manipulated"
    method: Optional[str]          # None for original, e.g. "Deepfakes"
    sequence_id: str               # identity of this video (target for manipulated)
    source_id: Optional[str]       # source identity for manipulated videos
    label: str                     # "real" or "fake"
    component_id: Optional[str] = None   # filled by build_identity_components


def parse_video_name(name: str) -> Tuple[str, str, Optional[str]]:
    """
    Interpret a FaceForensics++ video file name (without extension).

    Returns (kind, sequence_id, source_id):
        "183"        -> ("original", "183", None)
        "183_253"    -> ("manipulated", "183", "253")

    Raises ValueError when the name does not follow the documented convention.
    """
    parts = name.split("_")
    if len(parts) == 1:
        if not parts[0].isdigit() or len(parts[0]) == 0:
            raise ValueError(f"Not a valid FF++ sequence id: {name!r}")
        return "original", parts[0], None
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return "manipulated", parts[0], parts[1]
    raise ValueError(f"Not a valid FF++ video name: {name!r}")


def discover_videos(
    original_dir: Path | str = config.FFPP_ORIGINAL_VIDEOS_DIR,
    manipulated_dir: Path | str = config.FFPP_MANIPULATED_VIDEOS_DIR,
) -> List[FfppVideo]:
    """Find and describe every MP4 video in the two official FF++ folders."""
    videos: List[FfppVideo] = []

    original_dir = Path(original_dir)
    if original_dir.is_dir():
        for path in sorted(original_dir.glob("*.mp4")):
            kind, seq, _src = parse_video_name(path.stem)
            videos.append(
                FfppVideo(path=path, kind=kind, method=None,
                          sequence_id=seq, source_id=None, label="real")
            )

    manipulated_dir = Path(manipulated_dir)
    if manipulated_dir.is_dir():
        method = "Deepfakes"  # official method folder we are processing
        for path in sorted(manipulated_dir.glob("*.mp4")):
            kind, seq, src = parse_video_name(path.stem)
            videos.append(
                FfppVideo(path=path, kind=kind, method=method,
                          sequence_id=seq, source_id=src, label="fake")
            )

    return videos


def _component_key(ids: List[str]) -> str:
    """Stable component id built from sorted identity strings."""
    return "_".join(sorted(set(ids)))


def build_identity_components(videos: List[FfppVideo]) -> Dict[str, List[str]]:
    """
    Group identities into leak-free components.

    The manipulated videos form a graph: every video ``T_S`` connects the
    identity T (its target, the person whose sequence was altered) with the
    identity S (its source, the donor).  Two identities belong to the same
    component when they are connected through any chain of such edges, so that
    the whole component can be kept inside a single train/validation/test
    split.  Original videos that do not participate in any manipulated edge
    form a component on their own.

    Returns {component_id: sorted list of identity ids}.
    """
    parent: Dict[str, str] = {}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for video in videos:
        union(video.sequence_id, video.sequence_id)
        if video.source_id is not None:
            union(video.sequence_id, video.source_id)

    components: Dict[str, List[str]] = {}
    for identity in parent:
        root = find(identity)
        components.setdefault(root, []).append(identity)

    return {_component_key(ids): sorted(ids) for ids in components.values()}


def assign_components(videos: List[FfppVideo]) -> Tuple[List[FfppVideo], Dict[str, List[str]]]:
    """
    Attach a component_id to every video and return (videos, components).

    A video belongs to the component that contains its sequence id; for
    manipulated videos the same component also contains the source identity,
    which is exactly what the leakage audit relies on.
    """
    components = build_identity_components(videos)
    id_to_component = {
        identity: comp_id
        for comp_id, identities in components.items()
        for identity in identities
    }
    for video in videos:
        video.component_id = id_to_component.get(video.sequence_id)
        if video.source_id is not None:
            other = id_to_component.get(video.source_id)
            if other is not None and video.component_id != other:  # pragma: no cover
                # Union-find guarantees this cannot happen.
                raise ExtractionError(
                    f"Component mismatch for {video.path.name}: "
                    f"{video.component_id} != {other}"
                )
    return videos, components


def select_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    """
    Deterministically choose up to ``max_frames`` evenly-spaced frame indices.

    ``max_frames <= 0`` or ``max_frames >= total_frames`` returns every index.
    Indices are rounded, de-duplicated and sorted so the same run always picks
    exactly the same frames.
    """
    if total_frames <= 0:
        return []
    if max_frames <= 0 or max_frames >= total_frames:
        return list(range(total_frames))
    if max_frames == 1:
        return [0]
    step = (total_frames - 1) / (max_frames - 1)
    chosen = {round(i * step) for i in range(max_frames)}
    return sorted(chosen)


@dataclass
class FrameExtraction:
    """Statistics about one video's frame extraction."""

    video: Optional[FfppVideo] = None
    total_frames: int = 0
    extracted: int = 0
    skipped_existing: int = 0
    failed_frames: List[int] = field(default_factory=list)
    output_dir: Optional[Path] = None
    frame_files: List[Path] = field(default_factory=list)  # written or reused


def extract_video_frames(
    video: FfppVideo,
    output_dir: Path | str,
    max_frames: int = config.FFPP_FRAMES_PER_VIDEO,
    force: bool = False,
) -> FrameExtraction:
    """
    Decode ``max_frames`` deterministic frames from one video into PNG files.

    Existing files are skipped unless ``force`` is True.  A video that cannot
    be opened, or a frame that cannot be decoded, raises ExtractionError with
    a clear message (nothing is silently dropped).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cv2 is None or Image is None:  # pragma: no cover
        raise ExtractionError("OpenCV and/or Pillow are not installed.")

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        cap.release()
        raise ExtractionError(f"Cannot open video: {video.path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ExtractionError(f"Video has no readable frames: {video.path}")

    result = FrameExtraction(video=video, total_frames=total, output_dir=output_dir)
    for index in select_frame_indices(total, max_frames):
        out_file = output_dir / f"{video.component_id}__{video.path.stem}__frame_{index:06d}.png"
        if out_file.exists() and not force:
            result.skipped_existing += 1
            result.frame_files.append(out_file)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            result.failed_frames.append(index)
            continue
        image = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
        image.save(out_file, format="PNG")
        result.extracted += 1
        result.frame_files.append(out_file)

    cap.release()
    if result.failed_frames:
        raise ExtractionError(
            f"Failed to decode frame(s) {result.failed_frames} from {video.path}"
        )
    return result


@dataclass
class ExtractionReport:
    """Aggregate result of an extraction run."""

    videos: List[FfppVideo] = field(default_factory=list)
    components: Dict[str, List[str]] = field(default_factory=dict)
    extractions: List[FrameExtraction] = field(default_factory=list)
    failed_videos: List[str] = field(default_factory=list)

    @property
    def total_extracted(self) -> int:
        return sum(x.extracted for x in self.extractions)

    @property
    def total_skipped(self) -> int:
        return sum(x.skipped_existing for x in self.extractions)

    @property
    def real_frames(self) -> int:
        return sum(x.extracted for x in self.extractions if x.video.label == "real")

    @property
    def fake_frames(self) -> int:
        return sum(x.extracted for x in self.extractions if x.video.label == "fake")


def extract_all(
    raw_real_dir: Path | str = config.RAW_REAL_DIR,
    raw_fake_dir: Path | str = config.RAW_FAKE_DIR,
    original_dir: Path | str = config.FFPP_ORIGINAL_VIDEOS_DIR,
    manipulated_dir: Path | str = config.FFPP_MANIPULATED_VIDEOS_DIR,
    max_frames: int = config.FFPP_FRAMES_PER_VIDEO,
    force: bool = False,
) -> ExtractionReport:
    """
    Extract every discovered FF++ video into the raw real/fake folders.

    Real videos go to ``raw_real_dir/<component>/`` and fake videos to
    ``raw_fake_dir/<component>/`` so that the same component id exists in both
    class folders (used by the splitter for leakage-free group assignment).
    """
    report = ExtractionReport()
    videos = discover_videos(original_dir, manipulated_dir)
    if not videos:
        raise ExtractionError(
            "No FaceForensics++ videos found. Check that the download exists."
        )
    videos, components = assign_components(videos)
    report.videos = videos
    report.components = components

    for video in videos:
        output_dir = raw_real_dir if video.label == "real" else raw_fake_dir
        output_dir = Path(output_dir) / (video.component_id or "")
        try:
            report.extractions.append(
                extract_video_frames(video, output_dir, max_frames, force)
            )
        except ExtractionError as error:
            report.failed_videos.append(str(error))

    return report


def build_manifest(report: ExtractionReport, generated_at: str) -> dict:
    """Build the JSON-compatible provenance manifest for the extraction run."""
    frames = []
    totals = {}
    for extraction in report.extractions:
        video = extraction.video
        totals[video.path.name] = extraction.total_frames
        for entry in extraction.frame_files:
            name = entry.stem
            parts = name.split("__")  # <component>__<video>__frame_000012
            comp = parts[0] if len(parts) >= 1 else ""
            video_name = parts[1] if len(parts) >= 2 else ""
            frames.append(
                {
                    "manifest_index": len(frames) + 1,
                    "rel_path": str(entry.relative_to(PROJECT_ROOT)),
                    "source_dataset": "FaceForensics++",
                    "compression": config.FFPP_COMPRESSION,
                    "label": video.label,
                    "category": video.kind,
                    "method": video.method,
                    "source_video": video.path.name,
                    "sequence_id": video.sequence_id,
                    "source_id": video.source_id,
                    "component_id": video.component_id,
                    "frame_index": int(parts[2].replace("frame_", "")) if len(parts) >= 3 else -1,
                    "video_frame_count": extraction.total_frames,
                    "video_name": video_name,
                }
            )

    return {
        "dataset": "FaceForensics++",
        "compression": config.FFPP_COMPRESSION,
        "generated_at": generated_at,
        "sampling": {
            "strategy": "evenly spaced (deterministic)",
            "max_frames_per_video": config.FFPP_FRAMES_PER_VIDEO,
        },
        "components": {
            comp_id: ids for comp_id, ids in sorted(report.components.items())
        },
        "videos": [
            {
                "name": v.path.name,
                "label": v.label,
                "category": v.kind,
                "method": v.method,
                "sequence_id": v.sequence_id,
                "source_id": v.source_id,
                "component_id": v.component_id,
                "total_frames": totals.get(v.path.name),
            }
            for v in report.videos
        ],
        "frames": frames,
    }


def write_manifest(manifest: dict, raw_dir: Path | str = config.RAW_DIR) -> Path:
    """Write the provenance manifest JSON and return its path."""
    path = config.get_ffpp_manifest_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return path