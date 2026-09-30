"""
extract_dataset_frames.py
=========================
Extract deterministic, evenly-spaced frames from the downloaded
FaceForensics++ videos into the DeepGuard raw image folders.

Pipeline position
-----------------
    FaceForensics videos  ->  data/raw/real + data/raw/fake
                                  ->  prepare_dataset.py (leakage-free split)

Labels come exclusively from the official directory structure:
    original_sequences/youtube/...      -> real
    manipulated_sequences/Deepfakes/... -> fake

Every frame is saved inside an identity-component sub-folder; the SAME
component id is used in both ``real`` and ``fake`` folders so the existing
splitter can keep a whole identity inside one split (see src/extraction.py).

Nothing synthetic is ever created and the original videos under
``data/faceforensics`` are never modified.

Examples
--------
    python extract_dataset_frames.py                    # defaults
    python extract_dataset_frames.py --frames-per-video 40
    python extract_dataset_frames.py --force            # overwrite existing PNGs
    python extract_dataset_frames.py --dry-run          # report only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import extraction  # noqa: E402

LINE = "=" * 62


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract FaceForensics++ frames into raw real/fake folders.",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=config.FFPP_ORIGINAL_VIDEOS_DIR,
        help="FaceForensics++ original (real) videos folder.",
    )
    parser.add_argument(
        "--manipulated-dir",
        type=Path,
        default=config.FFPP_MANIPULATED_VIDEOS_DIR,
        help="FaceForensics++ manipulated (fake) videos folder.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=config.RAW_DIR,
        help="Output root that contains the 'real' and 'fake' folders.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=config.FFPP_FRAMES_PER_VIDEO,
        help=(
            "Maximum number of evenly-spaced frames to extract per video. "
            "0 extracts every frame."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing frame PNG files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover videos and show the plan without extracting anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(LINE)
    print("DeepGuard - FaceForensics++ frame extraction")
    print(LINE)

    videos = extraction.discover_videos(args.original_dir, args.manipulated_dir)
    if not videos:
        print("[!] No FaceForensics++ videos were found.")
        print(f"    original dir     : {args.original_dir}")
        print(f"    manipulated dir  : {args.manipulated_dir}")
        print(LINE)
        return 1

    videos, components = extraction.assign_components(list(videos))

    print("Discovered videos")
    original = [v for v in videos if v.label == "real"]
    manipulated = [v for v in videos if v.label == "fake"]
    print(f"  original videos : {len(original)}")
    print(f"  fake videos     : {len(manipulated)}")
    print(f"  identity components : {len(components)}")
    for comp_id in sorted(components):
        videos_in_component = [
            v.path.name for v in videos if v.component_id == comp_id
        ]
        print(f"    {comp_id}: identities {components[comp_id]} -> {videos_in_component}")

    print()

    if args.frames_per_video <= 0:
        sampling_note = "every frame of every video"
    else:
        sampling_note = f"up to {args.frames_per_video} evenly-spaced frames/video"
    print(f"Sampling: {sampling_note}  (deterministic)")
    print()

    if args.dry_run:
        print("[i] Dry run: nothing was extracted.")
        print(LINE)
        return 0

    raw_real_dir = args.raw_dir / "real"
    raw_fake_dir = args.raw_dir / "fake"

    print("Extracting frames...")
    report = extraction.extract_all(
        raw_real_dir=raw_real_dir,
        raw_fake_dir=raw_fake_dir,
        original_dir=args.original_dir,
        manipulated_dir=args.manipulated_dir,
        max_frames=args.frames_per_video,
        force=args.force,
    )
    print()

    if report.failed_videos:
        print("[!] Failures:")
        for failed in report.failed_videos:
            print(f"    {failed}")
        print(LINE)
        return 2

    print("Frame extraction summary")
    for x in report.extractions:
        print(
            f"  {x.video.path.name:16s} ({x.video.label}) "
            f"total={x.total_frames:5d} extracted={x.extracted:3d} "
            f"skipped={x.skipped_existing:3d}"
        )
    print()
    print(f"  real frames : {report.real_frames}")
    print(f"  fake frames : {report.fake_frames}")
    print(f"  total frames: {report.total_extracted}")
    print(f"  skipped     : {report.total_skipped} (already existing, --force to overwrite)")

    manifest = extraction.build_manifest(
        report, generated_at=datetime.now().isoformat(timespec="seconds")
    )
    manifest_path = extraction.write_manifest(manifest, raw_dir=args.raw_dir)
    print()
    print(f"[i] Provenance manifest: {manifest_path}")
    print(f"[i] Raw frames: {args.raw_dir}\\real  and  {args.raw_dir}\\fake")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())