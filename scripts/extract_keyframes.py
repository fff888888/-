#!/usr/bin/env python3
"""Extract keyframes from a video and store frame metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_search.frames import extract_keyframes
from video_search.metadata import VideoMetadata, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract keyframes using OpenCV")
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/frames"),
        help="Directory where extracted frames will be stored",
    )
    parser.add_argument(
        "--method",
        choices=["interval", "scene-diff"],
        default="interval",
        help="Frame extraction strategy",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds when method=interval",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=30.0,
        help="Mean pixel difference threshold when method=scene-diff",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        help="Image file extension (jpg or png)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality when saving frames",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional path where metadata JSON will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video
    output_dir = args.output_dir / video.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = extract_keyframes(
        video_path=video,
        output_dir=output_dir,
        method=args.method,
        interval=args.interval,
        scene_threshold=args.scene_threshold,
        image_format=args.image_format,
        quality=args.quality,
    )

    metadata = VideoMetadata(
        video_path=str(video),
        frames=frames,
        frame_interval=args.interval if args.method == "interval" else None,
        fps=fps,
        method=args.method,
    )
    metadata_path = args.metadata or Path("data/metadata") / f"{video.stem}.json"
    save_metadata(metadata, metadata_path)
    print(json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
