"""Utilities for extracting keyframes from video files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Literal, Tuple

import cv2
import numpy as np

from .metadata import FrameRecord


ExtractionMethod = Literal["interval", "scene-diff"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_frame(frame: np.ndarray, path: Path, quality: int) -> None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(str(path), frame)


def _seconds_from_frame(index: int, fps: float) -> float:
    if fps <= 0:
        return float(index)
    return index / fps


def _extract_with_interval(
    capture: cv2.VideoCapture,
    output_dir: Path,
    fps: float,
    interval: float,
    image_format: str,
    quality: int,
) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    frame_interval = max(int(round(interval * fps)) if interval > 0 else 1, 1)
    frame_index = 0
    saved_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % frame_interval == 0:
            timestamp = _seconds_from_frame(frame_index, fps)
            frame_path = output_dir / f"frame_{saved_index:06d}.{image_format}"
            _save_frame(frame, frame_path, quality)
            records.append(
                FrameRecord(
                    index=saved_index,
                    timestamp=float(timestamp),
                    image_path=str(frame_path),
                )
            )
            saved_index += 1
        frame_index += 1
    return records


def _extract_with_scene_diff(
    capture: cv2.VideoCapture,
    output_dir: Path,
    fps: float,
    threshold: float,
    image_format: str,
    quality: int,
) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    frame_index = 0
    saved_index = 0
    previous_frame: np.ndarray | None = None

    while True:
        success, frame = capture.read()
        if not success:
            break

        should_save = False
        if previous_frame is None:
            should_save = True
        else:
            diff = cv2.absdiff(frame, previous_frame)
            score = float(diff.mean())
            should_save = score >= threshold
        if should_save:
            timestamp = _seconds_from_frame(frame_index, fps)
            frame_path = output_dir / f"frame_{saved_index:06d}.{image_format}"
            _save_frame(frame, frame_path, quality)
            records.append(
                FrameRecord(
                    index=saved_index,
                    timestamp=float(timestamp),
                    image_path=str(frame_path),
                    extras={"scene_diff_index": frame_index},
                )
            )
            saved_index += 1
            previous_frame = frame
        frame_index += 1

    return records


def extract_keyframes(
    video_path: Path | str,
    output_dir: Path | str,
    method: ExtractionMethod = "interval",
    interval: float = 1.0,
    scene_threshold: float = 30.0,
    image_format: str = "jpg",
    quality: int = 95,
) -> Tuple[List[FrameRecord], float]:
    """Extract keyframes from a video.

    Parameters
    ----------
    video_path:
        Path to the input video.
    output_dir:
        Directory where frames will be written.
    method:
        Extraction strategy. "interval" samples every *interval* seconds, while
        "scene-diff" stores a frame when the mean absolute difference exceeds
        ``scene_threshold``.
    interval:
        Interval in seconds when ``method="interval"``.
    scene_threshold:
        Threshold for the mean absolute pixel difference when
        ``method="scene-diff"``.
    image_format:
        File extension for saved frames.
    quality:
        JPEG quality value when ``image_format`` is JPEG.

    Returns
    -------
    Tuple[List[FrameRecord], float]
        A tuple containing the frame metadata and the detected frame rate.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if math.isnan(fps) or fps <= 0:
        fps = 30.0

    if method == "interval":
        records = _extract_with_interval(
            capture=capture,
            output_dir=output_dir,
            fps=fps,
            interval=interval,
            image_format=image_format,
            quality=quality,
        )
    elif method == "scene-diff":
        records = _extract_with_scene_diff(
            capture=capture,
            output_dir=output_dir,
            fps=fps,
            threshold=scene_threshold,
            image_format=image_format,
            quality=quality,
        )
    else:
        raise ValueError(f"Unsupported extraction method: {method}")

    capture.release()
    return records, fps
