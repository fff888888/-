"""Metadata models for the video semantic search pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

NORMALIZED_TIMESTAMP_KEYS = (
    "timestamp",
    "time",
    "ts",
    "second",
    "seconds",
    "start",
    "frame_time",
)

IMAGE_PATH_KEYS = (
    "image_path",
    "frame_path",
    "path",
    "image",
    "file",
)

VIDEO_PATH_KEYS = (
    "video_path",
    "video",
    "source_video",
    "source",
)


@dataclass
class FrameRecord:
    """Metadata for a single extracted frame."""

    index: int
    timestamp: float
    image_path: str
    embedding_index: Optional[int] = None
    extras: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.extras is None:
            data.pop("extras")
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameRecord":
        return cls(**data)


@dataclass
class VideoMetadata:
    """High level description of extracted frames and embeddings."""

    video_path: str
    frames: List[FrameRecord] = field(default_factory=list)
    feature_file: Optional[str] = None
    embedding_dim: Optional[int] = None
    model_type: Optional[str] = None
    image_model_path: Optional[str] = None
    text_model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    frame_interval: Optional[float] = None
    fps: Optional[float] = None
    method: str = "interval"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "frames": [frame.to_dict() for frame in self.frames],
            "feature_file": self.feature_file,
            "embedding_dim": self.embedding_dim,
            "model_type": self.model_type,
            "image_model_path": self.image_model_path,
            "text_model_path": self.text_model_path,
            "tokenizer_path": self.tokenizer_path,
            "frame_interval": self.frame_interval,
            "fps": self.fps,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        frames = [FrameRecord.from_dict(frame) for frame in data.get("frames", [])]
        return cls(
            video_path=data["video_path"],
            frames=frames,
            feature_file=data.get("feature_file"),
            embedding_dim=data.get("embedding_dim"),
            model_type=data.get("model_type"),
            image_model_path=data.get("image_model_path"),
            text_model_path=data.get("text_model_path"),
            tokenizer_path=data.get("tokenizer_path"),
            frame_interval=data.get("frame_interval"),
            fps=data.get("fps"),
            method=data.get("method", "interval"),
        )


def save_metadata(metadata: VideoMetadata, path: Path | str) -> None:
    """Persist :class:`VideoMetadata` to a JSON file."""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata.to_dict(), fh, indent=2, ensure_ascii=False)


def load_metadata(path: Path | str) -> VideoMetadata:
    """Load :class:`VideoMetadata` from a JSON file."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    normalized = _normalize_metadata_payload(data, json_path)
    if not normalized.get("video_path"):
        normalized["video_path"] = str(json_path)
    return VideoMetadata.from_dict(normalized)


def merge_metadata(records: Iterable[VideoMetadata]) -> List[FrameRecord]:
    """Flatten the frame metadata across multiple videos."""

    flattened: List[FrameRecord] = []
    for metadata in records:
        flattened.extend(metadata.frames)
    return flattened


def _normalize_metadata_payload(data: Any, source: Path) -> Dict[str, Any]:
    """Accept both dict-based and list-based metadata payloads."""

    if isinstance(data, dict):
        return dict(data)

    if isinstance(data, list):
        frames: List[Dict[str, Any]] = []
        video_path: Optional[str] = None
        feature_file: Optional[str] = None
        embedding_dim: Optional[int] = None

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            if video_path is None:
                for key in VIDEO_PATH_KEYS:
                    if item.get(key):
                        video_path = str(item[key])
                        break

            if feature_file is None and item.get("feature_file"):
                feature_file = str(item["feature_file"])
            if embedding_dim is None and item.get("embedding_dim"):
                try:
                    embedding_dim = int(item["embedding_dim"])
                except (TypeError, ValueError):
                    embedding_dim = None

            image_path = _pick_first(item, IMAGE_PATH_KEYS)
            if not image_path:
                raise ValueError(f"{source}: 列表元素缺少 image_path/frame_path 等字段")

            timestamp = _coerce_timestamp(item, idx)
            frame_index = _coerce_index(item, len(frames))
            embedding_index = _coerce_embedding_index(item)
            extras = _collect_extras(item)

            frame_dict: Dict[str, Any] = {
                "index": frame_index,
                "timestamp": timestamp,
                "image_path": image_path,
            }
            if embedding_index is not None:
                frame_dict["embedding_index"] = embedding_index
            if extras:
                frame_dict["extras"] = extras
            frames.append(frame_dict)

        if not frames:
            raise ValueError(f"{source}: metadata 列表没有可解析的帧信息")

        normalized: Dict[str, Any] = {"video_path": video_path or str(source), "frames": frames}
        if feature_file:
            normalized["feature_file"] = feature_file
        if embedding_dim is not None:
            normalized["embedding_dim"] = embedding_dim
        return normalized

    raise TypeError(f"{source}: 不支持的 metadata 格式 {type(data)!r}")


def _pick_first(data: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return None


def _coerce_timestamp(item: Dict[str, Any], fallback_index: int) -> float:
    for key in NORMALIZED_TIMESTAMP_KEYS:
        if key in item and item[key] is not None:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                break
    return float(fallback_index)


def _coerce_index(item: Dict[str, Any], default_index: int) -> int:
    for key in ("index", "frame_index"):
        if key in item and item[key] is not None:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                break
    return default_index


def _coerce_embedding_index(item: Dict[str, Any]) -> Optional[int]:
    for key in ("embedding_index", "embedding_idx", "vector_index"):
        if key in item and item[key] is not None:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                return None
    return None


def _collect_extras(item: Dict[str, Any]) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    for key in ("text", "caption", "description", "label"):
        if item.get(key):
            extras[key] = item[key]
    for key in ("start_time", "end_time", "duration", "score"):
        if item.get(key) is not None:
            extras[key] = item[key]
    return extras
