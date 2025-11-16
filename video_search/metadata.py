"""Metadata models for the video semantic search pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
    return VideoMetadata.from_dict(data)


def merge_metadata(records: Iterable[VideoMetadata]) -> List[FrameRecord]:
    """Flatten the frame metadata across multiple videos."""

    flattened: List[FrameRecord] = []
    for metadata in records:
        flattened.extend(metadata.frames)
    return flattened
