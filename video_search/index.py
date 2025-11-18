"""FAISS index helpers for frame-level search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from .metadata import FrameRecord, VideoMetadata, load_metadata


@dataclass
class IndexFrame:
    video_path: str
    image_path: str
    timestamp: float
    metadata_path: Optional[str]
    frame_index: int
    embedding_index: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "video_path": self.video_path,
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "metadata_path": self.metadata_path,
            "frame_index": self.frame_index,
            "embedding_index": self.embedding_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "IndexFrame":
        return cls(
            video_path=data["video_path"],
            image_path=data["image_path"],
            timestamp=float(data["timestamp"]),
            metadata_path=data.get("metadata_path"),
            frame_index=int(data["frame_index"]),
            embedding_index=int(data["embedding_index"]),
        )


class FaissIndexer:
    """Helper around ``faiss.Index`` with frame bookkeeping."""

    def __init__(
        self,
        dimension: int,
        metric: str = "ip",
        normalize: bool = True,
    ) -> None:
        if metric not in {"ip", "l2"}:
            raise ValueError("metric must be either 'ip' or 'l2'")
        self.metric = metric
        self.normalize = normalize
        self.dimension = dimension
        if metric == "ip":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        self.frames: List[IndexFrame] = []

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        frames: Sequence[IndexFrame],
    ) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(embeddings)
        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")
        self.index.add(embeddings)
        self.frames.extend(frames)

    def add_metadata(self, metadata: VideoMetadata, metadata_path: Path | str | None = None) -> None:
        embeddings = None
        if metadata.feature_file is not None:
            feature_path = Path(metadata.feature_file)
            embeddings = np.load(feature_path)
        else:
            embeddings = _collect_inline_embeddings(metadata.frames)
        if embeddings is None:
            raise ValueError("Metadata 缺少 feature_file，且帧记录里也没有 embedding")
        if embeddings.shape[0] != len(metadata.frames):
            raise ValueError("Frame/embedding count mismatch")
        frames = [
            IndexFrame(
                video_path=metadata.video_path,
                image_path=frame.image_path,
                timestamp=frame.timestamp,
                metadata_path=str(metadata_path) if metadata_path else None,
                frame_index=frame.index,
                embedding_index=frame.embedding_index if frame.embedding_index is not None else idx,
            )
            for idx, frame in enumerate(metadata.frames)
        ]
        self.add_embeddings(embeddings, frames)

    def save(self, index_path: Path | str, manifest_path: Path | str | None = None) -> None:
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        manifest_path = Path(manifest_path) if manifest_path else index_path.with_suffix(".json")
        manifest = {
            "dimension": self.dimension,
            "metric": self.metric,
            "normalize": self.normalize,
            "frames": [frame.to_dict() for frame in self.frames],
        }
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, index_path: Path | str, manifest_path: Path | str | None = None) -> "FaissIndexer":
        index_path = Path(index_path)
        manifest_path = Path(manifest_path) if manifest_path else index_path.with_suffix(".json")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        index = faiss.read_index(str(index_path))
        obj = cls(
            dimension=int(manifest["dimension"]),
            metric=str(manifest.get("metric", "ip")),
            normalize=bool(manifest.get("normalize", True)),
        )
        obj.index = index
        obj.frames = [IndexFrame.from_dict(item) for item in manifest.get("frames", [])]
        return obj

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[IndexFrame, float]]:
        if query.ndim == 1:
            query = query[None, :]
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)
        results: List[Tuple[IndexFrame, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            frame = self.frames[idx]
            results.append((frame, float(score)))
        return results


def build_index_from_metadata(
    metadata_paths: Sequence[Path | str],
    metric: str = "ip",
    normalize: bool = True,
) -> FaissIndexer:
    if not metadata_paths:
        raise ValueError("metadata_paths cannot be empty")
    first = load_metadata(metadata_paths[0])
    dimension = _ensure_embedding_dim(first)
    indexer = FaissIndexer(dimension=dimension, metric=metric, normalize=normalize)
    indexer.add_metadata(first, metadata_paths[0])
    for path in metadata_paths[1:]:
        metadata = load_metadata(path)
        dim = _ensure_embedding_dim(metadata)
        if dim != indexer.dimension:
            raise ValueError("All metadata must share the same embedding dimension")
        indexer.add_metadata(metadata, path)
    return indexer


def _ensure_embedding_dim(metadata: VideoMetadata) -> int:
    if metadata.embedding_dim is not None:
        return int(metadata.embedding_dim)
    inline_dim = _infer_inline_dimension(metadata.frames)
    if inline_dim is not None:
        metadata.embedding_dim = inline_dim
        return inline_dim
    if metadata.feature_file is not None:
        feature_path = Path(metadata.feature_file)
        array = np.load(feature_path, mmap_mode="r")
        if array.ndim < 2:
            raise ValueError("无法从 feature_file 推断 embedding 维度")
        dim = int(array.shape[1])
        metadata.embedding_dim = dim
        return dim
    raise ValueError("metadata must include embedding_dim or inline embeddings")


def _infer_inline_dimension(frames: Sequence[FrameRecord]) -> Optional[int]:
    for frame in frames:
        embedding = getattr(frame, "embedding", None)
        if embedding:
            return len(embedding)
    return None


def _collect_inline_embeddings(frames: Sequence[FrameRecord]) -> Optional[np.ndarray]:
    vectors: List[List[float]] = []
    for frame in frames:
        embedding = getattr(frame, "embedding", None)
        if embedding is None:
            return None
        vectors.append(list(embedding))
    if not vectors:
        return None
    return np.asarray(vectors, dtype=np.float32)
