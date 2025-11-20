"""Helper utilities for ensuring metadata carries usable embeddings."""

from __future__ import annotations

from pathlib import Path

from .features import OnnxClipEncoder, build_frame_feature_cache
from .metadata import VideoMetadata


def metadata_has_embeddings(
    metadata: VideoMetadata, metadata_path: Path | str | None = None
) -> bool:
    """Return ``True`` when metadata already references frame embeddings."""

    feature_path = _resolve_feature_file(metadata, metadata_path)
    if feature_path is not None and feature_path.exists():
        metadata.feature_file = str(feature_path)
        return True

    frames = metadata.frames
    if not frames:
        return False
    for frame in frames:
        if frame.embedding is None:
            return False
    return True


def ensure_metadata_embeddings(
    metadata: VideoMetadata,
    metadata_path: Path | str,
    *,
    encoder: OnnxClipEncoder,
    batch_size: int = 32,
    feature_root: Path | str | None = None,
    store_inline: bool = False,
) -> Path:
    """Generate embeddings + ``feature_file`` for metadata lacking vectors."""

    metadata_file = Path(metadata_path)
    if metadata_has_embeddings(metadata, metadata_file):
        feature = metadata.feature_file
        return Path(feature) if feature else metadata_file

    if not metadata.frames:
        raise ValueError(f"{metadata_path}: metadata 没有任何帧记录，无法生成 embedding")

    _resolve_frame_image_paths(metadata, metadata_file)
    root = Path(feature_root) if feature_root else metadata_file.parent / "embeddings"
    video_name = _determine_video_name(metadata, metadata_file)
    feature_path = root / encoder.model_type / video_name / "frame_features.npy"
    cache = build_frame_feature_cache(
        frames=metadata.frames,
        encoder=encoder,
        output_path=feature_path,
        batch_size=batch_size,
    )

    metadata.frames = cache.frames
    metadata.feature_file = str(feature_path)
    metadata.embedding_dim = (
        int(cache.features.shape[1]) if cache.features.size else encoder.dimension
    )
    metadata.model_type = metadata.model_type or encoder.model_type
    if metadata.image_model_path is None:
        metadata.image_model_path = encoder.image_model_path
    if metadata.text_model_path is None:
        metadata.text_model_path = encoder.text_model_path
    if metadata.tokenizer_path is None:
        metadata.tokenizer_path = encoder.tokenizer_path

    if store_inline:
        vectors = cache.features.tolist()
        for idx, frame in enumerate(metadata.frames):
            frame.embedding = vectors[idx]
    else:
        for frame in metadata.frames:
            frame.embedding = None

    return feature_path


def _resolve_frame_image_paths(metadata: VideoMetadata, metadata_file: Path) -> None:
    base = metadata_file.parent
    for frame in metadata.frames:
        candidate = _resolve_path(frame.image_path, base)
        if candidate is None:
            raise FileNotFoundError(
                f"{metadata_file}: 找不到帧图像 {frame.image_path}，请检查路径是否存在"
            )
        frame.image_path = str(candidate)


def _resolve_feature_file(
    metadata: VideoMetadata, metadata_path: Path | str | None
) -> Path | None:
    if not metadata.feature_file:
        return None
    feature = Path(metadata.feature_file)
    if feature.exists():
        return feature
    if metadata_path is None:
        return feature if feature.exists() else None
    candidate = Path(metadata_path).parent / metadata.feature_file
    if candidate.exists():
        return candidate
    return None


def _determine_video_name(metadata: VideoMetadata, metadata_file: Path) -> str:
    raw = metadata.video_path
    if raw:
        return Path(raw).stem
    return metadata_file.stem


def _resolve_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    candidate = base / value
    if candidate.exists():
        return candidate
    return None


__all__ = ["metadata_has_embeddings", "ensure_metadata_embeddings"]
