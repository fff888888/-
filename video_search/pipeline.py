"""Reusable helpers for end-to-end video processing and index maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .features import OnnxClipEncoder, build_frame_feature_cache
from .frames import extract_keyframes
from .index import FaissIndexer, build_index_from_metadata
from .metadata import DEFAULT_EMBEDDING_DIM, VideoMetadata, save_metadata, load_metadata


@dataclass
class ProcessingResult:
    """Artifacts produced by processing a single video."""

    metadata: VideoMetadata
    metadata_path: Path
    frame_dir: Path
    feature_path: Path


def process_video_to_embeddings(
    video_path: Path | str,
    output_root: Path | str,
    *,
    model_type: str,
    image_model_path: Path | str | None,
    text_model_path: Path | str | None = None,
    tokenizer_path: str | None = None,
    method: str = "interval",
    interval: float = 1.0,
    scene_threshold: float = 30.0,
    image_format: str = "jpg",
    quality: int = 95,
    batch_size: int = 32,
    device: str = "cpu",
    metadata_path: Path | str | None = None,
    encoder: OnnxClipEncoder | None = None,
) -> ProcessingResult:
    """Extract frames + embeddings + metadata for a video."""

    video = Path(video_path)
    root = Path(output_root)
    frames_dir = root / "frames" / video.stem
    embeddings_dir = root / "embeddings" / model_type / video.stem
    feature_path = embeddings_dir / "frame_features.npy"
    metadata_file = Path(metadata_path) if metadata_path else root / "metadata" / f"{video.stem}.json"

    frames_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = extract_keyframes(
        video_path=video,
        output_dir=frames_dir,
        method=method,
        interval=interval,
        scene_threshold=scene_threshold,
        image_format=image_format,
        quality=quality,
    )

    local_encoder = encoder
    if local_encoder is None:
        if image_model_path is None:
            raise ValueError("image_model_path 不能为空，用于生成帧特征")
        local_encoder = OnnxClipEncoder(
            model_type=model_type,
            image_model_path=image_model_path,
            text_model_path=text_model_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )
    elif local_encoder.image_session is None:
        raise ValueError("传入的 encoder 未加载图像模型，无法生成帧特征")

    cache = build_frame_feature_cache(
        frames=frames,
        encoder=local_encoder,
        output_path=feature_path,
        batch_size=batch_size,
    )

    metadata = VideoMetadata(
        video_path=str(video),
        frames=cache.frames,
        feature_file=str(feature_path),
        embedding_dim=int(cache.features.shape[1]) if cache.features.size else local_encoder.dimension,
        model_type=model_type,
        image_model_path=str(image_model_path) if image_model_path else None,
        text_model_path=str(text_model_path) if text_model_path else None,
        tokenizer_path=tokenizer_path,
        frame_interval=interval if method == "interval" else None,
        fps=fps,
        method=method,
    )
    save_metadata(metadata, metadata_file)

    return ProcessingResult(
        metadata=metadata,
        metadata_path=metadata_file,
        frame_dir=frames_dir,
        feature_path=feature_path,
    )


def build_or_update_index(
    metadata_paths: Sequence[Path | str],
    index_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
    metric: str = "ip",
    normalize: bool = True,
    indexer: FaissIndexer | None = None,
    reset: bool = False,
    default_embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    convert_legacy: bool = True,
    write_converted: bool = False,
    converted_dir: Path | str | None = None,
) -> FaissIndexer:
    """Create a brand-new index or append new metadata into an existing one."""

    if not metadata_paths:
        raise ValueError("metadata_paths 不能为空")

    resolved_metadata = [Path(item) for item in metadata_paths]
    index_path = Path(index_path)
    manifest = Path(manifest_path) if manifest_path else index_path.with_suffix(".json")

    if reset:
        indexer = build_index_from_metadata(
            metadata_paths=resolved_metadata,
            metric=metric,
            normalize=normalize,
            default_embedding_dim=default_embedding_dim,
            convert_legacy=convert_legacy,
        )
        indexer.save(index_path, manifest)
        return indexer

    if indexer is None:
        if index_path.exists() and manifest.exists():
            indexer = FaissIndexer.load(index_path, manifest)
        else:
            indexer = build_index_from_metadata(
                metadata_paths=resolved_metadata,
                metric=metric,
                normalize=normalize,
                default_embedding_dim=default_embedding_dim,
                convert_legacy=convert_legacy,
            )
            indexer.save(index_path, manifest)
            return indexer

    for path in resolved_metadata:
        metadata = load_metadata(
            path,
            convert_legacy=convert_legacy,
            default_embedding_dim=default_embedding_dim,
            write_converted=write_converted,
            converted_path=_converted_metadata_path(path, converted_dir) if write_converted else None,
        )
        if metadata.embedding_dim is None:
            raise ValueError(f"metadata 缺少 embedding_dim: {path}")
        if metadata.embedding_dim != indexer.dimension:
            raise ValueError(
                f"索引维度 {indexer.dimension} 与 {path} 的 embedding_dim {metadata.embedding_dim} 不一致"
            )
        indexer.add_metadata(metadata, path)

    indexer.save(index_path, manifest)
    return indexer


def _converted_metadata_path(source: Path | str, converted_dir: Path | str | None) -> Path:
    source_path = Path(source)
    if converted_dir:
        target_dir = Path(converted_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / source_path.name
    if source_path.suffix:
        return source_path.with_name(f"{source_path.stem}_normalized{source_path.suffix}")
    return source_path.with_name(f"{source_path.name}_normalized.json")


def metadata_has_embeddings(metadata: VideoMetadata, metadata_path: Path | str | None = None) -> bool:
    """Check whether metadata already references usable embeddings."""

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
    """Generate embeddings + feature_file for metadata lacking vectors."""

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
    metadata.embedding_dim = int(cache.features.shape[1]) if cache.features.size else encoder.dimension
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


def _resolve_feature_file(metadata: VideoMetadata, metadata_path: Path | str | None) -> Path | None:
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
