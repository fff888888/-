"""Reusable helpers for end-to-end video processing and index maintenance."""

from __future__ import annotations

from dataclasses import dataclass
import time  # used for throttled progress reporting
from pathlib import Path
from typing import Callable, Optional, Sequence

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
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> ProcessingResult:
    """Extract frames + embeddings + metadata for a video."""

    last_report_ts = 0.0
    last_progress = 0.0

    def _report(stage: str, progress_pct: float, message: str, *, force: bool = False) -> None:
        nonlocal last_report_ts, last_progress
        if progress_callback is None:
            return
        now = time.monotonic()
        clamped = max(0.0, min(float(progress_pct), 100.0))
        if not force:
            if clamped < last_progress + 0.1 and (now - last_report_ts) < 1.0:
                return
            if (now - last_report_ts) < 1.0:
                return
        last_progress = max(last_progress, clamped)
        last_report_ts = now
        progress_callback(stage, clamped, message)

    def _fraction(done: int, total: int) -> float:
        if total > 0:
            return min(max(done / float(total), 0.0), 1.0)
        if done <= 0:
            return 0.0
        return min(done / float(done + 5), 1.0)

    EXTRACT_WEIGHT = 0.75

    video = Path(video_path)
    root = Path(output_root)
    frames_dir = root / "frames" / video.stem
    embeddings_dir = root / "embeddings" / model_type / video.stem
    feature_path = embeddings_dir / "frame_features.npy"
    metadata_file = Path(metadata_path) if metadata_path else root / "metadata" / f"{video.stem}.json"

    frames_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    _report("extracting", 10.0, "开始处理", force=True)
    _report("extracting_frames", 10.0, "正在抽帧", force=True)

    frames, fps = extract_keyframes(
        video_path=video,
        output_dir=frames_dir,
        method=method,
        interval=interval,
        scene_threshold=scene_threshold,
        image_format=image_format,
        quality=quality,
        progress_callback=lambda processed, total, saved: _report(
            "extracting_frames",
            10.0 + EXTRACT_WEIGHT * 100.0 * _fraction(processed, total),
            f"抽帧进度 {saved}/{total or '?'}",
        ),
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
        progress_callback=lambda processed, total: _report(
            "embedding_frames",
            10.0 + EXTRACT_WEIGHT * 100.0 * _fraction(processed, total),
            f"编码帧特征 {processed}/{total or len(frames)}",
        ),
    )

    _report("embedding_frames", 10.0 + EXTRACT_WEIGHT * 100.0, "帧特征生成完成", force=True)

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
