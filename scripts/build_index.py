#!/usr/bin/env python3
"""Build a FAISS index from one or more metadata JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from video_search.features import OnnxClipEncoder
from video_search.metadata import load_metadata, save_metadata
from video_search.pipeline import (
    build_or_update_index,
    ensure_metadata_embeddings,
    metadata_has_embeddings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_MODEL_CANDIDATES = (
    "models/clip-vit-b32-vision.onnx",
    "models/clip/image.onnx",
    "models/image/model.onnx",
)

DEFAULT_TEXT_MODEL_CANDIDATES = (
    "models/clip-vit-b32-text.onnx",
    "models/clip/text.onnx",
    "models/text/model.onnx",
)

DEFAULT_TOKENIZER_CANDIDATES = (
    "models/tokenizer",
    "models/clip/tokenizer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for frame embeddings")
    parser.add_argument(
        "metadata",
        type=Path,
        nargs="+",
        help="Paths to metadata JSON files produced by process_video.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/index/frame.index"),
        help="Output FAISS index path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path to save the frame manifest (JSON)",
    )
    parser.add_argument(
        "--metric",
        choices=["ip", "l2"],
        default="ip",
        help="Similarity metric for FAISS",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before adding vectors",
    )
    parser.add_argument(
        "--default-dim",
        type=int,
        default=512,
        help="Fallback embedding dimension when legacy JSON lacks embedding_dim",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("workspace/metadata/normalized"),
        help="Where to store normalized copies of legacy list metadata",
    )
    parser.add_argument(
        "--no-legacy-export",
        action="store_true",
        help="Skip writing normalized JSON copies for legacy metadata",
    )
    parser.add_argument(
        "--model-type",
        choices=["clip", "cnclip"],
        help="模型类别（clip/cnclip），用于 legacy metadata 生成缺失的 embedding",
    )
    parser.add_argument(
        "--image-model",
        type=Path,
        help="图像 ONNX 模型路径，用于 legacy metadata 自动生成 embedding",
    )
    parser.add_argument(
        "--text-model",
        type=Path,
        help="（可选）文本 ONNX 模型路径，会记录到规范化 JSON 中",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="（可选）tokenizer 名称/路径，记录到规范化 JSON 中",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="生成 legacy embedding 时使用的设备（cpu/cuda）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="生成 legacy embedding 时的 batch size",
    )
    parser.add_argument(
        "--legacy-feature-root",
        type=Path,
        default=Path("workspace/embeddings/legacy"),
        help="保存 legacy metadata 生成的 frame_features.npy 的根目录",
    )
    parser.add_argument(
        "--legacy-inline",
        action="store_true",
        help="生成 embedding 时同时写入逐帧 inline 数据（默认为仅保存 .npy）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _maybe_autofill_model_args(args)
    normalized_dir = None if args.no_legacy_export else args.normalized_dir
    metadata_inputs = _prepare_metadata_paths(
        args.metadata,
        normalized_dir,
        default_dim=args.default_dim,
        encoder_args=args,
    )
    build_or_update_index(
        metadata_paths=metadata_inputs,
        index_path=args.output,
        manifest_path=args.manifest,
        metric=args.metric,
        normalize=not args.no_normalize,
        reset=True,
        default_embedding_dim=args.default_dim,
        convert_legacy=True,
    )
    print(f"Index saved to {args.output}")


def _prepare_metadata_paths(
    paths: Iterable[Path],
    normalized_dir: Path | None,
    *,
    default_dim: int,
    encoder_args: argparse.Namespace,
) -> List[Path]:
    resolved: List[Path] = []
    if normalized_dir is not None:
        normalized_dir.mkdir(parents=True, exist_ok=True)
    encoder: Optional[OnnxClipEncoder] = None
    for source in paths:
        source = Path(source)
        metadata = load_metadata(
            source,
            convert_legacy=True,
            default_embedding_dim=default_dim,
        )
        target = source
        metadata_changed = False
        if normalized_dir is not None and getattr(metadata, "_legacy_payload", False):
            target = normalized_dir / source.name

        has_embeddings_source = metadata_has_embeddings(metadata, source)
        needs_target_rewrite = False
        if target != source:
            needs_target_rewrite = not metadata_has_embeddings(metadata, target)

        if not has_embeddings_source or needs_target_rewrite:
            encoder = encoder or _build_legacy_encoder(encoder_args)
            ensure_metadata_embeddings(
                metadata,
                metadata_path=source,
                encoder=encoder,
                batch_size=encoder_args.batch_size,
                feature_root=encoder_args.legacy_feature_root,
                store_inline=encoder_args.legacy_inline,
            )
            metadata_changed = True

        if target != source or metadata_changed:
            save_metadata(metadata, target)
            if target != source:
                suffix = "（已生成 embedding）" if metadata_changed else ""
                print(f"已将 legacy metadata {source} 转换为 {target}{suffix}")
            elif metadata_changed:
                print(f"已在 {source} 写入自动生成的 embedding 信息")

        disk_metadata = metadata
        if target.exists():
            # 重新读取一次磁盘里的 JSON，确保写出的结构也携带了 embedding
            disk_metadata = load_metadata(
                target,
                convert_legacy=True,
                default_embedding_dim=default_dim,
            )

        if not metadata_has_embeddings(disk_metadata, target):
            encoder = encoder or _build_legacy_encoder(encoder_args)
            base_path = source if source.exists() else target
            ensure_metadata_embeddings(
                disk_metadata,
                metadata_path=base_path,
                encoder=encoder,
                batch_size=encoder_args.batch_size,
                feature_root=encoder_args.legacy_feature_root,
                store_inline=encoder_args.legacy_inline,
            )
            save_metadata(disk_metadata, target)
            metadata = disk_metadata
            print(
                "已在 normalize 阶段补算 embedding ->",
                target,
            )

        if not metadata_has_embeddings(metadata, target):
            raise ValueError(
                f"{target}: legacy metadata 仍缺少 embedding，"
                "请为 build_index.py 追加 --model-type/--image-model（必要时再加 --text-model/--tokenizer）以自动生成向量。"
            )
        resolved.append(target)
    return resolved


def _build_legacy_encoder(args: argparse.Namespace) -> OnnxClipEncoder:
    if args.image_model is None:
        raise ValueError(
            "legacy metadata 缺少 embedding，需要通过 --image-model/--model-type 提供 ONNX 模型"
        )
    if args.model_type is None:
        raise ValueError("请通过 --model-type 指定 clip 或 cnclip，用于生成 legacy embedding")
    encoder = OnnxClipEncoder(
        model_type=args.model_type,
        image_model_path=args.image_model,
        text_model_path=args.text_model,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    return encoder


def _maybe_autofill_model_args(args: argparse.Namespace) -> None:
    """自动探测模型与 Tokenizer 路径，减少命令行参数。"""

    def _pick(candidates: Iterable[str]) -> Optional[Path]:
        for relative in candidates:
            candidate = (REPO_ROOT / relative).expanduser()
            if candidate.exists():
                return candidate
        return None

    if args.image_model is None:
        detected = _pick(DEFAULT_IMAGE_MODEL_CANDIDATES)
        if detected is not None:
            args.image_model = detected

    if args.text_model is None:
        detected = _pick(DEFAULT_TEXT_MODEL_CANDIDATES)
        if detected is not None:
            args.text_model = detected

    if args.tokenizer is None:
        detected = _pick(DEFAULT_TOKENIZER_CANDIDATES)
        if detected is not None:
            args.tokenizer = str(detected)

    if args.model_type is None:
        args.model_type = "clip"


if __name__ == "__main__":
    main()
