#!/usr/bin/env python3
"""Convert legacy metadata JSON files into embedding-ready structures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from video_search.embeddings import ensure_metadata_embeddings, metadata_has_embeddings
from video_search.features import OnnxClipEncoder
from video_search.metadata import load_metadata, save_metadata

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
    parser = argparse.ArgumentParser(
        description=(
            "将旧版 *_raw.json 或缺少 embedding 的 metadata 统一升级为带 feature_file 的新结构"
        )
    )
    parser.add_argument(
        "metadata",
        type=Path,
        nargs="+",
        help="需要转换的 metadata JSON，可以是 *_raw.json 或普通 JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workspace/metadata/normalized"),
        help="规范化 JSON 输出目录，默认为 workspace/metadata/normalized",
    )
    parser.add_argument(
        "--default-dim",
        type=int,
        default=512,
        help="legacy metadata 缺少 embedding_dim 时的默认维度",
    )
    parser.add_argument(
        "--model-type",
        choices=["clip", "cnclip"],
        help="使用的 ONNX 模型类型（clip/cnclip）",
    )
    parser.add_argument(
        "--image-model",
        type=Path,
        help="图像 ONNX 模型路径",
    )
    parser.add_argument(
        "--text-model",
        type=Path,
        help="（可选）文本 ONNX 模型路径，将写回 metadata",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="（可选）tokenizer 名称/路径，将写回 metadata",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="运行 ONNX Runtime 的设备（cpu/cuda）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="生成 embedding 时的 batch size",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("workspace/embeddings/legacy"),
        help="保存自动生成 frame_features.npy 的根目录",
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="除 .npy 以外，同时把 embedding 写进 frames[].embedding",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标 JSON 已存在则覆盖重写",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _maybe_autofill_model_args(args)
    encoder = _build_encoder(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    converted: List[Path] = []
    for source in args.metadata:
        source = Path(source)
        metadata = load_metadata(
            source,
            convert_legacy=True,
            default_embedding_dim=args.default_dim,
        )
        target = output_dir / source.name
        if target.exists() and not args.overwrite:
            print(f"跳过 {target}（已存在，如需重写请加 --overwrite）")
            converted.append(target)
            continue

        changed = False
        if not metadata_has_embeddings(metadata, source):
            ensure_metadata_embeddings(
                metadata,
                metadata_path=source,
                encoder=encoder,
                batch_size=args.batch_size,
                feature_root=args.feature_root,
                store_inline=args.inline,
            )
            changed = True

        save_metadata(metadata, target)
        converted.append(target)
        suffix = "（已生成 embedding）" if changed else ""
        print(f"已将 {source} 转换为 {target}{suffix}")

    print("全部完成，输出文件：")
    for path in converted:
        print(f" - {path}")


def _build_encoder(args: argparse.Namespace) -> OnnxClipEncoder:
    if args.image_model is None:
        raise ValueError("请通过 --image-model 指定 ONNX 图像模型路径")
    if args.model_type is None:
        raise ValueError("请通过 --model-type 指定 clip 或 cnclip")
    encoder = OnnxClipEncoder(
        model_type=args.model_type,
        image_model_path=args.image_model,
        text_model_path=args.text_model,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    return encoder


def _maybe_autofill_model_args(args: argparse.Namespace) -> None:
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

    if args.model_type is None and args.image_model is not None:
        args.model_type = "clip"


if __name__ == "__main__":
    main()
