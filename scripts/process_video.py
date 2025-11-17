#!/usr/bin/env python3
"""Full video processing pipeline: frames + embeddings + metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_search.features import OnnxClipEncoder, build_frame_feature_cache
from video_search.frames import extract_keyframes
from video_search.metadata import VideoMetadata, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames and CLIP embeddings")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory for frames, embeddings, and metadata",
    )
    parser.add_argument(
        "--method",
        choices=["interval", "scene-diff"],
        default="interval",
        help="Keyframe extraction strategy",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds")
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=30.0,
        help="Scene change threshold when method=scene-diff",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        help="Image format for extracted frames",
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality when saving frames"
    )
    parser.add_argument(
        "--model-type",
        choices=["clip", "cnclip"],
        default="clip",
        help="Which CLIP variant the ONNX checkpoints represent",
    )
    parser.add_argument("--image-model", type=Path, required=True, help="Image encoder ONNX path")
    parser.add_argument(
        "--text-model",
        type=Path,
        help="Optional text encoder ONNX path for later querying",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer name or path compatible with the text encoder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution provider: cpu or cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during image embedding",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional explicit output metadata path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video
    root = args.output_root
    frames_dir = root / "frames" / video.stem
    embeddings_dir = root / "embeddings" / args.model_type / video.stem
    metadata_path = args.metadata or root / "metadata" / f"{video.stem}.json"

    frames_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = extract_keyframes(
        video_path=video,
        output_dir=frames_dir,
        method=args.method,
        interval=args.interval,
        scene_threshold=args.scene_threshold,
        image_format=args.image_format,
        quality=args.quality,
    )

    encoder = OnnxClipEncoder(
        model_type=args.model_type,
        image_model_path=args.image_model,
        text_model_path=args.text_model,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    feature_path = embeddings_dir / "frame_features.npy"
    cache = build_frame_feature_cache(
        frames=frames,
        encoder=encoder,
        output_path=feature_path,
        batch_size=args.batch_size,
    )

    metadata = VideoMetadata(
        video_path=str(video),
        frames=cache.frames,
        feature_file=str(feature_path),
        embedding_dim=int(cache.features.shape[1]) if cache.features.size else encoder.dimension,
        model_type=args.model_type,
        image_model_path=str(args.image_model),
        text_model_path=str(args.text_model) if args.text_model else None,
        tokenizer_path=args.tokenizer,
        frame_interval=args.interval if args.method == "interval" else None,
        fps=fps,
        method=args.method,
    )
    save_metadata(metadata, metadata_path)
    print(json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
