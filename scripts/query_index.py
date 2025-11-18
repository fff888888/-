#!/usr/bin/env python3
"""Query the FAISS frame index with a text prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_search.features import OnnxClipEncoder
from video_search.index import FaissIndexer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query frame embeddings with text")
    parser.add_argument("query", type=str, help="Text query to search for")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/index/frame.index"),
        help="Path to the FAISS index file",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional manifest JSON path if different from index suffix",
    )
    parser.add_argument(
        "--model-type",
        choices=["clip", "cnclip"],
        default="clip",
        help="Which CLIP variant was used to create the embeddings",
    )
    parser.add_argument(
        "--image-model",
        type=Path,
        help="Optional image encoder ONNX path (not required for text-only queries)",
    )
    parser.add_argument(
        "--text-model",
        type=Path,
        required=True,
        help="Path to the text encoder ONNX checkpoint",
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
        help="Execution provider (cpu or cuda)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest frames to return",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexer = FaissIndexer.load(args.index, args.manifest)
    encoder = OnnxClipEncoder(
        model_type=args.model_type,
        image_model_path=args.image_model,
        text_model_path=args.text_model,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    text_embedding = encoder.encode_text([args.query])[0]
    results = indexer.search(text_embedding, top_k=args.top_k)
    payload = [
        {
            "score": score,
            "video_path": frame.video_path,
            "image_path": frame.image_path,
            "timestamp": frame.timestamp,
            "metadata_path": frame.metadata_path,
            "frame_index": frame.frame_index,
            "embedding_index": frame.embedding_index,
        }
        for frame, score in results
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
