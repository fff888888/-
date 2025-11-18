#!/usr/bin/env python3
"""Build a FAISS index from one or more metadata JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

from video_search.pipeline import build_or_update_index


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_or_update_index(
        metadata_paths=args.metadata,
        index_path=args.output,
        manifest_path=args.manifest,
        metric=args.metric,
        normalize=not args.no_normalize,
        reset=True,
    )
    print(f"Index saved to {args.output}")


if __name__ == "__main__":
    main()
