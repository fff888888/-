#!/usr/bin/env python3
"""Build a FAISS index from one or more metadata JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from video_search.metadata import load_metadata
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized_dir = None if args.no_legacy_export else args.normalized_dir
    metadata_inputs = _prepare_metadata_paths(args.metadata, normalized_dir)
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


def _prepare_metadata_paths(paths: Iterable[Path], normalized_dir: Path | None) -> List[Path]:
    resolved: List[Path] = []
    if normalized_dir is not None:
        normalized_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        source = Path(source)
        if normalized_dir is None:
            resolved.append(source)
            continue
        target = normalized_dir / source.name
        metadata = load_metadata(source, write_converted=True, converted_path=target)
        if getattr(metadata, "_legacy_payload", False):
            print(f"已将 legacy metadata {source} 转换为 {target}")
            resolved.append(target)
        else:
            resolved.append(source)
    return resolved


if __name__ == "__main__":
    main()
