"""一键启动 Web UI，自动寻找常见路径并运行服务。"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Iterable, Optional

import uvicorn

from video_search.webapp import WebAppConfig, create_app


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INDEX_CANDIDATES = [
    "workspace/index/faiss.index",
    "workspace/index/frame.index",
    "data/index/frame.index",
]

DEFAULT_MANIFEST_CANDIDATES = [
    "workspace/index/manifest.json",
    "workspace/index/frame.index.json",
    "data/index/frame.index.json",
]

DEFAULT_TEXT_MODEL_CANDIDATES = [
    "models/clip-vit-b32-text.onnx",
    "models/clip/text.onnx",
    "models/text/model.onnx",
]

DEFAULT_IMAGE_MODEL_CANDIDATES = [
    "models/clip-vit-b32-vision.onnx",
    "models/clip/image.onnx",
    "models/image/model.onnx",
]

DEFAULT_TOKENIZER_CANDIDATES = [
    "models/tokenizer",
    "models/clip/tokenizer",
]


def _find_first_existing(candidates: Iterable[str], description: str, required: bool) -> Optional[Path]:
    for relative in candidates:
        candidate = (REPO_ROOT / relative).expanduser()
        if candidate.exists():
            return candidate
    if required:
        raise SystemExit(
            f"找不到 {description}，请检查路径或通过命令行参数显式传入。"
        )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动寻找索引与模型后启动 Web UI"
    )
    parser.add_argument("--index", help="FAISS 索引文件路径，默认会自动寻找常见目录")
    parser.add_argument(
        "--manifest",
        help="索引 manifest JSON，默认会与索引同目录自动匹配",
    )
    parser.add_argument("--text-model", help="文本编码 ONNX 模型路径，默认自动寻找")
    parser.add_argument("--image-model", help="图像编码 ONNX 模型路径，可选")
    parser.add_argument("--tokenizer", help="Tokenizer 名称或本地目录，默认自动寻找")
    parser.add_argument("--model-type", default="clip", choices=["clip", "cnclip"], help="模型类型")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="ONNX Runtime 计算设备")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--default-top-k", type=int, default=9, help="默认展示候选数量")
    parser.add_argument("--preview-duration", type=float, default=3.0, help="悬停预览时长（秒）")
    parser.add_argument("--title", default="ClipFinder Web", help="网页标题")
    parser.add_argument("--no-browser", action="store_true", help="启动时不要自动打开浏览器")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    index_path = Path(args.index) if args.index else _find_first_existing(
        DEFAULT_INDEX_CANDIDATES, "索引文件", required=True
    )

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = _find_first_existing(
            DEFAULT_MANIFEST_CANDIDATES, "manifest 文件", required=True
        )

    text_model = Path(args.text_model) if args.text_model else _find_first_existing(
        DEFAULT_TEXT_MODEL_CANDIDATES, "文本 ONNX 模型", required=True
    )

    tokenizer = args.tokenizer or (
        _find_first_existing(DEFAULT_TOKENIZER_CANDIDATES, "tokenizer", required=True)
    )

    image_model: Path | None
    if args.image_model:
        image_model = Path(args.image_model)
    else:
        image_model = _find_first_existing(
            DEFAULT_IMAGE_MODEL_CANDIDATES, "图像 ONNX 模型", required=False
        )

    config = WebAppConfig(
        index_path=index_path,
        manifest_path=manifest_path,
        model_type=args.model_type,
        image_model=image_model,
        text_model=text_model,
        tokenizer_path=str(tokenizer) if tokenizer is not None else None,
        device=args.device,
        default_top_k=args.default_top_k,
        preview_duration=args.preview_duration,
        title=args.title,
    )

    app = create_app(config)
    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        webbrowser.open_new_tab(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main(sys.argv[1:])

