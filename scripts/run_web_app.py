"""启动交互式 Web UI 的命令行脚本。"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import uvicorn

from video_search.webapp import WebAppConfig, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动视频语义检索 Web UI")
    parser.add_argument("metadata_index", help="FAISS 索引文件路径，例如 data/index/frame.index")
    parser.add_argument("--manifest", dest="manifest", help="索引对应的 manifest JSON，默认与 index 同名")
    parser.add_argument("--text-model", required=True, help="文本编码 ONNX 模型路径")
    parser.add_argument("--image-model", help="可选的图像编码 ONNX 模型路径，用于后续扩展")
    parser.add_argument("--tokenizer", help="Tokenizer 名称或本地路径")
    parser.add_argument("--model-type", default="clip", choices=["clip", "cnclip"], help="模型类型")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="ONNX Runtime 计算设备")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--default-top-k", type=int, default=9, help="默认展示的候选数量")
    parser.add_argument("--preview-duration", type=float, default=3.0, help="预览片段时长（秒）")
    parser.add_argument("--title", default="视频语义检索", help="网页标题")
    parser.add_argument("--no-browser", action="store_true", help="启动时不要自动打开浏览器")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = WebAppConfig(
        index_path=Path(args.metadata_index),
        manifest_path=Path(args.manifest) if args.manifest else None,
        model_type=args.model_type,
        image_model=Path(args.image_model) if args.image_model else None,
        text_model=Path(args.text_model),
        tokenizer_path=args.tokenizer,
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
    main()

