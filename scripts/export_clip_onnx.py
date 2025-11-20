"""Export ViT-B/32 CLIP encoders from the official OpenAI weights to ONNX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

try:
    import clip  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "无法导入 openai/CLIP 库，请先执行 `pip install git+https://github.com/openai/CLIP.git`"
    ) from exc


LOGGER = logging.getLogger(__name__)


class _ImageEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model.encode_image(pixel_values)


class _TextEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:  # type: ignore[override]
        del attention_mask  # onnx 图不使用 mask，但为了兼容现有推理接口保留该参数
        return self.model.encode_text(input_ids)


def export_clip_onnx(
    output_dir: Path,
    model_name: str = "ViT-B/32",
    opset: int = 17,
    force: bool = False,
) -> tuple[Path, Path]:
    device = "cpu"
    model, _ = clip.load(model_name, device=device, jit=False)
    model.eval()
    image_out = output_dir / "clip-vit-b32-vision.onnx"
    text_out = output_dir / "clip-vit-b32-text.onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_out.exists() and not force:
        LOGGER.info("跳过图像模型导出：%s 已存在", image_out)
    else:
        LOGGER.info("正在导出图像 ONNX 模型到 %s", image_out)
        image_wrapper = _ImageEncoder(model)
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            image_wrapper,
            dummy,
            str(image_out),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=opset,
            dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        )

    if text_out.exists() and not force:
        LOGGER.info("跳过文本模型导出：%s 已存在", text_out)
    else:
        LOGGER.info("正在导出文本 ONNX 模型到 %s", text_out)
        text_wrapper = _TextEncoder(model)
        dummy_input = clip.tokenize(["export"], truncate=True)
        attention_mask = torch.ones_like(dummy_input)
        torch.onnx.export(
            text_wrapper,
            (dummy_input, attention_mask),
            str(text_out),
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeds"],
            opset_version=opset,
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "text_embeds": {0: "batch"},
            },
        )

    return image_out, text_out


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 CLIP ViT-B/32 ONNX 模型")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="保存 ONNX 模型的目录，默认写入 ./models",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="openai/CLIP 中的模型名称，默认 ViT-B/32",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="导出时使用的 ONNX opset 版本，默认 17",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若目标文件已存在，重新覆盖导出",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    image_model, text_model = export_clip_onnx(args.output_dir, args.model, args.opset, args.force)
    image_size = image_model.stat().st_size / (1024 * 1024)
    text_size = text_model.stat().st_size / (1024 * 1024)
    LOGGER.info("导出完成：%s (%.1f MB)", image_model, image_size)
    LOGGER.info("导出完成：%s (%.1f MB)", text_model, text_size)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
