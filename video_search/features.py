"""ONNX runtime helpers for CLIP-style encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .metadata import FrameRecord


MODEL_CONFIGS = {
    "clip": {
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "max_length": 77,
        "default_tokenizer": "openai/clip-vit-base-patch32",
    },
    "cnclip": {
        "image_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "max_length": 77,
        "default_tokenizer": "OFA-Sys/chinese-clip-vit-base-patch16",
    },
}


@dataclass
class FeatureCacheResult:
    features: np.ndarray
    frames: List[FrameRecord]


class OnnxClipEncoder:
    """Utility wrapper around ONNX Runtime CLIP/CN-CLIP checkpoints."""

    def __init__(
        self,
        model_type: str,
        image_model_path: Path | str | None,
        text_model_path: Path | str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        tokenizer_path: str | None = None,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.image_session: ort.InferenceSession | None = None
        if image_model_path is not None:
            self.image_session = ort.InferenceSession(
                str(image_model_path), providers=providers
            )
        self.text_session: ort.InferenceSession | None = None
        if text_model_path is not None:
            self.text_session = ort.InferenceSession(
                str(text_model_path), providers=providers
            )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            target_path = tokenizer_path or self.config.get("default_tokenizer")
            if target_path is None:
                raise ValueError("tokenizer_path must be provided when no default is set")
            self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        self.normalize = normalize
        self.max_length = int(self.config["max_length"])
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        if self.image_session is None:
            raise RuntimeError("Image model is not loaded; dimension is unknown")
        output_shape = self.image_session.get_outputs()[0].shape
        if len(output_shape) < 2 or output_shape[1] is None:
            raise RuntimeError("Unable to infer embedding dimension from model output")
        self._dimension = int(output_shape[1])
        return self._dimension

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        size = int(self.config["image_size"])
        image = image.resize((size, size))
        array = np.array(image).astype(np.float32) / 255.0
        mean = np.asarray(self.config["mean"], dtype=np.float32)
        std = np.asarray(self.config["std"], dtype=np.float32)
        array = (array - mean) / std
        array = array.transpose(2, 0, 1)
        return array

    def encode_image(self, images: Sequence[Path | str | Image.Image]) -> np.ndarray:
        if self.image_session is None:
            raise RuntimeError("encode_image requires an image model session")
        processed: List[np.ndarray] = []
        for image in images:
            if isinstance(image, (str, Path)):
                with Image.open(image) as img:
                    processed.append(self._preprocess_image(img))
            else:
                processed.append(self._preprocess_image(image))
        batch = np.stack(processed, axis=0)
        inputs = {self.image_session.get_inputs()[0].name: batch}
        outputs = self.image_session.run(None, inputs)[0]
        embeddings = outputs.astype(np.float32)
        if self._dimension is None:
            self._dimension = embeddings.shape[1]
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        return embeddings

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if self.text_session is None:
            raise RuntimeError("Text model is not available for this encoder")
        tokens = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        inputs = {name: value for name, value in tokens.items()}
        outputs = self.text_session.run(None, inputs)[0]
        embeddings = outputs.astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        return embeddings


def build_frame_feature_cache(
    frames: Sequence[FrameRecord],
    encoder: OnnxClipEncoder,
    output_path: Path | str,
    batch_size: int = 32,
) -> FeatureCacheResult:
    """Encode frames into CLIP embeddings and persist to disk."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        empty = np.empty((0, encoder.dimension), dtype=np.float32)
        np.save(output, empty)
        return FeatureCacheResult(features=empty, frames=[])

    all_embeddings: List[np.ndarray] = []
    updated_frames: List[FrameRecord] = []
    batch: List[FrameRecord] = []

    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            embeddings = encoder.encode_image([item.image_path for item in batch])
            all_embeddings.append(embeddings)
            updated_frames.extend(batch)
            batch = []
    if batch:
        embeddings = encoder.encode_image([item.image_path for item in batch])
        all_embeddings.append(embeddings)
        updated_frames.extend(batch)

    stacked = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, encoder.dimension), dtype=np.float32)
    np.save(output, stacked)

    for idx, frame in enumerate(updated_frames):
        frame.embedding_index = idx

    return FeatureCacheResult(features=stacked, frames=list(frames))
