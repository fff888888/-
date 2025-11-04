# Video semantic search pipeline

This repository provides a reproducible pipeline for building a frame-level video search system
with CN-CLIP/CLIP ONNX checkpoints. The workflow extracts keyframes, generates multimodal
embeddings, persists structured metadata, indexes vectors with FAISS, and exposes a simple
text-based retrieval script.

## Installation

Create a Python environment (3.9+) and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download or export the desired CLIP/CN-CLIP ONNX checkpoints. The pipeline expects separate
image and text encoder ONNX files as well as a compatible tokenizer (Hugging Face tokenizers
are supported).

## Repository layout

```
scripts/
  extract_keyframes.py   # Only extract frames + metadata
  process_video.py       # Full pipeline: frames + embeddings + metadata
  build_index.py         # Construct FAISS index from metadata
  query_index.py         # Query FAISS index with text
video_search/
  frames.py              # OpenCV based frame extraction helpers
  features.py            # ONNX Runtime wrapper for CN-CLIP/CLIP
  index.py               # FAISS indexing utilities
  metadata.py            # Dataclasses for metadata I/O
```

Generated assets follow this structure by default:

```
data/
  frames/<video-name>/frame_*.jpg
  embeddings/<model>/<video-name>/frame_features.npy
  metadata/<video-name>.json
  index/frame.index + frame.index.json
```

## Usage

### 1. Extract keyframes (optional standalone)

```bash
python scripts/extract_keyframes.py /path/to/video.mp4 \
  --method interval \
  --interval 1.0 \
  --output-dir data/frames \
  --metadata data/metadata/video.json
```

* `--method` can be `interval` (sample every *n* seconds) or `scene-diff` (mean pixel
  difference threshold).
* Timestamps and frame indices are stored in the metadata JSON.

### 2. Process a video end-to-end

```bash
python scripts/process_video.py /path/to/video.mp4 \
  --image-model /path/to/clip_image.onnx \
  --text-model /path/to/clip_text.onnx \
  --tokenizer /path/to/tokenizer_or_hub_id \
  --model-type clip \
  --interval 1.0 \
  --output-root data
```

This command extracts frames, computes CN-CLIP/CLIP embeddings for each frame, and writes a
metadata file capturing:

* `video_path`, `fps`, frame extraction settings
* `frames`: per-frame objects containing `timestamp`, `image_path`, and the embedding index
* `feature_file`: `.npy` array storing all frame vectors for the video
* ONNX/tokenizer paths to ensure reproducibility

Embeddings are cached on disk (`frame_features.npy`) to prevent recomputation.

### 3. Build a FAISS index

```bash
python scripts/build_index.py data/metadata/video.json \
  --output data/index/frame.index
```

You can provide multiple metadata files to index several videos simultaneously. The script saves
both the FAISS index and a manifest JSON describing every frame entry.

### 4. Query with natural language

```bash
python scripts/query_index.py "a dog running on the beach" \
  --index data/index/frame.index \
  --image-model /path/to/clip_image.onnx \
  --text-model /path/to/clip_text.onnx \
  --tokenizer /path/to/tokenizer_or_hub_id \
  --model-type clip \
  --top-k 5
```

The output is a JSON array with the top matches, each containing the frame image path and
timestamp for easy inspection.

## Metadata schema

Each metadata JSON file uses the following structure:

```json
{
  "video_path": "/absolute/path/video.mp4",
  "feature_file": "data/embeddings/clip/video/frame_features.npy",
  "embedding_dim": 512,
  "model_type": "clip",
  "image_model_path": "/models/clip_image.onnx",
  "text_model_path": "/models/clip_text.onnx",
  "tokenizer_path": "openai/clip-vit-base-patch32",
  "frame_interval": 1.0,
  "fps": 29.97,
  "method": "interval",
  "frames": [
    {
      "index": 0,
      "timestamp": 0.0,
      "image_path": "data/frames/video/frame_000000.jpg",
      "embedding_index": 0
    }
  ]
}
```

The `embedding_index` aligns each frame entry with the row inside `feature_file`. This metadata is
consumed by the FAISS builder and downstream tools.

## Notes

* Ensure the tokenizer matches the ONNX text encoder vocabulary (e.g. CN-CLIP requires a
  Chinese tokenizer such as `OFA-Sys/chinese-clip-vit-base-patch16`).
* FAISS normalization defaults to cosine similarity (`IndexFlatIP` with L2-normalized vectors).
  Disable normalization via `--no-normalize` if your embeddings are already normalized.
* The scripts operate purely on local files; feel free to adapt them into an API or scheduler for
  large-scale ingestion.
