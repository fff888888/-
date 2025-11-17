# 视频语义检索流水线

本仓库提供一套可复现的视频逐帧语义检索方案，涵盖以下模块：

- OpenCV/FFmpeg 抽帧与关键帧时间戳记录
- CLIP 与 CN-CLIP ONNX 模型推理封装
- 帧级别特征向量缓存与元数据管理
- 基于 FAISS 的向量索引构建与持久化
- 文本查询 → 特征匹配 → 返回关键帧路径和时间戳

> 💡 **仓库位置说明**：你在 Git 中看到的正是本目录的内容，所有脚本均位于 `scripts/`，可复用模块在 `video_search/` 中。克隆或下载本仓库即可获得全部代码。

## 0. 快速上手（一分钟了解）

1. **下载/克隆仓库**：确保你当前目录就是包含 `scripts/` 与 `video_search/` 的仓库根目录。
2. **准备运行环境**：创建 Python 虚拟环境，执行 `pip install -r requirements.txt` 安装依赖；macOS 用户额外用 Homebrew 安装 `ffmpeg` 与 `opencv`。
3. **下载模型权重**：准备 CLIP 或 CN-CLIP 的图像/文本 ONNX 文件以及对应 tokenizer 名称，并记住它们的路径。
4. **处理你的视频**：运行 `python scripts/process_video.py <视频路径> --image-model <图像模型.onnx> --text-model <文本模型.onnx> --tokenizer <tokenizer>`，脚本会自动抽帧、生成特征与元数据。
5. **构建索引并查询**：执行 `python scripts/build_index.py <metadata.json>` 生成向量索引，再用 `python scripts/query_index.py "你的文本描述" ...` 检索最相似的帧和时间戳。
6. **图形化使用界面**：若你沿用本仓库推荐的 `workspace/` 或 `data/` 目录结构，可直接执行 `python scripts/start_web.py` 自动寻找索引与模型并启动；需要自定义路径时，则使用 `python scripts/run_web_app.py data/index/frame.index --text-model /path/to/text.onnx --tokenizer openai/clip-vit-base-patch32` 明确传参。

下面的章节会对每个步骤做更详细的解释与可选项介绍，你可以根据需要深入阅读。

## 1. 环境准备

### 1.1 Python 依赖

- 支持 Python 3.9 及以上版本
- 建议使用虚拟环境隔离依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 平台支持

| 系统 | 说明 |
| --- | --- |
| Linux | 直接安装依赖即可 |
| macOS (Intel/Apple Silicon) | 需提前安装 Homebrew，并使用 `brew install ffmpeg` 获取 FFmpeg；pip 会自动选择合适的 wheels |

如使用 Apple Silicon (M1/M2) 且 pip 未提供 FAISS 预编译包，可改用 `conda install -c conda-forge faiss-cpu==1.7.4`。

### 1.3 额外工具

- FFmpeg：用于精确抽帧和视频信息读取
- OpenCV：用于读取与保存帧图

macOS 用户可通过 Homebrew 安装：`brew install ffmpeg opencv`。

## 2. 模型文件准备

仓库**不包含**任何预训练模型权重，你需要自行准备：

1. ONNX 图像编码器（`clip_image.onnx` 或 `cnclip_image.onnx`）
2. ONNX 文本编码器（`clip_text.onnx` 或 `cnclip_text.onnx`）
3. 与文本编码器匹配的分词器（可使用 Hugging Face Hub 上的 tokenizer 名称）

可以使用 Hugging Face `optimum` 或 `transformers` 导出 ONNX，也可直接下载社区提供的 ONNX 权重。后续所有脚本均通过命令行参数传入这些文件路径。

> ❗ 没有准备模型时，脚本会在推理阶段抛出 `FileNotFoundError`，因此下载本仓库后仍需补充模型文件才能完整运行。

## 3. 目录结构

```
scripts/
  extract_keyframes.py   # 抽帧与元数据生成
  process_video.py       # 从视频到特征向量的完整流程
  build_index.py         # 构建 FAISS 索引
  query_index.py         # 载入索引并执行文本检索
  run_web_app.py         # 启动交互式网站，完成搜索、预览与片段下载
  start_web.py           # 自动推断路径后一键启动 Web UI
video_search/
  frames.py              # 抽帧工具函数
  features.py            # CLIP/CN-CLIP ONNX 推理封装
  index.py               # 向量索引构建与查询
  metadata.py            # 元数据结构与读写
  webapp.py              # FastAPI Web UI 逻辑
```

默认产出目录：

```
data/
  frames/<video名称>/frame_*.jpg
  embeddings/<模型>/<视频名称>/frame_features.npy
  metadata/<视频名称>.json
  index/frame.index 与 frame.index.json
```

## 4. 使用流程

### 4.1 （可选）仅抽取关键帧

```bash
python scripts/extract_keyframes.py /path/to/video.mp4 \
  --method interval \
  --interval 1.0 \
  --output-dir data/frames \
  --metadata data/metadata/video.json
```

- `--method` 支持 `interval`（每隔 *n* 秒取一帧）或 `scene-diff`（基于帧差）
- 元数据 JSON 中会记录每一帧的时间戳和序号

### 4.2 视频到特征向量的一站式处理

```bash
python scripts/process_video.py /path/to/video.mp4 \
  --image-model /path/to/clip_image.onnx \
  --text-model /path/to/clip_text.onnx \
  --tokenizer openai/clip-vit-base-patch32 \
  --model-type clip \
  --interval 1.0 \
  --output-root data
```

该命令会完成：

1. 抽帧并保存 JPEG 图像
2. 调用 ONNX Runtime 计算每帧特征
3. 将全部帧向量保存为 `.npy` 文件
4. 生成包含视频路径、时间戳、特征文件路径等字段的元数据 JSON

元数据样例：

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

### 4.3 构建 FAISS 索引

```bash
python scripts/build_index.py data/metadata/video.json \
  --output data/index/frame.index
```

- 支持一次传入多个元数据文件，实现多视频联合检索
- 会额外生成 `frame.index.json`，记录索引中每一条向量对应的元数据

### 4.4 文本检索

```bash
python scripts/query_index.py "海滩上奔跑的狗" \
  --index data/index/frame.index \
  --image-model /path/to/clip_image.onnx \
  --text-model /path/to/clip_text.onnx \
  --tokenizer openai/clip-vit-base-patch32 \
  --model-type clip \
  --top-k 5
```

脚本会输出一个 JSON 数组，每个元素包含匹配帧的路径与时间戳，便于回放定位。

## 6. 交互式网站（Web UI）

当命令行结果已经可用时，你可以切换到网页界面，直接在浏览器里输入语义词、悬停预览并下载片段。

### 6.1 一键启动脚本（start_web.py）

如果你按照示例目录放置素材（例如 `workspace/index/faiss.index`、`workspace/index/manifest.json`、`models/clip-vit-b32-vision.onnx`、`models/clip-vit-b32-text.onnx`、`models/tokenizer/`），可以直接运行：

```bash
python scripts/start_web.py
```

脚本会自动在以下路径中按顺序寻找文件：

- 索引：`workspace/index/faiss.index` → `workspace/index/frame.index` → `data/index/frame.index`
- manifest：`workspace/index/manifest.json` → `workspace/index/frame.index.json` → `data/index/frame.index.json`
- 文本 ONNX：`models/clip-vit-b32-text.onnx` → `models/clip/text.onnx` → `models/text/model.onnx`
- 图像 ONNX（可选）：`models/clip-vit-b32-vision.onnx` → `models/clip/image.onnx` → `models/image/model.onnx`
- tokenizer：`models/tokenizer` → `models/clip/tokenizer`

找到即用，找不到就提示报错，并允许通过命令行选项覆盖，例如 `python scripts/start_web.py --index my_index/frame.index --text-model /tmp/clip_text.onnx`。

默认会监听 `0.0.0.0:8000` 并自动打开浏览器，方便在 Mac、iPad 或其它局域网设备上访问；若不希望自动打开，可加 `--no-browser`。

### 6.2 自定义启动（run_web_app.py）

```bash
python scripts/run_web_app.py data/index/frame.index \
  --text-model /path/to/clip_text.onnx \
  --tokenizer openai/clip-vit-base-patch32 \
  --manifest data/index/frame.index.json \
  --host 0.0.0.0 --port 8000
```

- `--image-model` 可选，仅在未来扩展需要图像模型时再传入。
- 默认会自动打开浏览器标签页，如果你计划把脚本做成桌面快捷方式，可新建一个 `.bat`（Windows）或 `.command`（macOS）文件，内容就是上述命令，双击即可启动。
- 需要 `ffmpeg` 可执行文件以便后台截取片段；若未安装请先按前文说明配置。

### 6.3 页面交互说明

1. **顶部搜索框**：输入任意语义词点击“开始搜索”，后台调用同一套 CLIP/CN-CLIP 推理逻辑并检索 FAISS 索引。
2. **候选卡片**：搜索结果以卡片形式展示，鼠标悬停时视频自动跳到匹配时间点并播放，便于快速预览；移开后自动暂停并回到起始位置。
3. **详情面板**：点击任意卡片即展开右侧面板，支持：
   - 查看视频名称、帧索引与精确时间戳；
   - 拖动/输入开始与结束时间，或使用 `±0.5s/±1s` 按钮微调；
   - 直接点击“下载这个片段”调用 FFmpeg 裁剪所选区间，浏览器会弹出下载对话框。

你也可以通过 `--default-top-k` 与 `--preview-duration` 调整默认展示数量与悬停预览的时间窗口，以适配不同素材密度。

## 5. 常见问题解答

### 5.1 我在 Git 看到了这些文件，是不是已经包含所有代码？

是的，`video_search/` 与 `scripts/` 目录中就是完整实现。只需克隆或下载本仓库，即可得到与当前环境一致的代码。

### 5.2 苹果电脑能跑吗？

可以。macOS 需安装 Homebrew，然后执行：

```bash
brew install ffmpeg opencv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装后即可使用脚本。若在 Apple Silicon 上遇到 FAISS 编译问题，可改用 `conda install -c conda-forge faiss-cpu==1.7.4`。

### 5.3 我只下载仓库，不提供模型能用吗？

下载仓库后可以直接运行抽帧、元数据与索引脚本，但推理和检索环节必须加载你提供的 ONNX 模型与 tokenizer。仓库仅提供执行逻辑，不包含任何预训练权重。

### 5.4 后续如何扩展？

- `video_search/features.py` 可扩展其它 ONNX 模型或量化版本
- `video_search/index.py` 支持替换为 HNSW、Annoy 等其它向量库
- 可以将 `scripts/` 中的命令行脚本改造成 API 或批量任务调度器

## 6. 快速验证

完成依赖安装后，可运行：

```bash
python -m compileall video_search scripts
```

该命令会检查 Python 语法是否正确，确保脚本在当前环境下可被解释执行。

## 7. 下一步建议

1. 准备目标视频并执行 `scripts/process_video.py`
2. 利用生成的元数据构建索引 `scripts/build_index.py`
3. 使用 `scripts/query_index.py` 输入中文或英文描述进行检索

祝你顺利搭建自己的视频语义检索流程！
