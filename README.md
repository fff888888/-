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
3. **导出/准备模型权重**：执行 `python scripts/export_clip_onnx.py --output-dir models`，脚本会自动下载 OpenAI 官方 ViT-B/32 权重并导出 `models/clip-vit-b32-vision.onnx` 与 `models/clip-vit-b32-text.onnx`（约 151 MB 与 95 MB）。随后使用 `ls -lh models/clip-vit-b32-*.onnx` 确认文件体积与路径。
4. **处理你的视频**：运行 `python scripts/process_video.py <视频路径> --image-model <图像模型.onnx> --text-model <文本模型.onnx> --tokenizer <tokenizer>`，脚本会自动抽帧、生成特征与元数据。
5. **迁移旧版 metadata（如 *_raw.json）**：如果你拿到的是 demo 那种只有帧列表、没有 `.npy` 的 JSON，请运行 `python scripts/migrate_metadata.py *_raw.json --output-dir workspace/metadata/normalized --feature-root workspace/embeddings/legacy --model-type clip --image-model models/clip-vit-b32-vision.onnx --text-model models/clip-vit-b32-text.onnx --tokenizer openai/clip-vit-base-patch32`。脚本会自动生成缺失的 embedding，把规范化 JSON 写到 `workspace/metadata/normalized/`，并把 `.npy` 缓存在 `workspace/embeddings/legacy/`。
6. **构建索引并查询**：执行 `python scripts/build_index.py <规范化 metadata.json>` 生成向量索引，再用 `python scripts/query_index.py "你的文本描述" ...` 检索最相似的帧与时间戳。
7. **图形化使用界面**：若你沿用本仓库推荐的 `workspace/` 或 `data/` 目录结构，可直接执行 `python scripts/start_web.py` 自动寻找索引与模型；macOS/Windows 用户也可以双击仓库根目录的 `start_app.py` 达到同样效果。需要自定义路径时，则使用 `python scripts/run_web_app.py data/index/frame.index --text-model /path/to/text.onnx --tokenizer openai/clip-vit-base-patch32` 明确传参。

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

### 1.4 macOS/Apple Silicon 稳定性建议

- 依赖选择：`requirements.txt` 会在 arm64 + macOS 上自动安装 `onnxruntime-silicon`，若之前装过 `onnxruntime`/`onnxruntime-openmp`，请先卸载。
- 环境变量：为避免 OpenMP/libomp 重复初始化导致的崩溃，建议在启动前设置：

  ```bash
  export OMP_NUM_THREADS=1
  export OMP_WAIT_POLICY=PASSIVE
  export KMP_DUPLICATE_LIB_OK=TRUE
  export ORT_USE_OPENMP=0
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  ```

- 一键启动：仓库新增 `./run_mac_silicon.sh`，会自动注入以上变量并运行 `start_app.py`，适合做成桌面快捷方式或直接双击：

  ```bash
  ./run_mac_silicon.sh \
    --index workspace/index/index.faiss \
    --manifest workspace/index/index.json \
    --text-model models/clip-vit-b32-text.onnx \
    --image-model models/clip-vit-b32-vision.onnx \
    --tokenizer openai/clip-vit-base-patch32
  ```

  如需覆盖默认模型或索引路径，只需在命令末尾追加相应参数即可。

## 2. 模型文件准备

仓库**不包含**任何预训练模型权重，你需要自行准备：

1. ONNX 图像编码器（`clip_image.onnx` 或 `cnclip_image.onnx`）
2. ONNX 文本编码器（`clip_text.onnx` 或 `cnclip_text.onnx`）
3. 与文本编码器匹配的分词器（可使用 Hugging Face Hub 上的 tokenizer 名称）

### 2.1 一键导出 ViT-B/32 ONNX（无需 Hugging Face）

```bash
# requirements.txt 已包含 torch 与 openai/CLIP，如已安装可忽略
pip install torch git+https://github.com/openai/CLIP.git

# 在仓库根目录执行，默认写入 ./models
python scripts/export_clip_onnx.py --output-dir models
ls -lh models/clip-vit-b32-*.onnx
```

- `clip-vit-b32-vision.onnx` 约 151 MB；`clip-vit-b32-text.onnx` 约 95 MB。
- 脚本内部通过 openai/CLIP 官方发布的 ViT-B/32 权重（由 `clip` 库自动从 `openaipublic.azureedge.net` 下载），不需要 Hugging Face 账号或登录。
- 如需重新导出，可加入 `--force`；如要放到其它目录，则调整 `--output-dir`。

如果你更偏好 CN-CLIP 或其它模型，也可以使用 Hugging Face `optimum`/`transformers` 自行导出 ONNX，再把文件路径传给 `--image-model/--text-model` 参数即可。

> ❗ 没有准备模型时，脚本会在推理阶段抛出 `FileNotFoundError`，因此下载本仓库后仍需补充模型文件才能完整运行。

## 3. 目录结构

```
scripts/
  extract_keyframes.py   # 抽帧与元数据生成
  process_video.py       # 从视频到特征向量的完整流程
  build_index.py         # 构建 FAISS 索引
  query_index.py         # 载入索引并执行文本检索
  export_clip_onnx.py    # 自动下载并导出 ViT-B/32 ONNX 权重
  run_web_app.py         # 启动交互式网站，完成搜索、预览与片段下载
  start_web.py           # 自动推断路径后一键启动 Web UI
  start_app.py           # macOS/Windows 可直接双击运行 Web UI
run_mac_silicon.sh       # Apple Silicon 上设置稳态环境变量后一键启动
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
workspace/
  metadata/normalized/*.json   # legacy *_raw.json 自动转换后的结果
  embeddings/legacy/<模型>/<文件>/frame_features.npy
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

> 📝 如果你拿到的是一整个帧列表（例如 demo 目录里的 `bicycle_raw.json` 这类 `[{...}, {...}]` 文件），本仓库的 `load_metadata()` 会自
动把它转换成如上结构，不需要你额外手动整理字段。
>
> 这些列表如果缺少 `embedding_dim` 字段，也不用担心 —— 只要每个元素里带有 `embedding`（或 `vector`/`features`）数组，程序就会根据数组长度自动推断维度。

### 4.3 构建 FAISS 索引

#### 4.3.1 旧式 metadata 一次性迁移

```bash
python scripts/migrate_metadata.py \
  bicycle_raw.json book_raw.json building_raw.json \
  --output-dir workspace/metadata/normalized \
  --feature-root workspace/embeddings/legacy \
  --model-type clip \
  --image-model models/clip-vit-b32-vision.onnx \
  --text-model models/clip-vit-b32-text.onnx \
  --tokenizer openai/clip-vit-base-patch32
```

- 会把所有 legacy `*_raw.json` 转成 `workspace/metadata/normalized/<同名>.json`
- 若 JSON 里缺少 `.npy` 或逐帧 embedding，会自动载入 ONNX 模型生成向量，并把 `.npy` 缓存在 `workspace/embeddings/legacy/<模型>/<文件>/frame_features.npy`
- 默认保留 `.npy` 路径即可满足后续索引流程；若想在 JSON 中直接查看每帧向量，可追加 `--inline`
- 目标 JSON 已存在时，脚本会提示“跳过”；如需覆盖则添加 `--overwrite`
- 旧版文档中提到的 `--legacy_metadata_dir`、`--embeddings_output` 参数已经被上述 `--output-dir`、`--feature-root` 所取代，请按本节命令执行

#### 4.3.2 使用规范化 JSON 构建索引

```bash
python scripts/build_index.py \
  workspace/metadata/normalized/bicycle_raw.json \
  workspace/metadata/normalized/book_raw.json \
  --output workspace/index/frame.index \
  --manifest workspace/index/manifest.json
```

- 支持一次传入多个元数据文件，实现多视频联合检索
- 会额外生成 `frame.index.json`，记录索引中每一条向量对应的元数据
- 如果命令里没有写 `--image-model`/`--text-model`/`--tokenizer`，脚本会尝试在 `models/clip-vit-b32-vision.onnx`、`models/clip-vit-b32-text.onnx`、`models/tokenizer/` 等常见目录下自动寻找；三者都找不到时会立即中断并提示需要补齐参数
- 在将向量写入 FAISS 前，脚本会重新加载 JSON 并校验 `feature_file`/inline embedding 是否存在，如有缺失会直接报错提示重新运行 `scripts/migrate_metadata.py`
- 只要所有 JSON 都完成迁移，`video_search.index.add_metadata` 就不再抛出 “metadata 缺少 feature_file” 之类的异常

> 💡如果你是增量更新（例如重新处理了少量视频），可以只将这些新增 metadata 传给 `build_index.py`，索引会在原有基础上附加新向量。

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

## 5. 交互式网站（Web UI）

当命令行结果已经可用时，你可以切换到网页界面，直接在浏览器里输入语义词、悬停预览并下载片段。

### 5.1 一键启动脚本（start_web.py / start_app.py）

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

默认会监听 `0.0.0.0:8000` 并自动打开浏览器，方便在 Mac、iPad 或其它局域网设备上访问；若不希望自动打开，可加 `--no-browser`。如果你更习惯图形化操作，macOS/Windows 可以直接双击仓库根目录的 `start_app.py`，其内部调用的也是 `scripts/start_web.py`。若命令行提示 `./start_app.py: No such file or directory`，请先 `cd` 到包含 `README.md` 的仓库根目录，再执行 `python start_app.py` 或 `chmod +x start_app.py && ./start_app.py`。

> ⚠️ Web UI 依赖已经构建好的索引与 manifest：先用上一节的 `build_index.py ... --output workspace/index/faiss.index --manifest workspace/index/manifest.json` 构建一次，`start_web.py`/`start_app.py` 才能在默认路径中找到它们；否则脚本会直接提示缺少文件。

### 5.2 自定义启动（run_web_app.py）

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

### 5.3 素材管理 / 视频批量处理

新版网页在搜索框下方新增了“素材管理 / 视频批量处理”区域，允许你直接上传 MP4/MOV 等文件，后台会完成下列动作：

1. 将上传文件保存到 `workspace/videos/`（或 `--workspace-dir` 指定的目录），文件名自动加上时间戳避免冲突。
2. 使用与命令行一致的流程抽帧、生成特征、写入元数据（内部复用 `process_video.py`/`build_index.py` 抽象出来的 pipeline）。
3. 把新视频的向量直接追加到当前的 FAISS 索引与 manifest，整个站点无需重启即可检索到最新素材。

前端会分阶段展示进度：

- **上传阶段**：点击“上传并处理”后开始展示 0–100% 的上传进度。
- **后台处理阶段**：上传完成后立即返回 `job_id`，前端每 1 秒轮询 `/api/add_video_status`，实时显示“上传 → 抽帧/生成向量 → 更新索引”等阶段性文案与百分比（`progress` 范围 0–100，数值即为百分比）。
- **完成/失败**：`status=completed` 时提示“处理完成，可以开始检索”；`status=error` 时展示后台的报错消息。

对应的接口定义：

```
POST /api/add_video
Form-Data: file=<UploadFile>
返回：{"success": bool, "message": str, "job_id": str, "video_id": str, "status": "pending|processing", "progress": 0-100, "stage": "uploading"}

GET /api/add_video_status?job_id=<id>
返回：{"job_id": str, "video_id": str, "status": "pending|processing|completed|error", "stage": "uploading|extracting_frames|indexing|completed|error", "progress": 0-100, "message": str, "eta_seconds": float|null, "created_at": str, "updated_at": str}

说明：
- 所有任务会被持久化到 `workspace/index/jobs.json`，重启后仍可查询；`job_id` 为自增数字字符串（1、2、3...）。
- 若查询时省略 `job_id`，接口默认返回最近创建的任务状态；上传接口成功返回的 `job_id` 一定会被写入上述 JSON，不会再出现“未找到对应任务”。
```

若服务器未配置图像模型（`--image-model` 未找到），该区域会自动隐藏上传表单并提示“仅可检索已有素材”。

### 5.4 页面交互说明

1. **顶部搜索框**：输入任意语义词点击“开始搜索”，后台调用同一套 CLIP/CN-CLIP 推理逻辑并检索 FAISS 索引。
2. **候选卡片**：搜索结果以卡片形式展示，鼠标悬停时视频自动跳到匹配时间点并播放，便于快速预览；移开后自动暂停并回到起始位置。
3. **详情面板**：点击任意卡片即展开右侧面板，支持：
   - 查看视频名称、帧索引与精确时间戳；
   - 拖动/输入开始与结束时间，或使用 `±0.5s/±1s` 按钮微调；
   - 直接点击“下载这个片段”调用 FFmpeg 裁剪所选区间，浏览器会弹出下载对话框。

你也可以通过 `--default-top-k` 与 `--preview-duration` 调整默认展示数量与悬停预览的时间窗口，以适配不同素材密度。

### 5.5 常见报错与排查

- **`Invalid input name: attention_mask`**：说明文本 ONNX 模型使用了带别名的输入名称。升级到本 README 对应的最新代码，或在命令行确认 `video_search/features.py` 的 `describe_text_inputs()` 输出中是否包含 `mask`/`segment`/`ids` 等字段。
- **`Tokenizer 输出没有匹配到任何文本模型需要的输入`**：通常是 tokenizer 与模型不匹配或 `--tokenizer` 参数填写错误。请确保 tokenizer 与导出的 ONNX 模型来自同一个 CLIP/CN-CLIP 版本，并且路径/名称正确无误。
- **下载片段时报 `ffmpeg` 找不到**：请确认已经安装 FFmpeg，并将其加入 `PATH`。macOS 用户可执行 `brew install ffmpeg`，Windows 用户需要将 FFmpeg 的 `bin/` 目录加入系统环境变量。

## 6. 常见问题解答

### 6.1 我在 Git 看到了这些文件，是不是已经包含所有代码？

是的，`video_search/` 与 `scripts/` 目录中就是完整实现。只需克隆或下载本仓库，即可得到与当前环境一致的代码。

### 6.2 苹果电脑能跑吗？

可以。macOS 需安装 Homebrew，然后执行：

```bash
brew install ffmpeg opencv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装后即可使用脚本。若在 Apple Silicon 上遇到 FAISS 编译问题，可改用 `conda install -c conda-forge faiss-cpu==1.7.4`。

### 6.3 我只下载仓库，不提供模型能用吗？

下载仓库后可以直接运行抽帧、元数据与索引脚本，但推理和检索环节必须加载你提供的 ONNX 模型与 tokenizer。仓库仅提供执行逻辑，不包含任何预训练权重。

### 6.4 后续如何扩展？

- `video_search/features.py` 可扩展其它 ONNX 模型或量化版本
- `video_search/index.py` 支持替换为 HNSW、Annoy 等其它向量库
- 可以将 `scripts/` 中的命令行脚本改造成 API 或批量任务调度器

## 7. 快速验证

完成依赖安装后，可运行：

```bash
python -m compileall video_search scripts
```

该命令会检查 Python 语法是否正确，确保脚本在当前环境下可被解释执行。

## 8. 下一步建议

1. 准备目标视频并执行 `scripts/process_video.py`
2. 利用生成的元数据构建索引 `scripts/build_index.py`
3. 使用 `scripts/query_index.py` 输入中文或英文描述进行检索

祝你顺利搭建自己的视频语义检索流程！
