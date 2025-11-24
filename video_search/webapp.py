"""FastAPI web界面，提供语义检索、预览、片段下载与素材管理。"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .features import OnnxClipEncoder, build_frame_feature_cache
from .frames import extract_keyframes
from .index import FaissIndexer
from .metadata import VideoMetadata, save_metadata
from .jobs import JobStore, JobRecord
from .pipeline import build_or_update_index


@dataclass
class WebAppConfig:
    """运行 Web UI 所需的关键配置。"""

    index_path: Path
    manifest_path: Optional[Path] = None
    model_type: str = "clip"
    image_model: Optional[Path] = None
    text_model: Optional[Path] = None
    tokenizer_path: Optional[str] = None
    device: str = "cpu"
    default_top_k: int = 9
    preview_duration: float = 3.0
    title: str = "视频语义检索"
    workspace_dir: Optional[Path] = None
    output_root: Optional[Path] = None
    upload_dir: Optional[Path] = None
    processing_method: str = "interval"
    processing_interval: float = 1.0
    processing_scene_threshold: float = 30.0
    processing_batch_size: int = 32
    processing_image_format: str = "jpg"
    processing_quality: int = 95


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class ClipRequest(BaseModel):
    video_path: str
    start: float
    end: float


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {candidate}")
    return candidate.resolve()


def _safe_stem(name: str) -> str:
    stem = Path(name).stem or "video"
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in stem)
    cleaned = cleaned.strip("_")
    return cleaned or "video"


def _save_upload(file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


def _render_template(config: WebAppConfig, upload_dir: Optional[Path]) -> str:
    payload = {
        "defaultTopK": config.default_top_k,
        "previewDuration": config.preview_duration,
        "title": config.title,
        "uploadsEnabled": bool(config.image_model),
        "uploadDir": str(upload_dir) if upload_dir else "",
    }
    data = json.dumps(payload, ensure_ascii=False)
    return (
        TEMPLATE.replace("__APP_CONFIG__", data)
        .replace("__APP_TITLE__", config.title)
        .replace("__APP_UPLOAD_TARGET__", str(upload_dir) if upload_dir else "")
    )


def create_app(config: WebAppConfig) -> FastAPI:
    if config.text_model is None:
        raise ValueError("text_model 不能为空，Web UI 需要文本编码模型")

    index_path = config.index_path.resolve()
    manifest_path = (
        Path(config.manifest_path).expanduser()
        if config.manifest_path
        else index_path.with_suffix(".json")
    )
    workspace_dir = (
        Path(config.workspace_dir).expanduser()
        if config.workspace_dir
        else index_path.parent.parent
    )
    output_root = (
        Path(config.output_root).expanduser()
        if config.output_root
        else workspace_dir
    )
    upload_dir = (
        Path(config.upload_dir).expanduser()
        if config.upload_dir
        else workspace_dir / "videos"
    )
    upload_dir.mkdir(parents=True, exist_ok=True)

    indexer = FaissIndexer.load(index_path, manifest_path)
    encoder = OnnxClipEncoder(
        model_type=config.model_type,
        image_model_path=config.image_model,
        text_model_path=config.text_model,
        tokenizer_path=config.tokenizer_path,
        device=config.device,
    )

    jobs_path = output_root / "index" / "jobs.json"
    job_store = JobStore(jobs_path)

    app = FastAPI(title=config.title)
    processing_lock = asyncio.Lock()

    progress_ranges = {
        # 与前端/文档约定保持一致：上传 0-10，抽帧+特征 10-60，建索引 60-99，完成 100
        "uploading": (0.0, 10.0),
        "extracting": (10.0, 60.0),
        "indexing": (60.0, 99.0),
        "completed": (100.0, 100.0),
        "error": (0.0, 0.0),
    }

    def _progress_for_stage(stage: str, fraction: float = 0.0) -> float:
        start, end = progress_ranges.get(stage, (0.0, 100.0))
        fraction = max(0.0, min(fraction, 1.0))
        return start + (end - start) * fraction

    def _set_stage(
        job_id: str,
        stage: str,
        *,
        status: Optional[str] = None,
        fraction: float = 0.0,
        message: Optional[str] = None,
    ) -> JobRecord:
        payload: Dict[str, object] = {
            "stage": stage,
            "progress": _progress_for_stage(stage, fraction),
        }
        if status:
            payload["status"] = status
        if message is not None:
            payload["message"] = message
        return _update_job(job_id, **payload)

    def _update_job(job_id: str, **kwargs: object) -> JobRecord:
        return job_store.update_job(job_id, **kwargs)

    def _new_job(video_id: str, message: str) -> JobRecord:
        return job_store.new_job(video_id=video_id, message=message)

    @app.get("/", response_class=HTMLResponse)
    def homepage() -> str:
        return _render_template(config, upload_dir)

    @app.post("/api/search")
    def search(request: SearchRequest) -> Dict[str, List[Dict[str, object]]]:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query 不能为空")
        top_k = request.top_k or config.default_top_k
        embedding = encoder.encode_text([query])
        results = indexer.search(embedding, top_k=top_k)
        payload = []
        for frame, score in results:
            preview_start = max(frame.timestamp - config.preview_duration / 2, 0.0)
            payload.append(
                {
                    "video_path": frame.video_path,
                    "image_path": frame.image_path,
                    "timestamp": frame.timestamp,
                    "score": score,
                    "preview_start": preview_start,
                    "preview_end": preview_start + config.preview_duration,
                    "display_name": Path(frame.video_path).name,
                    "metadata_path": frame.metadata_path,
                    "frame_index": frame.frame_index,
                }
            )
        return {"results": payload}

    @app.get("/api/frame_image")
    def frame_image(path: str) -> FileResponse:
        file_path = _resolve_path(path)
        return FileResponse(file_path, media_type="image/jpeg")

    @app.get("/api/video")
    def video_file(path: str) -> FileResponse:
        file_path = _resolve_path(path)
        return FileResponse(file_path, media_type="video/mp4")

    @app.post("/api/download_clip")
    def download_clip(request: ClipRequest) -> StreamingResponse:
        source = _resolve_path(request.video_path)
        start = max(request.start, 0.0)
        end = max(request.end, start + 0.1)
        duration = max(end - start, 0.1)
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-f",
            "mp4",
            "-movflags",
            "frag_keyframe+empty_moov",
            "pipe:1",
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        if process.stdout is None:
            raise HTTPException(status_code=500, detail="无法创建 ffmpeg 输出流")

        def iterator() -> Iterable[bytes]:
            try:
                while True:
                    chunk = process.stdout.read(8192)
                    if not chunk:
                        break
                    yield chunk
            finally:
                process.stdout.close()
                process.wait(timeout=1)

        filename = Path(source).stem
        filename += f"_{start:.2f}-{end:.2f}.mp4"
        headers = {
            "Content-Disposition": f"attachment; filename={urllib.parse.quote(filename)}",
        }
        return StreamingResponse(iterator(), media_type="video/mp4", headers=headers)

    @app.post("/api/add_video")
    async def add_video(file: UploadFile = File(...)) -> Dict[str, object]:
        if not config.image_model:
            raise HTTPException(status_code=400, detail="服务器未配置图像模型，暂不支持上传处理")
        if not file.filename:
            raise HTTPException(status_code=400, detail="请提供视频文件")

        suffix = Path(file.filename).suffix or ".mp4"
        stem = _safe_stem(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_id = f"{stem}_{timestamp}"
        job = _new_job(video_id, "正在排队")

        candidate = upload_dir / f"{video_id}{suffix}"
        counter = 1
        while candidate.exists():
            video_id = f"{stem}_{timestamp}_{counter}"
            candidate = upload_dir / f"{video_id}{suffix}"
            counter += 1

        try:
            _save_upload(file, candidate)
            _set_stage(job.job_id, "uploading", status="processing", fraction=0.9, message="上传完成，等待处理")
        except Exception as exc:  # pragma: no cover
            _update_job(job.job_id, status="error", stage="uploading", progress=0.0, message=f"保存视频失败: {exc}", error=str(exc))
            raise HTTPException(status_code=500, detail=f"保存视频失败: {exc}") from exc
        finally:
            file.file.close()

        async def _process_job() -> None:
            nonlocal indexer
            try:
                _set_stage(job.job_id, "uploading", status="processing", fraction=1.0, message="上传完成，准备处理")
                async with processing_lock:
                    _set_stage(job.job_id, "extracting", status="processing", fraction=0.05, message="正在抽帧")
                    frames_dir = output_root / "frames" / candidate.stem
                    embeddings_dir = output_root / "embeddings" / config.model_type / candidate.stem
                    feature_path = embeddings_dir / "frame_features.npy"
                    metadata_file = output_root / "metadata" / f"{candidate.stem}.json"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    embeddings_dir.mkdir(parents=True, exist_ok=True)

                    frames, fps = await asyncio.to_thread(
                        extract_keyframes,
                        candidate,
                        frames_dir,
                        config.processing_method,
                        config.processing_interval,
                        config.processing_scene_threshold,
                        config.processing_image_format,
                        config.processing_quality,
                    )

                    _set_stage(
                        job.job_id,
                        "extracting",
                        status="processing",
                        fraction=0.4,
                        message="抽帧完成",
                    )
                    _set_stage(job.job_id, "extracting", status="processing", fraction=0.5, message="开始生成向量")

                    def _encode_progress(done: int, total: int) -> None:
                        fraction = 0.0 if total <= 0 else done / float(total)
                        _set_stage(
                            job.job_id,
                            "extracting",
                            status="processing",
                            fraction=fraction,
                            message=f"生成向量 {done}/{total}",
                        )
                    cache = await asyncio.to_thread(
                        build_frame_feature_cache,
                        frames,
                        encoder,
                        feature_path,
                        config.processing_batch_size,
                        _encode_progress,
                    )

                    _set_stage(job.job_id, "extracting", fraction=1.0, message="向量生成完成，写入元数据")

                    metadata = VideoMetadata(
                        video_path=str(candidate),
                        frames=cache.frames,
                        feature_file=str(feature_path),
                        embedding_dim=int(cache.features.shape[1]) if cache.features.size else encoder.dimension,
                        model_type=config.model_type,
                        image_model_path=str(config.image_model) if config.image_model else None,
                        text_model_path=str(config.text_model) if config.text_model else None,
                        tokenizer_path=config.tokenizer_path,
                        frame_interval=config.processing_interval if config.processing_method == "interval" else None,
                        fps=fps,
                        method=config.processing_method,
                    )
                    await asyncio.to_thread(save_metadata, metadata, metadata_file)

                    _set_stage(job.job_id, "indexing", status="processing", fraction=0.1, message="正在更新索引")
                    indexer = await asyncio.to_thread(
                        build_or_update_index,
                        [metadata_file],
                        index_path,
                        manifest_path=manifest_path,
                        metric=indexer.metric,
                        normalize=indexer.normalize,
                        indexer=indexer,
                    )
                    _set_stage(job.job_id, "indexing", status="processing", fraction=0.95, message="索引更新完成")
                    _set_stage(job.job_id, "completed", status="completed", fraction=1.0, message="处理完成，可以开始检索")
                    _update_job(
                        job.job_id,
                        result={
                            "video_path": str(candidate),
                            "metadata_path": str(metadata_file),
                            "index_path": str(index_path),
                            "video_id": video_id,
                        },
                    )
            except Exception as exc:  # pragma: no cover
                _update_job(
                    job.job_id,
                    status="error",
                    stage="error",
                    progress=(job_store.get(job.job_id).progress if job_store.get(job.job_id) else 0.0),
                    message=f"处理失败: {exc}",
                    error=str(exc),
                )

        asyncio.create_task(_process_job())

        return {
            "job_id": job.job_id,
        }

    @app.get("/api/add_video_status")
    def add_video_status(job_id: Optional[str] = None) -> Dict[str, object]:
        target_id = job_id or (job_store.last().job_id if job_store.last() else None)
        if not target_id:
            raise HTTPException(status_code=404, detail="未找到对应任务")
        status = job_store.get(target_id)
        if not status:
            raise HTTPException(status_code=404, detail="未找到对应任务")

        progress = max(0.0, min(float(status.progress), 100.0))
        payload: Dict[str, object] = {
            "success": True,
            "job_id": status.job_id,
            "stage": status.stage,
            "progress": round(progress, 2),
            "eta_seconds": status.eta_seconds,
        }
        if status.result and isinstance(status.result, dict):
            for key in ["metadata_path", "video_path", "index_path"]:
                if key in status.result:
                    payload[key] = status.result[key]
        if status.status:
            payload["status"] = status.status
        if status.message:
            payload["message"] = status.message
        return payload

    return app


TEMPLATE = """<!DOCTYPE html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__APP_TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', sans-serif;
      background: #0c0c0f;
      color: #f2f2f2;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top, #1f1f35, #050507);
    }
    .app {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 20px 60px;
    }
    #upload-section {
      margin: 24px auto 32px;
      padding: 20px;
      border-radius: 18px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.12);
    }
    #upload-section h2 {
      margin: 0 0 8px;
      font-size: 20px;
    }
    #upload-form {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
    }
    #upload-form input[type="file"] {
      flex: 1 1 260px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.15);
      padding: 10px 14px;
      background: rgba(0,0,0,0.2);
      color: inherit;
    }
    #upload-status {
      margin-top: 8px;
      min-height: 20px;
      color: #c7c7e7;
      font-size: 14px;
    }
    #progress-wrapper {
      margin-top: 12px;
      display: none;
    }
    #progress-container {
      margin-top: 10px;
      height: 16px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
      position: relative;
    }
    #progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(135deg, #8a63ff, #3f7dfd);
      color: #fff;
      font-size: 12px;
      line-height: 16px;
      text-indent: 8px;
      transition: width 0.3s ease;
    }
    #progress-text, #eta-text, #stage-text {
      margin-top: 6px;
      font-size: 13px;
      color: #c7c7e7;
    }
    #upload-disabled {
      margin-top: 12px;
      color: #ffadad;
      font-size: 14px;
    }
    code.path {
      padding: 2px 6px;
      border-radius: 6px;
      background: rgba(0,0,0,0.35);
      border: 1px solid rgba(255,255,255,0.1);
      font-size: 13px;
    }
    header {
      text-align: center;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 1px;
    }
    #search-form {
      display: flex;
      gap: 12px;
      margin-top: 16px;
      justify-content: center;
      flex-wrap: wrap;
    }
    #query-input {
      flex: 1 1 320px;
      max-width: 520px;
      padding: 12px 16px;
      border-radius: 999px;
      border: 1px solid #3a3a5f;
      background: rgba(255,255,255,0.05);
      color: inherit;
      font-size: 16px;
    }
    button.primary {
      padding: 12px 24px;
      border-radius: 999px;
      border: none;
      font-size: 16px;
      background: linear-gradient(135deg, #8a63ff, #3f7dfd);
      color: #fff;
      cursor: pointer;
      transition: opacity 0.2s ease;
    }
    button.primary:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    #results {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 18px;
    }
    .card {
      background: rgba(255,255,255,0.04);
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.08);
      overflow: hidden;
      transition: transform 0.2s ease, border 0.2s ease;
      cursor: pointer;
    }
    .card:hover {
      transform: translateY(-4px);
      border-color: #5f7bff;
    }
    .card video {
      width: 100%;
      display: block;
      aspect-ratio: 16/9;
      object-fit: cover;
      background: #000;
    }
    .card img {
      width: 100%;
      aspect-ratio: 16/9;
      object-fit: cover;
      filter: brightness(0.85);
    }
    .card-body {
      padding: 12px 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .card-title {
      font-size: 16px;
      font-weight: 600;
    }
    .card-meta {
      font-size: 14px;
      color: #a7a7c7;
    }
    #detail-panel {
      margin-top: 32px;
      padding: 20px;
      border-radius: 16px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      display: none;
      gap: 18px;
      flex-wrap: wrap;
    }
    #detail-panel.active {
      display: flex;
    }
    #detail-video {
      width: min(480px, 100%);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.1);
      background: #000;
    }
    #clip-controls {
      flex: 1 1 280px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .control-group label {
      width: 64px;
      color: #b7b7d9;
      font-size: 14px;
    }
    .control-group input {
      flex: 1 1 120px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.15);
      background: rgba(0,0,0,0.3);
      color: inherit;
    }
    .control-buttons button {
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.2);
      background: rgba(255,255,255,0.05);
      color: inherit;
      cursor: pointer;
    }
    .download-btn {
      padding: 12px;
      border-radius: 10px;
      border: none;
      background: linear-gradient(135deg, #22c1c3, #2a9df4);
      color: #fff;
      font-size: 16px;
      cursor: pointer;
    }
    #status {
      margin-top: 12px;
      text-align: center;
      color: #b7b7d9;
      min-height: 20px;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div class=\"app\">
    <header>
      <h1>__APP_TITLE__</h1>
      <p>输入语义词，快速定位视频片段，悬停预览并一键下载。</p>
      <form id=\"search-form\">
        <input id=\"query-input\" placeholder=\"例如：跑步的男生、翻书的老师...\" autocomplete=\"off\" />
        <button type=\"submit\" class=\"primary\">开始搜索</button>
      </form>
      <div id=\"status\"></div>
    </header>


    <section id=\"upload-section\">
      <h2>素材管理 / 视频批量处理</h2>
      <p class=\"card-meta\">上传 MP4/MOV 等视频，系统会自动抽帧、提取特征并更新索引。保存目录：<code class=\"path\" id=\"upload-target\">__APP_UPLOAD_TARGET__</code></p>
      <form id=\"upload-form\">
        <input type=\"file\" id=\"video-input\" accept=\"video/mp4,video/*\" />
        <button type=\"submit\" class=\"primary\" id=\"upload-button\">上传并处理</button>
      </form>
      <div id=\"upload-status\"></div>
      <div id=\"progress-wrapper\">
        <div id=\"stage-text\"></div>
        <div id=\"progress-container\">
          <div id=\"progress-bar\"></div>
        </div>
        <div id=\"progress-text\"></div>
        <div id=\"eta-text\"></div>
      </div>
      <div id=\"upload-disabled\" style=\"display:none;\">服务器缺少图像模型，仅可检索已有素材。</div>
    </section>

    <section id=\"results\"></section>

    <section id=\"detail-panel\">
      <video id=\"detail-video\" controls muted></video>
      <div id=\"clip-controls\">
        <div>
          <div class=\"card-meta\" id=\"detail-title\"></div>
          <div class=\"card-meta\" id=\"detail-info\"></div>
        </div>
        <div class=\"control-group\">
          <label>开始</label>
          <input type=\"number\" step=\"0.1\" id=\"start-input\" />
          <div class=\"control-buttons\">
            <button type=\"button\" data-shift=\"-1\">-1s</button>
            <button type=\"button\" data-shift=\"-0.5\">-0.5s</button>
          </div>
        </div>
        <div class=\"control-group\">
          <label>结束</label>
          <input type=\"number\" step=\"0.1\" id=\"end-input\" />
          <div class=\"control-buttons\">
            <button type=\"button\" data-target=\"end\" data-shift=\"0.5\">+0.5s</button>
            <button type=\"button\" data-target=\"end\" data-shift=\"1\">+1s</button>
          </div>
        </div>
        <button class=\"download-btn\" id=\"download-btn\">下载这个片段</button>
      </div>
    </section>
  </div>

  <script>
    const APP_CONFIG = __APP_CONFIG__;
    const state = { results: [], selectedIndex: null };
    const resultsEl = document.getElementById('results');
    const statusEl = document.getElementById('status');
    const formEl = document.getElementById('search-form');
    const uploadForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const uploadStatus = document.getElementById('upload-status');
    const uploadTarget = document.getElementById('upload-target');
    const uploadDisabled = document.getElementById('upload-disabled');
    const uploadSection = document.getElementById('upload-section');
    const videoInput = document.getElementById('video-input');
    const queryInput = document.getElementById('query-input');
    const detailPanel = document.getElementById('detail-panel');
    const detailVideo = document.getElementById('detail-video');
    const detailTitle = document.getElementById('detail-title');
    const detailInfo = document.getElementById('detail-info');
    const startInput = document.getElementById('start-input');
    const endInput = document.getElementById('end-input');
    const downloadBtn = document.getElementById('download-btn');

    formEl.addEventListener('submit', async (evt) => {
      evt.preventDefault();
      const query = queryInput.value.trim();
      if (!query) {
        statusEl.textContent = '请输入语义词后再搜索';
        return;
      }
      setSearching(true);
      try {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: APP_CONFIG.defaultTopK }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || '搜索失败');
        }
        const data = await res.json();
        state.results = data.results || [];
        renderResults();
        statusEl.textContent = state.results.length ? `找到 ${state.results.length} 个候选片段` : '没有匹配项';
      } catch (err) {
        console.error(err);
        statusEl.textContent = err.message;
      } finally {
        setSearching(false);
      }
    });

    function setSearching(flag) {
      const button = formEl.querySelector('button');
      button.disabled = flag;
      if (flag) {
        button.textContent = '搜索中...';
      } else {
        button.textContent = '开始搜索';
      }
    }

    function setUploadProcessing(flag) {
      if (!uploadButton) return;
      uploadButton.disabled = flag;
      uploadButton.textContent = flag ? '处理中...' : '上传并处理';
    }

    if (uploadSection) {
      if (!APP_CONFIG.uploadsEnabled) {
        if (uploadDisabled) uploadDisabled.style.display = 'block';
        if (uploadForm) uploadForm.style.display = 'none';
      } else if (uploadForm) {
        if (uploadTarget && APP_CONFIG.uploadDir) {
          uploadTarget.textContent = APP_CONFIG.uploadDir;
        }
        const progressWrapper = document.getElementById('progress-wrapper');
        const progressContainer = document.getElementById('progress-container');
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const etaTextEl = document.getElementById('eta-text');
        const stageTextEl = document.getElementById('stage-text');

        const STAGE_LABELS = {
          uploading: '📤 正在上传视频…',
          extracting: '🧩 正在抽帧与提取特征…',
          indexing: '📚 正在写入索引…',
          completed: '✅ 处理完成，可以开始检索',
          finished: '✅ 处理完成，可以开始检索',
          error: '❌ 处理失败，请重新上传',
        };

        const stageName = (stage) => STAGE_LABELS[stage] || '⏳ 正在处理中…';

        const formatEta = (etaSeconds) => {
          if (typeof etaSeconds !== 'number' || etaSeconds <= 0) return '';
          const seconds = Math.round(etaSeconds);
          if (seconds < 60) return `剩余 ${seconds} 秒`;
          const mins = Math.floor(seconds / 60);
          const secs = seconds % 60;
          return secs ? `剩余 ${mins} 分 ${secs} 秒` : `剩余 ${mins} 分`;
        };

        const clampProgress = (value) => {
          if (typeof value !== 'number' || Number.isNaN(value)) return 0;
          return Math.min(100, Math.max(0, Math.round(value)));
        };

        const showProgress = () => {
          if (progressWrapper) progressWrapper.style.display = 'block';
          if (progressContainer) progressContainer.style.display = 'block';
        };

        const updateProgressBar = (pct, stageText, etaText) => {
          showProgress();
          progressBar.style.width = `${pct}%`;
          progressBar.textContent = `${pct}%`;
          stageTextEl.textContent = stageText || '';
          progressText.textContent = `${pct}%`;
          etaTextEl.textContent = etaText || '';
        };

        const resetProgressBar = () => {
          if (!progressWrapper) return;
          progressWrapper.style.display = 'none';
          progressBar.style.width = '0%';
          progressBar.textContent = '';
          progressText.textContent = '';
          stageTextEl.textContent = '';
          etaTextEl.textContent = '';
        };

        const pollStatus = (jobId) => {
          let timer = null;
          const stop = () => { if (timer) clearInterval(timer); timer = null; };
          const tick = async () => {
            try {
              const res = await fetch(`/api/add_video_status?job_id=${encodeURIComponent(jobId)}`);
              const data = await res.json().catch(() => ({}));
              if (!res.ok || data.success === false) {
                throw new Error(data.detail || data.message || '查询进度失败');
              }

              let stage = data.stage || 'uploading';
              if (stage === 'finished') stage = 'completed';
              const pct = clampProgress(data.progress);
              const etaText = stage === 'completed' || stage === 'error' ? '' : formatEta(data.eta_seconds);
              const stageText = stageName(stage);

              updateProgressBar(pct, stageText, etaText);
              const parts = [stageText, `${pct}%`];
              if (etaText) parts.push(etaText);
              uploadStatus.textContent = parts.filter(Boolean).join(' ｜ ');

              if (stage === 'completed' || pct >= 100) {
                stop();
                setUploadProcessing(false);
                updateProgressBar(100, stageName('completed'), '');
                uploadStatus.textContent = stageName('completed');
                videoInput.value = '';
              } else if (stage === 'error') {
                stop();
                setUploadProcessing(false);
                uploadStatus.textContent = stageName('error');
              }
            } catch (err) {
              stop();
              setUploadProcessing(false);
              uploadStatus.textContent = err.message;
            }
          };
          timer = setInterval(tick, 500);
          tick();
          return stop;
        };

        const startUpload = async (file) => {
          const formData = new FormData();
          formData.append('file', file);
          setUploadProcessing(true);
          showProgress();
          updateProgressBar(0, stageName('uploading'), '');
          uploadStatus.textContent = stageName('uploading');
          const res = await fetch('/api/add_video', { method: 'POST', body: formData });
          const data = await res.json().catch(() => ({}));
          if (!res.ok || !data.job_id) {
            const msg = data.detail || data.message || '上传失败';
            setUploadProcessing(false);
            uploadStatus.textContent = msg;
            throw new Error(msg);
          }
          return data.job_id;
        };

        uploadForm.addEventListener('submit', async (evt) => {
          evt.preventDefault();
          if (!videoInput || !videoInput.files || !videoInput.files[0]) {
            uploadStatus.textContent = '请选择要上传的视频文件';
            return;
          }
          resetProgressBar();
          try {
            const jobId = await startUpload(videoInput.files[0]);
            pollStatus(jobId);
          } catch (err) {
            if (err instanceof Error) uploadStatus.textContent = err.message;
            setUploadProcessing(false);
          }
        });
      }
    }

    function renderResults() {
      resultsEl.innerHTML = '';
      state.results.forEach((item, index) => {
        const card = document.createElement('article');
        card.className = 'card';
        const video = document.createElement('video');
        video.src = `/api/video?path=${encodeURIComponent(item.video_path)}`;
        video.poster = `/api/frame_image?path=${encodeURIComponent(item.image_path)}`;
        video.muted = true;
        video.preload = 'metadata';
        video.dataset.timestamp = item.timestamp;
        card.appendChild(video);
        const body = document.createElement('div');
        body.className = 'card-body';
        const title = document.createElement('div');
        title.className = 'card-title';
        title.textContent = item.display_name;
        const meta = document.createElement('div');
        meta.className = 'card-meta';
        meta.textContent = `时间：${item.timestamp.toFixed(2)}s  置信度：${item.score.toFixed(3)}`;
        body.appendChild(title);
        body.appendChild(meta);
        card.appendChild(body);
        card.addEventListener('mouseenter', () => {
          try {
            video.currentTime = item.timestamp;
            video.play();
          } catch (err) {
            console.warn('无法自动播放', err);
          }
        });
        card.addEventListener('mouseleave', () => {
          video.pause();
          video.currentTime = item.timestamp;
        });
        card.addEventListener('click', () => selectResult(index));
        resultsEl.appendChild(card);
      });
    }

    function selectResult(index) {
      const item = state.results[index];
      if (!item) return;
      state.selectedIndex = index;
      detailPanel.classList.add('active');
      detailVideo.src = `/api/video?path=${encodeURIComponent(item.video_path)}#t=${item.timestamp}`;
      detailVideo.currentTime = item.timestamp;
      detailTitle.textContent = item.display_name;
      detailInfo.textContent = `帧索引 ${item.frame_index} ｜ 时间戳 ${item.timestamp.toFixed(2)} 秒`;
      const start = Math.max(item.preview_start, 0).toFixed(2);
      const end = Math.max(item.preview_end, start).toFixed(2);
      startInput.value = start;
      endInput.value = end;
    }

    function shiftInput(targetInput, delta) {
      const value = parseFloat(targetInput.value || '0');
      const next = Math.max(value + delta, 0);
      targetInput.value = next.toFixed(2);
    }

    document.querySelectorAll('.control-group .control-buttons button').forEach((btn) => {
      btn.addEventListener('click', () => {
        const delta = parseFloat(btn.dataset.shift || '0');
        const target = btn.dataset.target === 'end' ? endInput : startInput;
        shiftInput(target, delta);
      });
    });

    downloadBtn.addEventListener('click', async () => {
      const item = state.results[state.selectedIndex];
      if (!item) return;
      const payload = {
        video_path: item.video_path,
        start: parseFloat(startInput.value || item.timestamp),
        end: parseFloat(endInput.value || (item.timestamp + APP_CONFIG.previewDuration)),
      };
      if (payload.end <= payload.start) {
        statusEl.textContent = '结束时间必须大于开始时间';
        return;
      }
      try {
        const res = await fetch('/api/download_clip', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error('下载失败');
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${item.display_name}_${payload.start.toFixed(2)}-${payload.end.toFixed(2)}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        statusEl.textContent = '片段已保存';
      } catch (err) {
        console.error(err);
        statusEl.textContent = err.message;
      }
    });
  </script>
</body>
</html>
"""

