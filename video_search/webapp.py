"""FastAPI web界面，提供语义检索、预览和片段下载。"""

from __future__ import annotations

import json
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .features import OnnxClipEncoder
from .index import FaissIndexer


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


def _render_template(config: WebAppConfig) -> str:
    payload = {
        "defaultTopK": config.default_top_k,
        "previewDuration": config.preview_duration,
        "title": config.title,
    }
    data = json.dumps(payload, ensure_ascii=False)
    return TEMPLATE.replace("__APP_CONFIG__", data).replace("__APP_TITLE__", config.title)


def create_app(config: WebAppConfig) -> FastAPI:
    if config.text_model is None:
        raise ValueError("text_model 不能为空，Web UI 需要文本编码模型")

    indexer = FaissIndexer.load(config.index_path, config.manifest_path)
    encoder = OnnxClipEncoder(
        model_type=config.model_type,
        image_model_path=config.image_model,
        text_model_path=config.text_model,
        tokenizer_path=config.tokenizer_path,
        device=config.device,
    )

    app = FastAPI(title=config.title)

    @app.get("/", response_class=HTMLResponse)
    def homepage() -> str:
        return _render_template(config)

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

