(() => {
  const APP_CONFIG = window.APP_CONFIG || {};
  const state = {
    results: [],
    selectedIndex: null,
    pollTimer: null,
    currentJobId: null,
  };

  const els = {
    results: document.getElementById('results'),
    status: document.getElementById('status'),
    searchForm: document.getElementById('search-form'),
    queryInput: document.getElementById('query-input'),
    uploadForm: document.getElementById('upload-form'),
    uploadButton: document.getElementById('upload-button'),
    uploadStatus: document.getElementById('upload-status'),
    uploadTarget: document.getElementById('upload-target'),
    uploadDisabled: document.getElementById('upload-disabled'),
    uploadSection: document.getElementById('upload-section'),
    videoInput: document.getElementById('video-input'),
    progressWrapper: document.getElementById('progress-wrapper'),
    progressBar: document.getElementById('progress-bar'),
    progressText: document.getElementById('progress-text'),
    stageText: document.getElementById('stage-text'),
    etaText: document.getElementById('eta-text'),
    detailPanel: document.getElementById('detail-panel'),
    detailVideo: document.getElementById('detail-video'),
    detailTitle: document.getElementById('detail-title'),
    detailInfo: document.getElementById('detail-info'),
    startInput: document.getElementById('start-input'),
    endInput: document.getElementById('end-input'),
    downloadBtn: document.getElementById('download-btn'),
  };

  const stageTextMap = {
    uploading: '⏳ 正在上传视频…',
    extracting: '🔍 正在抽帧与提取特征…',
    indexing: '📚 正在写入索引…',
    completed: '✅ 处理完成，可以开始检索',
    error: '❌ 处理出错，请重新上传',
  };

  function stageLabel(stage) {
    return stageTextMap[stage] || '⏳ 正在处理中…';
  }

  function clampProgress(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) return 0;
    return Math.min(100, Math.max(0, value));
  }

  function formatEta(seconds) {
    if (typeof seconds !== 'number' || seconds <= 0) return '';
    const sec = Math.round(seconds);
    if (sec < 60) return `剩余 ${sec} 秒`;
    const mins = Math.floor(sec / 60);
    const rem = sec % 60;
    return rem ? `剩余 ${mins} 分 ${rem} 秒` : `剩余 ${mins} 分`;
  }

  function setSearching(flag) {
    const button = els.searchForm ? els.searchForm.querySelector('button') : null;
    if (!button) return;
    button.disabled = flag;
    button.textContent = flag ? '搜索中...' : '开始搜索';
  }

  function setUploadProcessing(flag) {
    if (!els.uploadButton) return;
    els.uploadButton.disabled = flag;
    els.uploadButton.textContent = flag ? '处理中...' : '上传并处理';
  }

  function showProgressArea() {
    if (els.progressWrapper) {
      els.progressWrapper.style.display = 'block';
    }
  }

  function updateProgressUI(progress, stage, etaSeconds) {
    const pct = clampProgress(progress);
    const etaText = stage === 'completed' || stage === 'error' ? '' : formatEta(etaSeconds);
    if (els.progressBar) {
      els.progressBar.style.width = `${pct}%`;
      els.progressBar.textContent = `${pct}%`;
    }
    if (els.progressText) els.progressText.textContent = `${pct}%`;
    if (els.stageText) els.stageText.textContent = stageLabel(stage);
    if (els.etaText) els.etaText.textContent = etaText;
    showProgressArea();
    if (els.uploadStatus) {
      const parts = [stageLabel(stage), `${pct}%`];
      if (etaText) parts.push(etaText);
      els.uploadStatus.textContent = parts.filter(Boolean).join(' ｜ ');
    }
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  async function pollStatus(jobId) {
    stopPolling();
    const tick = async () => {
      try {
        const res = await fetch(`/api/add_video_status?job_id=${jobId}`);
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || '查询进度失败');

        const stage = data.stage || 'uploading';
        const pct = clampProgress(data.progress);
        const etaSeconds = data.eta_seconds;
        updateProgressUI(pct, stage, etaSeconds);

        if (stage === 'error' && els.uploadStatus) {
          const parts = [stageLabel(stage)];
          if (data.error) parts.push(data.error);
          else if (data.message) parts.push(data.message);
          els.uploadStatus.textContent = parts.filter(Boolean).join(' ｜ ');
        }

        if (stage === 'completed' || pct >= 100) {
          stopPolling();
          setUploadProcessing(false);
          updateProgressUI(100, 'completed', 0);
          if (els.videoInput) els.videoInput.value = '';
        } else if (stage === 'error') {
          stopPolling();
          setUploadProcessing(false);
        }
      } catch (err) {
        stopPolling();
        setUploadProcessing(false);
        if (els.uploadStatus) els.uploadStatus.textContent = err instanceof Error ? err.message : '查询进度失败';
      }
    };
    state.pollTimer = setInterval(tick, 500);
    tick();
  }

  async function startUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    setUploadProcessing(true);
    updateProgressUI(0, 'uploading', null);
    const res = await fetch('/api/add_video', { method: 'POST', body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.job_id) {
      const msg = data.detail || data.message || '上传失败';
      setUploadProcessing(false);
      if (els.uploadStatus) els.uploadStatus.textContent = msg;
      throw new Error(msg);
    }
    return data.job_id;
  }

  function initUpload() {
    if (!els.uploadSection) return;
    if (APP_CONFIG.uploadsEnabled === false) {
      if (els.uploadDisabled) els.uploadDisabled.style.display = 'block';
      if (els.uploadForm) els.uploadForm.style.display = 'none';
      return;
    }
    if (els.uploadTarget && APP_CONFIG.uploadDir) {
      els.uploadTarget.textContent = APP_CONFIG.uploadDir;
    }
    if (!els.uploadForm || !els.videoInput) return;
    els.uploadForm.addEventListener('submit', async (evt) => {
      evt.preventDefault();
      if (!els.videoInput.files || !els.videoInput.files[0]) {
        if (els.uploadStatus) els.uploadStatus.textContent = '请选择要上传的视频文件';
        return;
      }
      try {
        const jobId = await startUpload(els.videoInput.files[0]);
        state.currentJobId = jobId;
        pollStatus(jobId);
      } catch (err) {
        if (err instanceof Error && els.uploadStatus) els.uploadStatus.textContent = err.message;
      }
    });
  }

  function renderResults() {
    if (!els.results) return;
    els.results.innerHTML = '';
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
      meta.textContent = `时间：${Number(item.timestamp).toFixed(2)}s  置信度：${Number(item.score).toFixed(3)}`;
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
      els.results.appendChild(card);
    });
  }

  function selectResult(index) {
    const item = state.results[index];
    if (!item) return;
    state.selectedIndex = index;
    if (els.detailPanel) els.detailPanel.classList.add('active');
    if (els.detailVideo) {
      els.detailVideo.src = `/api/video?path=${encodeURIComponent(item.video_path)}#t=${item.timestamp}`;
      els.detailVideo.currentTime = item.timestamp;
    }
    if (els.detailTitle) els.detailTitle.textContent = item.display_name;
    if (els.detailInfo) els.detailInfo.textContent = `帧索引 ${item.frame_index} ｜ 时间戳 ${Number(item.timestamp).toFixed(2)} 秒`;
    const start = Math.max(item.preview_start || item.timestamp, 0).toFixed(2);
    const endVal = Math.max(item.preview_end || item.timestamp, start).toFixed(2);
    if (els.startInput) els.startInput.value = start;
    if (els.endInput) els.endInput.value = endVal;
  }

  function shiftInput(targetInput, delta) {
    const value = parseFloat(targetInput.value || '0');
    const next = Math.max(value + delta, 0);
    targetInput.value = next.toFixed(2);
  }

  function initControls() {
    document.querySelectorAll('.control-group .control-buttons button').forEach((btn) => {
      btn.addEventListener('click', () => {
        const delta = parseFloat(btn.dataset.shift || '0');
        const target = btn.dataset.target === 'end' ? els.endInput : els.startInput;
        if (target) shiftInput(target, delta);
      });
    });
    if (els.downloadBtn) {
      els.downloadBtn.addEventListener('click', async () => {
        const item = state.results[state.selectedIndex];
        if (!item) return;
        const payload = {
          video_path: item.video_path,
          start: parseFloat(els.startInput?.value || item.timestamp),
          end: parseFloat(els.endInput?.value || item.timestamp + (APP_CONFIG.previewDuration || 3)),
        };
        if (payload.end <= payload.start) {
          if (els.status) els.status.textContent = '结束时间必须大于开始时间';
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
          if (els.status) els.status.textContent = '片段已保存';
        } catch (err) {
          if (els.status) els.status.textContent = err instanceof Error ? err.message : '下载失败';
        }
      });
    }
  }

  function initSearch() {
    if (!els.searchForm || !els.queryInput) return;
    els.searchForm.addEventListener('submit', async (evt) => {
      evt.preventDefault();
      const query = els.queryInput.value.trim();
      if (!query) {
        if (els.status) els.status.textContent = '请输入语义词后再搜索';
        return;
      }
      setSearching(true);
      try {
        const res = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k: APP_CONFIG.defaultTopK || 9 }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || '搜索失败');
        state.results = data.results || [];
        renderResults();
        if (els.status) {
          els.status.textContent = state.results.length ? `找到 ${state.results.length} 个候选片段` : '没有匹配项';
        }
      } catch (err) {
        if (els.status) els.status.textContent = err instanceof Error ? err.message : '搜索失败';
      } finally {
        setSearching(false);
      }
    });
  }

  function init() {
    initSearch();
    initUpload();
    initControls();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
