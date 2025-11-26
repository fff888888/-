"""持久化上传/处理任务状态的工具。"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class JobRecord:
    job_id: str
    video_id: str
    status: str = "pending"  # pending / uploading / processing / completed / error
    stage: str = "uploading"  # uploading / extracting_frames / indexing / completed / error
    progress: float = 0.0  # 0 - 100
    message: str = "正在排队"
    eta_seconds: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, object]] = None
    total_items: int = 1
    completed_items: int = 0
    current_item_name: Optional[str] = None
    started_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(float(self.progress), 2),
            "message": self.message,
            "eta_seconds": self.eta_seconds,
            "error": self.error,
            "result": self.result,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "current_item_name": self.current_item_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "JobRecord":
        progress = float(data.get("progress", 0.0) or 0.0)
        if progress <= 1.0:
            progress = progress * 100.0

        return JobRecord(
            job_id=str(data.get("job_id", "")),
            video_id=str(data.get("video_id", "")),
            status=str(data.get("status", "pending")),
            stage=str(data.get("stage", data.get("phase", "uploading"))),
            progress=progress,
            message=str(data.get("message", "")),
            eta_seconds=data.get("eta_seconds"),
            error=data.get("error"),
            result=data.get("result"),
            total_items=int(data.get("total_items", 1) or 1),
            completed_items=int(data.get("completed_items", 0) or 0),
            current_item_name=data.get("current_item_name"),
            started_at=datetime.fromisoformat(str(data.get("started_at")))
            if data.get("started_at")
            else None,
            created_at=datetime.fromisoformat(str(data.get("created_at")))
            if data.get("created_at")
            else datetime.utcnow(),
            updated_at=datetime.fromisoformat(str(data.get("updated_at")))
            if data.get("updated_at")
            else datetime.utcnow(),
        )


class JobStore:
    """将作业状态持久化到 JSON 文件的轻量工具。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.jobs: Dict[str, JobRecord] = {}
        self.latest_id: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("jobs"):
                entries = raw.get("jobs", [])
            else:
                entries = raw if isinstance(raw, list) else []
            for item in entries:
                rec = JobRecord.from_dict(item)
                self.jobs[rec.job_id] = rec
            if self.jobs:
                self.latest_id = sorted(self.jobs.keys(), key=lambda x: int(x))[-1]
        except Exception:
            # 损坏的文件不影响服务启动
            self.jobs = {}
            self.latest_id = None

    def _save(self) -> None:
        payload = {"jobs": [job.to_dict() for job in self.jobs.values()]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _next_id(self) -> str:
        if not self.jobs:
            return "1"
        try:
            max_id = max(int(k) for k in self.jobs.keys())
            return str(max_id + 1)
        except Exception:
            return str(len(self.jobs) + 1)

    def new_job(
        self,
        video_id: str,
        message: str = "正在排队",
        *,
        status: str = "pending",
        stage: str = "uploading",
        progress: float = 0.0,
        total_items: int = 1,
        current_item_name: Optional[str] = None,
    ) -> JobRecord:
        with self._lock:
            job_id = self._next_id()
            rec = JobRecord(
                job_id=job_id,
                video_id=video_id,
                message=message,
                status=status,
                stage=stage,
                progress=progress,
                total_items=max(1, int(total_items)),
                current_item_name=current_item_name,
            )
            self.jobs[job_id] = rec
            self.latest_id = job_id
            self._save()
            return rec

    def update_job(self, job_id: str, **kwargs: object) -> JobRecord:
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            for key, value in kwargs.items():
                if key == "progress" and isinstance(value, (int, float)):
                    value = max(0.0, min(float(value), 100.0))
                if key in {"total_items", "completed_items"}:
                    try:
                        value = int(value)
                    except Exception:
                        value = getattr(job, key)
                    if key == "total_items":
                        value = max(1, value)
                    else:
                        value = max(0, value)
                setattr(job, key, value)
            now = datetime.utcnow()
            if job.started_at is None and kwargs.get("status") in {"processing", "uploading"}:
                job.started_at = now
            if job.progress >= 100.0 or job.status in {"completed", "error"}:
                job.eta_seconds = None
            elif job.started_at and 1.0 <= job.progress < 100.0 and "eta_seconds" not in kwargs:
                elapsed = max((now - job.started_at).total_seconds(), 0.0)
                fraction = job.progress / 100.0
                job.eta_seconds = max(5.0, min(elapsed * (1.0 / fraction - 1.0), 3600.0))
            job.updated_at = datetime.utcnow()
            self.jobs[job_id] = job
            self.latest_id = job_id
            self._save()
            return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def last(self) -> Optional[JobRecord]:
        if self.latest_id is None:
            return None
        return self.jobs.get(self.latest_id)
