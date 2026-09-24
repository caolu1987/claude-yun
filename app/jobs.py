"""Job queue: uploaded files are transcribed one at a time by a background worker."""

import itertools
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from engine import Cancelled
from writers import write_outputs

_ids = itertools.count(1)


@dataclass
class Job:
    filename: str
    src_path: Path
    language: str  # "auto" or a Whisper language code
    vad: bool
    id: int = field(default_factory=lambda: next(_ids))
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = "排队中"
    outputs: dict = field(default_factory=dict)
    detected_language: str | None = None
    duration: float | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "language": self.language,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "outputs": {fmt: path.name for fmt, path in self.outputs.items()},
            "detected_language": self.detected_language,
            "duration": self.duration,
            "elapsed": (self.finished or time.time()) - self.started if self.started else None,
        }


class JobManager:
    def __init__(self, engine, output_dir: Path):
        self.engine = engine
        self.output_dir = output_dir
        self.jobs: dict[int, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="transcribe-worker", daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, job: Job):
        with self._lock:
            self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def list(self):
        with self._lock:
            return [job.to_dict() for job in sorted(self.jobs.values(), key=lambda j: j.id)]

    def get(self, job_id):
        return self.jobs.get(job_id)

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job.cancel_requested = True
        if job.status == "queued":
            self._finish(job, "cancelled", "已取消")
        return True

    def clear_finished(self):
        with self._lock:
            for job_id in [j.id for j in self.jobs.values() if j.status in ("done", "error", "cancelled")]:
                del self.jobs[job_id]

    def _finish(self, job, status, message):
        job.status = status
        job.message = message
        job.finished = time.time()
        job.src_path.unlink(missing_ok=True)

    def _worker(self):
        while True:
            job = self._queue.get()
            if job.status != "queued":  # cancelled while waiting
                continue
            job.message = "等待模型加载…"
            self.engine.wait_ready()
            if self.engine.state != "ready":
                self._finish(job, "error", self.engine.message)
                continue
            self._run(job)

    def _run(self, job):
        job.status = "running"
        job.message = "正在识别…"
        job.started = time.time()

        def on_progress(value):
            job.progress = value

        try:
            segments, lang, duration = self.engine.transcribe(
                job.src_path,
                language=None if job.language == "auto" else job.language,
                vad=job.vad,
                on_progress=on_progress,
                should_cancel=lambda: job.cancel_requested,
            )
            job.detected_language = lang
            job.duration = duration
            job.outputs = write_outputs(self.output_dir, Path(job.filename).stem or "audio", segments)
            job.progress = 1.0
            message = "完成" if segments else "完成（未识别到语音）"
            self._finish(job, "done", message)
        except Cancelled:
            self._finish(job, "cancelled", "已取消")
        except Exception as exc:  # noqa: BLE001 - shown to the user
            if "Invalid data found" in str(exc) or type(exc).__module__.startswith("av."):
                self._finish(job, "error", "失败：无法读取这个文件，可能不是音频/视频文件，或文件已损坏")
            else:
                self._finish(job, "error", f"失败：{exc}")
