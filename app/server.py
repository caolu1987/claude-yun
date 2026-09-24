"""Local web API and page. Bound to 127.0.0.1 only."""

import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from engine import LANGUAGES
from jobs import Job
from paths import STATIC_DIR

APP_ID = "local-transcriber"
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_filename(name: str) -> str:
    name = _UNSAFE_CHARS.sub("_", Path(name or "").name).strip(" .")
    return name or "audio"


def open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import os

        os.startfile(path)  # noqa: S606 - local folder chosen by the app
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def create_app(engine, manager, upload_dir: Path, output_dir: Path) -> FastAPI:
    app = FastAPI(title="本地音频转文字", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/info")
    def info():
        return {
            "app": APP_ID,
            "engine": engine.info(),
            "languages": [{"code": code, "name": name} for code, name in LANGUAGES],
            "output_dir": str(output_dir),
        }

    @app.get("/api/jobs")
    def list_jobs():
        return manager.list()

    @app.post("/api/jobs")
    def create_jobs(
        files: list[UploadFile] = File(...),
        language: str = Form("auto"),
        vad: bool = Form(True),
    ):
        if language not in dict(LANGUAGES):
            raise HTTPException(400, "不支持的语言")
        upload_dir.mkdir(parents=True, exist_ok=True)
        created = []
        for upload in files:
            filename = safe_filename(upload.filename)
            dest = upload_dir / f"{uuid.uuid4().hex}_{filename}"
            with dest.open("wb") as out:
                shutil.copyfileobj(upload.file, out, length=1024 * 1024)
            created.append(manager.submit(Job(filename=filename, src_path=dest, language=language, vad=vad)))
        return [job.to_dict() for job in created]

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int):
        if not manager.cancel(job_id):
            raise HTTPException(404, "任务不存在")
        return {"ok": True}

    @app.post("/api/jobs/clear")
    def clear_jobs():
        manager.clear_finished()
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/download/{fmt}")
    def download(job_id: int, fmt: str):
        job = manager.get(job_id)
        path = job.outputs.get(fmt) if job else None
        if not path or not path.is_file():
            raise HTTPException(404, "文件不存在")
        media = "text/plain; charset=utf-8" if fmt == "txt" else "application/x-subrip"
        return FileResponse(path, media_type=media, filename=path.name)

    @app.post("/api/open-output")
    def open_output():
        open_folder(output_dir)
        return {"ok": True}

    return app
