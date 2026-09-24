import threading
import time

from fastapi.testclient import TestClient

from engine import Cancelled
from jobs import JobManager
from server import create_app, safe_filename


class FakeEngine:
    state = "ready"
    message = "就绪"

    def __init__(self):
        self.release = threading.Event()
        self.release.set()

    def info(self):
        return {"state": self.state, "message": self.message}

    def wait_ready(self, timeout=None):
        return True

    def transcribe(self, path, language=None, vad=True, on_progress=None, should_cancel=None):
        assert path.read_bytes() == b"fake audio"
        while not self.release.wait(0.01):
            if should_cancel():
                raise Cancelled()
        on_progress(1.0)
        return [{"start": 0.0, "end": 1.0, "text": "hello"}], language or "en", 1.0


def make_client(tmp_path, engine):
    manager = JobManager(engine, tmp_path / "output")
    manager.start()
    app = create_app(engine, manager, tmp_path / "uploads", tmp_path / "output")
    return TestClient(app)


def wait_for(client, job_id, status):
    for _ in range(200):
        job = next(j for j in client.get("/api/jobs").json() if j["id"] == job_id)
        if job["status"] == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status}: {job}")


def test_upload_transcribe_download(tmp_path):
    client = make_client(tmp_path, FakeEngine())
    assert client.get("/api/info").json()["app"] == "local-transcriber"
    resp = client.post(
        "/api/jobs",
        files=[("files", ("会议 录音.mp3", b"fake audio", "audio/mpeg"))],
        data={"language": "en", "vad": "true"},
    )
    assert resp.status_code == 200
    job = wait_for(client, resp.json()[0]["id"], "done")
    assert job["outputs"] == {"txt": "会议 录音.txt", "srt": "会议 录音.srt"}
    assert client.get(f"/api/jobs/{job['id']}/download/txt").content == "﻿hello\n".encode()
    assert client.get(f"/api/jobs/{job['id']}/download/srt").text.startswith("1\n00:00:00,000")
    assert client.get(f"/api/jobs/{job['id']}/download/exe").status_code == 404
    assert list((tmp_path / "uploads").iterdir()) == []  # upload removed after processing


def test_rejects_unknown_language(tmp_path):
    client = make_client(tmp_path, FakeEngine())
    resp = client.post("/api/jobs", files=[("files", ("a.wav", b"x"))], data={"language": "xx"})
    assert resp.status_code == 400


def test_cancel_running_job(tmp_path):
    engine = FakeEngine()
    engine.release.clear()
    client = make_client(tmp_path, engine)
    job_id = client.post("/api/jobs", files=[("files", ("a.wav", b"fake audio"))]).json()[0]["id"]
    wait_for(client, job_id, "running")
    client.post(f"/api/jobs/{job_id}/cancel")
    wait_for(client, job_id, "cancelled")


def test_safe_filename():
    assert safe_filename("../../evil.wav") == "evil.wav"
    assert safe_filename("a:b*c?.mp3") == "a_b_c_.mp3"
    assert safe_filename("") == "audio"
