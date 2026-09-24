"""Entry point: starts the local transcription server and opens the browser.

    python app/main.py              normal start
    python app/main.py --cpu        ignore the GPU
    python app/main.py --self-test  load the model, transcribe a test tone, exit (used by CI)
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Output redirected to a file or pipe on Windows defaults to the ANSI code page; keep Chinese printable.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from paths import MODELS_DIR, OUTPUT_DIR, UPLOAD_DIR  # noqa: E402

DEFAULT_PORT = 7860
HOST = "127.0.0.1"


def disable_quick_edit():
    """Clicking inside a Windows console with QuickEdit on pauses the program; turn it off."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ENABLE_QUICK_EDIT_MODE = 0x0040
            ENABLE_EXTENDED_FLAGS = 0x0080
            kernel32.SetConsoleMode(handle, (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS)
    except Exception:
        pass


def already_running(port):
    """True if another copy of this app is already serving on ``port``."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/info", timeout=2) as resp:
            return json.load(resp).get("app") == "local-transcriber"
    except Exception:
        return False


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def pick_port():
    for port in range(DEFAULT_PORT, DEFAULT_PORT + 50):
        if port_free(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def open_browser_when_up(url, port):
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((HOST, port)) == 0:
                webbrowser.open(url)
                return
        time.sleep(0.1)


def self_test(force_cpu):
    """Load the model and transcribe a short generated tone. Exit code 0 on success."""
    import wave

    import numpy as np

    from engine import Engine

    engine = Engine(force_cpu=force_cpu)
    engine.load()
    if engine.state != "ready":
        return 1
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "self-test.wav"
    t = np.linspace(0, 2, 32000, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(tone.tobytes())
    segments, lang, duration = engine.transcribe(path, vad=False)
    path.unlink(missing_ok=True)
    print(f"自检通过：device={engine.device} model={engine.model_name} "
          f"duration={duration:.1f}s segments={len(segments)} lang={lang}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="本地音频转文字")
    parser.add_argument("--cpu", action="store_true", help="只用 CPU")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    # Everything runs offline when the models are bundled.
    if MODELS_DIR.is_dir() and any(MODELS_DIR.glob("*/model.bin")):
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from engine import Engine, setup_cuda_dlls

    setup_cuda_dlls()

    if args.self_test:
        return self_test(args.cpu)

    disable_quick_edit()
    port = args.port or DEFAULT_PORT
    if already_running(port):
        print("程序已经在运行，正在打开浏览器页面…")
        webbrowser.open(f"http://{HOST}:{port}/")
        time.sleep(2)
        return 0
    if not port_free(port):
        port = pick_port()

    import uvicorn

    from jobs import JobManager
    from server import create_app

    engine = Engine(force_cpu=args.cpu)
    threading.Thread(target=engine.load, name="model-loader", daemon=True).start()
    manager = JobManager(engine, OUTPUT_DIR)
    manager.start()
    app = create_app(engine, manager, UPLOAD_DIR, OUTPUT_DIR)

    url = f"http://{HOST}:{port}/"
    print("=" * 56)
    print(" 本地音频转文字 已启动")
    print(f" 浏览器地址：{url}")
    print(f" 结果保存在：{OUTPUT_DIR}")
    print(" 使用期间请不要关闭本窗口；关闭窗口即退出程序。")
    print("=" * 56, flush=True)
    if not args.no_browser:
        threading.Thread(target=open_browser_when_up, args=(url, port), daemon=True).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
