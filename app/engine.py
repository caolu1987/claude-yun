"""Speech recognition engine: picks GPU or CPU, loads the model, transcribes files."""

import os
import shutil
import subprocess
import sys
import threading

from paths import CUDA_DIR, MODELS_DIR

# Environment overrides are for development only; the package ships these two models.
MAIN_MODEL = os.environ.get("LT_MAIN_MODEL", "large-v3-turbo")  # used on NVIDIA GPUs
CPU_MODEL = os.environ.get("LT_CPU_MODEL", "small")  # used when no usable GPU is found

# Below this much video memory, run the model with int8 weights instead of float16.
LOW_VRAM_MB = 4096

LANGUAGES = [
    ("auto", "自动识别"),
    ("en", "英语"),
    ("zh", "中文"),
    ("ja", "日语"),
    ("ko", "韩语"),
    ("fr", "法语"),
    ("de", "德语"),
    ("es", "西班牙语"),
    ("pt", "葡萄牙语"),
    ("it", "意大利语"),
    ("ru", "俄语"),
    ("ar", "阿拉伯语"),
    ("hi", "印地语"),
    ("th", "泰语"),
    ("vi", "越南语"),
    ("id", "印尼语"),
    ("ms", "马来语"),
    ("tr", "土耳其语"),
    ("nl", "荷兰语"),
    ("pl", "波兰语"),
    ("uk", "乌克兰语"),
]
LANGUAGE_NAMES = dict(LANGUAGES)

# Nudges Whisper towards simplified Chinese with punctuation when Chinese is chosen explicitly.
ZH_PROMPT = "以下是普通话的句子，使用简体中文。"


class Cancelled(Exception):
    pass


_GPU_ERROR_MARKERS = ("cuda", "cublas", "cudnn", "out of memory", "gpu", "device")


def is_gpu_error(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _GPU_ERROR_MARKERS)


def setup_cuda_dlls():
    """Make the bundled cuBLAS/cuDNN DLLs findable. Must run before ctranslate2 is imported."""
    if sys.platform != "win32" or not CUDA_DIR.is_dir():
        return
    os.add_dll_directory(str(CUDA_DIR))
    # cuDNN loads its sub-libraries with plain LoadLibrary, which searches PATH.
    os.environ["PATH"] = str(CUDA_DIR) + os.pathsep + os.environ.get("PATH", "")


def query_gpu():
    """Return (name, total_memory_mb) of the first NVIDIA GPU via nvidia-smi, or (None, None)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=flags,
        ).stdout
        name, mem = out.strip().splitlines()[0].rsplit(",", 1)
        return name.strip(), int(float(mem))
    except Exception:
        return None, None


def model_source(name):
    """Bundled model folder if present, otherwise the model name (downloaded on first use)."""
    local = MODELS_DIR / name
    if (local / "model.bin").is_file():
        return str(local)
    return name


class Engine:
    def __init__(self, main_model=MAIN_MODEL, cpu_model=CPU_MODEL, force_cpu=False):
        self.main_model = main_model
        self.cpu_model = cpu_model
        self.force_cpu = force_cpu
        self.state = "loading"  # loading | ready | error
        self.message = "正在加载模型…"
        self.device = None
        self.compute_type = None
        self.model_name = None
        self.gpu_name = None
        self.fallback_reason = None
        self._model = None
        self._ready = threading.Event()

    # ---------- loading ----------

    def info(self):
        return {
            "state": self.state,
            "message": self.message,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "model": self.model_name,
            "compute_type": self.compute_type,
            "fallback_reason": self.fallback_reason,
        }

    def wait_ready(self, timeout=None):
        return self._ready.wait(timeout)

    def load(self):
        try:
            self._load()
            self.state = "ready"
            where = f"GPU（{self.gpu_name}）" if self.device == "cuda" else "CPU"
            self.message = f"就绪：{where}，模型 {self.model_name}"
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.state = "error"
            self.message = f"模型加载失败：{exc}"
        finally:
            self._ready.set()
        print(self.message, flush=True)

    def _load(self):
        import ctranslate2

        if not self.force_cpu and ctranslate2.get_cuda_device_count() > 0:
            name, vram = query_gpu()
            self.gpu_name = name or "NVIDIA GPU"
            supported = ctranslate2.get_supported_compute_types("cuda")
            low_vram = vram is not None and vram < LOW_VRAM_MB
            if "float16" in supported and not low_vram:
                compute = "float16"
            elif "int8_float16" in supported:
                compute = "int8_float16"
            else:
                compute = "int8"
            try:
                self._load_model(self.main_model, "cuda", compute)
                self._warm_up()
                return
            except Exception as exc:  # noqa: BLE001
                self._model = None
                self.fallback_reason = f"GPU 不可用，已改用 CPU：{exc}"
                print(self.fallback_reason, flush=True)
        self._load_cpu()

    def _load_cpu(self):
        # Prefer the small model on CPU; use the main one if small is not bundled.
        name = self.cpu_model
        if model_source(name) == name and model_source(self.main_model) != self.main_model:
            name = self.main_model
        self._load_model(name, "cpu", "int8")

    def _load_model(self, name, device, compute_type):
        from faster_whisper import WhisperModel

        self.message = f"正在加载模型 {name}（{'GPU' if device == 'cuda' else 'CPU'}）…"
        print(self.message, flush=True)
        self._model = WhisperModel(
            model_source(name),
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() or 4,
        )
        self.model_name = name
        self.device = device
        self.compute_type = compute_type

    def _warm_up(self):
        """Run one second of silence through the model so missing CUDA DLLs or OOM show up now."""
        import numpy as np

        segments, _ = self._model.transcribe(
            np.zeros(16000, dtype=np.float32), language="en", vad_filter=False, beam_size=1
        )
        list(segments)

    def fall_back_to_cpu(self, reason):
        self.fallback_reason = f"GPU 出错，已改用 CPU：{reason}"
        print(self.fallback_reason, flush=True)
        self._model = None
        self._load_cpu()
        self.message = f"就绪：CPU，模型 {self.model_name}（GPU 出错后自动切换）"

    # ---------- transcription ----------

    def transcribe(self, path, language=None, vad=True, on_progress=None, should_cancel=None):
        """Transcribe ``path``; returns (segments, detected_language, duration_seconds)."""
        try:
            return self._transcribe(path, language, vad, on_progress, should_cancel)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            # Only GPU failures justify switching to CPU; a broken audio file must not.
            if self.device != "cuda" or not is_gpu_error(exc):
                raise
            self.fall_back_to_cpu(exc)
            return self._transcribe(path, language, vad, on_progress, should_cancel)

    def _transcribe(self, path, language, vad, on_progress, should_cancel):
        segments_iter, info = self._model.transcribe(
            str(path),
            language=language,
            vad_filter=vad,
            beam_size=5,
            initial_prompt=ZH_PROMPT if language == "zh" else None,
        )
        duration = info.duration or 0.0
        segments = []
        for seg in segments_iter:
            if should_cancel and should_cancel():
                raise Cancelled()
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
            if on_progress and duration > 0:
                on_progress(min(seg.end / duration, 1.0))
        return segments, info.language, duration
