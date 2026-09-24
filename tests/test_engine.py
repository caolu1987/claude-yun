import pytest

from engine import Engine, is_gpu_error


class BrokenModel:
    def __init__(self, exc):
        self.exc = exc

    def transcribe(self, *args, **kwargs):
        raise self.exc


def gpu_engine(exc):
    engine = Engine()
    engine.device = "cuda"
    engine._model = BrokenModel(exc)
    engine.fallbacks = []
    engine.fall_back_to_cpu = lambda reason: engine.fallbacks.append(reason) or setattr(
        engine, "_model", BrokenModel(RuntimeError("still broken"))
    )
    return engine


def test_bad_audio_file_does_not_switch_to_cpu():
    engine = gpu_engine(ValueError("[Errno 1094995529] Invalid data found when processing input"))
    with pytest.raises(ValueError):
        engine.transcribe("broken.mp3")
    assert engine.fallbacks == []


def test_gpu_error_switches_to_cpu_and_retries():
    engine = gpu_engine(RuntimeError("CUDA failed with error out of memory"))
    with pytest.raises(RuntimeError, match="still broken"):
        engine.transcribe("a.wav")
    assert len(engine.fallbacks) == 1


def test_is_gpu_error():
    assert is_gpu_error(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    assert is_gpu_error(RuntimeError("Could not locate cudnn_ops64_9.dll"))
    assert not is_gpu_error(ValueError("Invalid data found when processing input"))
