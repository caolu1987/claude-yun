"""Locations inside the portable package.

Layout (package root = parent of this ``app`` folder):

    python/          embedded Python runtime
    runtime/cuda/    cuBLAS / cuDNN DLLs
    models/<name>/   faster-whisper (CTranslate2) models
    app/             this code
    output/          transcripts written here
    tmp/             uploaded files while they wait to be processed
"""

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
MODELS_DIR = ROOT / "models"
CUDA_DIR = ROOT / "runtime" / "cuda"
OUTPUT_DIR = ROOT / "output"
UPLOAD_DIR = ROOT / "tmp" / "uploads"
