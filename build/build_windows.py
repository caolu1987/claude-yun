"""Build the portable Windows (NVIDIA GPU) package.

    python build/build_windows.py [--out dist] [--skip-models] [--zip]

Produces dist/LocalTranscriber/ with an embedded Python, all Python packages,
the cuBLAS/cuDNN DLLs and the models, so the target PC needs only an NVIDIA driver.

Runs best on Windows (it also copies the MSVC runtime DLLs from System32).
On other systems it still assembles everything else, which is useful for checking the script.
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "LocalTranscriber"

PYTHON_VERSION = "3.11.9"  # last 3.11 release with an official embeddable zip
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"

# Must match the CUDA/cuDNN versions ctranslate2 was built with (see requirements-win.txt).
CUDA_WHEELS = ["nvidia-cublas-cu12==12.8.4.1", "nvidia-cudnn-cu12==9.10.2.21"]
CUDA_DLL_SKIP = {"nvblas64_12.dll"}

MODELS = {
    "large-v3-turbo": "dropbox-dash/faster-whisper-large-v3-turbo",
    "small": "Systran/faster-whisper-small",
}

# Needed by ctranslate2/onnxruntime and not shipped with the embeddable Python.
MSVC_DLLS = [
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll", "vcomp140.dll",
]

PIP_WIN = ["--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.11", "--implementation", "cp"]


def log(msg):
    print(f"==> {msg}", flush=True)


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args, "--disable-pip-version-check"], check=True)


def download(url, dest: Path):
    log(f"下载 {url}")
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def build_python(pkg: Path, cache: Path):
    py_dir = pkg / "python"
    archive = cache / PYTHON_URL.rsplit("/", 1)[1]
    if not archive.exists():
        download(PYTHON_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(py_dir)
    # The ._pth file fixes sys.path for the embedded runtime.
    pth = next(py_dir.glob("python3*._pth"))
    stdlib_zip = pth.name.replace("._pth", ".zip")
    pth.write_text(f"{stdlib_zip}\n.\nLib\\site-packages\n..\\app\nimport site\n", encoding="ascii")

    log("安装 Python 依赖")
    site = py_dir / "Lib" / "site-packages"
    pip("install", *PIP_WIN, "--target", str(site), "-r", str(REPO / "build" / "requirements-win.txt"))
    for cache_dir in site.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    if sys.platform == "win32":
        system32 = Path(r"C:\Windows\System32")
        for name in MSVC_DLLS:
            src = system32 / name
            if src.exists():
                shutil.copy2(src, py_dir / name)
            else:
                log(f"警告：找不到 {src}")
    else:
        log("警告：不在 Windows 上构建，没有附带 MSVC 运行库（目标电脑需要安装 VC++ 运行库）")


def build_cuda(pkg: Path, cache: Path):
    cuda_dir = pkg / "runtime" / "cuda"
    cuda_dir.mkdir(parents=True, exist_ok=True)
    wheel_dir = cache / "cuda-wheels"
    wheel_dir.mkdir(exist_ok=True)
    log("下载 CUDA 运行库（cuBLAS、cuDNN）")
    pip("download", *PIP_WIN, "--no-deps", "-d", str(wheel_dir), *CUDA_WHEELS)
    for wheel in wheel_dir.glob("*.whl"):
        with zipfile.ZipFile(wheel) as zf:
            for member in zf.namelist():
                name = member.rsplit("/", 1)[-1]
                if member.endswith(".dll") and "/bin/" in member and name not in CUDA_DLL_SKIP:
                    with zf.open(member) as src, (cuda_dir / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    log("CUDA DLL：" + ", ".join(sorted(p.name for p in cuda_dir.glob("*.dll"))))


def build_models(pkg: Path):
    from huggingface_hub import snapshot_download

    for name, repo_id in MODELS.items():
        log(f"下载模型 {name}（{repo_id}）")
        snapshot_download(
            repo_id,
            local_dir=pkg / "models" / name,
            allow_patterns=["*.bin", "*.json", "*.txt"],
        )
        shutil.rmtree(pkg / "models" / name / ".cache", ignore_errors=True)


def copy_app(pkg: Path):
    shutil.copytree(REPO / "app", pkg / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for item in (REPO / "build" / "package").iterdir():
        shutil.copy2(item, pkg / item.name)
    (pkg / "output").mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "dist"))
    parser.add_argument("--skip-models", action="store_true", help="不下载模型（仅用于检查构建脚本）")
    parser.add_argument("--zip", action="store_true", help="额外生成 zip 压缩包")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    pkg = out / PACKAGE_NAME
    cache = out / "_cache"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)

    copy_app(pkg)
    build_python(pkg, cache)
    build_cuda(pkg, cache)
    if not args.skip_models:
        build_models(pkg)

    size = sum(p.stat().st_size for p in pkg.rglob("*") if p.is_file())
    log(f"完成：{pkg}（{size / 1024 ** 3:.2f} GB）")
    if args.zip:
        archive = shutil.make_archive(str(out / PACKAGE_NAME), "zip", out, PACKAGE_NAME)
        log(f"压缩包：{archive}")


if __name__ == "__main__":
    main()
