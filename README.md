# 本地音频转文字（Windows + NVIDIA 显卡离线包）

团队内部使用的离线语音转文字工具。解压后双击 `Start.bat` 就能用，浏览器里操作，结果导出为 **TXT** 和 **SRT**。

- 识别引擎：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（CTranslate2）
- 模型：显卡用 `large-v3-turbo`（英文和多语种都比较准），没有可用显卡时自动改用 CPU 和 `small`
- 完全离线：模型、Python 和 CUDA 运行库都打包在内，目标电脑**只需要装 NVIDIA 显卡驱动**

## 获取离线包

每次推送代码后，GitHub Actions 会在 Windows 构建机上自动打包：

1. 打开仓库的 **Actions** → **Build Windows package** → 选择最新一次成功的运行
2. 在页面底部 **Artifacts** 下载 `LocalTranscriber-win-x64-nvidia`（约 3.5 GB）
3. 解压到任意位置（路径里可以有中文和空格），把整个文件夹通过共享盘或 U 盘分发给同事

也可以在 Actions 页面点 **Run workflow** 手动触发打包。压缩包保留 30 天。

使用说明见离线包里的 `README.txt`（源文件：[`build/package/README.txt`](build/package/README.txt)）。

## 离线包结构

```
LocalTranscriber/
├─ Start.bat          双击启动，自动打开浏览器
├─ Start-CPU.bat      只用 CPU 启动（排查问题用）
├─ README.txt         给同事的使用说明
├─ app/               程序代码（本仓库的 app/ 目录）
├─ python/            Python 3.11 嵌入式运行环境和所有依赖
├─ runtime/cuda/      cuBLAS 12.8 和 cuDNN 9.10 的 DLL
├─ models/            large-v3-turbo、small
└─ output/            转写结果
```

## 显卡和 CPU 的选择逻辑

| 情况 | 行为 |
|---|---|
| 有 NVIDIA 显卡，显存 ≥ 4GB | GPU 运行 large-v3-turbo，float16 |
| 有 NVIDIA 显卡，显存 < 4GB 或不支持 float16 | GPU 运行 large-v3-turbo，int8 省显存模式 |
| 启动时 GPU 测试失败（驱动太旧、缺 DLL 等） | 自动改用 CPU 运行 small，页面上显示原因 |
| 转写过程中 GPU 报错（例如显存不足） | 切换到 CPU，并用 CPU 重新处理当前文件 |
| 没有 NVIDIA 显卡 | CPU 运行 small |

ctranslate2 4.8.2 的 Windows 版本是用 CUDA 12.8 和 cuDNN 9.10 编译的，所以驱动需要支持 CUDA 12（大约 2023 年以后的驱动）。

## 开发

```bash
pip install -r build/requirements-win.txt pytest httpx
python -m pytest -q tests

# 本地运行（Linux / macOS 也可以，模型首次使用时自动下载）
LT_MAIN_MODEL=tiny LT_CPU_MODEL=tiny python app/main.py
```

在 Windows 上手动打包：

```powershell
pip install huggingface_hub
python build/build_windows.py --out dist --zip
```

`build_windows.py` 在非 Windows 系统上也能运行（用来检查脚本），但不会附带 MSVC 运行库，所以正式的包要在 Windows 上打（CI 就是这样做的）。

| 文件 | 作用 |
|---|---|
| `app/main.py` | 程序入口：启动本地服务、打开浏览器、`--self-test` 自检 |
| `app/engine.py` | 选择 GPU/CPU、加载模型、转写、出错时切换到 CPU |
| `app/jobs.py` | 任务队列（逐个处理、进度、取消） |
| `app/server.py` | 本地网页接口，只监听 127.0.0.1 |
| `app/writers.py` | 生成 TXT / SRT |
| `app/static/index.html` | 中文网页界面 |
| `build/build_windows.py` | 组装离线包 |
| `.github/workflows/build-windows.yml` | 测试和 Windows 自动打包 |
