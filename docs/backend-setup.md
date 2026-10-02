# SkyReels-V2 独立模型环境

本文补齐控制工程的部署入口。目标基线为 Linux + NVIDIA CUDA；Windows 可开发控制程序，但本仓库没有验证 Windows 原生 FlashAttention/GPU 推理。以下安装及下载命令供目标机器执行，本次未下载权重或执行 GPU 推理。

## 固定来源与环境边界

| 项目 | 固定值 |
| --- | --- |
| 源码 | [SkyworkAI/SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2/tree/9351d13152207cc04de780e055346b08ade0b851)，commit `9351d13152207cc04de780e055346b08ade0b851` |
| 官方入口 | `generate_video_df.py`，已核对 Git blob `990ef610af1798b09aaba53d96201bc716f9d731` |
| 模型 | [Skywork/SkyReels-V2-DF-1.3B-540P](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P/tree/1100111771ba2d921f10e76991f54db9f73edb3d) |
| 权重 revision | `1100111771ba2d921f10e76991f54db9f73edb3d` |
| 模型环境 | Python 3.10，上游测试 3.10.12；torch 2.5.1、torchvision 0.20.1、NumPy < 2 |
| 控制环境 | Python 3.12，由本仓库 uv.lock 管理，与模型环境分开 |

源码使用 SkyReels 社区许可，模型卡为 `license=other`；不可套用其他组件的 MIT/Apache 许可。参见 [开源选型](./open-source-selection.md)。上游约 14.7 GB 显存数据是参考值，不是本项目实测，也不保证所有输入和配置都可运行。

## 在目标机器准备

以下 Bash 命令均从本仓库根目录执行。先确认 NVIDIA 驱动、匹配 Torch CUDA 构建的工具链和编译器；FlashAttention 的安装需要适配该机器，不能仅凭 pip 安装成功判断 GPU 可用。

```bash
uv sync --locked
git clone https://github.com/SkyworkAI/SkyReels-V2.git vendor/SkyReels-V2
git -C vendor/SkyReels-V2 checkout --detach 9351d13152207cc04de780e055346b08ade0b851
uv venv --python 3.10 vendor/SkyReels-V2/.venv
vendor/SkyReels-V2/.venv/bin/python -m ensurepip --upgrade
vendor/SkyReels-V2/.venv/bin/python -m pip install --upgrade pip setuptools wheel packaging ninja
vendor/SkyReels-V2/.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1
vendor/SkyReels-V2/.venv/bin/python -m pip install -r vendor/SkyReels-V2/requirements.txt
vendor/SkyReels-V2/.venv/bin/python -m pip install 'moviepy==1.0.3'
```

2026-10-02 重新只读核对固定提交：官方 requirements 包含 FlashAttention，但漏列入口无条件导入的 `moviepy.editor`。MoviePy 2 的导入结构不同，需补装 1.0.3。若 FlashAttention 构建隔离看不到已安装的 Torch，按其官方安装说明在该独立环境执行 `python -m pip install flash-attn --no-build-isolation` 后重试依赖安装；记录实际解析版本及失败日志。不要修改控制环境的 uv.lock 来容纳模型依赖。

上游 requirements 仍含未完全固定的间接依赖，这份指南不是经过 GPU 实测的完整模型锁文件。目标机器安装成功后应保存完整 freeze 和硬件信息，而不是假设每次解析等价。

## 固定下载模型

使用模型环境的 huggingface_hub，将固定快照下载到本地。不要把缓存的单个入口源码当作完整 vendor checkout。

```bash
vendor/SkyReels-V2/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Skywork/SkyReels-V2-DF-1.3B-540P",
    revision="1100111771ba2d921f10e76991f54db9f73edb3d",
    local_dir="models/skyreels-v2-df-1.3b-540p",
)
PY
mkdir -p var/environment
nvidia-smi > var/environment/nvidia-smi.txt
vendor/SkyReels-V2/.venv/bin/python -m pip freeze > var/environment/skyreels-requirements.txt
vendor/SkyReels-V2/.venv/bin/python -c 'import torch; from moviepy.editor import VideoFileClip; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

已核对上游 `download_model`：传入已存在的本地模型目录时不调用 snapshot_download。桥接始终传本地目录，但目前只验证目录存在，没有逐文件校验权重快照；run.json 中 `model_revision_expected` 表示期望 revision，`model_revision_verified=false` 如实保留。完整性仍须在目标机器核查。

## 配置和运行

```bash
export SKYREELS_REPO="$PWD/vendor/SkyReels-V2"
export SKYREELS_PYTHON="$PWD/vendor/SkyReels-V2/.venv/bin/python"
export SKYREELS_MODEL="$PWD/models/skyreels-v2-df-1.3b-540p"
export AVATAR_GENERATION_TIMEOUT_SECONDS=1800
uv run --locked avatar-runtime doctor
uv run --locked avatar-runtime generate --image /absolute/path/reference.png --output var/runs/dry --dry-run
uv run --locked avatar-runtime generate --image /absolute/path/reference.png --output var/runs/first
uv run --locked avatar-runtime generate --video /absolute/path/previous.mp4 --frames 257 --output var/runs/extension
uv run --locked avatar-runtime generate --image /absolute/path/reference.png --end-image /absolute/path/end.png --output var/runs/end-frame
```

每次使用新的 output 目录；失败实验也不会覆盖。默认固定 540P、97 帧基础窗口、17 帧重叠、24 FPS、30 步，具体有效参数记录于 argv。总预检/生成超时可通过环境变量或 CLI `--timeout` 调整；dry-run 只检查参数与输入路径，不启动模型、不写输出目录。

也可 `uv run --locked avatar-runtime serve`，通过 [README](../README.md) 的 API 流程初始化、resume、提交并轮询单段任务。API 使用初始化的可信原图；当前不支持通过 HTTP 自动续接或恢复。

## 验收与故障定位

- 先检查 run.json 的 status、error、returncode 和 backend.log；预检核对固定 Git HEAD、CUDA 可用性及 MoviePy 导入。HEAD 匹配不等于工作树无修改，实验前还应核查 vendor 的 `git status --short`。
- 记录真实输入哈希、种子、全部参数、源码/权重来源、GPU 型号/驱动、峰值显存、墙钟耗时和输出时长。run.json 自动记录其中的控制信息，未采集的硬件信息为 null，需要实验补充。
- 成功退出且存在非空 MP4 只通过产物检查；须实际播放/解码并评估身份、几何、运动、首尾连续性，保存失败样例。
- 首段、续接、首尾帧分别验证。模型缺失、超时、取消、OOM 和无输出应保留失败记录，不能记成完成。
- 单段成功不代表长期稳定、实时生成或自主恢复。完整闭环另按 #30 验收；模型优先级与显存调度由 #26 推进。

`vendor/`、`models/`、`var/` 均被忽略，不提交权重、私人原图或大型视频。
