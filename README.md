# Anime Avatar Runtime

从完整二次元角色图出发，研究持续生成、状态保持与异常恢复。原始三条路线保留原文；本仓库现已具备可安装、可启动和可测试的控制工程。

## 已落地与边界

- Python 3.12 + uv、FastAPI、Pydantic、Uvicorn；依赖锁定在 uv.lock。
- 单角色初始化：校验参考图、保存规范化 PNG、写入带版本的 JSON checkpoint，重启后校验参考图哈希。
- 统一配置与共享类型：生成请求/结果、任务、事件和独立观测信号；后续三条路线复用同一份契约。
- 单进程 Runtime：异步提交单段生成、任务互斥、状态查询、暂停取消、关停清理、中断任务识别；API 与 CLI 共用真实生成执行函数。
- CLI 可桥接固定版本的 SkyReels-V2 官方入口，支持单图生成、视频续接和首尾帧条件；保存输入哈希、有效配置、命令、耗时、日志和成功/失败/取消状态，支持超时。
- pytest、Ruff、mypy、wheel/sdist 构建入口和 Windows/Linux CI 配置。

当前没有自主待机循环、已加载的 Observer、自动 Recovery、LLM/语音或展示前端。初始化后的 AvatarState 为 paused；API 会明确报告 autonomous_loop=false。模型推理与长期稳定性尚未验证。

后续开发从 [Runtime 底座与接入说明](./docs/runtime-foundation.md) 开始，其中列出了模块职责、共享契约、控制状态、各条路线的接入点和对应 Issue。[现有 Project](https://github.com/users/proffitteoy/projects/2) 保留完整路线图；底座落地不代表所有功能 Issue 已完成。

## 开源选型

| 能力 | 首版选择 | 接入状态 |
| --- | --- | --- |
| 图生视频与续接 | SkyReels-V2 DF 1.3B 540P | 官方 CLI 桥接已实现；目标 CUDA 环境待执行 |
| 身份观测 | CCIP，ccip-caformer-24-randaug-pruned，经 dghs-imgutils / ONNX | 已完成源码、模型与许可选型，待接入 |
| 几何观测 | anime-face-detector 0.1.0，28 点 landmarks | 已完成选型，待接入 |
| 运动观测 | GMFlow | 已完成选型，待接入 |
| 恢复对照 | 基础 Reset → DRBA → ToonCrafter / AniSora | 后续实现顺序，尚未进入闭环 |

[开源调研与首版选型](./docs/open-source-selection.md) 包含 10 个官方项目的对比、代码提交、模型快照、许可区别、资源要求和取舍。生成后端使用独立 Python 3.10 / PyTorch 2.5.1 环境；不与控制工程或其他模型混装。

## 安装与启动

需要 Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在仓库根目录执行：

```powershell
uv sync --locked
uv run --locked avatar-runtime doctor
uv run --locked avatar-runtime serve
```

打开 [接口文档](http://127.0.0.1:8000/docs)。服务默认仅监听 127.0.0.1，使用一个进程；当前面向本地研究，不提供多用户服务或远程鉴权。

可用环境变量见 [.env.example](./.env.example)。uv 不会隐式读取 .env，需显式加载：

```powershell
Copy-Item .env.example .env
uv run --locked --env-file .env avatar-runtime serve
```

本次工作区把 uv 缓存和托管 Python 放在忽略目录内。如需沿用：

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.tools/python'
uv sync --locked
```

## 初始化角色

启动 API 后，在另一终端提交一张真实参考图（替换文件路径）：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/avatar/init -ContentType image/png -InFile 'reference.png'
Invoke-RestMethod -Uri http://127.0.0.1:8000/avatar/state
Invoke-RestMethod -Uri http://127.0.0.1:8000/runtime/status
```

输入为原始图片字节，限制 10 MiB、1600 万像素和单帧图像。重复初始化返回 409，不覆盖角色。若要研究另一角色，使用不同的 AVATAR_DATA_DIR 并启动独立实例。checkpoint 与参考图默认保存在 var/avatar；初始化只建立参考状态，不声称已提取身份向量或生成视频。

## 通过 API 提交单段生成

配置好模型环境后，使用初始化的参考图提交任务：

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/runtime/resume
$job = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/avatar/generate -ContentType application/json -Body '{"frames":97,"seed":42}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/runtime/jobs/$($job.run_id)"
Invoke-RestMethod -Uri http://127.0.0.1:8000/runtime/events
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/runtime/pause
```

生成返回 202 和任务 ID；查询直至 completed/failed/cancelled。resume 仅开放手动提交，pause 取消并等待当前任务释放。重复提交返回 409，后端路径未配置返回 503；实际环境预检失败写入任务错误。参考 checkpoint 不会因生成实验被覆盖，实时模式以 runtime/status 为准。

GET /observer/status 返回独立的未知指标及原因；POST /recovery/trigger 返回 503。没有模型或观测能力时不会伪装成功。当前限单进程、单服务实例；独立 CLI 实验不经过 API 的互斥锁，不要同时争用同一 GPU。

## 模型准备与生成

完整步骤见 [模型环境部署](./docs/backend-setup.md)。该文档固定源码提交与权重 revision，并处理上游缺失的 MoviePy 依赖。另一台本地 NVIDIA 机器上的模型安装与运行不阻塞当前控制工程。

准备完成后，调用示例：

```bash
uv run --locked avatar-runtime generate --python vendor/SkyReels-V2/.venv/bin/python --image /absolute/path/reference.png --output var/runs/first
uv run --locked avatar-runtime generate --python vendor/SkyReels-V2/.venv/bin/python --video /absolute/path/previous.mp4 --frames 257 --output var/runs/extension
```

加 --dry-run 只查看官方参数。真正运行时缺少模型、环境或输出将明确失败；不会使用占位视频伪装推理成功。生成实验尚未自动回写控制 API 的状态。

## 开发与验证

```powershell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv build
```

格式化修改使用 uv run --locked ruff format src tests。当前使用本地 JSON 状态，无数据库迁移和种子数据命令；不提供尚不存在的生产部署命令。

源码在 src/anime_avatar_runtime：config.py 统一配置，contracts.py 定义共享类型，runtime.py 编排单任务与生命周期，api.py 处理 HTTP，state.py 负责参考状态持久化，skyreels.py 执行独立后端并记录实验，cli.py 提供命令入口。tests/ 覆盖输入与状态损坏、并发互斥、失败、取消、重启和产物保护。[check 工作流](./.github/workflows/check.yml) 在 push 和 pull_request 时执行 Windows/Linux 检查；对应提交的实际结果见 [GitHub Actions](https://github.com/proffitteoy/anime-avatar-runtime/actions/workflows/check.yml)。

本次本地验收：43 项测试通过，Ruff、mypy、wheel/sdist 构建及实际 HTTP 控制流程通过；细节与初始冷启动记录见 [执行记录](./docs/冷启动.md)。新增契约、Runtime/API 控制及真实轻量子进程测试均不运行 GPU 模型。官方参数对照测试使用已核对 blob SHA 的源码缓存；全新 checkout 没有上游源码时该项明确 skip。其余测试不需要模型或联网。

## 文档

- [项目协作规则](./AGENTS.md)
- [Runtime 底座与后续开发入口](./docs/runtime-foundation.md)
- [冷启动流程与逐项执行记录](./docs/冷启动.md)
- [文档索引](./docs/README.md)
- [01：生成与长期状态保持](./01-avatar-generation-and-state-preservation.md)
- [02：状态恢复与稳定控制](./02-state-recovery-and-stability.md)
- [03：运行系统架构](./03-runtime-system-architecture.md)
