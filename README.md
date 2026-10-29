# Anime Avatar Runtime

## 停止维护与归档

本项目于 **2026-10-29** 停止维护，仓库作为历史代码、设计和研究记录归档，不再推进功能开发或接受新的 Issue / PR。

- 原路线图 [#45](https://github.com/proffitteoy/anime-avatar-runtime/issues/45) 及 #46–#66 共 22 个未完成事项按 `not_planned` 关闭；P1/P2/P3 里程碑与 [Project](https://github.com/users/proffitteoy/projects/2) 一并关闭。关闭表示停止推进，原验收条件仍保留。
- 旧工程处置 [#68](https://github.com/proffitteoy/anime-avatar-runtime/issues/68) 已完成关闭，对应 [PR #69](https://github.com/proffitteoy/anime-avatar-runtime/pull/69) 和 [PR #44](https://github.com/proffitteoy/anime-avatar-runtime/pull/44) 已合并。
- 保留源码、Git 历史、原始方案、实验结论及历史 Issue / PR；本次封仓仅调整文档和仓库管理状态，私有数据与模型保持原状。

以下命令和文档供历史复现与数据恢复使用，不代表持续维护承诺。资产编译、运动引擎、壁纸播放器和长期运行验收仍未交付。

## 历史目标与路线

从一张完整角色图出发，离线构建角色资产，再用轻量参数运动和渲染运行动态壁纸。原路线见 [#45](https://github.com/proffitteoy/anime-avatar-runtime/issues/45)。

- P1：自动资产编译，#49–#54。
- P2：长期角色运动，#55–#60。
- P3：轻量动态壁纸，#61–#66。

源码包含历史生成/观测控制工程。旧数据、checkpoint 兼容、旧模型隔离和历史 API 回归已由 [#68](https://github.com/proffitteoy/anime-avatar-runtime/issues/68) 收口；人工测试资产不代替自动编译的产品验收。

## 历史数据整理与恢复

仅在需要整理尚未迁移的旧数据时执行；已完成的迁移无需因封仓重新运行。

活动数据统一使用 `var/avatar`。旧实验、测试状态和缓存归档到 `.archive/`；仅保留显式活动 checkpoint 及其可信参考图，不自动导入其他测试角色。

停止 API、生成和观测进程后执行：

```powershell
uv sync --locked --no-dev
uv run --locked --no-dev avatar-runtime migrate-data --source var --archive .archive/legacy-data --no-backup
uv run --locked --no-dev avatar-runtime migrate-data --source var --archive .archive/legacy-data --no-backup --apply
uv run --locked --no-dev avatar-runtime verify-data --archive .archive/legacy-data
```

不加 `--apply` 只做盘点。`--no-backup` 不创建 ZIP 副本，核对全部哈希后同盘原子归档；归档保留原件。省略该选项才创建独立备份。切换失败会尽量把原目录放回；进程中断可使用相同归档续跑，目标冲突不覆盖。已完成的数据树不能再次迁移。归档和私有数据都不提交。

恢复到新的检查目录：

```powershell
uv run --locked --no-dev avatar-runtime restore-data --archive .archive/legacy-data --output .cache/legacy-recovered
```

恢复从归档原件或已有独立备份读取，不覆盖当前目录或自动启用第二套 Runtime。活动 checkpoint 不存在时保持未初始化。[完整流程与旧工程处置清单](./docs/data-migration.md)。

## 归档时保留的可执行能力

- Python 3.12 + uv、FastAPI、Pydantic 2；锁定依赖与 Windows/Linux CI。
- 可信参考图初始化、schema v1 状态校验与原子持久化。
- 离线数据盘点、可选备份、哈希校验、原子归档、失败回退与恢复。
- 历史研究工具：固定 SkyReels 生成桥接、单任务控制、独立 CPU 身份/几何观测及研究报告。

历史工具使用显式 `legacy` 入口；原 `serve/generate/observe` 命令仍为兼容别名。API 的路径、状态码及数据含义保留，OpenAPI 标为 deprecated。

```powershell
# 只读校验 checkpoint 和参考图，不需要旧模型配置。
uv run --locked --no-dev avatar-runtime legacy state --data-dir var/avatar
# 仅显式运行旧实验 API 时安装其轻量数值依赖。
uv sync --locked --no-dev --extra legacy
uv run --locked --no-dev --extra legacy avatar-runtime legacy serve
```

迁移和 checkpoint 查询不导入旧生成/观测模块。Torch、SkyReels、CCIP/人脸模型仍只在显式独立实验环境执行，不进入基本安装。用法与证据见 [控制工程](./docs/runtime-foundation.md)、[后端部署](./docs/backend-setup.md)、[观测研究](./docs/observers.md)。生成结果不会自动成为可信资产；CPU 测试和历史 GPU 基线不代表新产品或长期运行已验收。

## 历史开发与复现检查

```powershell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv build
```

格式化使用 `uv run --locked ruff format src tests`。常规测试不加载模型；CUDA、真实媒体、观测模型和实机长跑分别验证。当前没有数据库或生产部署入口。

协作规则见 [AGENTS.md](./AGENTS.md)，历史执行证据和原始三份方案见 [文档索引](./docs/README.md)。三份原始方案保留原文；其中的后续工作及历史 Issue 均不再构成活动开发计划。
