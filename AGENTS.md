# Anime Avatar Runtime 协作规则

## 工作方式

- 默认使用中文。优先遵守用户当前要求、就近 AGENTS.md、项目文档与实际代码；通用模板和 skills 按需参考。
- 修改前阅读本文件、[README](./README.md)、[冷启动执行记录](./docs/冷启动.md) 和对应设计方案。
- 优先做最小可验证修改，复用现有文件、函数与类型，不提前架构化，不覆盖用户已有改动。
- 三份根目录编号方案保留原文；它们是目标方案，当前可执行接口以源码和 README 为准。涉及设计取舍时同步说明差异。

## 当前技术栈与边界

- 控制程序为 Python 3.12 单进程本地 API/CLI，uv 管理 pyproject.toml 与 uv.lock；使用 FastAPI、Pydantic 2、Uvicorn 和 Pillow。
- 首个生成后端是 SkyReels-V2 DF 1.3B 540P。独立 Python 3.10 / Torch 2.5.1 / CUDA 环境，经 subprocess 参数列表调用；禁止 shell=True 或拼接 shell 命令。
- [开源选型](./docs/open-source-selection.md) 记录模型、源码提交与许可。代码开源不等于所有权重具有相同许可，升级必须重新核对接口和依赖。
- 当前只有参考图初始化、checkpoint、控制查询和官方生成 CLI 桥接；尚无自主生成/观测/恢复闭环。不得将 CPU 测试或 dry-run 报告成 GPU 推理通过。
- 控制程序可在 Windows 开发；模型推理部署基线见 [backend-setup](./docs/backend-setup.md)。不因未取得目标硬件细节而阻塞仓库初始化。
- V0 使用 FastAPI 自带接口文档作为操作入口；尚无展示前端、数据库、迁移或生产部署，不为它们创建空模块。

## 文件职责

| 路径 | 职责 |
| --- | --- |
| src/anime_avatar_runtime/api.py | HTTP 输入、错误映射和本地控制查询 |
| src/anime_avatar_runtime/state.py | AvatarState、图片校验、原子持久化与哈希校验 |
| src/anime_avatar_runtime/skyreels.py | 固定上游版本、生成/续接/首尾帧参数、模型环境检查 |
| src/anime_avatar_runtime/cli.py | 命令入口、服务启动、实验日志与退出状态 |
| tests/ | 控制行为、失败路径、上游参数契约；模型不作为常规测试依赖 |
| docs/ | 原始流程、开源选型、模型部署和初始化证据 |
| var/、models/、vendor/ | 被忽略的私有运行数据、权重、外部代码，不提交 |
| .cache/、.tools/、.venv/ | 被忽略的本地缓存、解释器和环境，不提交 |

## 领域约束

- A 路线负责生成与正常状态保持，B 路线负责异常恢复，C 路线负责组织、调度和存储。
- AvatarState 不等于当前帧。当前 schema_version=1 仅表达已知参考图和暂停状态；后续增加 pose、embedding、generation_context 时不得虚构已计算结果。
- Observer 提供信号，由 Runtime 决定后续动作；身份、几何、运动指标先独立记录，交互和待机条件需区分。
- Anchor 初期只使用原图或人工确认的图像，避免把生成错误加入可信参考。
- LLM、人格和语音在核心闭环之上扩展，避免与实时视觉状态混杂。
- 模型效果、实时性和长期稳定性必须绑定真实实验。记录输入、模型版本、种子、配置、硬件、运行时长和失败样例。

## 命令与代码约定

- 安装：uv sync --locked；启动：uv run --locked avatar-runtime serve；环境报告：uv run --locked avatar-runtime doctor。
- 测试：uv run --locked pytest；静态检查：uv run --locked ruff check . 和 uv run --locked mypy。
- 格式校验：uv run --locked ruff format --check .；修改格式：uv run --locked ruff format src tests。
- 构建：uv build。修改依赖时更新 uv.lock；CI 入口为 .github/workflows/check.yml。
- Python 使用类型注解、snake_case、Ruff 格式，mypy 检查 src。Pydantic 管状态校验，不重复实现另一套校验层。
- 领域函数明确抛异常；API 区分 404/409/413/422，CLI 失败返回非零。损坏状态不得静默重置，失败推理不得写成成功。
- 模型调用在独立环境中执行；API 导入不能加载模型或下载权重。当前存储仅保证单进程调用，不擅自开启多 worker。
- 文档使用 UTF-8、相对链接；不提交密钥、真实 .env、私人角色图或大型生成物。

## 验收与同步

- 变更后核对 README、docs、注释、测试、示例和配置是否需要同步；执行与改动相关的检查。
- 状态、持久化、模型参数和失败处理变更需要行为验证；纯文档调整检查链接和事实即可。
- 区分本地检查、可选上游契约检查、GPU 推理和远端 CI。缺少权重时明确未验证，不增加假生成器掩盖缺口。
- 最终说明具体改动、依据、已执行验证和未覆盖项；审查先列问题与位置。
- 安全审计按需参考 docs/代码审计.md；结构调整按需参考 docs/代码组织.md，不把整份通用模板叠加为强制规则。
