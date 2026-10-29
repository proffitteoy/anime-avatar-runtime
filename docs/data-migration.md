# 旧工程处置与数据恢复（#68）

本项目已于 2026-10-29 停止维护，见 [归档说明](../README.md#停止维护与归档)。#68 已完成，本文保留迁移与恢复用法；封仓无需重新迁移或清理本地数据。

当时的新路线是离线资产编译、参数运动和轻量壁纸。旧实验退出活动目录，不在 Runtime 内增加新旧格式双读或自动转换逻辑。本工具只整理已有数据，不生成尚不存在的网格、绑定或运动状态。

## 统一处置范围

按用户要求，所有涉及旧工程的收尾统一在 [#68](https://github.com/proffitteoy/anime-avatar-runtime/issues/68) 验收，不再分摊到新产品任务。

| 来源 | #68 的处置 | 新任务保留的工作 |
| --- | --- | --- |
| #1–#36、#67 | 保留历史完成/终止记录；旧功能不再作为当前待办 | P1/P2/P3 各自重新验收 |
| #49 | 旧 state.json v1 只读校验、参考图来源与原字节保留 | AvatarAsset 格式、合法/非法样例与校验 |
| #55/#57 | 旧 AvatarState 与旧 Runtime 语义保留，不翻译成 MotionState | 新运动状态、时间和安全恢复 |
| #63 | 迁移/只读状态工具不加载旧模块，旧数值依赖显式可选 | 新播放器资源与离线编译隔离策略 |
| #64 | 历史 API/CLI 兼容和回归、旧数据恢复 | 新资产导入、版本切换/回滚与播放器控制 |
| #54 | 旧研究保留来源和适用域，按显式离线命令复用 | 新资产质量研究与阈值验收 |
| PR #44 | 撤销上一版尚未合并的连续生成代码，原提交仅保留在 Git 历史 | 新壁纸不引入连续视频窗口 |

#68 的退出条件是旧数据可校验/恢复、可信参考不变、历史接口可明确选择且回归通过、基本安装与旧模型隔离。表中的新任务职责保留为当时的分工记录；新资产格式、播放器、资产版本回滚未交付，对应 Issue 已随封仓按不再计划关闭。

### #44 的旧代码退场

#44 原先为上一版路线新增的有限连续视频生成，不进入当前产品。本次通过普通回退提交撤销原实现，保留原始提交与研究记录；不以解决冲突为由重新接入旧功能。

| 位置 | 撤销的旧功能 |
| --- | --- |
| contracts.py | GenerationWindow、accepted_windows、windows/previous_run_id 请求与窗口事件 |
| runtime.py | 连续窗口调度、有效视频上下文提交、跨任务续接和等待入口 |
| skyreels.py | 连续窗口视频解码检查、来源校验、边界片段与媒体子命令 |
| cli.py | windows 命令及其 Runtime 调用 |
| tests/、README 和控制文档 | 旧窗口测试、使用入口和后续开发指引 |

#69 中的免备份迁移、只读 checkpoint、显式 legacy 入口、依赖隔离及已有单段生成/观测兼容仍保留。#69 与 #44 已于 2026-10-25 合并，#68 已完成关闭；原合并安排与代码撤销记录保留在对应 PR。#44 的最终代码与 #69 相同，只补充旧实现撤销记录，避免把旧代码再次带入主线。

## 目录与保留规则

```text
var/
  migration.json             本次归档位置
  avatar/                    唯一活动角色目录
    state.json               原有可信 checkpoint，存在时逐字节保留
    references/<sha256>.png   checkpoint 引用的参考图
.archive/<归档名称>/
  manifest.json              文件/空目录清单、SHA-256、字节数和事务阶段
  backup.zip                 可选独立备份；--no-backup 不创建
  data/                      原 var 目录，原子移动并再次验证
```

只从显式指定的 `--avatar-subdir` 加载活动 checkpoint。其他目录里的测试角色、历史 run/job、观测缓存、人工标签和实验脚本全部归档，不能自动提升为可信资产。活动目录没有 checkpoint 时保持未初始化。

已有 checkpoint 的 avatar_id、参考图哈希、时间和内容保持原值，不重新编码图片。历史视频、特征和任务记录保留原字节；视频不转换成绑定资产，已有日志中的绝对路径不静默改写。查询旧路径时，用 `manifest.json` 中的 source 前缀替换为归档的 data 目录。

`models/`、`vendor/`、解释器和环境不属于此次 var 迁移，不复制或删除。`.archive/`、`var/` 和恢复检查输出均不提交。

## 执行

先停止 API、生成和观测进程；命令要求离线运行，不提供跨进程写锁。数据不可读、含链接/联接、参考图损坏、源目录变化或空间不足都会失败，不跳过问题文件。

```powershell
uv run --locked --no-dev avatar-runtime migrate-data --source var --archive .archive/legacy-data --no-backup
uv run --locked --no-dev avatar-runtime migrate-data --source var --archive .archive/legacy-data --no-backup --apply
uv run --locked --no-dev avatar-runtime verify-data --archive .archive/legacy-data
```

第一条只读盘点。`--no-backup --apply` 校验并暂存可信参考、再次复查源目录，再同盘原子归档；不会制作 ZIP 副本。省略 `--no-backup` 时保留独立备份流程。两种模式都保留归档原件，不递归删除历史数据。归档必须与源目录同盘且互不嵌套，不能覆盖已有归档。

自定义活动目录使用 `--avatar-subdir`，例如 `--source var --avatar-subdir characters/alice`。它必须是相对路径，不会扫描其他 checkpoint 猜测用户角色。

事务阶段为 planned → prepared（免备份）或 backed_up（备份）→ archived → completed。manifest 的 backup_required 记录已选模式；旧清单缺少该字段时按有备份处理。续跑遵循原清单，不会补做免备份事务的 ZIP。

暂存数据在移动源目录前校验。切换遇到可捕获的写入/权限/激活失败时，目标位置仍空则将归档原件移回原路径；若回退也失败，原件仍留在归档并报错。强杀/断电不承诺原路径始终存在，使用相同 source/archive 再次执行 `--apply` 续跑；目标已被其他进程写入时不覆盖。planned 阶段尚未准备完成则保留源数据，改用新的归档重试。

## 恢复

```powershell
uv run --locked --no-dev avatar-runtime restore-data --archive .archive/legacy-data --output .cache/legacy-recovered
```

输出必须是不存在的新目录，不能覆盖活动目录或归档。工具从 manifest 指定的 ZIP 或免备份归档原件恢复，重新检查完整文件集、空目录、字节数及每个 SHA-256 后才启用输出目录。有备份模式下，即使 data 原件损坏/丢失也能使用完好的 ZIP；免备份模式依赖 data 原件完整，损坏或丢失时明确失败。不会自动改用其他来源掩盖损坏。

复制、校验或启用失败会清理本次 `.restore-<uuid>` 临时目录，再传播错误；若清理本身失败也明确报错，不静默忽略私人数据残余。强杀/断电无法保证执行清理。

恢复输出是离线检查目录，不会自动成为第二个 Runtime 数据根目录。需要回退时先停止服务、保留当前活动数据，再显式选择已经验证的恢复目录。ZIP 恢复保证文件字节和目录，不承诺还原原 NTFS ACL、所有者或所有文件系统元数据。

## 只读 checkpoint 与历史入口

```powershell
uv run --locked --no-dev avatar-runtime legacy state --data-dir var/avatar
uv sync --locked --no-dev --extra legacy
uv run --locked --no-dev --extra legacy avatar-runtime legacy serve
```

`legacy state` 复用 load_state，校验旧 schema 和参考 SHA-256，只输出现有 avatar_id、来源哈希、尺寸和时间。未初始化或损坏返回非零，不加载模型配置、不写回 state.json。新资产和运动快照应使用独立格式与文件，引用原图哈希，不能将旧参考状态宣称为已分层、已绑定或已运动。

`legacy serve/generate/observe` 是显式历史入口；原 serve/generate/observe 保持兼容，向 stderr 提示归属。HTTP 路径、返回体与状态码保持原义，OpenAPI 标记 deprecated。模型缺失仍返回原错误，不启动假生成器或自动循环。

基本 `uv sync --locked --no-dev` 不安装 NumPy、Torch 或观测模型。legacy extra 只补 NumPy；observers extra 保留独立 CPU 实验依赖。模型环境/权重仍按原部署文档显式安装。迁移、恢复和只读状态查询不导入旧 contracts、skyreels、observers、runtime 或 api；新播放器可独立实现，不应复用旧 Runtime 的生成循环。

## 验证与记录

迁移测试覆盖两种模式的盘点、校验、恢复、空间不足、权限失败、损坏数据、源变化、暂存损坏、切换回退和进程中断续跑；端到端覆盖可信参考迁移、旧 API 重启、缺失后端与字节不变。常规 Runtime/生成/观测测试继续验证旧语义。

SkyReels 参数契约测试先验证固定上游 blob，再只对测试副本应用仓库提交的入口补丁，通过 AST 核对实际部署参数；不执行模型。此前 `--negative_prompt` 的失败源于只检查未打补丁入口，现按已记录的部署契约验证。

本机历史数据已于 2026-10-23 归档，当前活动目录无可信 checkpoint，保持未初始化。本轮按要求不再创建备份，不重做已完成的迁移，不扫描测试角色自动填入。文件清单、哈希和权限记录保留在忽略目录；它们不进入源码分发。测试和构建不代表 GPU 推理、新资产编译、播放器或实机长期运行已验收。
