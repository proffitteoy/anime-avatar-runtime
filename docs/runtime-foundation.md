# Runtime 开发底座

本次底座对应 [Project #2](https://github.com/users/proffitteoy/projects/2) 的 [#24 共享契约](https://github.com/proffitteoy/anime-avatar-runtime/issues/24)，并落地 #25/#28/#29 中可独立于 GPU 实验推进的任务控制、API 与实验记录。三份编号方案仍是目标设计；本文描述当前代码，不代表这些 Issue 的全部验收已经完成。

## 模块与调用链

```text
API → Runtime → run_generation → 独立 SkyReels 进程
CLI ──────────→ run_generation → 独立 SkyReels 进程
        共享 RuntimeConfig、GenerationRequest、GenerationResult
Runtime → state.py（参考 checkpoint）/ job.json（任务记录）
```

| 文件 | 职责与后续修改入口 |
| --- | --- |
| [config.py](../src/anime_avatar_runtime/config.py) | 统一解析环境变量，校验路径和超时，导出有效配置 |
| [contracts.py](../src/anime_avatar_runtime/contracts.py) | 生成请求/结果、任务、事件与独立观测信号；三条路线复用同一份类型 |
| [runtime.py](../src/anime_avatar_runtime/runtime.py) | 单角色、单任务、手动生命周期与结果接收；不翻译模型私有参数 |
| [skyreels.py](../src/anime_avatar_runtime/skyreels.py) | 现有后端边界：参数翻译、环境预检、可取消子进程与实验产物 |
| [state.py](../src/anime_avatar_runtime/state.py) | 可信原图、schema v1 checkpoint、原子替换、重读哈希校验 |
| [api.py](../src/anime_avatar_runtime/api.py) | HTTP 输入、错误码与 lifespan 关停；不另写一套生成逻辑 |
| [cli.py](../src/anime_avatar_runtime/cli.py) | serve/doctor/generate 参数及退出状态；生成复用相同后端函数 |

保留平铺模块结构。当前只有一个后端，不增加插件注册表、消息中间件、抽象存储层或空的 LLM/Observer/Recovery 目录。切换后端时，在调用边界实现相同请求/结果及取消契约；加入第二个后端后再决定是否需要统一选择器。

## 共享契约

- `GenerationRequest`：frames、seed、prompt；frames 至少 97 且为 4n+1，seed 为 uint32，prompt 非空，未知字段拒绝。私有源图片路径由 Runtime 从可信 checkpoint 取得，HTTP 不接受任意输入/输出路径。
- `GenerationResult`：run_id、avatar_id、output、videos、elapsed_seconds；成功必须有产物。当前适配器验证进程成功退出与非空 MP4，尚不验证解码或视觉质量。
- `GenerationJob`：schema_version、run_id、avatar_id、state_schema_version、request、status、创建/完成时间、result/error。状态为 queued/running/completed/failed/cancelled；结果只允许属于同一角色和同一任务。
- `RuntimeEvent`：event_id、timestamp、kind、avatar_id、run_id、state_schema_version、mode、reason。初始化、提交、开始、完成、失败、取消、暂停、继续、关停使用同一时间轴。最近 200 条保留在内存，重启不保留；不是持久事件总线。
- `Observation`：identity_distance、geometry_distance、motion_distance 独立可空。测量值须带 avatar_id/run_id/measured_at；完全缺测须有 unavailable_reason，不能把未加载 Observer 显示为三个零值。当前 API 全部返回 null，后续部分模型接入时应说明其余缺测原因。

此处 `state_schema_version` 是格式版本，不是状态修订计数。当前一次 API 任务等于一个参考图生成实验，因此使用 run_id 即可关联；#8/#9 引入连续窗口和动态状态时再增加 window_id 与状态修订、提交校验，不预造 latent、pose 或 embedding。

## 生命周期与并发

启动加载 checkpoint：有参考图时 paused，没有时 uninitialized。初始化成功仍为 paused。

```text
paused --resume--> ready --generate--> generating --成功--> ready
                                       └--失败--> error --resume--> ready
generating --pause--> pausing --子进程释放--> paused
任意状态 --服务关停--> 等待活动任务释放 --> stopped
```

`resume` 只打开手动提交入口，不启动自主循环，也不检查 CUDA。生成能力是否配置见 backend_configured / backend_unavailable_reason；路径存在不代表模型健康。生成提交时检查后端路径，执行时检查源码 HEAD、CUDA 和 MoviePy。API 导入、启动、doctor 均不加载模型。

Runtime 串行处理控制操作，同一实例只能有一个活动任务；重复生成返回 409。pause 会设置取消信号，并等待模型进程清理后返回。失败进入 error，必须显式 resume 才能再次提交；没有自动重试或后台投递。迟到的取消结果不会被接受为成功，任务 ID 不匹配的结果记为失败。

阻塞模型调用在线程中等待独立子进程；API 仍能查询状态。预检最多 60 秒，且与生成共享总超时。Windows 通过 taskkill /T /F 清理所拥有的进程树，POSIX 通过独立进程组清理。进程清理若被权限拒绝会报告失败；控制进程被强杀、断电等情况下不能保证执行优雅清理。当前一个数据目录只能运行一个服务实例，CLI 独立实验不经过 Runtime 的单任务锁，勿与 API 同时争用同一 GPU。

## 存储与重启

```text
AVATAR_DATA_DIR/
  state.json
  references/<sha256>.png
  runs/<run_id>/
    job.json
    output/run.json
    output/backend.log
    output/*.mp4
```

AvatarState v1 的 mode=paused 表示可恢复的参考状态；实时控制状态以 `/runtime/status` 为准。生成不会改写原图、提升生成帧为可信 Anchor，也不会把视频自动提交为可续接 checkpoint。成功/失败/取消的 `job.json` 均原子保存，失败不会覆盖参考 checkpoint。

重启始终暂停，不自动重跑模型。查询遗留 queued/running 任务时将其标为 interrupted 失败。`job.json` 是 Runtime 是否接受结果的依据；`output/run.json` 保存后端执行证据，进程被强杀时可能仍为 running，或在取消竞态中后端完成但 Runtime 拒收，两者不能混同。持久任务和视频不自动删除；磁盘配额和连续窗口缓存清理由 #27 完成。

## HTTP 接口

| 接口 | 当前行为 |
| --- | --- |
| GET /health | 控制服务存活，不是模型健康检查 |
| POST /avatar/init | 原始图片字节；201；重复初始化 409，超限 413，无效图 422 |
| GET /avatar/state | 可信参考 checkpoint；未初始化 404，损坏状态报错 |
| GET /runtime/status | 实时模式、活动任务、能力和最后错误 |
| POST /runtime/resume | 开放手动提交；未初始化 404 |
| POST /avatar/generate | JSON 请求，202 返回 queued 任务；未初始化 404，暂停/繁忙/错误 409，参数无效 422，后端未配置 503 |
| GET /runtime/jobs/{run_id} | 查询本角色的持久任务；未知任务 404，非法 ID 422 |
| POST /runtime/pause | 取消并等待当前任务释放，进入暂停 |
| GET /runtime/events?limit=50 | 当前进程最近事件，limit 为 1–200 |
| GET /observer/status | 三种测量均未知，并返回原因 |
| POST /recovery/trigger | 503：未实现恢复，不创建假任务 |

API 仅接受初始化参考图的单段生成。CLI 仍提供手动视频续接和首尾帧参数。参数、默认值与返回类型也可在服务的 `/docs` 查看。

## 配置与开发次序

配置来源是进程环境，`.env` 需通过 uv 的 `--env-file` 显式加载；见 [.env.example](../.env.example)。CLI 的 --repo/--python/--model/--timeout 覆盖对应环境变量；环境配置本身须先合法。只支持当前实际生效的设置，不加入没有实现的精度/Observer/恢复阈值开关。

| 路线 | 下一步接入点及 Issue |
| --- | --- |
| A 生成保持 | #6 真实模型环境 → #7 首段基线 → #8 连续窗口；在 #9 中升级 state.py，保存带来源的动态状态与迁移 |
| B 观测恢复 | #13/#14/#15 各模型输出映射至 Observation → #16 聚合信号 → #17 由 Runtime 决策 Reset；Observer 不直接启动生成器 |
| C Runtime | #25 在现有生命周期上增加真实生成—观测循环；#26 GPU 资源调度；#27 续接 checkpoint；#28/#29 补齐真实恢复 API 与遥测；最后 #30 GPU 闭环验收 |

#5 的缺失部署文档已补齐；#24 的现有配置与调用契约已落地，模型精度、观测和恢复配置须随组件实现继续完善。#25/#27/#28/#29 的真实持续运行、恢复与资源验收仍未完成。远端 Issue 和 Project 不因本地底座测试通过自动关闭。

测试分层：契约/Runtime 使用显式测试替身验证控制行为；子进程测试实际启动短小 Python 程序验证产物、超时和子树取消；上游 AST 测试只核对固定源码的参数。以上均不是 CUDA 推理或视频质量证据。真实硬件验收步骤见 [backend-setup.md](./backend-setup.md)。
