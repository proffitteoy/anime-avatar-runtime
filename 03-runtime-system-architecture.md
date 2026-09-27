# 生成式二次元数字人：项目架构方案

## 1. 文档定位

本路线不研究单个生成模型，也不研究具体的 Drift 判定算法。

它负责回答：

> 如何把图片生成、状态保持、状态恢复、LLM、语音、交互、模型调用、资源管理和前端最终组织成一个可以长期运行的软件系统。

三条主线关系为：

```text
Track A
Avatar Generation
&
State Preservation

Track B
State Recovery
&
Stability Control

Track C
Runtime
&
System Architecture
```

其中本文档对应 Track C。

交互仍然是运行系统上的一条旁支，而不是新的核心主线。

---

## 2. 系统目标

系统需要同时满足：

```text
长期运行
模块解耦
模型可替换
状态可恢复
LLM 可接入
语音可接入
GPU 可调度
生成过程可观测
未来可扩展
```

架构不能写死为：

```text
SkyReels
+
CCIP
+
AniSora
```

因为具体模型未来一定会发生替换。

系统应该围绕：

```text
能力
```

而不是：

```text
模型名称
```

设计接口。

---

## 3. 总体架构

```text
                    External Input
                         │
              ┌──────────┴──────────┐
              │                     │
            Voice                  API
              │                     │
             ASR                    │
              │                     │
              └──────────┬──────────┘
                         ▼
                 ┌──────────────┐
                 │     LLM      │
                 │ Persona/Brain│
                 └──────┬───────┘
                        │
                   Intent / Event
                        │
                        ▼
             ┌────────────────────┐
             │  Runtime Controller │
             └──────────┬─────────┘
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
 Avatar Engine    Recovery Engine    Action Planner
        │               │                │
        └───────────────┼────────────────┘
                        │
                        ▼
                  AvatarState
                        │
                        ▼
                    Renderer
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
            Video                Audio
```

---

## 4. 核心原则

### 4.1 State-Centric

系统中心不是 Generator。

而是：

```text
AvatarState
```

所有模块围绕它工作。

```text
Generator
   │
   ▼
AvatarState
   ▲
   │
Recovery

LLM
   │
   ▼
Intent
   │
   ▼
AvatarState

Renderer
   ▲
   │
AvatarState
```

---

### 4.2 Model-Agnostic

例如不要让 Runtime 直接调用：

```python
skyreels.generate(...)
```

应该调用：

```python
avatar_generator.generate(...)
```

具体后端：

```text
SkyReels
AniSora
future_model_x
future_model_y
```

由 Adapter 决定。

---

### 4.3 Event-Driven

数字人运行不是单个 while-loop。

应该存在事件系统：

```text
USER_MESSAGE
VOICE_INPUT
ACTION_REQUEST
GENERATION_DONE
DRIFT_WARNING
RECOVERY_REQUEST
RECOVERY_DONE
MODEL_ERROR
RESOURCE_WARNING
SHUTDOWN
```

Runtime 根据事件改变系统状态。

---

## 5. AvatarState

所有路线共享统一状态：

```text
AvatarState
├── id
├── identity
├── appearance
├── pose
├── expression
├── motion
├── mode
├── generation_context
├── interaction_context
├── recovery_context
├── active_anchor
├── confidence
├── timestamp
└── metadata
```

其中：

```text
mode ∈ {
    idle,
    interaction,
    recovery,
    paused,
    error
}
```

`AvatarState` 应该成为：

```text
Track A
Track B
Track C
```

之间唯一稳定的核心接口。

---

## 6. Runtime Controller

Runtime Controller 负责整个系统生命周期。

逻辑类似：

```text
BOOT
 │
 ▼
INITIALIZE
 │
 ▼
IDLE
 │
 ├── interaction event
 │       ↓
 │   INTERACTION
 │       ↓
 │   POST_INTERACTION
 │       ↓
 │     IDLE
 │
 ├── drift event
 │       ↓
 │    RECOVERY
 │       ↓
 │     IDLE
 │
 └── fatal error
         ↓
       RESET
```

Runtime 本身不生成视频。

它负责：

```text
谁应该工作
什么时候工作
使用哪个模型
当前状态是什么
异常发生后怎么办
```

---

## 7. Module Boundary

第一版建议：

```text
src/
├── runtime/
├── state/
├── avatar/
├── recovery/
├── observer/
├── interaction/
├── llm/
├── audio/
├── models/
├── scheduler/
├── storage/
├── api/
├── config/
└── telemetry/
```

---

## 8. Avatar Engine

职责：

```text
Image → Avatar
Idle generation
State continuation
Motion generation
Expression generation
```

暴露：

```text
initialize(reference)
generate(state)
continue(state)
apply_action(state, action)
```

Runtime 不关心背后到底使用：

```text
SkyReels
AniSora
其他模型
```

---

## 9. Observer

职责：

```text
观察生成结果
计算状态信号
产生异常事件
```

输出：

```text
Observation
├── identity_score
├── geometry_score
├── motion_score
├── visual_quality
├── confidence
└── timestamp
```

Observer 不能直接调用 Recovery。

它只产生：

```text
DRIFT_WARNING
```

或：

```text
DRIFT_FAILURE
```

事件。

这样可以避免：

```text
Detector
→
Generator
```

之间产生紧耦合。

---

## 10. Recovery Engine

输入：

```text
AvatarState
Observation
Anchor Bank
RecoveryPolicy
```

输出：

```text
Recovered AvatarState
```

接口：

```text
diagnose()
select_strategy()
recover()
hard_reset()
```

Recovery 细节全部留在独立路线。

---

## 11. Model Manager

整个项目会同时存在大量模型：

```text
video generation
identity
face landmarks
optical flow
interpolation
LLM
ASR
TTS
```

如果所有模型同时驻留 GPU：

```text
VRAM
```

会迅速成为瓶颈。

因此需要：

```text
ModelManager
```

统一负责：

```text
load
unload
warmup
device placement
precision
model cache
health
```

例如：

```text
Idle
├── Avatar Generator      GPU
├── Observer              GPU/CPU
├── LLM                   Remote / GPU
└── Recovery Generator    Unloaded
```

进入 Recovery：

```text
Idle Generator
    ↓ pause / unload

Recovery Generator
    ↓ load

Recovery Done
    ↓ unload

Idle Generator
    ↓ resume
```

---

## 12. GPU Scheduler

长期目标不能假设：

```text
无限显存
```

需要显式处理：

```text
GPU memory
compute priority
model switching cost
concurrency
latency
```

建议定义任务：

```text
GenerationTask
ObservationTask
RecoveryTask
InteractionTask
AudioTask
```

优先级例如：

```text
Recovery
>
Interaction
>
Idle Generation
>
Background Analysis
```

Idle 本身可以暂停。

因此系统在资源不足时应该首先牺牲：

```text
idle frame rate
```

而不是：

```text
recovery correctness
```

---

## 13. Storage

需要保存两类状态。

### Persistent State

```text
character reference
persona
anchors
configuration
model preference
voice configuration
long-term memory
```

### Runtime State

```text
current AvatarState
generation context
recent frames
active model
current interaction
recovery state
```

两者生命周期不同。

---

## 14. Checkpoint

长期运行必须能够：

```text
crash
↓
restart
↓
continue
```

而不是：

```text
restart
↓
完全重新初始化角色
```

Checkpoint 建议保存：

```text
AvatarState
selected Anchor
recent generation context
runtime mode
timestamp
```

但大型 latent 或视频窗口是否保存，需要根据空间与恢复收益决定。

---

## 15. LLM 接入

LLM 属于系统架构的一部分，但不是长期视频运行的主闭环。

职责：

```text
conversation
persona
intent understanding
action decision
tool invocation
memory interaction
```

LLM 不直接控制：

```text
pixels
frames
optical flow
recovery threshold
```

LLM 输出高层意图：

```json
{
  "type": "action",
  "action": "wave",
  "emotion": "happy",
  "intensity": 0.4
}
```

再由 Action Planner 转换为具体生成请求。

---

## 16. Action Planner

LLM 不应该生成复杂 video prompt 后直接传入 Generator。

中间增加：

```text
Action Planner
```

例如：

```text
用户：
“跟我打个招呼”

LLM：
GREETING

Action Planner：
wave
small smile
look at camera
duration = 4s
return_to_idle = true

Generator：
实际生成
```

这样可以避免 prompt 不稳定直接污染视觉系统。

---

## 17. Interaction Side Branch

交互保持为旁支：

```text
                   interaction event
                          │
                          ▼
Idle ─────────────→ Action Planner
 ▲                        │
 │                        ▼
 │                  Action Generator
 │                        │
 │                        ▼
 └──── Recovery ← Post Interaction
```

默认运行闭环仍然是：

```text
Generate
→
Observe
→
Continue / Recover
```

交互只是临时打断 Idle。

---

## 18. ASR / TTS

Voice Pipeline：

```text
Microphone
    ↓
ASR
    ↓
LLM
    ↓
Response
    ├── TTS
    └── Action Planner
```

音频与视觉不应该完全独立。

例如说话时：

```text
TTS
↓
audio timing
↓
lip / expression generation
```

因此未来需要：

```text
audio-visual synchronization
```

但第一版可以暂时只做松耦合。

---

## 19. Persona

Persona 与 AvatarState 分开。

```text
Persona
=
这个角色是谁、怎么说话、如何回应

AvatarState
=
这个角色现在视觉上处于什么状态
```

不要把：

```text
personality
```

塞进生成模型的实时 state。

---

## 20. Memory

Memory 主要服务 LLM。

可以分为：

```text
short-term conversation memory
long-term user memory
character memory
world / session state
```

Memory 产生行为意图：

```text
Memory
↓
LLM
↓
Action
```

然后才影响 Avatar。

---

## 21. API

内部建议按照能力设计。

例如：

```text
POST /avatar/init
POST /avatar/generate
GET  /avatar/state

POST /interaction/message
POST /interaction/action

GET  /observer/status

POST /recovery/trigger

GET  /runtime/status
POST /runtime/pause
POST /runtime/resume
```

内部模块优先使用直接接口或 event bus。

HTTP API 主要服务：

```text
frontend
external plugins
remote controller
```

---

## 22. Event Bus

关键事件：

```text
AVATAR_INITIALIZED

FRAME_GENERATED
SEGMENT_GENERATED

DRIFT_WARNING
DRIFT_FAILURE

RECOVERY_STARTED
RECOVERY_FINISHED
RECOVERY_FAILED

USER_MESSAGE
VOICE_MESSAGE

ACTION_REQUESTED
ACTION_STARTED
ACTION_FINISHED

MODEL_LOADED
MODEL_UNLOADED
MODEL_ERROR

RESOURCE_WARNING

CHECKPOINT_CREATED
```

Event Bus 可以显著减少模块耦合。

---

## 23. Telemetry

这个项目如果没有 telemetry，后期很难调试。

至少记录：

```text
generation latency
FPS
GPU utilization
VRAM
model load time

D_id
D_geo
D_motion

recovery count
recovery duration
hard reset count

interaction latency

LLM latency
ASR latency
TTS latency
```

这些数据同时也是研究实验数据。

---

## 24. Configuration

避免将模型和参数硬编码。

例如：

```yaml
avatar:
  generator: skyreels-v2
  resolution: 540p

observer:
  identity: ccip
  geometry: anime-face-detector
  motion: gmflow

recovery:
  cheap: drba
  generative: tooncrafter

llm:
  provider: openai

runtime:
  checkpoint_interval: 30
```

未来替换模型时：

```text
改配置
```

而不是：

```text
改 Runtime 源码
```

---

## 25. 前后端关系

Frontend 只负责：

```text
avatar display
conversation UI
configuration
status display
debug panel
```

不能承担：

```text
model scheduling
state control
recovery
```

Backend 才是真正 Runtime。

---

## 26. 推荐的数据流

正常 Idle：

```text
AvatarState
    ↓
Avatar Engine
    ↓
Frames
    ↓
Observer
    ↓
Runtime
    ↓
AvatarState Update
```

发生 Drift：

```text
Observer
    ↓
DRIFT_WARNING
    ↓
Runtime
    ↓
Recovery Engine
    ↓
Recovered State
    ↓
Avatar Engine
```

交互：

```text
User
↓
ASR / Text
↓
LLM
↓
Action Planner
↓
Runtime
↓
Avatar Engine
↓
Recovery
↓
Idle
```

---

## 27. 第一版架构

V0 不追求复杂分布式系统。

可以保持：

```text
single process
+
async tasks
+
local models
```

模块：

```text
Runtime
AvatarEngine
Observer
RecoveryEngine
ModelManager
StateStore
LLMAdapter
```

足够。

---

## 28. V1

加入：

```text
event bus
checkpoint
telemetry
ASR
TTS
Action Planner
```

---

## 29. V2

解决：

```text
multi-GPU
remote inference
model service
dynamic scheduling
plugin system
```

---

## 30. 三条路线如何并行

项目开发不能再按：

```text
先全部做生成
→
再全部做 Recovery
→
最后做架构
```

而应该：

```text
                    Time →

Track A
Image→Avatar ─ State ─ Persistence ─ Adaptation
      │           │          │
      │           │          │

Track B
 Observer ─ Basic Recovery ─ Adaptive Recovery
      │           │          │
      │           │          │

Track C
 Runtime ─ State/API ─ LLM/Voice ─ Full System
```

三条线共享接口：

```text
AvatarState
Event
ModelAdapter
```

这样某一条研究线更换实现时，不会摧毁另外两条。

---

## 31. 第一个可运行版本

第一版目标不是完整 AI 主播。

而是：

```text
上传一张图片
↓
初始化角色
↓
持续 Idle
↓
Observer 持续监测
↓
严重异常可以 Reset
↓
Runtime 保存状态
↓
可以通过简单 API 控制
```

同时提前保留：

```text
LLMAdapter
ActionPlanner
AudioAdapter
```

接口，但不要求一次全部实现。

---

## 32. 第二个版本

增加：

```text
Anchor Bank
Adaptive Recovery
LLM conversation
basic actions
TTS
```

实现：

```text
Idle
↓
用户对话
↓
角色回答
↓
简单动作
↓
恢复 Idle
```

---

## 33. 第三个版本

增加：

```text
ASR
advanced Action Planner
audio-driven animation
adaptive GPU scheduling
learned recovery
long-term memory
plugins
```

逐步形成完整数字人 Runtime。

---

## 34. 最终架构定义

整个项目最终不是：

```text
几个 AI 模型串起来
```

而应该形成：

```text
Persistent Generative Avatar Runtime
```

其中：

```text
Track A
负责：
如何生成和保持 AvatarState

Track B
负责：
如何检测并恢复 AvatarState

Track C
负责：
如何组织、调度、保存和使用 AvatarState
```

交互则建立在这三者之上：

```text
User
↓
LLM / Voice
↓
Action
↓
Avatar Runtime
↓
Recovery
↓
Idle
```

因此整个项目最核心的系统关系可以压缩为：

```math
\boxed{
\text{Generate \& Preserve}
+
\text{Recover}
+
\text{Orchestrate}
}
```

三条路线同时推进，并最终在统一 Runtime 中汇合。