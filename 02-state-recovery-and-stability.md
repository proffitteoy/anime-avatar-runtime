# 数字人状态恢复与稳定控制

## 1. 目标

本路线负责处理数字人在长期生成过程中已经出现异常之后的问题。

基本流程：

```text
Avatar Running
      ↓
Observation
      ↓
Drift Detection
      ↓
Diagnosis
      ↓
Recovery Decision
      ↓
Recovery
      ↓
Valid AvatarState
```

本路线不负责正常视频生成模型如何训练，也不负责完整应用架构。

核心问题是：

> 当生成式数字人的状态开始偏离合法角色状态时，如何尽早发现、判断严重程度，并以最低视觉代价和计算代价恢复到稳定状态。

---

## 2. 为什么必须存在 Recovery

任何 autoregressive 或 window-based generative system 都可能产生误差积累。

典型异常包括：

```text
身份漂移
五官畸变
发型变化
衣服变化
多脸
肢体错误
姿态跳变
运动异常
局部闪烁
角色逐渐变成其他角色
```

因此系统不能假定：

```text
Generator 永远正确
```

而必须采用：

```text
Generator
+
Observer
+
Controller
+
Recovery
```

形成闭环。

---

## 3. Recovery 与 State Preservation

两者必须分开。

```text
State Preservation
```

负责：

> 正常状态如何持续。

```text
State Recovery
```

负责：

> 状态已经损坏以后如何返回合法区域。

抽象地说：

```math
S_t \in \mathcal M_{\mathrm{valid}}
```

是正常运行。

如果：

```math
S_t \notin \mathcal M_{\mathrm{valid}},
```

则进入 Recovery。

---

## 4. Observer

Observer 不直接修改生成结果。

它负责估计当前状态。

第一版 Observer 分为：

```text
Observer
├── Identity
├── Geometry
├── Motion
├── Visual Quality
└── Runtime Signals
```

---

## 5. Identity Drift

当前首选：

```text
CCIP
```

提取角色 embedding：

```math
z_t = E_{\mathrm{CCIP}}(I_t).
```

对于 Anchor Bank：

```math
\mathcal A =
\{A_1,\ldots,A_n\},
```

对应：

```math
z_j^A =
E_{\mathrm{CCIP}}(A_j).
```

身份距离：

```math
D_{\mathrm{id}}(t)
=
\min_j d(z_t,z_j^A).
```

它回答：

> 当前生成结果还属于原来的角色吗？

Identity 是 Recovery 系统最重要的信号之一。

---

## 6. Geometry Drift

Identity embedding 不能覆盖所有结构错误。

可能出现：

```text
眼睛位置错误
脸宽变化
下巴畸变
嘴部结构异常
五官比例改变
```

但整体 embedding 仍然接近。

因此需要独立的 Geometry Detector。

当前可以使用：

```text
anime-face-detector
+
anime facial landmarks
```

设：

```math
L_t=
\{p_1,\ldots,p_k\}.
```

经过：

```text
translation normalization
scale normalization
rotation alignment
Procrustes alignment
```

获得：

```math
\hat L_t.
```

再比较：

```math
D_{\mathrm{geo}}(t)
=
\min_j
\|
\hat L_t-\hat L_j^A
\|.
```

---

## 7. Motion Anomaly

角色可能外观正常，但运动开始异常。

例如：

```text
突然跳头
局部瞬移
头发运动方向错误
身体周期性抽动
相邻帧结构跳变
```

当前候选：

```text
GMFlow
```

获得：

```math
F_t =
\operatorname{Flow}(I_{t-1},I_t).
```

但不能简单使用：

```math
\|F_t\|.
```

因为正常大动作也可能产生大 optical flow。

应该定义：

```math
D_{\mathrm{motion}}
=
\operatorname{Deviation}
(
F_t,
\mathcal D_{\mathrm{normal}}
).
```

其中：

```math
\mathcal D_{\mathrm{normal}}
```

表示正常 idle motion distribution。

---

## 8. 不应过早合并成单一分数

早期实验阶段保留：

```math
D_{\mathrm{id}}(t)
```

```math
D_{\mathrm{geo}}(t)
```

```math
D_{\mathrm{motion}}(t)
```

三条独立曲线。

而不是立即使用：

```math
D_t =
\alpha D_{\mathrm{id}}
+
\beta D_{\mathrm{geo}}
+
\gamma D_{\mathrm{motion}}.
```

原因是需要先回答：

```text
哪个指标最先发生异常？
哪个是真正 failure indicator？
哪个具有预测价值？
```

例如可能观察到：

```text
motion anomaly
      ↓
5–10 秒后
      ↓
geometry drift
      ↓
identity collapse
```

那么 Motion 就可能成为 early-warning signal。

这比简单构造一个加权分数更有研究价值。

---

## 9. State Machine

Controller 不应该使用：

```text
if D > threshold:
    recover()
```

这样的单阈值逻辑。

建议状态机：

```text
NORMAL
  │
  │ weak anomaly
  ▼
WATCH
  │
  │ persistent anomaly
  ▼
RECOVER
  │
  ├── success → NORMAL
  │
  └── failure → HARD_RESET
```

状态定义：

```text
NORMAL
WATCH
RECOVER
HARD_RESET
```

---

## 10. Hysteresis

为了避免系统在阈值附近频繁切换：

```math
\tau_{\mathrm{enter}}
>
\tau_{\mathrm{exit}}.
```

例如：

进入 Recovery：

```math
D_t >
\tau_{\mathrm{high}}.
```

退出 Recovery：

```math
D_t <
\tau_{\mathrm{low}}.
```

从而避免：

```text
NORMAL
RECOVER
NORMAL
RECOVER
...
```

快速震荡。

---

## 11. Temporal Decision

单帧异常通常不足以触发 Recovery。

需要考虑：

```text
duration
trend
velocity
acceleration
```

例如：

```math
D_t
```

较高但下一帧立即恢复，可能只是检测噪声。

而：

```math
D_t,
D_{t+1},
D_{t+2},
...
```

持续上升，则应该提高严重度。

后续 Controller 输入可以包含：

```math
x_t =
(
D_t,
\Delta D_t,
\Delta^2D_t,
T_{\mathrm{abnormal}}
).
```

---

## 12. Anchor Bank

Anchor 是 Recovery 的基础设施之一。

第一版：

```math
\mathcal A = \{A_0\}.
```

之后：

```math
\mathcal A =
\{
A_0,
A_1,
\ldots,A_n
\}.
```

每个 Anchor 除了图像之外还应该保存：

```text
Anchor
├── image
├── identity embedding
├── landmarks
├── pose
├── expression
├── timestamp
└── source
```

选择 Recovery Anchor 时不应该只比较 Identity。

可以计算：

```math
j^\star
=
\arg\min_j
D(S_t,A_j).
```

其中综合考虑：

```text
identity
pose
expression
visual continuity
```

---

## 13. Recovery 分级

Recovery 不应该只有一个模型。

应根据异常类型和严重程度选择不同策略。

```text
Drift
  │
  ▼
Diagnosis
  │
  ├── motion-only
  ├── mild semantic drift
  ├── medium semantic drift
  └── severe failure
```

---

## 14. Level 0 — Continue

异常不足以处理：

```text
NORMAL / WATCH
```

继续生成。

避免不必要的干预。

---

## 15. Level 1 — Cheap Recovery

如果：

```text
identity stable
geometry stable
motion abnormal
```

优先采用低成本修复。

候选：

```text
DRBA
AnimeInterp
frame interpolation
short temporal smoothing
```

目标是：

```text
不重新生成角色
```

而仅修复时间连续性。

---

## 16. Level 2 — Generative Interpolation

如果已经出现轻微：

```text
identity drift
geometry drift
```

普通 interpolation 无法解决 semantic error。

此时可以使用：

```text
ToonCrafter
```

执行：

```text
Current Frame
     ↓
Generative Transition
     ↓
Selected Anchor
```

形式上：

```math
I_t
\longrightarrow
A_j.
```

生成自然回归路径。

---

## 17. Level 3 — Controlled Generation

更复杂状态可以考虑：

```text
SkyReels start/end control
AniSora arbitrary-frame control
```

目标仍然是：

```math
I_t
\rightarrow
A_j
```

但中间允许生成更复杂的动作。

---

## 18. Level 4 — Hard Reset

如果发生：

```text
identity collapse
multiple faces
severe geometry corruption
body corruption
unrecoverable temporal state
```

则不应该继续尝试平滑恢复。

执行：

```text
Hard Reset
```

即：

```text
stop corrupted generation
↓
load stable AvatarState / Anchor
↓
restart generator
```

---

## 19. Recovery Policy

最终 Controller 学习：

```math
\pi
(
D_{\mathrm{id}},
D_{\mathrm{geo}},
D_{\mathrm{motion}},
S_t
)
\rightarrow
a_t.
```

其中：

```text
a_t ∈ {
    continue,
    watch,
    cheap_recovery,
    generative_recovery,
    hard_reset
}
```

V0 不需要机器学习。

直接规则：

```python
if severe_failure:
    hard_reset()

elif semantic_drift:
    generative_recovery()

elif motion_anomaly:
    cheap_recovery()

else:
    continue_generation()
```

---

## 20. Recovery Policy 的训练路线

### V0 — Rule Based

人工规则。

### V1 — Data-Driven Threshold

利用实验数据估计：

```math
P(D_{\mathrm{id}}\mid\mathrm{normal})
```

以及其他指标。

采用：

```text
percentile
median / MAD
ROC
false-positive rate
false-negative rate
```

确定阈值。

### V2 — Failure Predictor

训练：

```math
f(
D,
\Delta D,
\text{state}
)
\rightarrow
P(
\text{failure in next }k\text{ frames}
).
```

模型可以很小：

```text
XGBoost
MLP
Tiny Transformer
```

### V3 — Learned Recovery Policy

积累足够运行数据之后，再考虑学习：

```math
\pi(S_t)\rightarrow a_t.
```

---

## 21. Interaction Mode

交互仍然是一条旁支。

但交互过程中 Observer 必须切换 profile。

正常 Idle：

```text
Identity
Geometry
Motion
```

都使用严格阈值。

Interaction：

```text
Identity → 仍严格
Geometry → 放宽
Motion → 大幅放宽
```

因为用户可能要求：

```text
挥手
转身
大幅动作
```

这时正常 idle motion threshold 已经不适用。

因此：

```math
C =
C(
s_t,
D_t
),
```

其中：

```math
s_t
\in
\{
idle,
interaction,
recovery
\}.
```

---

## 22. Recovery Metrics

### Failure Detection Recall

真正发生异常时是否检测出来。

### False Recovery Rate

```math
R_{\mathrm{false}}
=
\frac{
N_{\mathrm{unnecessary\ recovery}}
}{
T
}.
```

### Recovery Success Rate

```math
R_{\mathrm{recovery}}
=
\frac{
N_{\mathrm{successful}}
}{
N_{\mathrm{recovery}}
}.
```

### Recovery Discontinuity

例如：

```math
J_{\mathrm{recovery}}
=
\operatorname{LPIPS}
(
I_{t^-},
I_{t^+}
).
```

也可以使用 optical flow spike。

### Recovery Cost

```math
C_{\mathrm{recovery}}
=
\frac{
\text{GPU time used by recovery}
}{
T
}.
```

### Intervention Rate

```math
R_{\mathrm{intervention}}
=
\frac{
N_{\mathrm{recovery}}
}{
T
}.
```

---

## 23. 对照实验

### Baseline A

```text
No Recovery
```

### Baseline B

固定时间：

```text
每 20 秒 reset
```

### Method C

```text
Adaptive Recovery
+
single anchor
```

### Method D

```text
Adaptive Recovery
+
Anchor Bank
```

### Method E

```text
Adaptive Recovery
+
Multi-Backend Policy
```

比较：

```text
identity lifetime
visual continuity
recovery frequency
GPU cost
failure rate
```

---

## 24. 阶段路线

### B0 — Observer

实现：

```text
CCIP
landmark
optical flow
```

并输出三条曲线。

### B1 — Basic Recovery

实现：

```text
threshold
+
hard reset
```

### B2 — Anchor Recovery

加入：

```text
Anchor Bank
+
nearest valid state
```

### B3 — Multi-Level Recovery

加入：

```text
DRBA
ToonCrafter
controlled generation
```

### B4 — Failure Prediction

由：

```text
detect failure
```

升级到：

```text
predict failure
```

### B5 — Adaptive Policy

根据：

```text
failure type
recovery quality
compute cost
```

自动选择 Recovery。

---

## 25. 本路线最终目标

最终结构：

```text
              Generated Avatar
                    │
                    ▼
                Observer
                    │
                    ▼
              Drift Estimate
                    │
          ┌─────────┼─────────┐
          │         │         │
       Stable      Mild      Severe
          │         │         │
       Continue   Recover    Reset
          │         │         │
          └─────────┼─────────┘
                    ▼
              Valid AvatarState
```

最终目标不是让 Recovery 不断工作。

而是：

```math
\boxed{
\text{在必要时以最低代价把系统重新送回合法状态}
}
```