# 图片到数字人：生成与长期状态保持

## 1. 目标

本路线负责解决数字人的生成与正常运行问题：

> 给定一张完整的二次元角色图，在不进行 Live2D 式切图、骨骼绑定和动作预制的前提下，将角色初始化为可以持续运行的生成式数字人，并在长时间运行过程中尽可能保持角色身份、外观、姿态和运动状态的连续性。

这一条路线关注的是：

```text
Single Image
    ↓
Avatar Initialization
    ↓
Avatar State
    ↓
Motion / Expression Generation
    ↓
State Transition
    ↓
Long-Horizon State Preservation
```

这里必须明确区分：

```text
State Preservation
        ≠
State Recovery
```

本路线处理正常状态下如何持续运行。

已经发生异常之后如何恢复，由独立的 Recovery 路线负责。

---

## 2. 核心研究问题

整个问题可以拆成五部分：

```text
A1. Image → Avatar
A2. Character Representation
A3. Motion / Expression Generation
A4. State Transition
A5. Long-Horizon State Preservation
```

最终目标不是简单实现：

```text
image → video
```

而是建立：

```text
image → persistent avatar
```

即角色不仅能够运动，而且能够作为一个持续存在的动态对象长期运行。

---

## 3. AvatarState

项目不能只保存当前视频帧。

需要区分：

```math
I_t = \text{当前渲染结果}
```

和：

```math
S_t = \text{当前数字人状态}.
```

第一版统一状态结构建议为：

```text
AvatarState
├── identity
├── appearance
├── pose
├── expression
├── motion
├── generation_context
├── interaction_context
├── temporal_context
└── checkpoint_metadata
```

抽象写为：

```math
S_t =
(
z_{\mathrm{id}},
z_{\mathrm{app}},
z_{\mathrm{pose}},
z_{\mathrm{expr}},
z_{\mathrm{motion}},
z_{\mathrm{ctx}}
).
```

### 3.1 Identity

描述角色本身的身份特征。

这一部分应该在正常运行过程中保持高度稳定。

例如：

```text
脸部主要结构
发型
角色特征
服装主体
配饰
整体画风
```

不能因为生成时间增加而逐渐发生角色替换。

### 3.2 Appearance

描述允许缓慢变化的视觉状态：

```text
光照
头发局部形态
衣物局部状态
遮挡
视角
局部阴影
```

Identity 与 Appearance 不应该完全混为一体。

### 3.3 Pose

描述角色当前姿态，例如：

```text
头部方向
身体朝向
身体重心
手臂位置
局部关节状态
```

### 3.4 Expression

描述当前面部状态，例如：

```text
眼睛开合
嘴部状态
眉毛
微笑
情绪
视线
```

### 3.5 Motion

保存运动趋势，而不是只有静态姿态。

例如：

```text
头部正在向左移动
正在眨眼
身体正在回正
头发仍存在惯性
呼吸处于吸气阶段
```

如果只保存单帧：

```math
I_t
```

那么下一段生成时会不断丢失之前的动态趋势。

### 3.6 Generation Context

保存生成模型继续运行所需要的信息，例如：

```text
reference image
recent frames
latent state
prompt
seed
motion context
generation history
```

第一版不要求所有生成模型都显式支持内部状态。

可以把这些信息共同视为：

```text
generation_context
```

的工程近似。

---

## 4. 状态演化

理想情况下，数字人的运行过程应该表达为：

```math
S_t
\overset{F_\theta}{\longrightarrow}
S_{t+1}
\overset{R_\theta}{\longrightarrow}
I_{t+1}.
```

其中：

```math
F_\theta
```

表示状态演化；

```math
R_\theta
```

表示视觉生成。

现阶段使用的视频生成模型通常没有显式提供完整的 `AvatarState`。

因此第一版采用：

```text
显式 Runtime State
+
生成模型隐式 temporal state
```

共同表示数字人的真实状态。

即：

```text
AvatarState
        │
        ├── Runtime 显式维护
        │
        └── Video Model 隐式维护
```

后续再研究两者是否需要进一步融合。

---

## 5. Image → Avatar

输入为：

```math
I_{\mathrm{ref}}.
```

初始化过程建议为：

```text
Reference Image
      │
      ├── identity extraction
      ├── appearance extraction
      ├── face / pose analysis
      ├── generation initialization
      └── reference embedding
      │
      ▼
Initial AvatarState S₀
```

第一阶段不要求建立复杂的显式角色模型。

主要目标是：

```text
单图
→
稳定启动生成模型
→
产生合理的第一段待机视频
→
建立连续状态
```

---

## 6. Idle Generation

待机是整个系统最重要的默认状态。

目标不是产生丰富动作，而是产生：

```text
自然
低频
低幅度
长时间不重复
身份稳定
```

的动态。

推荐的 idle motion distribution 包括：

```text
呼吸
眨眼
眼神移动
轻微头部移动
轻微身体重心变化
头发微动
衣物微动
极轻微表情变化
```

不应该默认包含：

```text
大幅转身
复杂手势
快速运动
场景切换
明显镜头运动
```

原因是待机状态越复杂：

```math
P(\mathrm{drift})
```

通常越高。

---

## 7. 当前生成后端

第一阶段优先验证现有 pretrained video generation model。

当前主要候选：

```text
SkyReels-V2
```

它负责：

```text
reference image
+
current generation context
+
idle prompt
        ↓
continuous video segments
```

第一阶段原则：

```text
不训练 foundation model
```

不是因为未来永远不训练，而是因为需要先回答：

> 现有模型的能力究竟在哪里失效？

如果连失效模式都没有确定，直接训练一个新模型没有明确优化目标。

---

## 8. 长时间生成问题

短视频生成成功并不等于数字人成功。

真正需要研究的是：

```math
T \rightarrow \text{minutes / hours}
```

之后：

```text
identity 是否仍一致
appearance 是否逐渐漂移
pose 是否累积畸变
motion 是否进入异常状态
generation context 是否逐渐污染
```

这也是本路线的主要研究问题。

---

## 9. State Preservation

状态保持分为三层。

### 9.1 Identity Preservation

保证：

```math
z_{\mathrm{id}}(t)
\approx
z_{\mathrm{id}}(0).
```

这里不是要求像素不变。

角色可以：

```text
转头
眨眼
说话
改变表情
```

但仍应该被认为是同一个角色。

---

### 9.2 Dynamic State Preservation

例如角色此时：

```text
头部向右转
眼睛正在闭合
身体略向前
```

下一次 generation window 不应该重新从：

```text
标准正脸静态图
```

开始。

需要保留：

```text
pose trajectory
expression trajectory
motion trajectory
```

从而实现：

```text
Segment_n
    ↓
state extraction / context continuation
    ↓
Segment_n+1
```

---

### 9.3 Context Preservation

对于窗口式视频模型：

```text
previous frames
→
next generation window
```

长期运行过程中必须控制 context。

不能无限保存全部视频。

因此需要研究：

```text
rolling context
keyframe context
anchor context
compressed state
```

等策略。

---

## 10. Anchor 在本路线中的作用

Anchor 同时服务于：

```text
State Preservation
```

和：

```text
State Recovery
```

但两种用途不同。

在本路线中，Anchor 的作用主要是提供长期身份约束。

初始版本：

```math
\mathcal A = \{A_0\}
```

其中：

```text
A₀ = 用户上传的原始角色图
```

之后可以扩展：

```math
\mathcal A =
\{
A_0,
A_1,
\ldots,
A_n
\}.
```

例如：

```text
A0 正脸
A1 左微侧
A2 右微侧
A3 微笑
A4 眨眼
A5 嘴微张
```

State Preservation 可以使用 Anchor 作为：

```text
identity reference
pose reference
appearance reference
```

而不是每次强行回到 Anchor。

真正的强制恢复由 Recovery Track 管理。

---

## 11. Anchor Contamination

自动扩充 Anchor Bank 有明显风险：

```text
轻微错误
↓
被加入 Anchor
↓
错误成为合法状态
↓
下一轮生成继续向错误方向漂移
```

最终形成：

```text
Anchor Contamination
```

因此第一阶段建议：

```text
禁止自动添加 Anchor
```

只允许：

```text
原始图
+
人工确认的少量 Anchor
```

后续如果需要自动加入，应通过严格检查。

---

## 12. 是否需要训练

项目初期采用：

```text
V0 = Zero Training
```

然后根据实验结果决定训练对象。

可能存在四个层次。

### V0 — Pretrained Model

完全使用已有模型。

目的：

```text
建立 baseline
确认 failure mode
收集长时间运行数据
```

### V1 — Prompt / Conditioning Optimization

不修改模型参数。

研究：

```text
prompt
reference frame
context window
generation schedule
seed strategy
```

对长期稳定性的影响。

### V2 — Adapter / LoRA

如果现有模型在动漫角色长期生成方面存在明确系统性缺陷，可以训练：

```text
anime idle LoRA
character consistency adapter
motion adapter
```

而不是直接训练完整模型。

### V3 — Avatar-Specific Model

只有当实验明确证明：

```text
foundation video model architecture
```

本身不适合 persistent avatar 时，再考虑更深层训练。

---

## 13. 数据集需求

如果进入训练阶段，需要的数据不是普通动漫视频集合。

更重要的是：

```text
same character
+
long temporal sequence
+
small motion
+
stable camera
+
stable appearance
```

可以分成：

```text
Idle Dataset
Expression Dataset
Pose Transition Dataset
Interaction Dataset
Recovery Dataset
```

其中 Idle Dataset 优先级最高。

---

## 14. 评估指标

本路线主要关注正常运行质量。

### Identity Stability

```math
D_{\mathrm{id}}(t)
```

### State Continuity

评估相邻生成窗口的：

```text
pose continuity
expression continuity
motion continuity
```

### Long-Horizon Survival

```math
T_{\mathrm{stable}}
```

表示不进入 Recovery 的平均持续时间。

### Motion Diversity

避免最终退化为一个短循环。

### Motion Naturalness

待机不能表现为：

```text
随机抖动
机械循环
突然停止
周期性复制
```

### Compute Cost

```math
C =
\frac{
\text{GPU seconds}
}{
\text{generated video seconds}
}.
```

---

## 15. 第一阶段实验

选择：

```text
20–50 个不同风格角色
```

分别生成：

```text
1 min
3 min
5 min
```

第一轮重点记录：

```text
identity stability
pose stability
motion continuity
generation window boundary
failure timestamp
GPU usage
VRAM usage
```

暂时不进行复杂 Recovery。

目的是先获得：

> 纯生成系统在没有外部修正时究竟能够稳定运行多久。

---

## 16. 阶段路线

### A0 — Single Image Animation

```text
Image
→
短时间自然待机
```

### A1 — Continuous Idle

```text
Image
→
连续多个 generation window
```

### A2 — Explicit AvatarState

建立统一状态结构。

### A3 — Long-Horizon Preservation

解决：

```text
context
identity
motion
pose
```

的长期连续问题。

### A4 — Training / Adaptation

根据真实 failure mode 决定是否需要：

```text
LoRA
Adapter
Fine-tuning
新模型
```

---

## 17. 本路线最终目标

最终希望得到：

```text
Reference Image
      ↓
Avatar Initialization
      ↓
Persistent AvatarState
      ↓
Generative Dynamics
      ↓
Long-Horizon State Preservation
      ↓
Continuous Avatar
```

数字人正常运行时，Recovery 系统应该尽量少介入。

这条路线最终追求的是：

```math
\boxed{
\text{让角色自己稳定运行，而不是依赖不断修复才能运行}
}
```