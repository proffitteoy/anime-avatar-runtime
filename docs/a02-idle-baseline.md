# A02 单图自然待机配置与首段动画基线

> 对应 issue：#7（Track A · M0 生成基线，P0）。前置：#6（真实推理已打通）。

## 1. 可复现 idle 配置

### 1.1 prompt / negative prompt

代码里已固化为默认值（`src/anime_avatar_runtime/skyreels.py`）：

- `DEFAULT_PROMPT`（idle 正向）：
  > An anime character standing still and idling naturally: slow gentle breathing,
  > occasional soft blinks, natural gaze with subtle eye shifts, very slight head sway,
  > light hair and clothing movement from breathing, consistent facial identity and outfit,
  > static camera, stable background, no turning, no large gestures.

- `DEFAULT_NEGATIVE_PROMPT`（idle 负向，用于压制默认限制项）：
  > turning around, spinning, large body movement, fast motion, running, jumping,
  > complex hand gesture, camera pan, camera zoom, camera shake, scene change,
  > background change, flickering, jitter, shaking, looping, repeated motion,
  > warped face, deformed face, extra fingers, blurred, low quality, watermark, text, oversaturated

**默认限制项**（必须在 baseline 中压制，靠 negative prompt + 输入约束实现）：

| 限制 | 说明 |
| --- | --- |
| 转身 / 大幅肢体动作 | 禁止 turning/spinning/large body movement |
| 复杂手势 | 禁止 complex hand gesture / extra fingers |
| 快速动作 | 禁止 fast motion / running / jumping |
| 镜头移动 | 禁止 camera pan/zoom/shake |
| 场景切换 | 禁止 scene/background change |

### 1.2 采样参数（默认，可经 CLI 覆盖）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| frames | 97 | 4n+1，首段 ~4s @24fps |
| inference_steps | 30 | 完整基线；冒烟用 1 |
| guidance_scale | 5.0 | |
| shift | 5.0 | |
| fps | 24 | |
| resolution | 540P（544×960） | |
| seed | 固定集合（42 / 1 / 7 / 2024…） | 每个输入跑固定 seed |

### 1.3 8GB 显存必开项

- **`QUANTIZE_INT8=1`**：transformer 权重 int8 量化（`vendor/SkyReels-V2` 的
  `get_transformer` 里门控）。不开会 bf16 权重 + 峰值 7.1GB 触发 WDDM 换页，
  FFN 每 block 从 ~0.2s 涨回 3–7s。
- `--offload`：VAE 懒加载/卸载。

## 2. 首段质量评审表（人工评审）

对每段首段输出，按下列维度打分（1–5），并记录失败样例（**失败样例不得剔除**）。

| 维度 | 1（差） | 3（可用） | 5（好） | 评分 |
| --- | --- | --- | --- | --- |
| 身份一致性 | 人脸/发型明显漂移 | 偶发轻微偏差 | 全程稳定 | |
| 衣饰一致性 | 服装结构崩坏 | 轻微纹理漂移 | 全程稳定 | |
| 脸部结构 | 明显畸形/五官错位 | 偶发局部变形 | 结构完整自然 | |
| 自然度 | 机械感/木偶感 | 大体自然 | 呼吸眨眼视线自然 | |
| 短循环倾向 | 明显重复循环 | 偶发周期感 | 无明显循环 | |

### 2.1 三类动作必须可区分（评审核心）

| 类别 | 判定特征 |
| --- | --- |
| **自然运动** | 呼吸起伏、眨眼、视线微移、轻微头/发/衣摆动，幅度小且低频 |
| **静止** | 画面几乎不动（应判为不达标，非"稳定"） |
| **抖动/闪烁** | 帧间无规律高频跳变 |
| **重复循环** | 可辨识的周期性来回（如呼吸用同一帧对折） |

### 2.2 输入风格适用范围

对每个输入记录画风、构图、是否含复杂背景/多人，判断 idle 配置是否适用；不适用的输入单独标注，不作为"失败"掩盖。

## 3. 运行记录（逐 run 填）

| 输入 | sha256 | seed | frames | steps | 产物路径 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| priority.png | `d4d2a6dd0d7fdcb38f34ceede8bb213b0a2b584baea4a8dfaf295cbdfff3b414` | 42 | 97 | 30 | `var/runs/a02-idle-priority/` | 竖图 544×960 |
| test1.jpg | `10dab4073738a36bc695fbaf3574a2ff6f425342f1ad907409e45bf13cd11c06` | 1 | 97 | 30 | `var/runs/a02-idle-test1/` | 横图 960×544 |
| wechat_a.png | `909260b02b68e62b8546407947ae4c6f963f7d7e54af26e036181c597039cb1e` | 7 | 97 | 30 | `var/runs/a02-idle-wechat_a/` | 横图 960×544 |
| wechat_b.jpg | `729e705bbc4c169f132644e0bb72f6badeff597de249db626e938c346392c4e4` | 2024 | 97 | 30 | `var/runs/a02-idle-wechat_b/` | 横图 960×544 |

实测（540P、97 帧、30 步、`QUANTIZE_INT8=1`、`--offload`、种子固定集合 42/1/7/2024）：

- 4/4 成功，均可解码播放（24fps、97 帧、4.04s）。
- 峰值显存 ~5.83GB（去噪，`[DEBUG-BLOCK]` 采样），解码 ~62s。
- 单段墙钟 ~1932–2198s（含模型加载 5–8min + 去噪 ~50s/步 + 解码）。
- 冒烟 1 步与全量 30 步共用同一 prompt / negative prompt / 采样配置。

> 说明：4 段全量 run 的 `run.json` 由早期 `skyreels.py` 生成，不含后补的 `quantize_int8` 字段；int8 由运行脚本 `var/run_idle_full.py` 显式 `QUANTIZE_INT8=1` 启用。此后新增的 run 会在 manifest 记录 `quantize_int8` 与 vendor 补丁指纹。

## 4. 结论边界（验收要求）

- 首段成功 ≠ 长期稳定 / 实时性；本任务**不**引入训练或复杂恢复作为前置。
- 所有产物可通过 `run.json` 追溯：输入 sha256、模型 revision、seed、argv 全参数。
- 评审表（第 2 节）为人工评审框架；本记录只落事实，不替代逐段视觉打分。
