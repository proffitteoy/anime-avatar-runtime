# 开源调研与首版选型

核验日期：2026-09-27。依据是官方仓库的 README、依赖、许可和实际入口源码，以及 Hugging Face 模型元数据。本文区分上游提供的能力、本项目已接入的能力和后续工作；未进行跨模型画质或吞吐基准测试。

## 已确定的工程基线

| 层次 | 采用方案 | 依据与边界 |
| --- | --- | --- |
| 控制程序 | Python 3.12 + uv + FastAPI + Pydantic 2 + Uvicorn | 模型生态以 Python 为主；FastAPI/Pydantic 提供接口与校验，uv 固定依赖。无需引入第二种后端语言 |
| 开发工具 | pytest、Ruff、mypy、Hatchling；uv.lock | 已安装、执行检查并构建 wheel/sdist。锁文件记录实际版本 |
| 当前操作入口 | CLI + FastAPI 自带 OpenAPI 文档 | 初始化阶段可操作，不提前引入 React/Electron 或构建一套展示前端 |
| 状态存储 | 单角色、单进程、JSON checkpoint + 按 SHA-256 命名的参考图 | 当前没有关系查询或多用户需求；原子替换并校验参考图，暂不引入数据库和迁移 |
| 生成基线 | SkyReels-V2 DF 1.3B 540P，独立 Python 3.10 / PyTorch 2.5.1 环境 | 官方支持图生视频、视频续接、首尾帧控制；保持其原始实现，控制程序通过参数列表启动独立进程 |
| 观测接入顺序 | CCIP → anime-face-detector 0.1.0 → GMFlow | 分别覆盖身份、几何和运动，避免用一个 embedding 分数替代所有异常判定 |
| 恢复接入顺序 | 显式异常处理/基础 Reset → DRBA → ToonCrafter / AniSora | 先获得失效数据，再接入插帧和生成式回归；这些尚未进入自主闭环 |
| 交互 | 核心闭环后接 LLM/ASR/TTS | 只借鉴现有项目的交互组织，保持项目“不依赖 Live2D”的约束 |

## 生成与续接对比

| 项目 | 核验到的实现 | 环境、资源与许可 | 决策 |
| --- | --- | --- | --- |
| [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2/blob/9351d13152207cc04de780e055346b08ade0b851/README.md) | generate_video_df.py 实际包含 --image、--video_path、--end_image；调用 DiffusionForcingPipeline.extend_video | 官方测试 Python 3.10.12；torch 2.5.1、NumPy < 2、FlashAttention。官方 540P 1.3B 峰值参考约 14.7 GB，14B DF 约 51.2 GB；这是上游数据，不是本项目实测。[自定义社区许可](https://github.com/SkyworkAI/SkyReels-V2/blob/9351d13152207cc04de780e055346b08ade0b851/LICENSE.txt)，不是 Apache/MIT | 首版固定 DF 1.3B；已有官方命令桥接和参数对照测试，CUDA 推理未运行 |
| [SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3/blob/28c771e8456341be6a213e3d1133ed1fd19bf75d/README.md) | R2V 14B、V2V 14B、A2V 19B；single_shot_extension 支持指定 5–30 秒扩展 | README 推荐 Python 3.12+、CUDA 12.8+。本轮未完成其独立许可证正文核验，不推断沿用 V2 许可 | 作为后续升级候选；没有因版本号更新而直接替换更小的 V2 基线 |
| [AniSora](https://github.com/bilibili/Index-anisora/blob/6cdce3a17548d7ff0f2e05978469f134da25e68e/README.md) | V3.2 基于 Wan2.2；V3 系列支持任意帧条件；[V3 入口](https://github.com/bilibili/Index-anisora/blob/6cdce3a17548d7ff0f2e05978469f134da25e68e/anisoraV3/README.md) 为 generate-pi-i2v-any.py | V3 环境 Python 3.10，两份依赖清单及独立安装。仓库和模型卡为 Apache-2.0；README 的 12 GB 优化发行物与原生模型不能混同 | 动漫质量与受控恢复的后续对照；尚未接入 |
| [FramePack](https://github.com/lllyasviel/FramePack/blob/97fe5dbe06ac1f337ece08935b1076a35eefeeb9/README.md) | 以历史帧压缩处理长视频，官方 Gradio 入口 | 官方要求支持 fp16/bf16 的 NVIDIA GPU，宣称最低 6 GB；同页报告 RTX4090 约 1.5–2.5 秒/帧，不能据低显存宣称实时。仓库 Apache-2.0，底座权重许可需另核 | 低显存长视频备选；当前不引入其整套 UI 和模型链 |

选择 V2 DF 的原因是它与原方案的跨窗口续生成问题直接对应，且 1.3B 规格、源码入口与帧参数都可核对；这不是对“最佳画质”或“长期无漂移”的结论。

## Observer 与 Recovery

| 组件 | 固定依据 | 许可与集成注意 | 当前用途 |
| --- | --- | --- | --- |
| [deepghs/imgutils](https://github.com/deepghs/imgutils/blob/46df848dc4d20ac93f4919a40e2636d5f7c19766/imgutils/metrics/ccip.py) / CCIP | ccip_extract_feature、ccip_difference；默认 ccip-caformer-24-randaug-pruned | [库代码 MIT](https://github.com/deepghs/imgutils/blob/46df848dc4d20ac93f4919a40e2636d5f7c19766/LICENSE)，deepghs/ccip_onnx 模型卡标注 OpenRAIL；不能把库许可套到权重上。ONNX 路线支持先做 CPU 身份观测 | 已选身份模型与 API，下一步接入；其默认分类阈值不直接等同长期漂移阈值 |
| [hysts/anime-face-detector](https://github.com/hysts/anime-face-detector/blob/98a9fb480fa04bdd96acbe4d98da191a898267f3/README.md) | [pyproject 0.1.0](https://github.com/hysts/anime-face-detector/blob/98a9fb480fa04bdd96acbe4d98da191a898267f3/pyproject.toml) 要求 Python ≥ 3.12；28 点 HRNetV2 + YOLOv3/Faster R-CNN | 主代码 MIT，内含 Apache-2.0 派生代码；新版已去除 OpenMMLab 运行依赖，不能照旧版 mmcv 配置安装 | 已选几何模型；初期只评估近正脸适用范围，尚未接入 |
| [GMFlow](https://github.com/haofeixu/gmflow/blob/b5123431164d01ec14526a1c3d22218aecb62024/README.md) | 光流与双向一致性；官方 main.py 支持图像目录推理 | [代码 Apache-2.0](https://github.com/haofeixu/gmflow/blob/b5123431164d01ec14526a1c3d22218aecb62024/LICENSE)；官方基线 Python 3.8、Torch 1.9、CUDA 10.2，现代环境兼容需验证 | 已选运动信号来源；不能把光流幅值直接当异常 |
| [DRBA](https://github.com/routineLife1/DRBA/blob/d93e9128f4859df702d08aa9e860f1ea4691f483/README.md) | infer.py 支持 rife / gmfss / gmfss_union；输入输出视频与目标 FPS | [包装代码 MIT](https://github.com/routineLife1/DRBA/blob/d93e9128f4859df702d08aa9e860f1ea4691f483/LICENSE)，底层模型许可另核；CuPy 是上游可选加速 | 后续低成本时间连续性修复；不能修复身份语义错误 |
| [ToonCrafter](https://github.com/Doubiiu/ToonCrafter/blob/b0c47ff339c5e5ec45b84d0c6587850f242d41ef/README.md) | 两帧卡通插值，官方 512×320、最多 16 帧 | [代码 Apache-2.0](https://github.com/Doubiiu/ToonCrafter/blob/b0c47ff339c5e5ec45b84d0c6587850f242d41ef/LICENSE)；Python 3.8.5，官方约 24–27 GB，社区优化规格不等于原版。权重独立条款仍需核验 | 后续到 Anchor 的生成式过渡；成功率尚未验证 |

## 为什么不直接套现成数字人应用

核对了 [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/blob/992309c0aa19845960228f880013d4685fde93b5/README.md) 及其 [依赖清单](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/blob/992309c0aa19845960228f880013d4685fde93b5/pyproject.toml)。它已有 Python/FastAPI、语音与多模型接入，但视觉核心是 Live2D，不能直接满足本项目从完整角色图生成视频的目标。后续借鉴其语音中断和交互组织；本次不复制它的 UI、示例角色或整套依赖。其代码为 MIT，Live2D 示例素材另有许可。

## 可复现来源

下列为本轮查询到的仓库 HEAD；上文链接均固定到相应提交，不会随 main 漂移：

- SkyworkAI/SkyReels-V2：9351d13152207cc04de780e055346b08ade0b851
- bilibili/Index-anisora：6cdce3a17548d7ff0f2e05978469f134da25e68e
- deepghs/imgutils：46df848dc4d20ac93f4919a40e2636d5f7c19766
- hysts/anime-face-detector：98a9fb480fa04bdd96acbe4d98da191a898267f3
- haofeixu/gmflow：b5123431164d01ec14526a1c3d22218aecb62024
- Doubiiu/ToonCrafter：b0c47ff339c5e5ec45b84d0c6587850f242d41ef
- routineLife1/DRBA：d93e9128f4859df702d08aa9e860f1ea4691f483
- lllyasviel/FramePack：97fe5dbe06ac1f337ece08935b1076a35eefeeb9
- SkyworkAI/SkyReels-V3：28c771e8456341be6a213e3d1133ed1fd19bf75d
- Open-LLM-VTuber/Open-LLM-VTuber：992309c0aa19845960228f880013d4685fde93b5

模型元数据核验：

- [Skywork/SkyReels-V2-DF-1.3B-540P](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P/tree/1100111771ba2d921f10e76991f54db9f73edb3d)：revision 1100111771ba2d921f10e76991f54db9f73edb3d，license=other，非 gated。
- [deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx/tree/eb2acdd29af1703388d3d0c04221add322bc9110)：revision eb2acdd29af1703388d3d0c04221add322bc9110，license=openrail。
- [IndexTeam/Index-anisora](https://huggingface.co/IndexTeam/Index-anisora/tree/b134a8e677e4b22269827af7d596a4f2d9d3430a)：revision b134a8e677e4b22269827af7d596a4f2d9d3430a，license=apache-2.0。

许可证栏只报告本轮读取的文件/模型卡，不替未读取的底座模型、衍生权重或数据集作统一授权判断。

## 本轮核验范围

已完成公开源码和依赖核对、模型元数据核对、控制工程安装和验证。Git 直接 clone 遇到连接重置/超时；通过 GitHub 连接器读取官方源码，缓存了生成入口并核对 Git blob SHA（990ef610af1798b09aaba53d96201bc716f9d731），用于实际参数对照。没有把缓存入口冒充完整可运行的模型仓库。

模型权重未下载，CUDA 推理、Observer 模型加载、长期运行、恢复质量及吞吐未验证。下一步按 [模型环境说明](./backend-setup.md) 在目标 NVIDIA 机器完成生成基线，不阻塞当前控制工程的使用。

