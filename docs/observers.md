# 身份与几何观测（#13 / #14）

当前提供独立 CPU 观测进程、离线图片/视频分析和手动 HTTP 单帧观测。它们复用 `Observation`，保存身份、几何及表情残差；不自动调用 Recovery，不启动自主生成循环，不改写参考 checkpoint。运动观测仍不可用。结果解读与逐项验收见 [研究报告](./reports/2026-10-12-observers/README.md)，其中包含可提交的数值证据和图表，原始图片/视频仍留在忽略目录。

## 安装与固定来源

控制环境只新增 NumPy。可选 `observers` 依赖锁在同一份 `uv.lock`，安装到独立环境，避免与 SkyReels Python 3.10/CUDA 混装。以下为 Windows 命令；Linux 将 Python 路径换为 `.tools/observer-env/bin/python`。

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.tools/python'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $PWD '.tools/observer-env'
uv sync --locked --extra observers --no-dev
Remove-Item Env:UV_PROJECT_ENVIRONMENT
$env:HF_HOME = Join-Path $PWD '.cache/huggingface'
.tools/observer-env/Scripts/python.exe -m anime_avatar_runtime.observers prepare --models models/observers
```

只有 `prepare` 下载权重。推理通过本地路径加载，验证 `manifest.json` 的源码/权重 revision 和文件 SHA-256，核对已安装包的源码档案 URL。API 导入、服务启动、doctor 不下载或加载模型。可选环境固定 Torch 2.5.1 / torchvision 0.20.1 CPU；本版没有 GPU Observer 设置。

| 组件 | 固定来源 | 权重/代码许可 |
| --- | --- | --- |
| CCIP 代码 | deepghs/imgutils `46df848dc4d20ac93f4919a40e2636d5f7c19766` | MIT |
| CCIP 特征与距离模型 | deepghs/ccip_onnx `eb2acdd29af1703388d3d0c04221add322bc9110`，ccip-caformer-24-randaug-pruned | 模型卡 OpenRAIL，不能用库 MIT 替代 |
| 人脸检测代码 | hysts/anime-face-detector 0.1.0，`98a9fb480fa04bdd96acbe4d98da191a898267f3` | MIT；内含 Apache-2.0 派生代码 |
| YOLOv3 | hysts/anime-face-detector-yolov3 `afdd4226a79ae8bb81f334dbcffd34f8cc000c38` | 模型卡 MIT |
| HRNetV2 | hysts/anime-face-detector-hrnetv2 `9b3435248b26aeb82e2a8578fe9d86d5d57158af` | 模型卡 MIT |

`prepare` 保存上述模型卡。公开图片/视频的可访问性不代表其许可与模型代码相同；实验素材、裁剪、权重和结果仅存在忽略目录，不随源码分发。完整选型背景见 [开源选型](./open-source-selection.md)。

OpenCV 仅安装 `opencv-contrib-python-headless==4.11.0.86`，同时提供 contrib 和无 GUI 的 `cv2`。根目录 `[tool.uv].exclude-dependencies` 排除上游请求的 `opencv-python`、`opencv-python-headless` 与 `opencv-contrib-python`，避免它们重装同名文件。须使用仓库固定的 uv 0.11.5 和锁文件安装；普通 pip 不解释该配置。上游包的 metadata 仍要求原发行包名称，因此 `uv pip check` 会报告两个名称缺失；这不是已修改上游依赖的声明。此替代已用真实 CCIP/HRNet 推理验证，后续升级需重新核对。已有环境升级时可加 `--reinstall-package opencv-contrib-python-headless`，避免卸载旧 provider 后残缺的共享文件。

## 离线图片与视频

```powershell
uv run --locked avatar-runtime observe --python .tools/observer-env/Scripts/python.exe --anchor reference.png --image frame.png --output var/observations/still-01
uv run --locked avatar-runtime observe --python .tools/observer-env/Scripts/python.exe --anchor reference.png --anchor manually-approved.png --video clip.mp4 --sample-seconds 0.3 --max-samples 120 --output var/observations/video-01
```

每次使用新目录，失败目录也不能覆盖。Anchor 必须由调用者选定为原始/人工确认的参考图，程序不会提升采样帧为 Anchor。`--anchor` 可重复，每个信号分别取参考中的最小距离及匹配路径。多脸 Anchor 可记录身份候选，但最近身份参考为多脸时标记 ambiguous，不进入异常判定；缺少可对齐几何 Anchor 时身份候选仍可独立记录。Anchor embedding 和 detections 缓存在 `AVATAR_DATA_DIR/anchor-cache`，key 包含图片哈希、模型/源码 manifest 哈希、预处理版本和检测置信度；图片、模型变化时不复用旧缓存，损坏缓存重新提取。两套模型独立校验与加载，一套失败不阻止另一套输出；部分失败的 Anchor 不写入缓存，以免将缺测永久缓存。两套模型均不可用时运行失败。

视频按解码器报告的 FPS / frame count 抽样，时间为 `frame_index / FPS`，适用于可索引的恒定帧率视频。变帧率素材需先转为保留时间轴的恒定帧率版本；本版不宣称任意 VFR/流媒体的精确 PTS 支持。`max_samples` 限制采样时刻数，超过范围在结果中标记 `truncated`。无法解码的采样记录为 `frame_decode_failed`，哈希为源视频哈希；成功帧的哈希为传入模型的规范 RGB PNG 哈希。

产物：

| 文件 | 内容 |
| --- | --- |
| request.json / run.json | 配置、输入/Anchor 哈希、进程参数、时间、成功/失败及异常原因 |
| models.json / environment.json | 实际已核验源码/权重、许可、Python/平台/CPU 设备与依赖版本 |
| samples.jsonl | 帧序号、时间、测量时间、独立距离/状态、匹配 Anchor、耗时、bbox、28 点及原始置信度 |
| features/*.npy | 成功提取的 768 维 float32 CCIP 向量，不使用 pickle |
| identity.svg / identity_candidate.svg / geometry.svg | 身份可信距离、候选距离及几何独立曲线；缺测用红色 x 标记，不连线或填零 |
| overlays/*.png | bbox 与点编号；青色结构点、橙色表情点、红色低置信点 |
| result.json / backend.log | 最后样本、样本数、截断状态、进程日志 |

`run.json.elapsed_seconds` 包含子进程启动、模型加载、Anchor 准备、采样和写盘。`result.elapsed_seconds` 仅统计 Anchor 准备完成后的分析阶段；样本身份/几何耗时包含对应推理与距离计算，不含叠图和图片写盘。它们不是实时性能承诺。ONNX/PyTorch 各使用 2 个 CPU 计算线程；视频大小、采样上限、磁盘产物仍由调用者管理。

## 特征与几何计算

身份调用固定源码的 `ccip_extract_feature` / `ccip_difference`，保持上游 RGB、384×384 双线性预处理及 ONNX 距离模型。独立进程仅替换固定源码的两个内部模型加载函数，强制使用已校验本地 CPU sessions；不将距离替换为余弦/欧氏距离。升级源码必须重新核对这两个加载入口。

检测器需要 BGR，转换只发生在检测边界。保存 bbox、28 个 `(x, y, heatmap score)`，置信度原样记录，权重计算时裁剪至 [0,1]；这些分数不是概率。结构点为轮廓 0–4、眼角 11/13/17/19、鼻点 23；其他眉、眼睑和唇部点作为表情信号。

结构计算以两组点最小置信度为权重，要求至少 8/10 个可靠结构点；进行加权平移、正尺度和旋转 Procrustes 对齐，禁止镜像反射。结构距离为稳定点加权 RMS 残差除以参考脸加权 RMS 半径，是无量纲值。表情残差使用相同变换单独计算，不参与结构拟合，可靠表情点不足则为 null。

无脸、多脸、低置信/遮挡、缺少姿态点、侧脸代理检查失败、退化点集和没有可对齐 Anchor 均有明确状态。CCIP 始终独立尝试计算 `identity_candidate_distance`，不会因人脸失败而停止提取。`subject_status` 为 single_face_proxy / unverified / ambiguous，另有 face_count 与原因；只有单脸代理允许候选进入既有 `identity_distance` 和阈值判定。无脸/检测失败/多脸仍可有候选特征，但可信身份值为 null、身份判定为 unknown。单脸只能说明检出一张脸，不保证画面只有一个角色或已追踪到目标。

近正脸检查仅使用鼻点沿双眼连线的偏移和眼宽比例，是保守的启发式代理，不能保证检出所有侧脸、遮挡或多人全身图，也可能拒绝有效画面。本版没有稳定的人脸跟踪或多人身份分配。鲁棒 IRLS 与局部残差只在 [对照实验脚本](../scripts/observer_geometry_experiment.py) 中比较，Runtime 仍使用原有置信度加权最小二乘。

## HTTP 与异常接口

设置 `AVATAR_OBSERVER_PYTHON` 为上述独立解释器，`AVATAR_OBSERVER_MODELS=models/observers`，按常规命令启动服务。初始化的原图为 HTTP 观测的唯一可信 Anchor。

```powershell
$env:AVATAR_OBSERVER_PYTHON = Join-Path $PWD '.tools/observer-env/Scripts/python.exe'
uv run --locked avatar-runtime serve
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/observer/observe' -ContentType image/png -InFile frame.png
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/observer/status'
```

`POST /observer/observe` 接收 10 MiB / 1600 万像素以内的原始单帧图片，返回 `ObservationResult`；未初始化 404、与活动生成/观测冲突或结果已取消/过期 409、超限 413、非法图/阈值 422、独立环境或模型不可用 503。运行在线程中等待独立进程，GET 状态/health 可继续访问；任务有 `AVATAR_OBSERVATION_TIMEOUT_SECONDS` 上限（默认 600）。Runtime 短时间持锁注册任务，锁外准备图片及执行模型，重新持锁校验 run/avatar、参考哈希、取消信号与 `state_revision` 后接收。

pause/close 主动发出独立的观测取消信号，锁外等待拥有的进程清理后才返回。进入停止阶段即拒绝新提交；resume 实际改变模式时递增修订并取消旧观测。请求调用方断开/取消也等待清理，迟到结果不会更新 observer/status。`runtime/status` 新增 observation_active 与 state_revision；已接收测量标注 state_revision，离线结果无 Runtime 修订号。磁盘后端产物是执行记录，不等于 Runtime 已接收，仍不提供后台观测任务或进度流。

`GET /observer/status` 返回本进程最后一次已接收的测量；重启后为未知，磁盘实验仍保留。`GET /runtime/status` 的 `offline_observers_configured` 只说明配置了路径，`observers_enabled=[]` 仍表示自主循环中没有启用 Observer，不等于模型已健康加载。

`POST /observer/anomalies` 是不加载模型的纯判定接口，输入独立距离和显式阈值：

```json
{"identity_distance":0.05,"geometry_distance":0.18,"identity_threshold":0.2,"geometry_threshold":0.1}
```

上述数值仅为接口示例，未校准。身份/几何各返回 exceeded / within / unknown；任一超阈值则整体 anomaly，全部有测量和阈值且未超限才是 within_thresholds，否则 unknown。等于阈值视为 within。HTTP 观测可通过查询参数、CLI 可通过同名参数显式提供阈值；默认不设置阈值，不继承 CCIP 分类阈值，不把未知解释为正常。

没有实现 #17 的滞回、持续时间约束和恢复状态机。Observer 不暂停/重启生成器；Runtime 仅接收手动观测，记录 `observation_recorded` 事件，后续 #16/#17 再聚合并决策。运动指标继续为 null；不把当前范围写成自主闭环或长期稳定性验收。

## 本次实际验证

下面保留 `e181817` 的初始实测；后续身份解耦、单 OpenCV 环境、真实取消和局部几何/表情对照见 [PR #38 修订报告](./reports/2026-10-12-observers/revision-p0.md)。旧记录的空白/多脸身份 null 指可信身份；修订后这些输入仍可提取候选距离。

2026-10-12，在 Windows / Python 3.12.13、固定 CPU 模型上使用现有动漫图片及公开 AniSora 演示素材运行。实验参考是离线显式选定的输入，不自动加入 Runtime 的可信 Anchor Bank。原始素材、派生图、日志、特征和叠图保留于 `var/observer-assets`、`var/observer-experiments`，不提交到仓库。

| 现有图样例 | CCIP 距离 | 结构残差 | 说明 |
| --- | --- | --- | --- |
| 原图自身 | 约 0.000000294 | 0 | 真实模型输出，有浮点误差 |
| 同图旋转 12° | 0.01369 | 0.01435 | 检测噪声后仍有小残差，不宣称绝对零 |
| 同图横向拉宽 65% | 0.00001095 | 0.18756 | 身份接近仍存在显著几何改变 |
| 不同角色 | 0.38559 | null | 本例被近正脸检查拒绝，未填几何正常值 |
| 空白 / 多脸拼图 | null | null | 明确缺测，保留可回看叠图 |

公开视频来源为 [AniSora 固定版本 README](https://github.com/bilibili/Index-anisora/blob/6cdce3a17548d7ff0f2e05978469f134da25e68e/README.md) 的 [公开 MP4 演示](https://github.com/user-attachments/assets/a207d43e-26f6-445b-883e-9b28c129607f)，以及同提交的 `assets/video_sr_video_3.gif`。GIF 转为 100 FPS MP4，以 10 ms 时基重复对应帧保留原始 duration；对后者固定裁剪 `(700,320)-(1250,1000)` 为单人近景，裁剪和参考选择需随源素材一起记录。未将这些公开视频称为本机 SkyReels 推理结果。

实际 HTTP 调用返回 200，拉宽图身份在显式示例阈值内、几何超限；`/observer/status` 可读回实际数值，参考 checkpoint 哈希保持不变。首次近景实验的首帧漏检，几何全段 unavailable，这一失败记录仍保留；对照实验另选输入视频的 0.6 秒帧作为离线参考，不覆盖首次记录。

| 视频实验 | 样本与有效性 | 实测距离 |
| --- | --- | --- |
| 约 90.47 秒公开 MP4，每 5 秒采样 | 19 个时刻；9 无脸、7 多脸、2 几何有效、1 低置信 | 有效身份 0.29154–0.32172；有效几何 0.18443–0.25984。本片角色与参考图不同，示例阈值仅验证异常接口 |
| 3.56 秒公开近景动图转码，每 0.3 秒采样，0.6 秒输入帧为参考 | 12 个时刻；8 几何有效、1 无脸、1 姿态不支持、2 姿态点置信度不足 | 有效身份约 0–0.07638；有效几何约 0–0.15463。同角色片段也有明显几何波动，尚未逐帧标注正常/异常，不能据此设置普适阈值 |

两次视频运行启动/加载/分析总耗时分别约 76.64 和 83.91 秒，包含同期其他验证负载，不能作为串行性能基准。CPU 与线程配置见各自 environment.json。首次近景约 39.36 秒但所有几何样本缺测，也保留为失败范围，不以更快耗时评价模型质量。

自动测试覆盖相似变换、结构畸变、反射、表情、置信度、退化点、独立缺测、缓存失效、HTTP 和失败产物；控制测试使用明确替身，真实模型证据来自上述独立实验。GPU 生成、真实 SkyReels 长序列、阈值误报/漏报率、跨镜头/侧脸质量、VFR PTS 和自主恢复未验收；#13/#14 的真实生成端集成验收仍需目标环境输入。远端 CI 与 Issue 状态没有因本地实现自动变更。

真实最小推理测试为显式 opt-in；给定配置后缺少模型或检测失败会 fail，不会悄悄 skip。参考图片须是已知近正脸且可输出 28 点的输入：

后续离线研究提供 [单角色局部几何、姿态参考与统计校准](./local-geometry-calibration.md)，采用人工盲审、来源分组和独立验证；它不修改本页的 HTTP 判定与 Runtime 几何计算。

```powershell
$env:OBSERVER_TEST_PYTHON = Join-Path $PWD '.tools/observer-env/Scripts/python.exe'
$env:OBSERVER_TEST_MODELS = Join-Path $PWD 'models/observers'
$env:OBSERVER_TEST_IMAGE = Join-Path $PWD 'reference.png'
uv run --locked pytest tests/test_observers.py -k real_observer
```
