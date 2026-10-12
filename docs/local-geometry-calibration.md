# 单角色局部几何与离线统计校准

本流程实现单角色待机场景的局部几何、姿态参考、人工盲审、来源分组和 5% 秩校准。它不改变 Runtime、HTTP Observation 或现有 geometry_distance；不触发恢复。真实执行与数据缺口见 [研究报告](./reports/2026-10-12-local-geometry/README.md)。

当前 dataset-v1 的 GIF 裁剪经视觉复核，有明显网格抖色、彩色条带和构图变化，已降级为流水线回归／低质量输入压力测试。不要继续为它补标以完成正式训练或校准；以下命令保留为该历史实验的复现入口。开始新主实验前，应先筛选清晰正脸和同角色动作来源，逐段检查五官完整性、压缩痕迹、运动模糊与裁剪范围，再生成变形并复核变形是否连带扭曲头发／背景。图像质检须先于正式数据冻结，不得按模型分数事后挑选“正常”统计主样本。

## 数据与盲审

复用 [实验脚本](../scripts/observer_geometry_experiment.py) 的 `study` 子命令。新一轮已用清晰正脸原图重做为 dataset-front-v1：1 张原图、±8° 旋转和0.9倍缩放三个相似变换对照、眼距／鼻口／下巴各三档变形，共13张。该批人工盲测已完成（正常9、畸变4、不确定0），正常原图获准作为参考；最新对照见 [报告第八节](./reports/2026-10-12-local-geometry/README.md#8-完成人工盲测后的局部几何对照)。旧标签没有迁移。原图与全部派生图仍只有一个独立来源，旋转／缩放也不冒充真实头部姿态或正常表情动作。

```powershell
.tools/observer-env/Scripts/python.exe scripts/observer_geometry_experiment.py study prepare --image var/observer-assets/public-anchor.png --character public-frontal-01 --models models/observers --similarity-controls --compact-warps --output var/local-geometry-study/dataset-front-v1
```

`--image` 与 `--gif` 二选一；静态图默认使用整张图和预选第0帧，可用 `--crop` 指定图内裁剪。`--compact-warps` 使用有限椭圆范围的逆向位移，范围外映射为原坐标；范围内仍可能改变邻近发梢／皮肤，需人工判断。三个部位的椭圆半径按双眼中心距离计，依次为眼部 `(0.55,0.16)`、鼻口 `(0.5,0.42)`、下巴 `(0.65,0.35)`；位移权重为 `max(1-r²,0)²`。5%／10%／20% 是机械位移幅度，不能直接当作视觉异常程度。

旧 GIF 的历史复现入口如下。生成器固定裁剪、帧号与统计主样本，从可检测的原始帧派生变形；源片段及图片不提交。

```powershell
.tools/observer-env/Scripts/python.exe scripts/observer_geometry_experiment.py study prepare --gif var/observer-assets/anisora-portrait.gif --crop 700 320 1250 1000 --character anisora-portrait-01 --models models/observers --output var/local-geometry-study/dataset-v1
```

产物包括 manifest.json、splits.json、images/ 和自包含 review.html。盲审使用完整尺寸无损PNG，点击可放大，只显示匿名样本与中文控件，隐藏分数、变形参数和关键点叠图；新样本分类初始为“请选择分类”，逐张明确选择后才能导出，无法判断时可选 uncertain。页面顶部汇总分类、未知字段及与分类冲突的参考勾选。共同姿态／模式／遮挡可由复核者明确批量选择，分类、程度和异常部位仍逐张填写。填写人工复核者标识并声明实际复核后导出 annotations.json。未复核、助手标签或仍有分歧的标签不能进入正式拟合。人工声明是一份可审计记录，程序不能证明声明人确实完成了复核。

| 字段 | 页面含义与填写规则 |
| --- | --- |
| label | normal 正常（包含正常眨眼、张嘴、轻微转头）；geometry_anomaly 五官错位、比例或脸形不自然；uncertain 无法判断 |
| pose | front 正脸；left_turn / right_turn 向画面左／右侧转头；down / up 低头／抬头；unsupported 大角度侧脸等超出范围；unknown 无法判断。左右按画面方向，侧倾无需另设三维角度 |
| mode | idle 待机／轻微动作；interaction 明确互动场景；unknown 无法判断 |
| occlusion | none 无遮挡；mild 少量遮挡；severe 关键五官严重遮挡；unknown 无法判断 |
| severity | none 无畸变；mild / moderate / severe 轻微／中等／严重畸变；unknown 无法判断。描述畸变程度，不是转头或张嘴幅度 |
| parts | 畸变时勾选画面左／右眼、双眼关系、鼻、嘴或轮廓；正常图不勾异常部位 |
| approve_anchor | 允许作为正常参考，不能代替 label；程序只接受开发集人工确认正常的非派生原图 |

导入时必须保留同一数据目录；旧清单不覆盖，可让不同复核者追加记录：

```powershell
uv run --locked python scripts/observer_geometry_experiment.py study review --manifest var/local-geometry-study/dataset-v1/manifest.json --annotations var/local-geometry-study/dataset-v1/annotations.json --output var/local-geometry-study/dataset-v1/reviewed.json
```

标注包含 normal / geometry_anomaly / uncertain，异常部位、严重度、姿态、idle / interaction / unknown 及遮挡。分歧保持 uncertain；复核导入项显式设置 `adjudication=true` 可保存人工裁决，必须已有人工原始记录。仅开发集的人工确认正常、显式 approve_anchor、非派生原图能成为参考。无标签时可以完成流水线，但所有正式统计结论为未验收。

补填时可从已导入清单生成新中文页面，保留已填写字段与勾选，不加载模型、不推断新标签，也不覆盖旧页面：

```powershell
uv run --locked python scripts/observer_geometry_experiment.py study review-page --manifest var/local-geometry-study/dataset-v1/reviewed.json --output var/local-geometry-study/dataset-v1/review-cn.html
```

页面尝试在当前浏览器按数据哈希暂存字段，并只恢复同一批数据的进度；恢复时仍须重新勾选人工声明。浏览器禁用存储时显示提示，不阻止导出。完成后必须导出文件，浏览器暂存不能代替版本化记录。同一复核者更正未完成的首版标注时，保留原导出文件，将完整修订导出导入原 manifest.json 并保存为新的 reviewed 文件；不要把同一人的草稿与修订当成两个独立意见叠加，否则原本的 unknown 字段会形成分歧。不同复核者的独立意见仍使用追加／裁决流程。

通用输入采用 `StudyManifest`，一个清单只能包含一个 character_id。多来源数据应使用相同结构：每条样本有 sample_id、image、input_sha256、source_id、原始来源 source_sha256、parent_id、预先选定的 primary、帧号/时刻、transform 和 reviews。image 可相对清单或为绝对路径。相同作品集数／原片段／生成运行归为同一个 source_id；同一源哈希不得伪装多个独立来源。派生样本必须继承来源，每个来源最多一个原始统计主样本。

按来源而不是帧随机划分，固定种子 42，约 60% 开发、20% 校准、20% 验证；少量来源按整数取整，单来源全部归开发。同源裁剪与变形不会跨集合；增加来源需要重新冻结整个数据版本。统计主样本应在查看模型分数前固定，不能事后挑“模型能测”的帧。

## 提取与诊断

模型环境沿用 [observers.md](./observers.md)，每次离线提取在一个独立环境进程复用已加载的 CCIP／28 点模型，保存模型来源、原始置信度、失败与耗时；无脸也独立尝试 CCIP。基础控制环境不新增模型依赖。

```powershell
.tools/observer-env/Scripts/python.exe scripts/observer_geometry_experiment.py study extract --manifest var/local-geometry-study/dataset-v1/reviewed.json --models models/observers --output var/local-geometry-study/extraction-v1
uv run --locked python scripts/observer_geometry_experiment.py study explore --manifest var/local-geometry-study/dataset-v1/reviewed.json --extraction var/local-geometry-study/extraction-v1/extraction.json --reference s0001 --output var/local-geometry-study/diagnostics-v1
```

尚未人工复核时改用 manifest.json。`explore` 必须显式选原始参考，输出带 diagnostic_only 与 not_human_ground_truth；它不产生校准、可信 Anchor 或异常真值。公开图表与盲审需要由不同查看流程使用，复核者不应预先阅读诊断分数。

| 指标 | 点索引 | 评分与局部辅助量 |
| --- | --- | --- |
| 左／右眼 | 11/13、17/19 | 眼角 RMS、宽度误差；12/15、18/21 开合单独保存 |
| 双眼关系 | 11/13/17/19 | 眼角 RMS、双眼中心关系误差 |
| 鼻 | 23 | 鼻点位移 RMS |
| 嘴 | 24/26 | 嘴角 RMS、宽度误差；25/27 张嘴高度独立保存 |
| 轮廓 | 0–4 | 轮廓 RMS，至少 3 点且必须有下巴点 2 |

每个部位从现有 10 个结构拟合点中排除该部位，至少 4 个且不少于剩余点 75% 可靠；拟合必须非退化、正尺度且禁止反射。参考尺度来自参考结构点，评分使用原始最小置信度权重；下限仍为 0.3，不用 Huber 降权剔除异常。眼角、鼻、嘴评分要求对应目标点全部可靠；失败只影响对应部位。现有检测器没有瞳孔点，不虚构视线估计。

## 冻结、校准与独立评估

```powershell
uv run --locked python scripts/observer_geometry_experiment.py study freeze --manifest var/local-geometry-study/dataset-v1/reviewed.json --extraction var/local-geometry-study/extraction-v1/extraction.json --candidate pose_local --output var/local-geometry-study/frozen-v1
uv run --locked python scripts/observer_geometry_experiment.py study calibrate --manifest var/local-geometry-study/dataset-v1/reviewed.json --extraction var/local-geometry-study/extraction-v1/extraction.json --frozen var/local-geometry-study/frozen-v1/frozen.json --output var/local-geometry-study/calibration-v1
uv run --locked python scripts/observer_geometry_experiment.py study evaluate --manifest var/local-geometry-study/dataset-v1/reviewed.json --extraction var/local-geometry-study/extraction-v1/extraction.json --frozen var/local-geometry-study/frozen-v1/frozen.json --calibration var/local-geometry-study/calibration-v1/calibration.json --output var/local-geometry-study/evaluation-v1
```

所有输出目录必须不存在。输入图片、标签、模型、实现或配置变化使后续绑定失效；需要保留旧版本并重新冻结／校准，不能修改旧阈值文件。

冻结阶段按 front / left_turn / right_turn / down / up 选择人工确认正常原图的代表形状（结构对齐距离的 medoid）。用不包含参考来源的正常开发主样本拟合逐部位姿态门限：正确匹配误差 95% 分位数、最优次优差额 5% 分位数，至少 20 个独立来源。姿态候选比较不包含当前被检测部位；仅一个姿态参考、数据不足、歧义和越界都不输出可信姿态。各有效部位匹配姿态不一致时联合判定 unknown，仍保存原始局部残差和匹配原因。

固定输出四个版本：生产 WLS、单正脸参考的局部对齐 single_local、自动姿态参考 pose_local、人工姿态 oracle。前两者使用一个固定正脸参考与正脸配置，按混合动作数据校准；自动版本由匹配结果选择姿态配置，只有 oracle 使用人工真值姿态路由。oracle 仅诊断，不参与采用。候选由 freeze --candidate 在校准前选定，默认 pose_local，不能根据验证集重新挑选。

正常开发主样本拟合 median／1.4826 MAD，至少 20 个独立来源；尺度下限 1e-6 并记录 mad_floored。以非负标准化局部残差的最大值形成几何辅助统计量，不混入身份/运动，不代替原始局部输出。每姿态配置至少 39 个独立正常校准主样本，排序位置 `ceil((n+1)*0.95)`，严格大于才标候选异常。稀疏部位、未知有效性组合或姿态歧义为 unknown；仅部分部位未超限时整体仍未知。

秩校准要求同分布和近独立，不能保证每个罕见缺测子组误报均为 5%。报告必须展示有效性组合与分层表现，输入分布变化要重新评估。代码中的 statistical_flag 表达支持观测上的阈值事件；decision 表达整张脸几何结论，两者分别报告，避免把部分部位未超限写成整脸正常。

验证仅使用验证来源、人工确认的 idle、受支持姿态及非严重遮挡数据。误报率分母为有可用统计判定的正常主样本；完整判定覆盖率的分母为所有符合范围的正常主样本。异常 unknown 计未检出；按姿态、部位、人工严重度、机械变形强度及合成／真实分别统计。Clopper–Pearson 单侧 95% 上界：59 个独立有效正常观测、0 误报时约 4.95%；不能用相邻帧替代这些来源。

采用要求误报上界 ≤5%、正常完整判定覆盖率 ≥80%，WLS 也经过同目标校准，且人工确认畸变上的配对来源 bootstrap 区间下界大于 0。bootstrap 固定种子 42、2000 次；合成与真实畸变的区间和检出率分别报告；合成数据上的通过不能宣称自然生成异常已验收。缺数据时明确 not_accepted；任何通过也仅为离线范围，不自动启用 Runtime 或恢复。
