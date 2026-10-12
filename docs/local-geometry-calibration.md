# 单角色局部几何与离线统计校准

本流程实现单角色待机场景的局部几何、姿态参考、人工盲审、来源分组和 5% 秩校准。它不改变 Runtime、HTTP Observation 或现有 geometry_distance；不触发恢复。真实执行与数据缺口见 [研究报告](./reports/2026-10-12-local-geometry/README.md)。

## 数据与盲审

复用 [实验脚本](../scripts/observer_geometry_experiment.py) 的 `study` 子命令。生成器读取本地 GIF，固定裁剪、帧号与统计主样本；从可检测的原始帧派生眼距、鼻口、下巴的 5%／10%／20% 位移场。生成标签只说明发生了变换，不代表视觉异常；源片段及图片不提交。

```powershell
.tools/observer-env/Scripts/python.exe scripts/observer_geometry_experiment.py study prepare --gif var/observer-assets/anisora-portrait.gif --crop 700 320 1250 1000 --character anisora-portrait-01 --models models/observers --output var/local-geometry-study/dataset-v1
```

产物包括 manifest.json、splits.json、images/ 和自包含 review.html。盲审只显示匿名样本与标注控件，隐藏分数、变形参数和关键点叠图；默认标签为 uncertain。填写人工复核者标识并声明实际复核后导出 annotations.json。未复核、助手标签或仍有分歧的标签不能进入正式拟合。人工声明是一份可审计记录，程序不能证明声明人确实完成了复核。

导入时必须保留同一数据目录；旧清单不覆盖，可让不同复核者追加记录：

```powershell
uv run --locked python scripts/observer_geometry_experiment.py study review --manifest var/local-geometry-study/dataset-v1/manifest.json --annotations var/local-geometry-study/dataset-v1/annotations.json --output var/local-geometry-study/dataset-v1/reviewed.json
```

标注包含 normal / geometry_anomaly / uncertain，异常部位、严重度、姿态、idle / interaction / unknown 及遮挡。分歧保持 uncertain；复核导入项显式设置 `adjudication=true` 可保存人工裁决，必须已有人工原始记录。仅开发集的人工确认正常、显式 approve_anchor、非派生原图能成为参考。无标签时可以完成流水线，但所有正式统计结论为未验收。

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
