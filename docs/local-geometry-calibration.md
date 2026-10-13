# 单角色局部几何与离线统计校准

本流程实现单角色待机场景的局部几何、姿态参考、人工盲审、来源分组和 5% 秩校准。它不改变 Runtime、HTTP Observation 或现有 geometry_distance；不触发恢复。真实执行与数据缺口见 [研究报告](./reports/2026-10-12-local-geometry/README.md)。

当前 dataset-v1 的 GIF 裁剪经视觉复核，有明显网格抖色、彩色条带和构图变化，已降级为流水线回归／低质量输入压力测试。不要继续为它补标以完成正式训练或校准；以下命令保留为该历史实验的复现入口。开始新主实验前，应先筛选清晰正脸和同角色动作来源，逐段检查五官完整性、压缩痕迹、运动模糊与裁剪范围，再生成变形并复核变形是否连带扭曲头发／背景。图像质检须先于正式数据冻结，不得按模型分数事后挑选“正常”统计主样本。

## 数据与盲审

复用 [实验脚本](../scripts/observer_geometry_experiment.py) 的 `study` 子命令。新一轮已用清晰正脸原图重做为 dataset-front-v1：1 张原图、±8° 旋转和0.9倍缩放三个相似变换对照、眼距／鼻口／下巴各三档变形，共13张。该批人工盲测已完成（正常9、畸变4、不确定0），正常原图获准作为参考；最新对照见 [报告第八节](./reports/2026-10-12-local-geometry/README.md#8-完成人工盲测后的局部几何对照)。旧标签没有迁移。原图与全部派生图仍只有一个独立来源，旋转／缩放也不冒充真实头部姿态或正常表情动作。

本轮另取得 [CC0 作者表情包](https://opengameart.org/content/headmugshot-packs-chibi-emotions) 的三个角色、23张正脸表情，以及 [ExpressEdit](https://huggingface.co/datasets/JiashengGuo/ExpressEdit/tree/85ea6cb43553c705a9301630985279b917190ee3) 固定版本 `85ea6cb43553c705a9301630985279b917190ee3` 的两个角色、10张扩散表情编辑结果，已完成真实CPU提取及诊断。各角色只与自己的参考候选配对；作者表情意图和生成标签只作弱标记，不是人工正常／异常真值。新增两批尚无人工复核标签或可信Anchor，ExpressEdit的无表情编辑结果也不能冒充未公开的原始图。重复表情和随机种子不能增加独立来源数，不能据此报告误报率或正式采用结论。新结果、鼻点缺测与表达域重叠见 [报告第十四节](./reports/2026-10-12-local-geometry/README.md#14-2026-10-13正脸表达域续实验与审查修复)。

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

## 拟合耦合机制实验

`coupling` 阶段复用已提取的真实关键点，不加载模型。需要人工确认正常、获准作为参考的非派生原图；输入图片、样本和提取记录仍逐项核对哈希。输出目录必须不存在：

```powershell
uv run --locked python scripts/observer_geometry_experiment.py study coupling --manifest var/local-geometry-study/dataset-front-v1/reviewed-human-v1.json --extraction var/local-geometry-study/extraction-front-v1/extraction.json --reference s0000 --output var/local-geometry-study/coupling-repeat-v1
```

固定比较原结构点排除被测部位、再去掉鼻点、添加嘴角后排除被测部位三种拟合。每种都禁止被测点参与自身拟合。固定置信度权重和有效点集合，在参考位置线性化；用56个坐标方向的中心差分、准确相似变换与只移动下巴的控制检查。保存六组局部传播增益、理想各向同性噪声敏感性、13例实际重拟合与线性预测误差，以及下巴／两侧轮廓分项。

`coupling.json` 的数字为开发诊断，不输出异常阈值或采用结论；双眼与左右眼区域有重叠，增益矩阵不是互斥部位分解。固定权重的导数不覆盖置信度变化、缺测切换、重新检测或大幅姿态变化。噪声敏感性假设坐标独立同方差，不表示检测器的真实误差分布。新增拟合方案不接入 `single_local` / `pose_local` 或已有冻结产物；若后续成为候选，需重新声明、冻结并独立验证。实测结果见 [报告第九节](./reports/2026-10-12-local-geometry/README.md#9-拟合耦合机制与拟合点对照实验)。

## 局部形状稳定性与检测器等变诊断

按2026-10-13确认的优先级，人工盲审暂缓，先分析稳定性，不继续增加合成时序回放。新增`study stability`固定比较现有全局WLS、补集轮廓拟合、六维端点坐标与五点等权保向Procrustes；后者只是研究对照，不加入冻结候选或Runtime。前三种复用原定义，Procrustes使用五点RMS／参考轮廓宽；原始数值尺度不同，比较噪声响应与最小目标响应之比。

`stability`固定置信度及0.3有效集合，解析推导端点雅可比与相似变换切空间投影，用两个步长、56坐标方向复核。目标类限定为五点形状中正交于相似变换切空间的全部单位方向；理想独立／端点／内部／共同端点噪声因子不是真实检测器协方差。结果同时保留相对其他部位的轮廓等比例大小盲区，不能仅凭更小分数宣布异常检测改进。

`repeatability`在固定模型CPU环境执行三种已知图像探针：整数平移(+3,-2)、+4°旋转、0.95倍取整缩放，再按像素中心约定逆变换检测点。每张与自身原提取比较；误差混合检测、参考、重采样及坐标约定，不是定位真值或随机噪声标定。全部点与失败留在私有记录；不可靠端点宽度不用于归一化误差。三层分别记录可靠点、原始量可计算性和未经校准的异常判断，生产姿态门单独报告。

```powershell
uv run --locked python scripts/observer_geometry_experiment.py study stability --manifest var/local-geometry-study/dataset-front-v1/reviewed-human-v1.json --extraction var/local-geometry-study/extraction-front-v1/extraction.json --reference s0000 --output var/local-geometry-study/stability-repeat
.tools/observer-env/Scripts/python.exe scripts/observer_geometry_experiment.py study repeatability --manifest var/research-continuation/frontal-emotions-v1/aranel/dataset/manifest.json --extraction var/research-continuation/frontal-emotions-v1/aranel/extraction/extraction.json --models models/observers --output var/local-geometry-study/repeatability-repeat
```

输出目录必须不存在，原图哈希和模型清单须匹配。已完成六参考形状的解析对照及33图的99次检测探针；当前帧／参考的双边误差、近共线正常／异常方向及有界正常范围另有受控计算，详见 [第十五节](./reports/2026-10-12-local-geometry/README.md#15-局部形状稳定性条件性优势与测量等变实验)。这些结果没有学习正常表达分布或增加独立来源，未改变正常校准39来源、有效验证59来源及覆盖要求。

## 冻结、校准与独立评估

新增正脸候选 `intrinsic_local`：眼、双眼关系、鼻、嘴沿用 `single_local`，仅把轮廓替换为五点内部形状比值。点0/4确定轮廓宽度和局部坐标轴，以两端点中点为原点。保留既有前四个分量：下巴点2的法向高度和沿轴偏移、两侧中间点1/3的法向高度；追加点1/3的沿轴偏移，共六维，均除以轮廓宽度；全部五点需可靠。对于端点不同、索引固定的二维五点轮廓，六维坐标完整保留去除平移、旋转和正尺度后的六个形状自由度，避免遗漏中间点沿轴移动。它消除跨部位拟合传播，不消除检测器噪声或耦合，也不保证三维姿态鲁棒性；轮廓整体等比例大小变化仍是盲区。

单独诊断使用 `study intrinsic`，正式阶段可明确 `freeze --candidate intrinsic_local`，其余参数与下方命令一致。候选仍使用固定正脸参考／配置，不由人工姿态标签路由；每个部位重新用独立正常开发来源拟合尺度，再共同校准最大标准化统计量。新的轮廓比值不能共用原轮廓RMS阈值。[历史报告第十节](./reports/2026-10-12-local-geometry/README.md#10-正脸轮廓内部比值与可辨识性负对照) 保留历史v1的四维数值；新版六维评分须重新运行，不能把旧值改标为新版结果或沿用其阈值。源码变化使旧冻结产物失效，需使用新输出目录重新冻结；历史报告保留原版本。

```powershell
uv run --locked python scripts/observer_geometry_experiment.py study freeze --manifest var/local-geometry-study/dataset-v1/reviewed.json --extraction var/local-geometry-study/extraction-v1/extraction.json --candidate pose_local --output var/local-geometry-study/frozen-v1
uv run --locked python scripts/observer_geometry_experiment.py study calibrate --manifest var/local-geometry-study/dataset-v1/reviewed.json --extraction var/local-geometry-study/extraction-v1/extraction.json --frozen var/local-geometry-study/frozen-v1/frozen.json --output var/local-geometry-study/calibration-v1
uv run --locked python scripts/observer_geometry_experiment.py study evaluate --manifest var/local-geometry-study/dataset-v1/reviewed.json --extraction var/local-geometry-study/extraction-v1/extraction.json --frozen var/local-geometry-study/frozen-v1/frozen.json --calibration var/local-geometry-study/calibration-v1/calibration.json --output var/local-geometry-study/evaluation-v1
```

所有输出目录必须不存在。输入图片、标签、模型、实现或配置变化使后续绑定失效；需要保留旧版本并重新冻结／校准，不能修改旧阈值文件。

冻结阶段按 front / left_turn / right_turn / down / up 选择人工确认正常原图的代表形状（结构对齐距离的 medoid）。用不包含参考来源的正常开发主样本拟合逐部位姿态门限：正确匹配误差 95% 分位数、最优次优差额 5% 分位数，至少 20 个独立来源。姿态候选比较不包含当前被检测部位；仅一个姿态参考、数据不足、歧义和越界都不输出可信姿态。各有效部位匹配姿态不一致时联合判定 unknown，仍保存原始局部残差和匹配原因。

固定输出五个版本：生产 WLS、单正脸参考的局部对齐 single_local、自动姿态参考 pose_local、人工姿态 oracle、正脸轮廓比值 intrinsic_local。WLS、single_local和intrinsic_local使用固定正脸参考／配置，不自动取得姿态鲁棒性；自动版本由匹配结果选择姿态配置，只有 oracle 使用人工真值姿态路由。oracle 仅诊断，不参与采用。候选由 freeze --candidate 在校准前选定，默认 pose_local，不能根据验证集重新挑选。当前intrinsic_local实验输入限定为已有人工确认的正脸批次；其他姿态适用性未验收。

正常开发主样本拟合 median／1.4826 MAD，至少 20 个独立来源；尺度下限 1e-6 并记录 mad_floored。以非负标准化局部残差的最大值形成几何辅助统计量，不混入身份/运动，不代替原始局部输出。每姿态配置至少 39 个独立正常校准主样本，排序位置 `ceil((n+1)*0.95)`，严格大于才标候选异常。稀疏部位、未知有效性组合或姿态歧义为 unknown；仅部分部位未超限时整体仍未知。

秩校准要求同分布和近独立，不能保证每个罕见缺测子组误报均为 5%。报告必须展示有效性组合与分层表现，输入分布变化要重新评估。代码中的 statistical_flag 表达支持观测上的阈值事件；decision 表达整张脸几何结论，两者分别报告，避免把部分部位未超限写成整脸正常。

验证仅使用验证来源、人工确认的 idle、受支持姿态及非严重遮挡数据。误报率分母为有可用统计判定的正常主样本；完整判定覆盖率的分母为所有符合范围的正常主样本。异常 unknown 计未检出；按姿态、部位、人工严重度、机械变形强度及合成／真实分别统计。Clopper–Pearson 单侧 95% 上界：59 个独立有效正常观测、0 误报时约 4.95%；不能用相邻帧替代这些来源。

采用要求误报上界 ≤5%、正常完整判定覆盖率 ≥80%，WLS 也经过同目标校准且误报上界 ≤5%，人工确认畸变至少来自20个独立来源；配对来源簇 bootstrap 区间下界大于0，并且单侧精确来源符号检验的 p 值 ≤0.05。20个来源只是流程最低样本门槛，不保证充分检验功效；仍需按预期效应、来源差异和实际异常组成估算样本量。

bootstrap 固定种子42、2000次，每次按来源簇重采样，保留簇内配对样本，计算按样本数加权的候选与WLS检出率差；输出百分位近似区间，不是严格的均值置信保证。来源符号检验按每个来源的检出率差统计正／负方向，排除差为0的来源，检验非零差来源中正向概率是否大于1/2；单侧精确二项 p 值依赖独立来源及零假设下正负方向等可能，检验对象不是样本加权平均检出率差，也不能将其改述为均值的置信区间。合成与真实畸变的区间和检出率分别报告；合成数据上的通过不能宣称自然生成异常已验收。缺数据时明确 not_accepted；任何通过也仅为离线范围，不自动启用 Runtime 或恢复。

## 正脸光流、时序回放与统计预算

`study flow` 在独立CPU模型环境中，对已人工标注的正脸图与正常参考作真实GMFlow双向推理，不下载模型。固定源码提交 `b5123431164d01ec14526a1c3d22218aecb62024`，使用官方模型表链接的 `gmflow-scale1-things-e9887eda.pth`，SHA-256为 `e9887eda6165e0813c03f35998d6af86e265aafce2e3349af93d691a59696088`。准备路径及许可边界见 [开源选型](./open-source-selection.md#正脸光流诊断补充2026-10-12)。需要10个上游源码文件的本地哈希清单 `models/gmflow/provenance.json`；严格加载权重，固定256×256、CPU、2线程、注意力分块2和全局匹配。

```powershell
.tools/observer-env/Scripts/python.exe scripts/observer_geometry_experiment.py study flow --manifest var/local-geometry-study/dataset-front-v1/reviewed-human-v1.json --extraction var/local-geometry-study/extraction-front-v1/extraction.json --reference s0000 --gmflow-repo vendor/gmflow --gmflow-model models/gmflow/gmflow-scale1-things-e9887eda.pth --gmflow-provenance models/gmflow/provenance.json --output var/local-geometry-study/front-flow-repeat
uv run --locked python scripts/observer_geometry_experiment.py study flow-regions --manifest var/local-geometry-study/dataset-front-v1/reviewed-human-v1.json --extraction var/local-geometry-study/extraction-front-v1/extraction.json --reference s0000 --flow-report var/local-geometry-study/front-flow-repeat/flow.json --output var/local-geometry-study/front-flow-regions-repeat
uv run --locked python scripts/observer_geometry_experiment.py study intrinsic --manifest var/local-geometry-study/dataset-front-v1/reviewed-human-v1.json --extraction var/local-geometry-study/extraction-front-v1/extraction.json --reference s0000 --output var/local-geometry-study/front-intrinsic-repeat
uv run --locked python scripts/observer_geometry_experiment.py study replay --manifest var/local-geometry-study/dataset-front-v1/reviewed-human-v1.json --extraction var/local-geometry-study/extraction-front-v1/extraction.json --reference s0000 --output var/local-geometry-study/front-replay-repeat
uv run --locked python scripts/observer_geometry_experiment.py study budget --output var/local-geometry-study/statistical-budget-repeat
```

所有输出目录须不存在。光流原始数组留在私有输出，`flow-regions`复用它们，不重新推理。固定参考点矩形加0.15倍参考RMS边距作为部位范围，输出中位数、95%分位数及双向一致性；全图光流或一致性均不能直接当异常分数。图对没有真实时间间隔，不输出速度、加速度或每秒运动指标。

`replay` 是专用于本批已标注s0000/s0005/s0012的合成时间线：从静态测量构造正常、短暂变化、持续畸变及缺测场景，验证因果状态处理。显式工程夹具阈值0.03/0.015、持续0.5秒、最大间隔0.15秒未校准，不进入Runtime。超过进入线开始WATCH，保持在退出线上方持续达标才建议恢复；缺测、过长间隔或无交互配置断开持续计时。返回RECOVER_RECOMMENDED仅表示影子建议，不调用恢复后端，不表示真实视频预警或恢复成功。

`budget` 计算独立有效正常观测的Clopper–Pearson要求及来源划分预算。39来源秩阈值提供交换性假设下的边际保证，不能保证每份冻结阈值的真实误报率都≤5%；独立验证仍必需。80%有效率预算另假设独立二项观测，不是当前覆盖率实测。完整续作结果、正式阶段数据不足与验收逐项状态见 [报告第十至十三节](./reports/2026-10-12-local-geometry/README.md#10-正脸轮廓内部比值与可辨识性负对照)。
