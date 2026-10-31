# CBI/WMD：固定 BrainIAC embedding 的三模态验证

使用现有独立导出程序产生的逐 study NPZ，建立相同配对患者的 T1、T2、T1+T2 分类实验。Encoder 保持冻结，只训练线性头。本文中的路径是示例，请替换为实际输入位置。

## 1. 实验定义

- 仅从 `final_pre_process.csv` 列出的序列选择。
- 质量条件固定为 `coverage_mm >= 100`、`20 <= frame_count <= 40`，全部包含边界。
- 描述不区分大小写匹配 T1/T2；同时命中两个模态的描述列为歧义候选，不用于选取。
- 每个 study、每个模态在合格候选内取数值最大的 SeriesNumber。
- 最大 SeriesNumber 并列则排除该 study；选中 embedding 缺失或不符合格式时排除该 study，不回退到其他序列。
- 主分析要求同一 study 同时具备合格 T1 和 T2；三种输入使用相同患者、标签和分组。
- CBI、WMD 分别训练；目标标签为空的患者只从该目标的三种输入实验中共同排除。
- 每次训练读取一份 manifest，使用 `final_embed` 或 `block_embed` 中指定的 block 3/6/9/12。单模态 768 维，T1+T2 使用两模态的同一指定层，按 T1 在前、T2 在后的顺序拼接为 1536 维。
- 采用 Harvey 10k 的五组 site 留出映射。其模型为二维切片 MIL；这里评估三维 BrainIAC 的全局向量，不能称为完整复现论文表征。

现有上游 manifest 负责 UID/SeriesNumber 分组与最大 AcquisitionNumber；本程序保留该身份，只在下游选择 T1/T2，不重新选择 acquisition 或合并 DICOM。

## 2. 环境与入口

Python 3.10 或更新版本；依赖 NumPy、PyTorch、scikit-learn。已有预训练环境通常已经包含这些依赖，无需 MONAI 参与本程序运行。

在仓库根目录执行：

```bash
python -m pip install -r src/CBI_WMD_predict/requirements.txt
python src/CBI_WMD_predict/manifest.py --help
python src/CBI_WMD_predict/train.py --help
```

也支持从其他工作目录使用脚本的绝对路径运行。所有代码均在本目录，不导入预处理、SimCLR 或旧分类程序。

## 3. Manifest 构建输入

| 参数 | 必需字段 / 内容 |
|---|---|
| `--manifest` | 上游 `manifest.csv`；`unique_id,study_id,series_instance_uid,series_number,acquisition_number` |
| `--final-csv` | `final_pre_process.csv`；`unique_id` |
| `--selected-csv` | 同批次原始 `selected_series.csv`；`unique_id,study_id,series_instance_uid,series_number,series_description` |
| `--quality-csv` | 同批次 `image_number_coverage.csv`；`unique_id,frame_count,coverage_mm`，可含 `series_description` |
| `--labels-csv` | 一人一个 index study 的标签表，见下文 |
| `--embedding-dir` | 一个固定 checkpoint 的导出目录，包含 `<study_id>.npz` |
| `--output-dir` | 尚不存在的下游 manifest 输出目录 |
| `--patient-id-column` | 患者身份列，默认 `Study_ID`（用户已确认）；可指定其他列 |

标签表默认字段为 `Study_ID,SiteID,idCBI,idWMD`。可分别通过 `--study-id-column`、`--site-id-column`、`--cbi-column`、`--wmd-column` 指定实际列名。以下仅为虚构格式示例：

```csv
Study_ID,SiteID,idCBI,idWMD
STUDY_001,12,0,1
STUDY_002,13,1,
```

CBI/WMD 接受 0/1（包括 0.0/1.0）、空白或 NaN。空白/NaN 视为目标缺失，其他非二元值报错。Study_ID 或患者 ID 重复时停止，要求先提供明确的 index-study 标签表；不自动挑选某次检查。**用户于 2026-10-28 确认本队列使用 Study_ID 作为患者身份，程序据此采用该默认值。** 若其他队列使用独立患者列，可通过 `--patient-id-column Patient_ID` 指定。

完整 `unique_id` 原样用于连接各输入和 NPZ key，不从字符串拆解患者或 UID。上游与 selected 表的 study、UID、SeriesNumber 必须一致；若 selected 含 AcquisitionNumber、quality 含描述，也会核对对应字段。上游缺失 AcquisitionNumber 的 `unknown` 状态继续保留，不因此排除序列。

新导出格式的每个 NPZ key 对应一个零维 structured array，其中 `final_embed` 为 `(768,)`，`block_embed` 为 `(4,768)`，四行依次为 block 3、6、9、12。manifest 构建保持使用 final_embed 验证选中序列；训练只读取和验证指定的表征。旧式固定模态轴、对象数组或其他维度不能直接使用。

## 4. 构建命令与输出

下面是一条完整命令，适用于 PowerShell 或 Bash（请替换示例路径和患者列名）：

```bash
python src/CBI_WMD_predict/manifest.py --manifest work/pretraining/inventory/manifest.csv --final-csv work/pretraining/processed/final_pre_process.csv --selected-csv work/pretraining/stats_uid/selected_series.csv --quality-csv work/pretraining/quality/image_number_coverage.csv --labels-csv resources/labels.csv --embedding-dir work/embeddings_epoch198 --output-dir work/cbi_wmd/manifests_epoch198
```

输出目录必须尚不存在；更换输入、标签或 checkpoint 时使用新目录。

| 输出 | 内容 |
|---|---|
| `manifest_t1.csv` | 配对患者，只引用选中 T1 的特征字段 |
| `manifest_t2.csv` | 相同患者，只引用选中 T2 的特征字段 |
| `manifest_t1t2.csv` | 相同患者，同时引用两模态 |
| `folds.csv` | 每名患者在五组方案中的 train/val/test 身份 |
| `series_selection.csv` | final 清单中每个候选的质量、资格、是否选中和原因 |
| `excluded_studies.csv` | 缺模态、并列、选中 embedding 不可用、缺标签行、非目标 site 等排除原因 |
| `cohort_counts.csv` | 按 site、目标标签列出初始标签人数、配对保留/排除人数、候选/合格/选中序列数和阳性/阴性/缺失数 |
| `selection_config.json` | 输入路径、列名、固定规则、分组；导出目录有 `embedding_config.json` 时一并保存其内容 |

三份 manifest 以患者/index study 为行，保留完整选中序列 ID、SeriesNumber、AcquisitionNumber、描述、帧数、coverage 和 embedding 路径。未使用的模态字段留空。`series_selection.csv` 的 selected 表示该模态胜出，不代表整个 study 最终保留；患者排除须同时查看 `excluded_studies.csv`。

全体输入没有任何合格配对患者时直接报错，不发布空队列。已有上游 CSV、MRI、embedding 均只读。

## 5. 固定的五组 site 留出

| Fold | Train | Val | Test |
|---|---|---|---|
| 1 | 1,2,3,6,7,9,10,11 | 12 | 13,14 |
| 2 | 1,2,3,6,7,9,13,14 | 10,11 | 12 |
| 3 | 1,2,3,6,7,12,13,14 | 9 | 10,11 |
| 4 | 1,2,3,6,10,11,12,13,14 | 7 | 9 |
| 5 | 1,2,3,9,10,11,12,13,14 | 6 | 7 |

其他 SiteID 不加入本实验并记录原因。这里不是让全部患者恰好测试一次的完整五折 OOF。三个模态使用同一张 `folds.csv`；训练入口会核对其身份与固定映射，避免手工改表产生不同划分。

## 6. 分类训练

模型是 `Linear(768,1)` 或 `Linear(1536,1)`。不增加 PCA、特征标准化、EHR、切片 attention 或 encoder 微调。

默认设置：

- 1000 epochs，batch size 64，seed 1001；有 CUDA 时默认使用 `cuda:0`，可指定 `--device cpu`。
- SGD，momentum 0.9，optimizer weight_decay=0；梯度范数上限 1.0。
- BCEWithLogitsLoss；L1/L2 惩罚均包括线性层权重与偏置，分别为 `alpha/2 * sum(abs(params))` 和 `alpha/2 * sum(params**2)`。
- 学习率：0.1、0.01、0.001、0.0001。
- L1 和 L2 各搜索 alpha：1、0.1、0.01、0.001、0.0001、1e-5、1e-6、0。
- 打乱训练患者并 `drop_last=True`。各 trial 的初始化及 DataLoader 分别固定随机种子；批次顺序不依赖输入维度。
- Harvey checkpoint 条件：本轮 online train AUROC 严格高于 epoch 结束后的 val AUROC，才有资格参与选择；在合格候选中取最高 val AUROC，平局取最早遇到者。

online train AUROC 来自该 epoch 训练过程中收集的预测，与最终权重对完整训练集重评估的 AUROC 不是同一个值。若全部候选都不满足条件，记录 `no_eligible_checkpoint`，不替换为其他选择规则，也不生成测试预测。

每个请求的目标/fold，过滤缺失标签后 train/val/test 都必须非空且包含两类，train 人数须至少达到 batch size；不满足时在训练前停止。

**train.py 现在只接受一份 `--manifest` 文件。** 表内 `modality` 必须一致且为 T1、T2 或 T1T2，程序据此确定输入维度。每行需有 patient_id、study_id、site_id、CBI、WMD、embedding_path，以及该模态使用的 t1_unique_id / t2_unique_id。患者和 study 都必须唯一；T1T2 两个选中 UID 都必须存在且不同。相对 embedding_path 按该 manifest 所在目录解析。

五组划分直接由表中的 site_id 和 common.FOLDS 生成，**不读取 folds.csv，也不读取其他模态 manifest**。三模态配对一致性由 manifest.py 生成三份表时保证，独立训练不再跨文件核对。

完整实验：

```bash
python src/CBI_WMD_predict/train.py --manifest work/cbi_wmd/manifests_epoch198/manifest_t1.csv --embedding-field final_embed --output-dir work/cbi_wmd/results_t1_final --device cuda:0
```

使用 block 6 的双模态表征：

```bash
python src/CBI_WMD_predict/train.py --manifest work/cbi_wmd/manifests_epoch198/manifest_t1t2.csv --embedding-field block_embed --block 6 --output-dir work/cbi_wmd/results_t1t2_block6 --device cuda:0
```

`--embedding-field` 默认 final_embed。指定 block_embed 时必须提供 `--block 3/6/9/12`，表示实际 encoder 层号，而不是数组下标；选择 final_embed 时不允许提供 --block。block 12 是最终 LayerNorm 前的表征，与 LayerNorm 后的 final_embed 分别读取，不相互替代。

**单份 manifest 默认运行两目标 × 五组，共 10 个 task、640 个 trial，每个 trial 1000 epochs。** 三模态分别调用三次时总计 1920 个 trial。这是代码提供的统一预定义网格，不代表已经取得论文 10k 实验的完整历史搜索记录。可通过以下参数缩小首轮运行或按组提交：

```bash
python src/CBI_WMD_predict/train.py --manifest work/cbi_wmd/manifests_epoch198/manifest_t2.csv --output-dir work/cbi_wmd/smoke_t2 --device cpu --epochs 20 --batch-size 8 --learning-rates 0.1 --alphas 0 0.001 --penalties L2 --folds 1 --targets CBI WMD
```

参数包括 `--manifest`、`--embedding-field`、`--block`、`--folds`、`--targets`、`--epochs`、`--batch-size`、`--learning-rates`、`--alphas`、`--penalties`、`--seed`、`--device`。旧的 --manifest-dir 和 --modalities 已移除。只比较采用同一候选网格和明确表征选择的各模态；参数根据验证集分别选择，不照搬 ABMIL 的最佳参数。

运行输出目录必须是新目录。本版不恢复中断的超参数搜索；按 fold/目标分开执行时使用不同输出目录。

## 7. 训练结果与模型保存

```text
results_t1_final/
├── run_config.json
├── results.csv
├── summary.csv
└── T1/                          # 仅所选 manifest 的模态
    └── CBI / WMD/
        └── fold_1 ... fold_5/
            ├── history.csv
            ├── trials.csv
            ├── metrics.json
            ├── best_model.pt
            └── predictions.csv
```

- `history.csv`：每个 trial/epoch 的 online train 与 val 指标、候选资格。
- `trials.csv`：各超参数组合的最佳合格 epoch 或无合格候选状态。
- `best_model.pt`：选中线性头的 CPU state_dict，以及模态、embedding_field、block、目标、fold、输入维度、epoch、超参数、seed 等元数据。
- `predictions.csv`：选定后才计算的逐测试患者标签、logit 和阳性概率，同时记录 embedding_field 和 block。
- `metrics.json`：选中模型的 train_eval、val、test 指标；无合格候选时仅保存状态与人数。
- `results.csv`：每个已完成目标/fold 的状态、人数、表征字段/层号、选中参数和测试指标。
- `summary.csv`：本次模态/表征、各目标成功 fold 的均值和样本标准差（ddof=1）；同时给出已处理的 fold 数及成功数，缺失结果不能当作完整五组结果。

测试特征仅在该项全部超参数与 epoch 选定后读取和评估。验证标签用于选模；测试指标不能回用于修改网格、选择 checkpoint 或筛序列。summary 的均值随任务完成更新，长任务中途的文件可能仅包含已完成部分。

每次运行只处理一个 manifest 和一种表征，结果不自动跨目录汇总，也不生成 paired_differences.csv。比较模态时可对照不同运行的逐患者 predictions.csv；应确认患者和标签一致。所有评估针对 CBI/WMD 当下状态，不是卒中生存预测。

## 8. 测试与验证边界

```bash
python -m unittest discover -s src/CBI_WMD_predict/tests -v
```

测试覆盖质量边界、数值 SeriesNumber 排序、先筛后选、并列/歧义序列、缺少最高候选 embedding 不回退、配对患者与 site 映射一致性、缺失/非法标签、重复患者、全部四个 block 的精确读取、final 与 block 12 区分、768 维表征与非有限值、L1/L2 偏置惩罚、严格选模、测试特征隔离，以及三份 manifest 分别隔离后通过真实 CLI 完成合计 30 项合成数据训练与模型输出。

本地验证使用 Python 3.10、NumPy 2.2.6、PyTorch 2.13.0+cpu、scikit-learn 1.7.2。真实患者输入与 GPU 尚需在对应服务器验证；测试分数不代表队列预测能力。

## 9. 方法依据

- [Harvey2026](https://arxiv.org/html/2604.26807v1)：冻结表征、二分类、site 留出及实验设置。
- [10k site/模型配置记录](https://github.com/YueHu00000/neuroimage-classifiers/blob/8bdb9fafe3169174529a8eccad078853d7b39b8c/MODELS.md)。
- [Harvey 分类入口](https://github.com/YueHu00000/neuroimage-classifiers/blob/8bdb9fafe3169174529a8eccad078853d7b39b8c/src/kpsc.py)。
- [Harvey loss 与正则化](https://github.com/YueHu00000/neuroimage-classifiers/blob/8bdb9fafe3169174529a8eccad078853d7b39b8c/src/losses.py)。
- [Harvey 训练步骤与梯度裁剪](https://github.com/YueHu00000/neuroimage-classifiers/blob/8bdb9fafe3169174529a8eccad078853d7b39b8c/src/utils.py)。

与原训练入口的差异：使用三维全局 final_embed 或某个 block 均值表征；上游显式患者配对与 metadata 审计；独立单 manifest 训练，按 site 自动分组；测试评估延后到完整选模之后；显式 DataLoader 随机生成器。没有移用原编码脚本的尺寸或模态切分文档。
