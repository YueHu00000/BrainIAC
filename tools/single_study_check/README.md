# 单个 study 的 DICOM / raw NIfTI 检查

两个单样本检查脚本，以及一个 DICOM 队列统计脚本。Python 3.10+，只读取源影像。无需 GPU、SimpleITK 或 ANTs。

GitHub 发布位置：`YueHu00000/BrainIAC` 仓库的 `tools/single_study_check/`。在仓库根目录运行 `cd tools/single_study_check` 后，即可使用下列命令；工作区原始文件保留在 `E:\codex\MRI_project\code\single_study_check`。

## 环境

在已有 `vit_survival` 环境中，`numpy`、`nibabel` 已包含在项目的依赖清单里；额外安装 DICOM 读取库：

```bash
python -m pip install -r requirements.txt
```

上面的 requirements.txt 指本目录的文件。也可直接运行 `python -m pip install pydicom==3.0.1`。

## 1. 列出所有 T1 series

```bash
python inspect_dicom_t1.py "/data/STUDY_1234" --output "/reports/STUDY_1234_T1_series.txt"
```

- 输入推荐是单个 study 的目录；递归读取其中全部文件，不依赖 `.dcm` 后缀。
- 如果输入某个 DICOM 文件，则递归检查其父目录，以便找到其他 series；不会只检查该文件。
- 按 `StudyInstanceUID + SeriesInstanceUID` 分组；只要组内任意文件的 `SeriesDescription` 包含 `T1`（忽略大小写），就列出整组。缺少 SeriesInstanceUID 的文件逐个列出。
- 这只是复现 Harvey 的 T1 描述匹配口径，并不保证识别所有真正的 T1 加权扫描，例如描述仅含 MPRAGE 的序列不会匹配。
- 报告包含 UID、series number/description、protocol、文件数/帧数、TR/TE/TI、几何、每个文件路径和 SOP UID。Enhanced MR 的 shared/per-frame 几何也会列出。
- 图像位置采用 DICOM LPS 坐标；PixelSpacing 是 `[row, column]`。帧数不等于不同空间层数，重复位置、多 echo、时间维都可能造成差异。
- 不读取 PixelData；无需安装 JPEG 解码器。无法读取的文件会列入报告末尾，不会静默忽略。
- Harvey 按 `(SeriesNumber, SeriesDescription)` 分组，选最大 SeriesNumber；本报告按 UID 保留真实序列身份，不执行 series 选择。可结合报告里的 number/description 检查 Harvey 是否可能混合多个 UID。
- `--study-id STUDY_0230` 可显式指定用于展示硬编码排除项的 ID，默认使用输入目录名。它仅展示规则，不过滤报告里的 series。

## 2. 检查 raw T1 NIfTI

```bash
python inspect_raw_nifti.py "/data/harvey/STUDY_1234/T1.nii.gz" --output "/reports/STUDY_1234_raw_T1.txt"
```

报告包含 shape、spacing、方向、affine、qform/sform、空间范围、强度缩放与统计、完整 header、扩展中的 UTF-8 文本。统计会读取完整图像到内存，使用应用强度缩放后的 float32 数据。

NIfTI affine 使用 RAS+，DICOM 使用 LPS。对比物理坐标时需反转 X/Y 符号，并核对空间单位。NIfTI 第三轴不一定是解剖轴向，也不能在多帧/重排/重采样情况下直接当作原始 DICOM 文件数。

标准 NIfTI 没有专门的 SeriesInstanceUID 字段，Harvey 的转换通常没有额外保存这一信息。几何匹配只能支持来源判断，唯一确认还需候选 DICOM 重建后的体素对照；这两个脚本不执行体素匹配。

## 输出

`--output` 可省略：DICOM 默认写到当前工作目录的 `<目录名>_t1_series.txt`；NIfTI 默认写到 `<文件名去扩展名>_info.txt`。输出父目录须已存在，现有文件不会覆盖；再次运行请换输出文件名。推荐把报告放在 DICOM 目录之外，避免下次扫描时把报告当成非 DICOM 文件记录。

合成数据检查可在本目录执行：

```bash
python -m unittest -v test_inspect.py
```

## 3. 对目标 CSV 中全部 studies 做 T1/T2 统计

程序：`summarize_dicom_t1t2.py`。使用 `numpy`、`pandas`、`pydicom`；前两者已有于 `vit_survival` 依赖清单。程序复用同目录 `inspect_dicom_t1.py` 的几何读取函数和原有排除项；部署到服务器时请复制整个 `single_study_check` 目录。

**统计全部 CSV studies，包含用户报告的那个 embedding 失败样本。** 不读取 raw NIfTI 或 BrainIAC 输出，不检查 embedding 成功状态，不把统计队列称为“仅成功样本”。

### 输入文件和运行

目标 CSV，例如 `target_studies.csv`：

```csv
Study_ID
STUDY_0001
STUDY_0002
```

允许其他列，程序只使用 ID 列。默认列名 `Study_ID`；若是 `STUDY_ID`，加 `--study-id-column STUDY_ID`。ID 以字符串读取，保留前导零。空 ID 或重复 ID 会报错，不会静默重复统计。

根目录列表 `folder_address.txt`，与转换程序使用同一份列表，每行一个目录，不加引号（建议使用绝对路径）：

```text
/data/TUFTS_MRI_Training_Dataset_Part1
/data/TUFTS_MRI_Training_Dataset_Part2
```

程序在这些目录下查找 `<根目录>/<Study_ID>`，与现有转换程序一致。根目录列表中的相对路径相对于 TXT 所在目录解析。空行忽略；根目录必须存在且不能重复。

不同 studies 可以分布在不同根目录中，但每个 study 的全部 DICOM 文件应完整地位于一个 `<根目录>/<Study_ID>` 内。例如：

```text
/data/TUFTS_MRI_Training_Dataset_Part1/STUDY_0001/*.dcm
/data/TUFTS_MRI_Training_Dataset_Part2/STUDY_0002/*.dcm
```

程序逐个 study 搜索全部根目录，不会把不同 studies 的文件混合。跨根目录文件名是否相同不影响定位；同一 Study_ID 出现在多个根目录中仍报告 `study_ambiguous`。仅列出根目录，不需要逐个列出 study 子目录或 DICOM 文件。

在脚本目录运行（下列命令在 PowerShell 和 Linux shell 中都可写成一行）：

```bash
python summarize_dicom_t1t2.py --csv "/data/target_studies.csv" --folder-list "/data/folder_address.txt" --output-dir "/reports/dicom_t1t2_statistics_v1"
```

若 `folder_address.txt` 位于运行命令时的当前工作目录，可省略 `--folder-list`：

```bash
python summarize_dicom_t1t2.py --csv "/data/target_studies.csv" --output-dir "/reports/dicom_t1t2_statistics_v1"
```

默认列表文件相对于**当前工作目录**定位，不是脚本目录。仍可用 `--folder-list` 指定任何其他文件名或位置。列表内若使用相对根目录路径，统计程序相对于 TXT 目录解析，而当前转换程序相对于工作目录解析；使用同一份绝对路径列表可避免这一区别。

`--output-dir` 必须是一个尚不存在的新目录，且位于 DICOM 根目录之外。再次运行请使用 `v2` 等新名字。报告期间每 100 个 studies 显示一次进度；不改动源 DICOM 或原有结果。

### 选择规则与错误处理

- 只读取 study 目录的直接子项，不递归；按 `SeriesNumber, InstanceNumber` 排序，再按 `(SeriesNumber, SeriesDescription)` 分组。
- 应用原有四个 study 的 SeriesNumber 排除规则，然后分别匹配描述中的 T1/T2，取最大 SeriesNumber。最大编号并列时，按 pandas 默认排序后的分组顺序取首组，与原代码 `idxmax()` 一致。
- 不按 UID 重新分组、不选择最多切片的序列、不因层数少而回退到别的序列。UID 混组会记录在报告中。
- 与原始严格转换代码一致，任意直接子项读取失败或缺少 `SeriesNumber/SeriesDescription/InstanceNumber`，该 study 的选择无法确认，两个模态标为 `read_error`，而不是跳过文件后猜测结果。其他 studies 继续统计。若 study 下有子目录，也会按此规则记录读取错误。
- 找不到或在多个根目录找到同一 study 时，分别报告 `study_not_found`、`study_ambiguous`。某模态没有候选时报告 `no_candidate`；另一个模态若可选择仍保留统计，`pair_selected=False` 提示原始成对转换不能据此视为成功。
- `status=selected` 只表示规则选中了候选组，不表示影像质量合格或 BrainIAC 成功。

### 四个输出文件

| 输出 | 内容 |
|---|---|
| `selected_series.csv` | 每个 CSV study 固定两行（T1/T2），包含错误状态、来源 UID、选中文件名、InstanceNumber、各项指标与 flags |
| `distribution_summary.csv` | 按 T1/T2 分开的总体分布，每个指标都有总数、有效数、缺失数和分位数 |
| `flagged_series.csv` | 包含低数量、几何异常、metadata 缺失、派生图像标签或选择失败的行；标记不自动删除样本 |
| `summary.txt` | 英文摘要，包含 cohort 定义、有效分母、低数量比例、异常计数、study 任一模态低数量计数和分布表 |

CSV 为带 UTF-8 BOM 的文本，便于 Excel 读取；列表字段采用 JSON 文本，未知值写为 `unknown`。源文件路径可由 `study_directory` 与 `file_names` 组合得到。

### 指标定义与统计口径

| 字段 | 定义 |
|---|---|
| `file_count` | 选中组的 DICOM 文件数 |
| `frame_count` | 帧总数；未声明 NumberOfFrames 的普通 DICOM 按 1 帧计 |
| `unique_slice_count` | 几何完整且方向一致时，沿共同层法向的不同空间层位置数；不等于所有情形下的文件数 |
| `rows` / `columns` | 原始像素矩阵，缺失或组内不一致时未知 |
| `pixel_row_*_mm` / `pixel_column_*_mm` | 原始 PixelSpacing；`*` 分别为 min/median/max |
| `thickness_*_mm` | 原始 SliceThickness 的 min/median/max |
| `tag_spacing_*_mm` | 原始 SpacingBetweenSlices 的 min/median/max |
| `gap_min_mm` / `gap_median_mm` / `gap_max_mm` | 不同层位置按空间排序后，相邻实际距离的 min/median/max |
| `coverage_mm` | 首尾层中心沿层法向的跨度，不代表已验证完整脑覆盖 |
| `duplicate_plane_count` | 总帧数减去不同层位置数；重复层可能来自多 echo、多时间点，不能直接当作重复文件 |
| `max_in_plane_shift_mm` | 相对首帧，图像原点在层平面内的最大位移 |

每个模态、每个指标输出 `min、P1、P5、P10、P25、median、P75、P90、P95、P99、max`，使用 NumPy `method='linear'`；计数指标的分位数也可能是小数。每个 study/modality 对每个指标贡献一个值，不按切片数加权。

像素间距、层厚、标记层间距在全部帧均有有效正值时才输出该序列的 min/median/max；部分缺失时结果为 `unknown`，并保留 `*_known_frames`。不以少数有值帧代表整组。空间位置或方向缺失/不一致时，空间层数、实际间距及跨度为未知；文件数/帧数等已知量仍可统计。

几何容差固定并写入英文摘要：

- 不同层位置按投影距离区分，容差为 `0.01 mm`；按排序后距当前组首位置超过此容差建立新层。
- 方向余弦一致性及正交/单位向量检查容差为 `1e-4`。
- 不等间距判据：`max(gaps) - min(gaps) > 0.01 mm + 0.01 × median(gaps)`；至少 3 个不同层位置才可评估均匀性。
- 保留 `projected_positions_mm`、`ordered_steps_mm`、`sorted_unique_gaps_mm` 供追查；原始堆叠顺序非单调也会标记。
- Enhanced MR 使用 shared/per-frame 信息。多帧文件缺少完整 per-frame 位置时不会把同一个顶层位置复制成所有帧的真实几何。

例如之前的异常：位置为 `0、63、70 mm`，结果应为 `file_count=3`、`frame_count=3`、`unique_slice_count=3`、`gap_min=7`、`gap_median=35`、`gap_max=63`、`coverage=70`，并标记 `nonuniform_spacing`。**这里的 35 是两个真实间距的中位数，不表示实际采集均匀，更不是读取 raw NIfTI 得到的值。**

英文摘要分别对文件数、帧数和不同空间层数列出 `<4`、`<10`、`<20` 的计数与有效分母比例；另给出 T1/T2 任一模态触发时的 study 数（yes/no/unknown）。一个单文件多帧体积可能文件数低但空间层数充足，这三个口径不能混用。阈值只是描述分布，不是已验证的合格标准。

如有缺失/读取失败，总体分位数仅使用该指标的有效记录，但同时保留全 CSV 分母和缺失数。没有任何可用值时分位数为 `unknown`，不是零。

### 验证

```bash
python -m unittest -v test_inspect.py test_summarize_dicom_t1t2.py
```

测试使用合成数据，不访问真实患者数据。全量统计仍需在可读取目标 CSV 与 DICOM 根目录的环境中执行。
