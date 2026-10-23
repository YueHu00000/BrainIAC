# 全 series 统计：SeriesInstanceUID 唯一 ID 与最大 acquisition

## 精简 metadata：直接读取原始 DICOM

新增入口 `export_series_metadata.py`，输入为 Study ID CSV 和 DICOM 根目录列表，**不需要以前的统计输出**。依赖与下方统计入口相同；复制到服务器时同时保留 `_series_statistics.py`。

```bash
python src/brainIAC_pretraining/stat/export_series_metadata.py \
  --csv resources/labels.csv \
  --folder-list resources/folders.txt \
  --output-dir resources/series_metadata

python -m unittest discover -s src/brainIAC_pretraining/stat -p 'test_export_series_metadata.py' -v
```

- CSV 默认 ID 列为 `Study_ID`，可用 `--study-id-column` 指定；ID 按文本读取，保留前导零和字面值 `NA`。
- folder list 为 UTF-8 文本，一行一个根目录；相对路径以列表所在目录解析。每个 study 必须唯一对应一个 `root/Study_ID`，递归扫描其中的 MR DICOM headers。
- 身份为 `{study_id}_{SeriesInstanceUID}_{SeriesNumber}`。每个组合仅保留数值最大的 AcquisitionNumber；10 大于 9，已知 0 不是缺失。全部 acquisition 缺失或非法时保留全部文件，否则舍弃没有编号的文件；相关问题记入 `scan_issues.csv`。
- **保留所有可识别的 MR unique_id**，包含 coverage 不足、重复平面、单平面、localizer 和投影图像；不运行第二阶段筛选。身份字段缺失或文件不可读时记录问题，不编造 unique_id。
- 字段汇总、文件列表和 coverage 全部来自选中的 acquisition。文件按 InstanceNumber、完整路径稳定排序，缺失或非法 InstanceNumber 记录问题。

输出目录必须尚不存在，且位于 DICOM 根目录之外。三个 CSV 均为 UTF-8 BOM，即使没有记录也保留固定表头：

| 文件 | 内容 |
|---|---|
| `series_meta.csv` | 每个 unique_id 一行的精简 metadata |
| `series_multivalue.csv` | `unique_id,multiple_fields`，只记录四个字段存在不同非缺失值的 series |
| `scan_issues.csv` | `study_id,unique_id,file,reason`，记录目录、读取、身份、缺失字段和 coverage 无法计算等问题 |

主表固定列：

```text
unique_id,study_id,SeriesInstanceUID,SeriesNumber,SeriesDescription,AcquisitionNumber,ImageType,SliceThickness,SpacingBetweenSlices,dicom_file_list,coverage_mm
```

`SeriesDescription`、`ImageType`、`SliceThickness`、`SpacingBetweenSlices`：对选中组的非缺失值去重，一个值输出标量，多个值输出方括号列表文本，不为元素添加引号，例如 `[AX T1, AX T2]`、`[4.0, 5.0]`。数值 5、5.0、5.000 视为相同，不提前四舍五入合并不同值；全部缺失输出 `unknown`，部分缺失保留已知值并记录问题。层厚和层间距读取普通顶层及 Enhanced MR shared/per-frame PixelMeasures；acquisition 选择仍基于顶层 AcquisitionNumber。

`ImageType` 将每个文件的完整有序组成项用 `|` 连接，例如 `DERIVED|SECONDARY|OTHER`。它整体算一个值；只有不同完整组合才触发多值记录，多值示例为 `[DERIVED|SECONDARY|OTHER, ORIGINAL|PRIMARY]`。`multiple_fields` 也使用不带元素引号的列表文本，例如 `[SeriesDescription, SliceThickness]`，每个 unique_id 至多一行。

`dicom_file_list` 始终是方括号列表文本，例如 `[DICOM_00001, DICOM_00002]`，**只保留文件名，不含目录和扩展名**：`acq2/IM0001.dcm` 输出为 `IM0001`。无扩展名时保留原文件名；不同目录下文件名相同的条目仍分别保留，不去重。

`coverage_mm` 沿用现有几何计算：有效一致几何下，首末独立切片平面中心的投影跨度，单位 mm；只有一个平面时为 0，无法计算时为 `unknown`，均保留该 unique_id。它不是按文件数乘层厚估算，也不是全脑覆盖确认。

上述列表按展示文本读取，不再是 JSON，不能使用 `json.loads()`。包含逗号的整个单元格仍由标准 CSV writer 添加外层引号，CSV reader 会正确去掉该外层引号；元素不再因 JSON 引号产生双重双引号。字段原始文本本身的逗号或引号仍会保留，列表文本不提供无歧义的元素转义。例如：

```python
import pandas as pd

table = pd.read_csv("series_meta.csv", keep_default_na=False)
files_text = table.iloc[0]["dicom_file_list"]  # [DICOM_00001, DICOM_00002]
```

## 当前版本：按 SeriesInstanceUID 区分 series

新入口为 `summary_dicom_SeriesInstanceUID.py` 和 `summary_dicom_SeriesInstanceUID_v2.py`。旧 `summariy_dicom.py`（原文件名有拼写错误）及 `summary_dicom_v2.py` 保留，供旧结果复现。

- `unique_id = "{Study_ID}_{SeriesInstanceUID}_{SeriesNumber}"`；CSV 同时保存标量 `series_instance_uid`。`SeriesNumber` 按整数处理，不包含 AcquisitionNumber。
- 每个 study 内按 `(SeriesInstanceUID, SeriesNumber)` 分组；相同 SeriesNumber、不同 UID 的图像不会再合并。每组独立选择数值最大的 AcquisitionNumber，再计算统计。
- 缺少 SeriesInstanceUID 的 MR 文件写入 `scan_issues.csv` 后跳过，不生成替代 UID。受扫描问题影响的 study 保留 `incomplete`／`partial_study_scan` 标记。
- 第一步仍统计全部 MR，包含 `PJN`、`FL:A/PJN`、`FL:B/PJN` 和 `ImageType=PROJECTION IMAGE`，以便理解投影图像和 unknown；预训练 manifest 单独排除投影图像。
- 新 v2 要求 `series_instance_uid` 列，并核对完整 unique_id；旧 ID 的 CSV 会报错，需要重新运行新第一步。覆盖、位置和重复平面的统计规则沿用旧版。
- 第一步 `selected_series.csv` 保存完整 ID 和 UID，是预训练 quality 的 `--selected-csv` 输入。v2 的 `csv/` 仍采用索引、不公开 UID；`csv_with_abnormal_study_ids/` 仅在异常行的 `abnormal_unique_ids` 中记录完整新 ID。v2 的这两套报告不作为 quality 输入。

依赖为 Python、numpy、pandas、pydicom；新入口与 `_series_statistics.py` 放在同一目录。

```bash
python src/brainIAC_pretraining/stat/summary_dicom_SeriesInstanceUID.py \
  --csv resources/labels.csv \
  --folder-list resources/folders.txt \
  --output-dir resources/all_series_uid_stats

python src/brainIAC_pretraining/stat/summary_dicom_SeriesInstanceUID_v2.py \
  --selected-csv resources/all_series_uid_stats/selected_series.csv \
  --output-dir resources/all_series_uid_quality \
  --coverage-threshold-mm 100 \
  --position-error-percent 20

python -m unittest discover -s src/brainIAC_pretraining/stat -p 'test_series_instance_uid_statistics.py' -v
```

新版本的 7 项合成测试覆盖：SeriesNumber 冲突时独立 UID 分组及各自最大 acquisition、同 UID 不同 SeriesNumber、缺失 UID 报告、统计保留 PJN、v2 拒绝旧 schema/ID、仅排除重复平面的对应 UID，以及正常／重复／空队列 CLI 和报告隐私。未在服务器真实 DICOM 数据上运行。

## 复制问题 series 所属的完整 DICOM study 供人工检查

入口为 `copy_problem_dicom_studies.py`。只需新版第一阶段的原始 `selected_series.csv`；其中有完整身份、描述、来源目录及选中最大 acquisition 的 frame_count／coverage_mm，不需要再运行 v2 或拼接 manifest、quality。v2 的分析报告隐藏了正常 ID、路径和描述，quality 不含描述且可能已筛掉部分 series，不能单独作为这里的输入。第一阶段保留 PJN，因此本程序能选出 manifest 已排除的投影序列。

```bash
python src/brainIAC_pretraining/stat/copy_problem_dicom_studies.py \
  --selected-csv /reports/all_series_uid_stats/selected_series.csv \
  --output-dir /data/dicom_manual_review \
  --frame-count-threshold 20 \
  --coverage-threshold-mm 100
```

20 帧、100 mm 仅为示例，两个阈值必须明确指定。以下任一条件满足即记录为问题 series：

- 统计中的 `series_description` 完全不含 T1 或 T2，不区分大小写；例如 PJN、LOCALIZER、仅写 FLAIR 的描述会触发，T2 FLAIR 不会因描述触发。空描述也按不含 T1/T2 处理。
- `frame_count < --frame-count-threshold`。image count 统一使用 frame_count，不使用 DICOM 文件数或独立平面数。
- `coverage_mm < --coverage-threshold-mm`。等于阈值不触发；coverage 沿用统计中的首末切片平面中心跨度。

未知、非有限、负数指标不当作低于阈值，写入 unknown_metrics.csv；若描述规则或另一个已知指标已触发，仍复制该 study。这里不额外按 v2 screen_status、位置误差或重复平面判定问题。

输出目录必须尚不存在，且不得与源 study 目录嵌套。程序在每个命中的 Study_ID 下**递归复制整个原 study 文件夹**到 `<output-dir>/<Study_ID>/`，包括正常序列、问题序列、所有 acquisition、嵌套目录和其他原有文件；不只复制统计选中的文件。同一 study 只复制一次，原件保留。CSV 的 study_directory 必须在执行机器上可访问。

| 输出 | 内容 |
|---|---|
| `problem_series.csv` | 每个问题 unique_id 一行：study_id、series_instance_uid、series_number、选中的 acquisition_number、frame_count、coverage_mm、series_description、reason、study_directory |
| `copied_studies.csv` | 每个命中 study 一行，包含来源与目标目录、问题 series 数、copied／copy_failed／pending 状态和错误 |
| `unknown_metrics.csv` | 数量或 coverage 无法比较的 series，保留完整身份及原因；不把 unknown 当作低值 |
| `<Study_ID>/` | 对应 study 的完整原始目录副本 |

problem_series.csv 的数量、coverage 和 acquisition 描述统计选中的组，复制目录则包含整个 study。多个问题原因用分号分隔。单个 study 完整复制后才发布正式目录；复制失败记录并继续其他 study，中断时报告仍保留已完成及 pending 状态。已有输出目录拒绝覆盖，再执行需指定新目录。

```bash
python -m unittest discover -s src/brainIAC_pretraining/stat -p 'test_copy_problem_dicom_studies.py' -v
```

8 项合成测试覆盖 OR 规则、大小写、严格阈值边界、unknown、完整 study 复制、按 study 去重、原件保留、复制失败继续、目标冲突、身份和来源冲突、空队列及 CLI 输出。

## 旧版入口与历史 CSV 说明

以下说明对应旧的 `{Study_ID}_{SeriesNumber}` 版本，不能用于新版预训练 pipeline 的输入。

这是 `code/single_study_check/summarize_dicom_t1t2.py` 与 `summarize_dicom_t1t2_v2.py` 的新版本，旧程序保留。

新入口名称：`summariy_dicom.py`（保留用户指定的拼写）和 `summary_dicom_v2.py`。

用户的 acquisition 审查结果：`R01_Study_002904` 是已审查数据中唯一出现两个 acquisition number 的 study；第一组质量较低，因此采用编号较大的组。新程序把该选择规则统一用于每个 series，**不把 acquisition 大小本身当作图像质量测量**。

## 身份与选择规则

- `unique_id = "{Study_ID}_{SerisNumber}"`。`SerisNumber` 使用 DICOM 正式字段 `SeriesNumber` 的整数值。例如：`R01_Study_002904_10`。
- ID 不含 AcquisitionNumber，不根据 T1/T2 描述命名。CSV 字段仍使用 `series_number` 对应 DICOM 字段。
- 对 labels CSV 中每个 study 递归读取所有 MR DICOM headers；同一个 SeriesNumber 只输出一行。
- 同一个 series 内有多个 AcquisitionNumber 时，选择**数值最大**的那一组；例如 10 大于 9，统计只使用选中组的图像。
- 不再选择最大的 T1/T2 SeriesNumber，也不要求 T1/T2 配对，不沿用旧脚本的四个序列例外。
- 不因描述不含 T1/T2 而排除 FLAIR、localizer 等 MR 序列。这里只统计和检查，未决定最终预训练资格。
- 所有 AcquisitionNumber 缺失时保留该 series 的所有文件，值写为 `unknown` 并标记；已知和缺失混杂时仅选已知最大组，缺失编号文件计入舍弃数。缺失不等于 0。

这些身份规则依照用户对当前数据的审查结论执行。所扫描范围内出现多个 UID、不同几何等情况仍按旧统计函数记录 flags，但不改用 UID 命名或自动细分新 ID。

## 运行

依赖：Python、numpy、pandas、pydicom。三个程序文件需要放在同一目录；无需访问本机旧脚本目录。`_series_statistics.py` 复用了原脚本的几何、切片位置和一致性统计函数。

第一步直接扫描 DICOM：

```bash
python src/brainIAC_pretraining/stat/summariy_dicom.py \
  --csv resources/labels.csv \
  --folder-list resources/folders.txt \
  --output-dir resources/all_series_stats
```

CSV 默认列名为 `Study_ID`，可通过 `--study-id-column` 指定其他列名。folder list 为一行一个根目录，每个 study 存放在 `root/Study_ID` 下；相对根路径以列表文件所在目录解析。一个运行内 study 必须恰好匹配一个根目录，缺失或多目录匹配记录在报告中，不静默合并。

第二步读取新版本第一步生成的 CSV，不再读取 DICOM：

```bash
python src/brainIAC_pretraining/stat/summary_dicom_v2.py \
  --selected-csv resources/all_series_stats/selected_series.csv \
  --output-dir resources/all_series_quality \
  --coverage-threshold-mm 100 \
  --position-error-percent 20
```

输出目录必须是新目录，第一步输出不能位于 DICOM 根目录内部。旧版两行一个 study 的 CSV 不可直接用于新 `v2`，因为缺少全 series 的唯一 ID 与 acquisition 选择结果。

## 第一步输出

| 文件 | 内容 |
|---|---|
| `selected_series.csv` | 所有唯一 series，已选最大 acquisition；保留全部 unique_id 和研究标识 |
| `flagged_series.csv` | 带统计 flags 的 series，包括 metadata 限制及选择说明，不仅是质量缺陷 |
| `study_summary.csv` | 每个 CSV study 的扫描状态、series 数、选中及舍弃文件数；包含找不到的 study |
| `scan_issues.csv` | 无法读取文件、缺失 SeriesNumber、study 目录匹配问题 |
| `distribution_summary.csv` | 按每个唯一 series 等权的切片数、覆盖跨度、spacing 等分位数，逐指标保留 valid/missing 分母 |
| `sequence_study_counts.csv` | 按选中组的原始描述汇总去重 study 数和唯一 series 数，不推断 MRI 对比类别 |
| `summary.txt` | 扫描范围、选择规则、状态和汇总 |

每个 series 记录 `acquisition_number`、全部已知 `acquisition_numbers`、`source_file_count`、`discarded_file_count` 和 `missing_acquisition_file_count`。`file_count`、`frame_count`、`unique_slice_count`、几何数组和文件清单仅来自**选中的 acquisition**。多个原始描述以 ` | ` 拼接并标记，仍不生成额外 ID。

覆盖指标是首末切片平面中心间的跨度，不是人工确认的完整脑覆盖。DICOM 文件数、frame 数和独立切片平面数分别统计，不读取或解码像素。

## 第二步输出及与旧版的差别

保留旧覆盖与位置阈值含义：覆盖跨度严格小于阈值为 fail；任一切片位置误差严格大于该 series 跨度的指定百分比为 fail；等于阈值不算失败。位置模型按导出的 InstanceNumber 顺序与首末端点等间距模型比较，不是实际配准误差。

重复切片平面只排除**对应的 unique_id**，其他 series 即使来自同 study 也保留。`study_quality_v2.csv` 汇总该 study 的 series 个数和各结果数量，不再强制 T1/T2 同时通过。

两个 CSV 目录：

- `csv/`：使用 `study_index` 和 `series_index`，不复制原始 study/unique ID、文件名、路径或 UID。
- `csv_with_abnormal_study_ids/`：同样表格增加异常 study ID；逐 series/切片表还增加 `abnormal_unique_ids`。正常行不填原始 ID；所有正常 ID 的完整映射仍在第一步 `selected_series.csv`，index 按第一步输入行顺序建立。

每个目录包含 `series_quality_v2.csv`、`excluded_repeated_series.csv`、`study_quality_v2.csv`、`slice_position_deviations.csv`、`slice_count_analysis.csv`、`slice_count_thresholds.csv` 和 `secondary_checks.csv`；另有 `summary_v2.txt`。

缺失几何、缺失 InstanceNumber、扫描不完整及未建模的 multiframe 位置检查保持 unknown。Enhanced MR 的 acquisition 选择依赖顶层 AcquisitionNumber，不支持按 frame 内的 acquisition 重新选择。阈值 pass 只是这些 header 检查的结果，不等于图像质量、全脑覆盖或模型可用性已获确认。

## 旧版 CSV 的逐 series 低 coverage 临床导出

此导出器保持旧版行为，只接受旧 `{Study_ID}_{SeriesNumber}` 的统计 CSV；目前不接受 `all_series_uid_stats/selected_series.csv`。下方 `all_series_stats` 指旧版统计目录。

`export_low_coverage_clinical.py` 输出 `coverage_mm < 阈值` 的 `{Study_ID}_{SeriesNumber}`，每个 series 一行，并标注是否是旧 T1/T2 流程实际选中的 series。

```bash
python src/brainIAC_pretraining/stat/export_low_coverage_clinical.py \
  --selected-csv resources/all_series_stats/selected_series.csv \
  --old-selected-csv resources/old_t1t2_stats/selected_series.csv \
  --labels-csv resources/labels.csv \
  --output-csv resources/low_coverage_series_clinical.csv \
  --coverage-threshold-mm 100
```

- `--selected-csv` 为新全 series 统计第一阶段的原始 CSV；`--old-selected-csv` 为旧 T1/T2 统计第一阶段的原始 CSV，必须有 study_id、modality、status、series_number。旧临床导出 CSV 没有 SeriesNumber，不能用于判断旧选择。
- 使用旧 CSV 的实际选择，不重新按 SeriesDescription、最大 SeriesNumber 或例外规则推测。标注只比较 Study_ID + SeriesNumber，不要求新旧 acquisition 相同，也不表示旧 embedding 成功。
- `is_previous_t1t2_selected` 为 yes/no/unknown；`previous_selected_modality` 为 T1、T2、T1;T2 或空值。没有旧 study 记录或旧选择失败时，无法确认的 series 标为 unknown，不误记为 no。已确认的选中记录仍标 yes。
- 输出 unique_id、Study_ID、series_number、临床列、frame_count、coverage_mm 及两个旧选择标注列。临床列沿用旧程序：labels.csv 第一列和最后一列不导出，Study_ID 只用于连接；同一 study 的多个 series 分别保留，缺失临床记录仍输出并留空。字符串 ID、前导零和字面值 NA 保留。
- 默认阈值 100 mm，严格小于才导出，等于阈值不导出；未知、非有限或负 coverage 不纳入。只使用 status=selected 的新统计记录，不额外强制纳入重复平面 series，也不添加 excluded_ ID 前缀。需要涵盖 [100,101) 时使用阈值 101。
- 直接使用第一阶段的 coverage，不要求通过预训练 quality 筛选。不会读取 DICOM、执行转换、移动图像或引入哈希校验。输出 CSV 为 UTF-8 BOM，已有输出拒绝覆盖。

## 验证

```bash
python -m unittest discover -s src/brainIAC_pretraining/stat -p 'test_series_statistics.py' -v
python -m unittest discover -s src/brainIAC_pretraining/stat -p 'test_export_low_coverage_clinical.py' -v
```

8 项合成测试包含：46 文件双 acquisition 选择为 23 文件、数值最大值、所有描述及旧例外序列都纳入、递归扫描、缺失 acquisition/instance/几何、重复平面只影响本 series，以及正常/全部排除/空队列的两步 CLI 运行。尚未在服务器的真实数据上运行这两个新脚本。
