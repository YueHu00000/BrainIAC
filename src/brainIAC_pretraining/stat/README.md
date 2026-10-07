# 全 series 统计：SeriesInstanceUID 唯一 ID 与最大 acquisition

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
