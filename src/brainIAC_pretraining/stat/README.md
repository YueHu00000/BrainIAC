# 全 series 统计：按唯一 ID 选择最大 acquisition

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

## 验证

```bash
python -m unittest discover -s src/brainIAC_pretraining/stat -p 'test_series_statistics.py' -v
```

8 项合成测试包含：46 文件双 acquisition 选择为 23 文件、数值最大值、所有描述及旧例外序列都纳入、递归扫描、缺失 acquisition/instance/几何、重复平面只影响本 series，以及正常/全部排除/空队列的两步 CLI 运行。尚未在服务器的真实数据上运行这两个新脚本。
