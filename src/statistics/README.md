# 每个 study 的全部 MRI 序列统计

SeriesNumber/AcquisitionNumber 多值及 UID 碰撞检查：见 [README_acquisitions.md](README_acquisitions.md)，使用 `check_series_acquisitions.py` 扫描全部 DICOM，已知 `R01_Study_002904` 单独汇总。

程序：`summarize_dicom_sequences.py`。只依赖 pydicom 和 Python 标准库，不需要 GPU，不读像素，不修改源 DICOM。将脚本放到服务器即可运行，无需其他项目脚本。

GitHub 位置：`YueHu00000/BrainIAC` 的 `src/statistics/`。在仓库根目录运行 `cd src/statistics` 后使用以下命令；环境需已安装 pydicom（本地测试版本为 3.0.1）。

```bash
python summarize_dicom_sequences.py --csv /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/resources/labels.csv --folder-list /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/resources/folders.txt --output-dir /home/a470829/brainMRI/brainIAC_stat/reports/all_mr_sequences_v1
```

上述输入路径来自现有真实统计摘要；运行前确认服务器上仍有效。输出目录必须尚不存在，并位于 DICOM 根目录之外。

## 计数口径

- 每个 CSV study 对应一个 `<根目录>/<Study_ID>`，递归读取全部文件，不依赖扩展名。folder list 每行一个根目录，相对路径相对于 TXT 所在目录。默认 ID 列为 `Study_ID`，可用 `--study-id-column` 指定。
- 只统计 DICOM `Modality=MR`。其他已识别模态文件计入 `non_mr_file_count`。
- 独立序列由 `(StudyInstanceUID, SeriesInstanceUID)` 区分。同 UID 多张切片只算一个序列；不同 UID 即使描述相同，也分别计数。
- “序列种类”使用 `SeriesDescription` 原文，仅去掉两端空白。不合并大小写、协议别名、方向或参数差异，也不推断 T1、T1CE、T2、FLAIR 等标准类别。
- 同一个 study 中出现多个同名序列，该描述的 `study_count` 只加一，`series_count` 则累加全部独立序列。
- 包含定位像、派生图像及重复采集，不应用原 T1/T2 选择或四个排除规则。一个 UID 下不同 acquisition 仍是一个 series。

## 输出

| 文件 | 内容 |
|---|---|
| `study_sequence_counts.csv` | 每个 study 一行：`series_count` 独立序列数、`sequence_type_count` 不同描述数、描述列表、文件数及状态 |
| `all_mr_series.csv` | 每个可识别 MR series 一行：study/series UID、序列号、描述列表、文件数 |
| `sequence_study_counts.csv` | 每种描述覆盖多少 studies、占比、独立序列总数；按 study 数降序 |
| `series_count_distribution.csv` | 含 0、1、2…个序列的 study 分别有多少；只计总数已知的记录 |
| `scan_issues.csv` | 文件读取失败、字段缺失、描述冲突、study 缺失或多根目录重复等问题 |
| `summary.txt` | 统计口径、状态数量、序列数分布与每种描述覆盖数 |

列表字段以 JSON 写入 CSV。ID 按字符串保留前导零。CSV 使用 UTF-8 BOM。

## 不完整记录

读取失败、缺少 Modality 或 study/series UID 时，无法确认序列总数，记为 `unknown`，不补零。若只是描述缺失或同 UID 描述不一致，仍报告可确认的 UID 序列数，但描述种类数为 `unknown`。

`sequence_study_counts.csv` 只统计状态为 `complete` 的 studies；百分比分母是这些完整且有 MR 序列的 studies。`incomplete`、`study_not_found`、`study_ambiguous`、`no_mr_series` 分别记录，不进入该分母。因而发生读取问题时，此表不是完整队列的最终覆盖数；先检查 `scan_issues.csv`。可读取的部分 series 仍保存在明细中。

未读取到 MR 且没有读取错误时，状态为 `no_mr_series`，两个计数为 0。无法读取的普通文本等非 DICOM 文件也会记为问题，不会静默跳过。

## 验证

```bash
python -B -m unittest -v test_summarize_dicom_sequences
```

合成 DICOM 验证 UID 去重、同名多序列的 study 去重、递归读取、非 MR 文件、缺失/冲突字段、目录缺失与歧义，以及 CLI 输出、前导零、重复 ID 和拒绝覆盖。本地验证不代表真实服务器队列已运行。
