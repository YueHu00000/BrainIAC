# SeriesNumber 与 AcquisitionNumber 检查

`check_series_acquisitions.py` 输入 labels CSV 和 DICOM 根目录列表，递归只读全部 DICOM headers，检查同一 CSV study 内一个 SeriesNumber 是否出现多个非缺失 AcquisitionNumber。不读像素，不转换、拆分或修改图像，不计算任何文件校验值。依赖 pydicom 及同目录的 `summarize_dicom_sequences.py`（CSV 写出函数）。

在 `src/statistics` 目录运行：

```bash
python check_series_acquisitions.py --csv /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/resources/labels.csv --folder-list /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/resources/folders.txt --output-dir /home/a470829/brainMRI/brainIAC_stat/reports/series_acquisitions_v1
```

输入路径来自既有真实报告，服务器运行前确认仍有效。默认 ID 列为 `Study_ID`，可用 `--study-id-column` 更改。ID 以字符串读取，保留前导零；重复 ID 报错。folder list 每行一个包含 study 子目录的根目录，相对路径相对于 TXT。输出目录须不存在且位于 DICOM 根目录之外。

## 输出

| 文件 | 内容 |
|---|---|
| `summary.txt` | 除已知病例外是否观察到多 acquisition；study/编号组数、选中 T1/T2 命中、UID 碰撞和问题数量 |
| `study_summary.csv` | 所有输入 study 的读取状态、命中组数、碰撞数 |
| `all_series_number_groups.csv` | 每个 study/SeriesNumber 一行，含 acquisition 值及文件数、UID、描述和选择标记 |
| `multiple_acquisition_groups.csv` | acquisition 值超过一个的组，排除 `R01_Study_002904` |
| `known_case_groups.csv` | 已知病例的全部编号组，单独留作核对；只有其在输入 CSV 中时才有记录 |
| `multiple_acquisition_files.csv` | 所有命中组的逐文件 header 信息，含已知病例标记、UID、instance、echo/time、位置/方向及帧数 |
| `pair_collisions.csv` | 同一 study 内同一 SeriesNumber+AcquisitionNumber 对应多个 `(StudyUID,SeriesUID)` 的组 |
| `scan_issues.csv` | 读取失败、字段缺失/非法、多帧对象、study 缺失/歧义 |

列表采用 JSON，CSV 使用 UTF-8 BOM。`files_per_acquisition` 是各非缺失 acquisition 的文件数，缺失值另列；文件数不等于独立层数。

## 如何解读

1. 首先查看 `summary.txt` 的 `Other studies with observed multiple acquisitions`，再查 `multiple_acquisition_groups.csv` 定位病例。
2. `selected_t1t2` 表示编号组内包含原规则选中的直接子文件；`selected_t1_acquisition_numbers`、`selected_t2_acquisition_numbers` 仅统计实际选中描述组，`selected_multiple_acquisition_modalities` 给出真正的所选 T1/T2 多值情况。同编号其他描述的多值，不自动算作所选 T1/T2 多值。
3. 原规则用直接子文件的 `(SeriesNumber,SeriesDescription)` 分组，描述含 T1/T2，取最大编号，保留四个历史例外。并列最大编号时按排序后的第一个描述展示，同时报告 `max_number_tie`。直接子文件读取失败或旧规则必需字段不足时标为 `not_evaluable`。嵌套文件参与全部库存，但不加入旧选择规则。
4. `series_identity_count > 1` 表示同一个编号跨多个 UID 身份。`uid_with_multiple_acquisitions > 0` 才直接支持“同一 UID series 内有多次 acquisition”。这两种情况应分开理解。
5. `pair_collisions.csv` 中有记录，证明仅用编号组合不能区分所有 UID series 身份。没有记录也不能证明全局唯一或 volume 正确：study、UID、echo/time、几何和帧信息仍需核对。
6. AcquisitionNumber 缺失不当作 0；合法值 0 是实际编号。不完整 study 也保留已经观察到的明确命中；其“无命中”不能证实没有多 acquisition。
7. 多帧对象会记为问题，并保留顶层 AcquisitionNumber；本工具不分析 per-frame acquisition。不能用本结果宣称所有 Enhanced MR 帧均已核对。
8. 本工具不自动决定拆分还是合并，不生成最终图像 unique name。命中组的逐文件信息用于后续核对；唯一文件命名也不能替代正确 volume 分组。

本地合成验证：

```bash
python -B -m unittest -v test_check_series_acquisitions
```

覆盖所选/未选序列、同编号不同描述、UID 碰撞、缺失 acquisition、合法0值、递归文件、读取失败、多帧、旧例外/并列规则、已知病例独立汇总与 CLI 输出/拒绝覆盖。真实服务器 cohort 需另行运行，合成通过不代表队列没有其他命中。
