# Summary v2：覆盖范围与层位置偏差统计

`summarize_dicom_t1t2_v2.py` 读取原统计程序生成的 `selected_series.csv`，沿用其中已经选好的 T1/T2，不重新选择序列或扫描 DICOM。原 `summarize_dicom_t1t2.py` 保持不变。重复层面的 study 使用 `inspect_repeated_slice_study.py` 单独检查。

## 运行

在 `vit_survival` 环境中，与现有检查程序放在同一个目录；依赖仍为 NumPy、pandas、pydicom，无新增第三方依赖。

```bash
python summarize_dicom_t1t2_v2.py --selected-csv /reports/selected_series.csv --output-dir /reports/summary_v2 --coverage-threshold-mm 100 --position-error-percent 20
```

输出目录必须尚不存在，避免覆盖既有结果。输入须包含每个 study 的 T1、T2 两行，包括原程序的选择失败记录；输入 ID 作为字符串读取，保留前导零。

## 主分析的判断

1. `unique_slice_count` 用作分组变量：展示精确层数分布及累计分布，不强加最低层数。
2. `coverage_mm < 100` 为覆盖检查失败；等于 100 通过。此值是沿层法向首尾中心的跨度，不是已验证的完整脑覆盖。
3. 按输入 CSV 中 `projected_positions_mm` 的原始文件顺序，比较实际位置与首末点之间等距排列的位置：

```text
q[i] = p[0] + i * (p[N-1] - p[0]) / (N-1)
error[i] = abs(p[i] - q[i])
threshold_mm = coverage_mm * 20 / 100
```

任何一层 `error > threshold_mm` 即为位置检查失败；等于阈值通过。例如实际位置 `0,63,70 mm`，等距位置 `0,35,70 mm`，第二层偏差 `28 mm`、占覆盖范围 `40%`，有一层超过默认 `14 mm` 阈值。

位置列表不能先排序：排序会掩盖原文件顺序的问题。反向但均匀的序列可以通过。这里实现的是已约定的**沿层方向一维模型**，不计平面内偏移，不是完整三维 ITK 模拟，也不是配准后实测误差。位置不完整、少于两层、覆盖为零、无法确认唯一层面或多帧转换模型未覆盖时，位置检查为 `unknown`。ITK 实际三维间距计算可对照 [ImageSeriesReader 源码](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/IO/ImageBase/include/itkImageSeriesReader.hxx)。

一个 series 的主筛查结果：任一主检查明确失败则 `fail`；两个检查均通过且层数已知才是 `pass`；其余为 `unknown`。一个 study 的 T1/T2 均通过才算 study 通过；任一模态失败即 study 失败；其余未知。辅助检查不改变主筛查结果。

**100 mm 和 20% 是可调整的研究筛查规则。`pass` 表示通过这两项规则，不等于图像质量、采样密度、完整脑覆盖或模型输入适用性已得到验证。** 例如只有三层、位置 `0,70,140 mm`，可以通过这两项规则，但采样仍很稀疏；因此报告保留完整的层数分布。

## 重复层面与统计分母

从 `duplicate_plane_count > 0` 或 `repeated_slice_plane` 标记自动识别 study；只要任一模态命中，整个 study 的 T1/T2 均从主分析剔除，记录到独立排除表。不会硬编码某个 Study ID 或假设恰好只有一个。重复平面可能来自不同 echo、时间点或同位置重复采集，不能仅据此认定像素重复或数据不可用。

主分析比例的分母为剔除后的 cohort。报告同时列出 `pass/fail/unknown`；`screen_pass_percent_all` 的分母包括 unknown，`screen_pass_percent_classified` 的分母仅为 pass+fail。位置异常切片比例按已评估切片总数计算，不是 series 比例。

## 两套 CSV

每个程序的输出目录下都有两个子目录，文件一一对应：

- `csv/`：不含原始 Study ID。
- `csv_with_abnormal_study_ids/`：只在对应表的最后增加 `abnormal_study_ids` 列，其他列、行顺序和数值完全相同。

最后一列使用 JSON 列表，例如 `["000123", "000456"]`，ID 去重；该行无已识别异常时留空。**空白不能单独解释为正常，必须同时查看状态列。** 正常和 unknown 记录不会在末列输出 ID。

明细使用 `study_index` 关联记录：它是本次输入中 study 首次出现顺序对应的 1 起始内部编号，重复层面 study 被剔除后可能出现跳号；不是原始 Study ID，也不保证输入重排后编号不变。两套共同列均不输出源目录、源文件名、原始 UID 或自由文本错误消息，避免这些字段带出 Study ID。此处理是本次字段导出规则，不代表通用医疗数据匿名化认证。

| CSV 文件 | 内容 | 最后一列列出哪些异常 ID |
|---|---|---|
| `series_quality_v2.csv` | 每个保留 study 的 T1/T2 指标及主检查状态 | 当前 series 的主筛查为 fail |
| `study_quality_v2.csv` | T1、T2 及合并后的 study 状态 | 当前 study 为 fail |
| `slice_position_deviations.csv` | 每层实际位置、等距位置、偏差与阈值；`slice_index` 从 0 开始 | 当前切片超过位置偏差阈值 |
| `slice_count_analysis.csv` | 按模态和精确层数分组，计数及通过比例 | 当前组主筛查失败的 study，去重 |
| `slice_count_thresholds.csv` | 按模态及候选层数阈值分组，分 below / at_or_above | 当前区间主筛查失败的 study，去重 |
| `secondary_checks.csv` | 一致性检查及原始 flags 的计数 | 当前辅助检查标记异常的 study |
| `excluded_repeated_studies.csv` | 因重复平面整 study 排除的 T1/T2 记录 | 被排除的 study，两行均列出 |

同一 study 可以在多个异常分组出现；不能将这些 ID 数直接相加。辅助检查异常 ID 也不等于主筛查失败 ID。study 仅 T1 失败时，series 明细中正常 T2 的末列为空，study 汇总行仍列出该 ID。

累计表的候选阈值是每个已观察层数加 1，`below` 为严格小于该值，`at_or_above` 为大于等于；未知层数不进入这两个区间，另列 `unknown_slice_count`。英文摘要中的 observed boundary 取某类已观察异常的最大层数加 1，用来概括当前数据，**不是经验证的合格层数阈值**，不假定层数越多一定越可靠。

辅助检查对 pixel_row、pixel_column、thickness、tag_spacing 的 series 内 min/median/max 做一致性比较，容差 `atol=1e-5, rtol=1e-4`；缺失或无效值归 unknown，不能作为一致性通过。Rows/Columns 使用原统计的缺失/不一致标记。其他已出现的 flags 作为观察结果计数；未出现某个 flag 不等于独立验证无缺陷。辅助参数在 series 内一致，不代表其绝对数值适合本研究。

`summary_v2.txt` 是英文主报告，仍包含输入路径、被排除的 Study ID，用于内部追查；本次“两套无 ID / 异常 ID”的约定专指 CSV。

## 单独检查重复层面的 study

从带异常 ID 的排除表定位源 study 目录，然后运行：

```bash
python inspect_repeated_slice_study.py /dicom/root/STUDY_ID --output-dir /reports/repeated_study_check
```

该程序使用原程序的 T1/T2 选择规则，只读取该 study 的 DICOM headers，不读像素。输出目录必须是 study 目录之外的新目录。

两套 CSV 各有：

- `selected_series.csv`：选中序列及统计；若该 study 存在重复平面，两行末列均列出排除 ID。
- `frames.csv`：逐帧位置、层面分组、echo/time 等信息；只有重复平面的帧在末列列出 ID。
- `repeated_plane_groups.csv`：每个重复层面及其成员的 `order_indices`；每行末列列出 ID。

此处 `study_index` 固定为 1；帧的 `order_index`、`frame_index_in_file` 从 0 开始。`file_index`、`sop_index`、`series_uid_index`、`frame_of_reference_index` 从 1 起，为当前检查中源文件/UID 的替代编号；相同源值具有相同编号，保留比较关系，不输出原始值。不同模态的 `order_index` 要结合 modality 解释。

`repeated_slice_report.txt` 保留源文件名、UID 等内部追查信息。元数据一致不能证明像素相同；echo/time 不同则可能提示多维采集，需要结合图像继续判断。

## 验证及范围

```bash
python -B -m unittest -v test_summary_v2.py
```

14 项合成测试覆盖位置偏差、严格阈值、反向/非单调顺序、未知值、多帧边界、整 study 排除、非单调质量分布、辅助检查、空 cohort、ID 字符串保留、两个 CLI、重复 echo/SOP、Enhanced MR 元数据，以及两套 CSV 共同列完全一致、末列异常 ID 与对应检查匹配。

当前本地只有原真实 `summary.txt`，没有真实 cohort 的 `selected_series.csv`。因此程序已用合成数据验证，但尚未生成真实约一万 study 的 v2 结果。请使用服务器上的原 CSV 运行上述命令。
