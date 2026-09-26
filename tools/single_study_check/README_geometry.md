# DICOM T1/T2 coverage、最远距离与 Z 跨度对比

程序：`summarize_dicom_geometry.py`。在 `vit_survival` 环境运行，只需已有的 numpy、pandas、pydicom，不需要 GPU、SimpleITK 或像素解码。

```bash
conda activate vit_survival
python summarize_dicom_geometry.py --csv /path/to/labels.csv --folder-list /path/to/folder_address.txt --output-dir /path/to/reports/geometry_v1
```

将程序与同目录的 `summarize_dicom_t1t2.py`（本次更新版本）及 `inspect_dicom_t1.py` 一起部署。原统计程序默认行为不变。

`labels.csv` 默认 ID 列为 `Study_ID`，可通过 `--study-id-column` 指定。folder list 每行一个包含 Study_ID 子目录的根目录；一个 study 应完整位于一个根目录中。相对路径以 folder list 所在目录为基准。输出目录必须是新的目录，且位于 DICOM 根目录之外。

## 选择规则和研究范围

直接复用 `summarize_dicom_t1t2.inspect_study`：读取 study 目录直接子文件，按 SeriesNumber、InstanceNumber 排序，按 `(SeriesNumber, SeriesDescription)` 分组，应用原有四个例外规则，分别从描述包含 T1/T2 的组中选 SeriesNumber 最大者。

这对应当前基于 neuroimage-classifiers 规则的 BrainIAC embedding 转换流程，不是 BrainIAC 上游按文件名排序的原始转换脚本。

全部 CSV study 均保留，包括低 coverage 和重复采集 study。一个 modality 缺失不会阻止另一个 modality 的统计。不读取 labels 中的临床信息。

## 五个指标

设每帧的 `ImagePositionPatient (0020,0032)` 为 P，`ImageOrientationPatient (0020,0037)` 中行列方向的叉乘经归一化为 n。

| 输出字段 | 计算及含义 |
|---|---|
| `coverage_mm` | `max(P·n) - min(P·n)`，垂直于切片的投影跨度 |
| `distance_mm` | 所有图像对中最大的 `norm(Pb-Pa)`，不是只比较排序后的首尾图像 |
| `z_mm` | 上述同一最远图像对的 `abs(Pb.z-Pa.z)`；并非独立寻找全组最大 Z 跨度 |
| `distance_minus_coverage_over_distance` | `(distance-coverage)/distance` |
| `z_minus_coverage_over_distance` | `(z-coverage)/distance`，保留正负号 |

距离单位 mm，比值为小数而非百分数，例如 0.01 表示 1%。第二个比值可为正或负。由于浮点误差，第一个比值可能出现极小的负数，不强制裁剪。

P 是每张 image **第一个像素中心**，不是图像中心，也不是整个图像平面或脑组织范围。整组倾斜但沿法向整齐排列时，coverage 与 distance 相等，而 z 可能更小。

新程序直接计算精确投影极差。旧统计先按 0.01 mm 容差合并相近平面再求跨度；如果极端位置被合并，新旧 coverage 可能有不超过 0.01 mm 的差异。选组、unique_slice_count 和重复平面标志仍来自旧统计函数。

最远对通过遍历所有点对确定，时间复杂度 O(N²)，临时内存 O(N)。距离相等时保留转换顺序中先遇到的一对；不同并列最远对可能有不同 z。CSV 保存两个端点坐标及从 0 开始的帧索引，便于核对。单帧 DICOM 时帧索引也是选中文件列表的索引；多帧 DICOM 时按文件顺序和文件内帧顺序展平。

## 输出

- `study_geometry.csv`：每个 study 一行，T1/T2 各自的五个指标及状态，含 Study_ID。
- `series_geometry.csv`：每个 study 两行，含选组信息、帧数、重复平面标志、端点坐标及帧索引。
- `distribution_summary.csv`：T1/T2 分开统计，列出总数、有效数、unknown 数、mean、min、P1/P5/P10/P25/median/P75/P90/P95/P99/max。
- `summary.txt`：输入、公式、统计范围、状态计数及上述分布表。队列分布本身不包含 Study_ID，逐项明细含 Study_ID。

每个 study/modality 权重相同，不按切片数加权。有效值参与统计，unknown 不补零。

## 无法计算的情况

沿用原程序的几何检查：位置缺失、多帧位置不完整、切片方向不一致、非法方向向量、多个不兼容坐标系时，五个指标整体记为 unknown；原因保存在 flags 中。目录缺失、多根目录出现同一 study、文件读取失败等也保留状态。

只有一帧或所有位置完全相同时，distance 为 0，两个比值因分母为零记为 unknown，几何状态为 `zero_distance`。重复位置本身不阻止计算，但保留警告，不能把距离正常理解为转换后的三维图像有效。

## 验证

```bash
python -m unittest test_summarize_dicom_geometry test_summarize_dicom_t1t2
```

合成测试包括：30° 倾斜无横向偏移、2 mm 横向偏移、最远对不是首尾图像、同一最远对的 Z 定义、正负比值、重复位置、零距离、unknown 统计分母、多帧 DICOM，以及多个根目录的命令行完整流程。
