# 检查 T1/T2 是否选中了同一批 DICOM 文件

`check_dicom_t1t2_same_files.py` 根据 labels CSV 和 DICOM 根目录列表，逐个 study 检查原 T1/T2 筛选最终使用的文件。它直接调用 `summarize_dicom_t1t2.py` 的 `inspect_study()`，沿用原分组、字符串匹配、SeriesNumber 最大值、并列处理及原有例外规则。

## 运行

将新程序与 `summarize_dicom_t1t2.py`、`inspect_dicom_t1.py` 放在同一目录，在原 `vit_survival` 环境运行，无新增第三方依赖：

```bash
python check_dicom_t1t2_same_files.py --csv /data/labels.csv --folder-list /data/folder_address.txt --output-dir /reports/t1t2_same_files
```

输入参数与原队列统计程序相同：

- `--csv`：含 Study ID 的 labels CSV，只使用 ID 列，不导出 clinical 数据。
- `--folder-list`：默认当前工作目录中的 `folder_address.txt`。每行一个 DICOM 根目录，空行忽略；相对路径以 TXT 所在目录为基准。
- `--study-id-column`：默认 `Study_ID`，可指定其他列名。
- `--output-dir`：尚不存在的新目录，必须位于 DICOM 根目录之外。

每个 study 应完整位于一个 `<root>/<Study_ID>/` 中。程序检查全部输入 study，包括低 coverage、重复层面或先前 embedding 失败的 study。每完成 100 个 study 打印一次进度。

## 如何判定

原程序按 `(SeriesNumber, SeriesDescription)` 分组；T1、T2 分别在描述包含对应字符串的组中选择 SeriesNumber 最大者，匹配不区分大小写。因此，两边可以选中同一个包含 `T1`、`T2` 的描述组。

新程序比较同一个已定位 study 目录内的实际文件名集合，另比较文件列表的顺序。不是根据层数相同、coverage 相同或 SeriesInstanceUID 相同推断。

| comparison_status | 含义 |
|---|---|
| `same_files` | 两边选中的文件集合完全相同；检查 `same_file_order` 可知排列顺序是否也相同 |
| `disjoint` | 两边都成功选中，但没有共同文件 |
| `partial_overlap` | 有共同文件，但集合不完全相同；按当前互斥分组规则不应出现，若出现需复核筛选实现 |
| `unknown` | 无法比较：study 缺失、同一 study 位于多个根目录、某个直接子文件读取失败、或缺少 T1/T2 候选组等 |

文件不重合不代表像素内容必然不同：不同路径下的副本不属于本次检查范围。程序只读 DICOM headers，不读取像素进行比较，也不执行 NIfTI 转换。调用原 `inspect_study()` 时也会完成它已有的几何统计，因此扫描工作量与原队列统计接近；几何标记不会用于排除 study。

## 输出

| 文件 | 内容 |
|---|---|
| `study_file_overlap.csv` | 所有输入 study，每个 study 一行，含成功比较和 unknown |
| `overlapping_studies.csv` | 文件集合相同或部分重合的 study，供直接追查 |
| `unknown_studies.csv` | 无法比较的 study，保留每个模态的状态和错误信息 |
| `summary.txt` | 全部 study 数、可比较数量、重合/不重合/未知数量，以及相同文件的比例 |

三个 CSV 使用相同表头，即使某个子集为空也保留表头。包含以下信息：

- `Study_ID`、`study_directory`：定位源 study。
- `comparison_status`、`same_file_set`、`same_file_order`、`shared_file_count`：主要判断。
- `shared_file_names`：共同文件列表。
- `T1_*` / `T2_*`：各自的选择状态、SeriesNumber、SeriesDescription、候选组数、文件数、SeriesInstanceUID 列表、文件列表、原 flags、错误信息。

列表字段使用 JSON 文本，CSV 编码为 UTF-8 BOM。路径与文件名为源数据的内部追查信息。`unknown` 行的 `shared_file_count` 写为 `unknown`，不写为 0；其中空共同文件列表不代表已经确认没有重合。

摘要同时报告相同文件 study 占全部输入 study 的比例，以及占两模态都成功选择 study 的比例。无法判断者不算作不重合。

## 验证

```bash
python -B -m unittest -v test_check_dicom_t1t2_same_files.py
```

8 项合成测试覆盖同组同文件、重复层面仍纳入、同 UID/同几何但不同文件、最高编号与并列规则、原例外规则、缺失/歧义/读取失败、集合与顺序的区别、两个根目录、默认及显式 folder-list、ID 前导零/字面 NA、自定义 ID 列、输出及空子集表头。

程序已在本地通过合成数据验证；真实 cohort 的检查需在可读取 labels CSV 和全部 DICOM 根目录的服务器运行。
