# 旧 ViT NORM-A embedding 转换

将 `vit_survival_simple` 已产生的成对 T1/T2 切片 CLS 表征转换为当前 `CBI_WMD_predict/train.py` 的输入。转换程序只依赖 Python 标准库和 NumPy，不加载编码器、不重新编码，也不重新训练或拟合 NORM-A。原始数据只读；输出目录必须不存在。

## 输入与前提

| 参数 | 含义 |
| --- | --- |
| `--legacy-results-dir` | 旧结果根目录；默认递归扫描其 `embeddings/` |
| `--legacy-embedding-dir` | 可选，精确限定旧 embedding 子目录；有多个修正版时由调用者明确选择 |
| `--selected-csv` | 全量、第一阶段的 `selected_series.csv`，来自 `summary_dicom_SeriesInstanceUID.py` |
| `--labels-csv` | 默认含 `Study_ID,SiteID,idCBI,idWMD` 的标签表 |
| `--output-dir` | 新输出目录；包括 audit-only，存在即拒绝 |
| `--audit-only` | 验证序列和向量，输出报告与配置，不输出 NPZ 和训练 manifest |

标签列名可用 `--study-id-column`、`--site-id-column`、`--cbi-column`、`--wmd-column` 修改。按用户已确认的规则，`Study_ID` 同时是患者身份，必须唯一。标签只允许 0、1 或缺失；两个任务各自如何处理缺失由现有训练程序决定。Harvey 分组以外的 site 排除。

统计输入必须包含下列字段：

```text
unique_id,study_id,series_instance_uid,series_number,series_description,
series_descriptions,acquisition_number,acquisition_numbers,
source_file_count,discarded_file_count,missing_acquisition_file_count,
file_count,frame_count,coverage_mm,status,flags
```

其中 `series_descriptions`、`acquisition_numbers`、`flags` 是统计程序原样输出的 JSON 列表。不要使用第二阶段匿名筛选表、仅包含预处理成功序列的表，或已经做过质量筛选的表。CSV 本身无法证明不存在被提前删掉的候选，因此输入全量性是运行前提。

身份映射采用用户确认的前提：原始数据和旧 Harvey 选择规则没有改变。旧 NPZ 不包含足以逐张证明 DICOM 身份的元数据；这里从全量统计反推身份，报告中记录为 `inferred_harvey`。本程序不要求补交历史选择记录，也不把推导结果称为直接来自旧 NPZ 的 UID 证明。

## 选择与排除

1. 对每个 Study_ID，分别匹配描述中不区分大小写的 `T1`、`T2`。
2. **先**按数值取最大的 SeriesNumber，再检查这个旧选择是否唯一、可信及满足质量。失败时排除整个 Study，不回退到较小序列。
3. 质量要求为原始 `coverage_mm >= 100` 且 `20 <= frame_count <= 40`，边界包含。不会用重采样后的切片数替代原始 frame count。
4. 最大值有多个候选、描述同时包含 T1/T2、描述不唯一、`partial_study_scan`、缺失 InstanceNumber、普通序列存在 acquisition 合并/丢弃证据，均排除。`largest_acquisition_selected` 等描述性 flag 不单独导致排除。AcquisitionNumber 全部缺失但文件全部保留的情况允许。
5. 用户已确认 `R01_Study_002904` T2 完成修正：接受统计中选择的 acquisition 2，质量采用其选中部分，例如 23 帧；不因历史来源总数为 46 帧就排除。实际 Series UID/SeriesNumber 来自输入，未硬编码。若磁盘存在多份同名 Study 的旧 embedding，仍视为歧义；用 `--legacy-embedding-dir` 明确限定已修正分支。
6. 仅接收元数据中 `normalization_id=tufts_train_global_v1` 的 NORM-A；NORM-B、身份不符、来源缺失、非法形状或非有限向量均记录并跳过。有效候选间若模型、转换参数或完整 NORM-A 参数/拟合名单不同，则在创建输出目录前报错，不混用。

三份 manifest 具有**完全相同且同序的成对 Study_ID**，每个 Study 恰好一个 T1 和一个 T2。单模态 manifest 仍保留两侧身份列，但训练只读取所指定的模态。任一侧不可确认/不合格，三份 manifest 都排除该 Study。

## 向量变换

| 旧输入 | 操作 | 新记录字段 |
| --- | --- | --- |
| v1 `block_cls[2,S,4,768]`，层 3/6/9/12 | 仅沿 S 取算术均值 | 每模态 `block_embed[4,768]` |
| v2 `block_cls[2,S,12,768]`，层 1..12 | 取下标 2/5/8/11，再仅沿 S 取均值 | 同上 |
| `final_cls[2,S,768]` | 仅沿 S 取算术均值 | 每模态 `final_embed[768]` |

旧模态轴 0=T1、1=T2；不跨模态、不跨层平均。block 12 是最终 LayerNorm 之前的 CLS，final 是之后的 CLS，分别保存。支持正整数切片数，实际文件通常为 179；不强制把测试或旧数据切片数写死。

每个 `<Study_ID>.npz` 有两个完整键：`<Study_ID>_<SeriesInstanceUID>_<整数SeriesNumber>`。值是零维 structured array，字段为 Unicode `series_description`、float32 `block_embed(4,768)`、float32 `final_embed(768,)`。读取不需要对象反序列化。

## 运行

在 BrainIAC 仓库根目录安装 NumPy；下面命令使用服务器路径，两个 CSV 路径需替换为实际的全量统计表和标签表。

```bash
python -m pip install -r src/vit_survival_embeding_convert/requirements.txt

python src/vit_survival_embeding_convert/convert.py \
  --legacy-results-dir /mnt/image_test/C2D2AI/vit_survival/results \
  --selected-csv /absolute/path/to/full_uid_stats/selected_series.csv \
  --labels-csv /absolute/path/to/labels.csv \
  --output-dir /mnt/image_test/C2D2AI/vit_survival/results/cbi_wmd_norm_a_audit \
  --audit-only

python src/vit_survival_embeding_convert/convert.py \
  --legacy-results-dir /mnt/image_test/C2D2AI/vit_survival/results \
  --selected-csv /absolute/path/to/full_uid_stats/selected_series.csv \
  --labels-csv /absolute/path/to/labels.csv \
  --output-dir /mnt/image_test/C2D2AI/vit_survival/results/cbi_wmd_norm_a_converted
```

audit-only 是可选的检查方式，不是正式转换的强制前置审批步骤。两次运行使用不同的新输出目录。

输出：

```text
output/
  embeddings/
    <Study_ID>.npz
    embedding_config.json
  manifests/
    manifest_T1.csv
    manifest_T2.csv
    manifest_T1T2.csv
  legacy_series_mapping.csv
  embedding_source_issues.csv
  conversion_summary.json
  source_metadata.jsonl
```

`legacy_series_mapping.csv` 逐个标签 Study 记录选择身份、状态与排除原因；`embedding_source_issues.csv` 记录扫描到的非法或重复旧文件；`conversion_summary.json` 汇总人数和原因。没有生成需要重算的清单或任务。`source_metadata.jsonl` 保留每个有效源的路径和专属 metadata，共同模型信息及完整 NORM-A provenance 集中存放于 `embeddings/embedding_config.json`，避免重复保存同一份拟合名单。

audit-only 不生成 NPZ 或三份 manifest。正常转换若初筛后完全没有候选则报错，建议查看 audit-only 报告；若候选的实际数组全部验证失败，则保留排除报告和空 manifest，命令返回非零。发生其他运行错误时可能留下部分新输出；检查该目录后为重试选择新目录，程序不会覆盖已有结果。

## 接入现有训练

每次只传一份 manifest。现有 `train.py` 自动按其 `modality` 确定 T1/T2/T1T2，并沿用既有 Harvey site 分组和训练逻辑。

```bash
python src/CBI_WMD_predict/train.py \
  --manifest /mnt/image_test/C2D2AI/vit_survival/results/cbi_wmd_norm_a_converted/manifests/manifest_T1.csv \
  --embedding-field final_embed \
  --output-dir /mnt/image_test/C2D2AI/vit_survival/results/cbi_wmd_t1_final \
  --device cuda:0

python src/CBI_WMD_predict/train.py \
  --manifest /mnt/image_test/C2D2AI/vit_survival/results/cbi_wmd_norm_a_converted/manifests/manifest_T1T2.csv \
  --embedding-field block_embed --block 6 \
  --output-dir /mnt/image_test/C2D2AI/vit_survival/results/cbi_wmd_t1t2_block6 \
  --device cuda:0
```

T2 使用 `manifest_T2.csv`；支持的实际 block 层号为 3、6、9、12。T1T2 使用同一层，按 T1 在前、T2 在后拼接为 1536 维；单模态为 768 维。每次实验指定不同的训练输出目录。

**NORM-A 沿用旧拟合参数和名单，未按当前 Harvey 五组重新拟合。** 其旧拟合名单可能覆盖当前验证/测试人员，因此本实验应标记为“复用旧 NORM-A 的固定表征评估”。转换不消除此限制。与 BrainIAC 表征比较时，还须对齐实际 Study_ID/序列和评估分组，不能仅因输入格式相同就认为已控制了样本差异。

## 验证

在仓库根目录执行：

```bash
python -m unittest discover -s src/vit_survival_embeding_convert/tests -v
```

测试使用合成数据，覆盖选择次序、UID/acquisition 歧义、边界质量、修正病例、两种旧 schema 的精确聚合、三份配对清单、元数据冲突、audit-only、原文件保护及实际分类器接入。CPU 训练冒烟测试需要现有分类器的 PyTorch/scikit-learn 等依赖；转换程序自身只需 NumPy。这些测试不代表已经对服务器真实患者数据完成转换或验证了临床性能。
