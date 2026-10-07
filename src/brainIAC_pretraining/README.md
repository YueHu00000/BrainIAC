# BrainIAC 全 Series 自监督预训练

实现日期：2026-10-04。每个 series 独立处理，ID 为 `{Study_ID}_{SeriesNumber}`，使用最大的数值 AcquisitionNumber，不要求 T1/T2 配对。保留原始 DICOM 和旧 embedding 流程。

## 输入、输出和失败行为

| 阶段 | 输入 | 输出 | 失败行为 |
|---|---|---|---|
| manifest | labels CSV、folder list | manifest.csv | 扫描问题记录；多帧 MR DICOM 报错中断 |
| quality | manifest.csv、可选的 selected_series.csv；未指定统计 CSV 时读取 DICOM header | image_number_coverage.csv | 多帧或文件数与 frame 数不一致报错中断；独立切片数不一致筛掉对应 series |
| convert | manifest.csv | raw/<ID>.nii.gz、converted.csv | 多帧 DICOM 中断；其他单 series 转换错误报告后继续 |
| pre_process | converted.csv、raw 目录 | processed/<ID>.nii.gz、pre_process.csv | 单 series 失败报告后继续 |
| exclude | 质量 CSV、pre_process.csv、processed 目录 | excluded_preprocess/、final_pre_process.csv | 移动冲突报错，不覆盖 |
| train | final_pre_process.csv、processed 目录 | 日志、完整 checkpoint | 直接读取；任何读取、增强、训练错误中断 |

quality 正式结果只有 `unique_id,frame_count,coverage_mm`。不记录 file_count、unique_slice_count、重复平面、位置检查或 flags。内部从已有统计 CSV 读取数量，或从 manifest 所列 DICOM header 计算数量：未知数量或 frame 数与独立平面数不一致时筛掉该 series，rejected_quality.csv 仅记录 ID 和原因；多帧属于致命错误，不作为普通排除处理。

coverage 按切片方向法向量投影后计算首末独立切片平面中心跨度，单位 mm；独立平面容差仍为 0.01 mm，方向容差仍为 1e-4。frame 和 coverage 的阈值只在 exclude 执行，严格小于阈值排除，等于阈值保留。缺失质量记录或未知 coverage 也排除并说明原因。

manifest.csv 保存 unique_id、study_id、series_number、study_directory、acquisition_number、file_names；file_names 是 JSON 相对路径列表。quality 指定统计 CSV 时确认其文件列表、目录及 acquisition 与 manifest 一致，不读取 DICOM；未指定统计 CSV 时只读取 manifest 列出的 header，不重新扫描目录或选择 acquisition，不读取像素。

2026-10-06：六个阶段均不导入或调用 `stat/` 中的程序文件。扫描与必要的数量、coverage 计算放在本目录的 `_dicom.py`；`stat/` 继续作为独立统计工具保留。quality 可以通过 `--selected-csv` 使用 stat 输出的 CSV，CSV 可存放在任意目录；不指定该参数也能运行。

converted.csv、pre_process.csv 和 final_pre_process.csv 均只有 unique_id 一列。目录提供文件位置，文件名统一为 `<unique_id>.nii.gz`。训练只使用 final_pre_process.csv 所列文件，不预先核对目录，也不自动扫描增加样本。

convert 直接转换 manifest 的全部 series，不读取质量 CSV，`--quality-csv` 参数已移除。quality 可在 convert 之前或之后运行，只需在 exclude 前完成。exclude 依据质量结果执行筛选；quality 未保留的 series 若预处理成功，则按 missing_quality 排除。转换和预处理可能因此处理一些最终被排除的 series。

## 依赖

先安装运行机器适用的 PyTorch 与对应 torchvision，然后从仓库根目录执行：

```bash
python -m pip install -r requirements-pretraining.txt
```

保持作者的 MONAI 1.3.2，并限制 NumPy < 2，避免旧随机增强在 NumPy 2 上出现 seed OverflowError。预处理继续使用 src/preprocessing、src/brainiac_embedding/_preprocess_bridge.py、模板和 HD-BET 权重，运行环境须满足原有成像依赖。

复制后的作者代码在 `src/simclr/`。原 `E:\codex\MRI_project\BrainIAC\simclr` 保留。新的训练入口是本目录的 train.py；复制后的 train_multigpu.py 提供增强函数，不再作为直接启动入口。

## 分阶段运行

以下在仓库根目录执行。示例 `work/pretraining` 为独立输出位置，各图像目录不要混入旧文件、mask 或其他队列结果。每一步成功后再执行下一步；阈值必须根据本次实验指定。

生成 manifest：

```bash
python src/brainIAC_pretraining/manifest.py --csv resources/labels.csv --folder-list resources/folders.txt --output-dir work/pretraining/inventory
```

folder list 一行一个 DICOM 根目录；相对路径以 folder list 所在目录解析。labels 默认使用 Study_ID 列。manifest 使用本目录的扫描函数，检测到任意扫描到的 MR 文件 NumberOfFrames > 1 时立即报出 ID、路径并中断，包含之后会被 acquisition 选择舍弃的文件。无需预先运行统计程序。

生成质量 CSV：

```bash
python src/brainIAC_pretraining/quality.py --manifest work/pretraining/inventory/manifest.csv --output-dir work/pretraining/quality
```

已有全 series 统计 CSV 时可直接使用，避免重读 DICOM：

```bash
python src/brainIAC_pretraining/quality.py --manifest work/pretraining/inventory/manifest.csv --selected-csv work/pretraining/stats/selected_series.csv --output-dir work/pretraining/quality
```

`--selected-csv` 使用全 series 的原始 `selected_series.csv`，需包含 unique_id、选中文件信息、file_count、frame_count、unique_slice_count 和 coverage_mm；不使用旧 T1/T2 两行一个 study 的 CSV。缺少某个 unique_id 或选中文件不一致时记录到 rejected_quality.csv。

转换 manifest 中的全部 series：

```bash
python src/brainIAC_pretraining/convert.py --manifest work/pretraining/inventory/manifest.csv --output-dir work/pretraining/raw
```

逐 series 预处理：

```bash
python src/brainIAC_pretraining/pre_process.py --converted-csv work/pretraining/raw/converted.csv --raw-dir work/pretraining/raw --output-dir work/pretraining/processed
```

预处理仍是原 BrainIAC N4、模板配准与 HD-BET 成像操作；每次只暂存一个 series，临时输入固定为 volume.nii.gz，最终 volume_0000.nii.gz 发布时恢复 `<unique_id>.nii.gz` 名称，避免上游截断带点的 ID 或把带 `_mask` 的 ID 当作 mask。失败和缺少最终图像都报告并继续，同 study 的其他 series 不受影响。mask 和配准中间结果不发布到正式 processed 目录。

筛选移动，以下 20 帧和 100 mm 仅为命令示例，替换为实际实验阈值：

```bash
python src/brainIAC_pretraining/exclude.py --quality-csv work/pretraining/quality/image_number_coverage.csv --pre-process-csv work/pretraining/processed/pre_process.csv --input-dir work/pretraining/processed --excluded-dir work/pretraining/excluded_preprocess --frame-count-threshold 20 --coverage-threshold-mm 100 --dry-run
```

确认输出后去掉 `--dry-run` 执行移动。dry-run 不创建目录、不写 CSV、不移动文件；实际执行前检查全部同名目标冲突，已排除的原路径再次运行时跳过。excluded_preprocess.csv 保留已移动 ID 和排除原因；final_pre_process.csv 重新读取移动后的正式输出目录生成。

开始预训练：

```bash
python src/brainIAC_pretraining/train.py --final-csv work/pretraining/processed/final_pre_process.csv --input-dir work/pretraining/processed --output-dir work/pretraining/training --accelerator gpu --device 0 --batch-size 16 --num-workers 4
```

batch size 和 workers 是机器资源相关的运行参数，上述值仅为示例。默认配置为原 200 epochs、batch size 160、workers 24，CLI 可覆盖。SimCLR batch 至少包含两个独立样本；drop_last=True，只有一个不完整 batch 时应降低 batch size。训练不提前打开或检查图像，每个文件加载后分别执行两次随机增强，输入错误直接中断。

## 断点续跑和训练初始化

convert、pre_process 在隔离临时目录计算，完成后发布正式 `<ID>.nii.gz`；再次运行同一命令按正式输出文件存在跳过，不额外读取或校验完成文件。临时目录不计入完成清单。批次结束或正常 KeyboardInterrupt 时，完成清单按正式输出目录重建；机器被强制终止后再次运行也会重建。

因此续跑必须使用同一队列、同一选中 acquisition、同一参数。改变数据来源或预处理参数时使用新输出目录。完成文件不会自动重算，已排除的图像也不会自动恢复。

convert_errors.csv、pre_process_errors.csv 记录最近一次运行失败的 unique_id 和原因。程序遇到错误继续时仍可正常结束，必须查看报告以确认哪些 series 完成。

训练默认随机初始化。公开 BrainIAC encoder 或同结构 checkpoint 初始化只加载 `state_dict` 中的 `backbone.` 参数，严格匹配结构：

```bash
python src/brainIAC_pretraining/train.py --final-csv work/pretraining/processed/final_pre_process.csv --input-dir work/pretraining/processed --output-dir work/pretraining/training_continue --init-checkpoint /path/to/BrainIAC.ckpt --accelerator gpu --batch-size 16
```

恢复同一次实验的完整 checkpoint：

```bash
python src/brainIAC_pretraining/train.py --final-csv work/pretraining/processed/final_pre_process.csv --input-dir work/pretraining/processed --output-dir work/pretraining/training --resume work/pretraining/training/checkpoints/last.ckpt --accelerator gpu --batch-size 16
```

`--init-checkpoint` 与 `--resume` 互斥。完整 checkpoint 包含模型、optimizer、scheduler、epoch 和 global step；使用同一输入 CSV、目录及训练配置续跑。--max-epochs 表示恢复后的总 epoch 上限。

## 训练行为和参数

- 输入 `[B,1,96,96,96]`；ViT classification=False，输出 216 个空间 patch token；新模型平均全部 patch，表示维度 768；projection 为 2048 维。
- 复用原 SimCLR projection head 和 NTXentLoss；保留原实际 AdamW weight decay=0.0005、lr=0.0005、CosineAnnealingWarmRestarts 行为。
- 使用安装版 Lightly 的原默认 temperature；本次测试 Lightly 1.5.26 的实际值为 0.5，写入 run_config.yml。
- affine 保留作者原填写范围，不在数据接口改动时调整增强强度。MONAI 1.3.2 将 scale_range 作为相对 1 的偏移，因此当前范围实际产生前两轴 1.85–2.15、第三轴 1.9–2.1 的缩放参数；完整训练前应检查真实增强视图再决定强度。新的全 patch pooling 与旧附件平均 `1:` 的行为不同，表示版本记为 all_216_patch_tokens_mean_v1。
- 本地 CSVLogger 记录日志，不依赖 W&B 账号或网络。完整配置记录在 run_config.yml，恢复时记录 resume_config.yml。
- 第一版单卡，支持 CPU smoke；没有加入多卡对比负样本实现。

## 验证与边界

```bash
python -m unittest discover -s src/brainIAC_pretraining/tests -v
```

测试覆盖不含 stat 目录时六个入口启动及 manifest→quality 执行、最大数值 acquisition、已知双 23 切片选择、精简 CSV、法向量投影、单帧 shared/per-frame 几何、数量筛选、多帧中断、真实合成 DICOM 转换、断点续跑、预处理单 series 失败继续、阈值边界、移动冲突与 dry-run、训练错误传播、真实 NIfTI 增强、真实 ViT-B 前向、对比损失与参数更新、完整 optimizer/scheduler 恢复，以及串联的数据阶段。

本地验证使用合成数据。预处理调度通过 subprocess mock 验证，尚未对患者图像执行真实 N4/HD-BET；优化更新和 checkpoint 恢复使用小 backbone 替身及真实 Lightly/Lightning，完整 ViT 仅完成 CPU 前向。尚未运行真实 GPU 预训练，也没有产生预训练效果结论。
