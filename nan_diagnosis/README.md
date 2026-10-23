# BrainIAC 独立 NaN 来源诊断

本目录只调查这次SimCLR预训练中NaN的来源。程序不导入项目原训练模块，不修改原程序、配置、数据或已有checkpoint。新增输出必须使用尚不存在的目录，避免覆盖已有结果。

`diagnose_nan.py` 提供 `inspect → run → replay` 三步。相同架构、增强顺序与优化方式在本文件中独立实现；不会自动降低学习率、改变增强或把NaN替换成0。

## 环境与范围

建议在原Linux训练环境运行。原项目锁定版本为Python 3.10、PyTorch 2.5.1、MONAI 1.3.2、Lightly 1.5.26、pytorch-lightning 2.6.6、NumPy 1.26.4、nibabel 5.4.2、PyYAML 6.0.2，并使用原来对应的torchvision。无需安装新的服务或实验跟踪平台。

- 仅支持本次的单GPU、AdamW、无梯度累积、96×96×96输入。
- 默认batch_size=160、num_workers=24、lr=0.0005、200 epochs、梯度裁剪2、50 epoch学习率重启。
- GPU默认16-mixed，CPU默认32-true；可显式选择32-true或bf16-mixed作独立实验。
- 全部数据直接读取，不进行哈希或checksum校验。
- 原训练程序与本程序之间不共享代码。源码仅作为复现依据；测试会只读核对原架构和增强参考行为。

以下示例从仓库根目录执行；`/path/to/...` 为必须替换的实际路径，输出目录每次换新名称。把整个 `nan_diagnosis` 目录复制到服务器即可，程序运行不需要项目的 `src` 目录；测试中的原实现对照需要原仓库的 `src`。

## 1. inspect：检查健康起点

```bash
python nan_diagnosis/diagnose_nan.py inspect \
  --checkpoint /path/to/epoch149.ckpt \
  --output-dir /path/to/diagnosis/inspect149_01
```

优先检查epoch 149，不通过时再考虑epoch 99。静态检查包括参数／buffer／optimizer中的非有限张量、optimizer／scheduler／scaler中的非有限标量、负的BatchNorm running variance，并列出epoch、global_step、scheduler及AMP scaler状态。静态检查通过不代表所有输入都能正常前向。

可选传入固定T1、T2、FLAIR样本，至少两个输入，以满足训练模式的projection BatchNorm：

```bash
python nan_diagnosis/diagnose_nan.py inspect \
  --checkpoint /path/to/epoch149.ckpt \
  --sample /path/to/T1.nii.gz \
  --sample /path/to/T2.nii.gz \
  --sample /path/to/FLAIR.nii.gz \
  --device cuda:0 \
  --output-dir /path/to/diagnosis/inspect149_samples_01
```

两个模式分别重新加载原权重并恢复同一随机数状态，不回写checkpoint。这项测试使用resize及nonzero/channel-wise标准化后的图像、FP32、无随机增强，用于检查模式差异，不代替实际增强batch的训练测试。报告标明已检查和未检查项目。

## 2. run：按原条件捕获异常

```bash
python nan_diagnosis/diagnose_nan.py run \
  --final-csv /path/to/final_pre_process.csv \
  --input-dir /path/to/final_nifti \
  --resume /path/to/epoch149.ckpt \
  --accelerator gpu --device 0 \
  --output-dir /path/to/diagnosis/run149_01
```

不传 `--config` 即使用上述原参数。若需读取原配置，可加 `--config /path/to/config.yml`，该文件只读。`--resume` 必须是含optimizer和scheduler的完整训练checkpoint；先检查其有限值。程序通过Lightning恢复完整训练状态，不只加载encoder。未提供 `--resume` 时从头初始化。

诊断在原训练计算之间加入只读检查，顺序为：

```text
NIfTI读取 → resize → 每一步增强 → 两个最终视图
→ backbone tokens → 768维平均池化 → projection → normalize
→ loss → 解除AMP缩放且裁剪前的梯度 → 更新后的参数与optimizer状态
```

数据以CSV中的完整unique_id加载，不重新拼接或截断：

```text
{Study_ID}_{SeriesInstanceUID}_{SeriesNumber}
```

原有自定义裁剪在调用时临时创建MONAI随机裁剪对象，本程序保持这个行为。Linux中保持父进程构建transform的方式；Windows spawn worker中单独构建transform以便启动。这是诊断工具，不保证恢复旧训练的历史数据顺序／worker状态，也不保证旧epoch 161的异常在相同位置重现。改变worker数也会改变增强轨迹。

### 异常处理与返回码

| 情况 | 行为 |
| --- | --- |
| 数据／增强非有限 | worker保存样本ID、视图、变换和异常张量，停止；此时未必存在完整batch |
| 前向／loss／buffer非有限 | 保存更新前现场并停止 |
| AMP梯度非有限，loss有限 | 首次保存现场，允许GradScaler跳步和降scale，继续观察 |
| FP32／无scaler的梯度非有限 | 保存现场并停止 |
| 更新后参数／状态非有限 | 保存异常后状态并停止，注明可重放范围 |

返回码0表示命令完成；返回码2表示inspect检测异常，或run因已记录的非有限值停止。replay命令完成时返回0，即使已重现NaN；结论查看报告的 `first_nonfinite`／`finite`。普通读写错误、显存不足和其他程序错误保留异常堆栈，不冒充NaN定位结果。

### 更新阶段的精确现场

默认不为每个batch复制全部参数和optimizer。前向／反向异常可保存更新前参数，并恢复前向前的buffer；更新后才发现异常时，默认只保留异常后状态，不能声称复现了原更新。

只有怀疑optimizer更新时，短时间开启：

```bash
python nan_diagnosis/diagnose_nan.py run \
  --final-csv /path/to/final_pre_process.csv \
  --input-dir /path/to/final_nifti \
  --resume /path/to/healthy.ckpt \
  --accelerator gpu --device 0 \
  --capture-update-state \
  --max-steps 25010 \
  --output-dir /path/to/diagnosis/update_window_01
```

`--max-steps` 是Lightning的**绝对global_step上限**。例如起点global_step=25000时，25010大约再运行10个batch；不能填写10并期望再训练10步。也可通过 `--max-epochs` 限制结束epoch，上限同样为完整训练的累计值。

完整快照每batch复制参数和optimizer到CPU，时间与内存开销明显，只用于短区间。

## 3. replay：固定张量与初始状态对照

```bash
python nan_diagnosis/diagnose_nan.py replay \
  --scene /path/to/diagnosis/run149_01/diagnostics/first_forward_nonfinite \
  --device cuda:0 \
  --output-dir /path/to/diagnosis/replay_forward_01
```

默认分别执行16-mixed和32-true。每次重新构建模型，严格加载同一参数，恢复所有buffer（包括不写入checkpoint的Lightly buffer）、随机数及训练模式；直接使用捕获的完整增强视图，不重新采样增强。反向会恢复原optimizer和scaler状态，但默认不执行optimizer更新。

FP32沿用现场记录的TF32／矩阵计算设置，不额外改变它们；因此32-true不等同于禁用TF32的严格浮点参考。

FP32显存不足时，先只做前向。此模式不建立反向图，也不加载optimizer状态到GPU：

```bash
python nan_diagnosis/diagnose_nan.py replay \
  --scene /path/to/diagnosis/run149_01/diagnostics/first_forward_nonfinite \
  --device cuda:0 --precisions 16-mixed 32-true --forward-only \
  --output-dir /path/to/diagnosis/replay_forward_only_01
```

需要精确重放一次更新时，现场必须用 `--capture-update-state` 捕获：

```bash
python nan_diagnosis/diagnose_nan.py replay \
  --scene /path/to/diagnosis/update_window_01/diagnostics/first_update_nonfinite \
  --device cuda:0 --update \
  --output-dir /path/to/diagnosis/replay_update_01
```

定位反向算子可额外加 `--detect-anomaly`；它改变异常报告方式并增加开销，先保存普通重放结果再使用。数据worker现场也能传给replay，但只检查保存的异常张量与变换记录，不重跑增强，不伪造完整batch。

## 输出和解释

```text
独立输出目录/
├── environment.json
├── checkpoint_inspection.json       # inspect或run
├── run_config.yml                  # run
├── checkpoints/last.ckpt            # 诊断运行自己的完整checkpoint
├── logs/                           # Lightning日志
└── diagnostics/
    ├── steps.csv
    ├── events.jsonl
    ├── data_<worker>_<time>/         # 数据异常现场
    ├── first_forward_nonfinite/     # 仅在对应异常发生时创建
    ├── first_buffer_nonfinite/
    ├── first_gradient_nonfinite/
    └── first_update_nonfinite/
        ├── summary.json
        ├── batch.pt
        ├── state.pt
        └── post_update_state.pt     # 开启完整快照时另存异常后状态
```

- `steps.csv`：每个完成的batch一行，实际学习率、loss、解除缩放且裁剪前的梯度、scale变化、实际optimizer更新事件和更新后状态检查。未完成的异常batch查看events及现场summary。
- `events.jsonl`：每batch的模型阶段统计、数据增强阶段的样本统计范围、完整batch ID；异常事件包含首个异常位置。
- 2维表示额外记录每个向量范数的最小／最大值，帮助定位归一化前的极端值或零向量。
- `batch.pt`：完整实际增强张量和样本元数据，保留dtype。batch=160、两个FP32视图仅图像约1.05 GiB，现场另含权重及optimizer。
- `summary.json` 的 `replay_scope` 区分 `forward_backward`、`forward_backward_update` 和 `post_update_inspection_only`。最后一种不能重放原故障；程序会拒绝。
- AMP scaler调节可能产生偶发跳步。`grad_finite=False`、scale下降且参数仍有限，不能单独证明模型损坏。
- 已定位的第一个阶段不一定就是具体根因。比如projection首次Inf且FP32正常，应进一步调查projection的算子与幅度。

## 验证与限制

执行本目录新增测试：

```bash
python -m unittest discover -s nan_diagnosis -p test_diagnose_nan.py -v
```

测试使用合成NIfTI及小型backbone，覆盖NaN／Inf阶段注入、BatchNorm污染、现场重放、真实Lightning回调与续训、worker数据加载、AMP跳步计数及更新异常的保存边界。架构构造参数与原源码做AST对照，增强输出与只读原实现参考比较，不修改原文件。

本地验证环境为Windows、Python 3.10、PyTorch 2.13.0+cpu；临时依赖中的Lightning 2.6.6、Lightly 1.5.26、MONAI 1.3.2、NumPy 1.26.4与项目对应版本对照。临时安装只用于本地测试，不改变服务器环境或原依赖文件。

本次验证结果：19项测试中18项通过，1项CUDA AMP测试因本机没有GPU跳过。已验证合成NIfTI的worker加载、完整run命令及续训、异常现场重放、更新阶段故障复现；未把这些小型模型测试当作完整ViT的服务器验证。

CUDA AMP测试需真实GPU。本机CPU测试不能证明服务器FP16行为，也不能确认此次真实NaN的根因。真实checkpoint检查、完整ViT在原GPU上的运行和FP16／FP32对照须在服务器执行。不要因尚未复现而自动同时修改学习率、精度和增强。
