# 保存现场的 NaN 层级与算子定位

`replay_localize.py` 是本次诊断新增的重放入口，配合本目录的 `diagnose_nan.py` 使用。不改原训练程序、数据、checkpoint 或已有诊断程序；不执行 optimizer 更新，不保存替换权重。所有文件直接读取。

在产生现场的原 Linux 环境中，从仓库根目录运行。依赖沿用 Python 3.10、PyTorch 2.5.1、MONAI 1.3.2、Lightly 1.5.26、Lightning 2.6.6、NumPy 1.26.4、nibabel 5.4.2 及对应 torchvision。精细 attention 定位针对 MONAI 1.3.2 的实际实现。

## 1. 先定位 FP16 前向异常

```bash
python nan_diagnosis/replay_localize.py \
  --scene /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/diagnosis/run149_01/diagnostics/first_forward_nonfinite \
  --device cuda:0 \
  --precisions 16-mixed 32-true \
  --forward-only \
  --output-dir /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/diagnosis/localize_forward_01
```

输出目录必须尚不存在。`--scene` 指向包含 `summary.json`、`batch.pt` 和 `state.pt` 的完整现场目录；只接受可信的本次保存现场。它直接载入已增强的完整 batch，不重新加载 MRI 或采样增强。仅下载的 summary／replay JSON 不能完成重放。

程序先用原 forward 加只读 hooks，检查 patch embedding、各 Transformer block 和后续表示的输入／输出。发现异常后，恢复同一初始模型、buffer、模式和随机数，再对可定位的首个异常模块增加精细观察；不会把首次 hooks 定位结果当作已经确认的具体根因。

对首个异常 Transformer block，检查归一化、QKV、q／k／v、未缩放 QKᵀ、缩放后 logits、softmax、attention×V、输出投影和 MLP 子模块。异常张量记录样本索引及保存 batch 中的完整 unique_id：

```text
{Study_ID}_{SeriesInstanceUID}_{SeriesNumber}
```

每次重放遇到首个非有限值立即停止；之后的阶段尚未计算，不应将其缺少统计理解为正常。精细重放也会遵守这一规则，因此例如 QKᵀ 首先异常时，不会继续计算 softmax。

索引以实际 batch 轴为准；参数梯度索引不用于识别样本。如果未缩放 QKᵀ 异常，再以同一 q／k 的 FP32 副本重算，至多取第一个异常样本，并在报告中注明限制。此对照检验同一已量化输入下的算子，不能代替整模型 FP32 训练测试。

## 2. 验证 BF16 前向及反向

```bash
python nan_diagnosis/replay_localize.py \
  --scene /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/diagnosis/run149_01/diagnostics/first_forward_nonfinite \
  --device cuda:0 \
  --precisions bf16-mixed \
  --output-dir /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/diagnosis/localize_bf16_backward_01
```

这里不加 `--forward-only`，会在前向通过后执行反向，检查梯度有限性。程序始终不调用 optimizer.step。BF16 需要支持它的 CUDA GPU。若需要 FP32 反向，可另用新输出目录运行 `--precisions 32-true`，同样不加 `--forward-only`；完整 ViT 的显存开销较大。

## 3. 报告与解释

新增输出为 `localization_report.json`，逐个 trial 写入，后续 trial 出错时保留此前结果。报告包括：

- 现场来源、原环境及当前环境、请求的精度和执行状态。
- 原 forward 的最早异常位置、模块／阶段统计、异常样本索引和 unique_id。
- 可执行时的精细重放，以及 QKᵀ 的局部 FP32 对照。
- 前向、反向是否实际执行／完成、梯度检查、optimizer 更新情况。

应先检查各 trial 是否执行，再看异常位置及反向状态。前向失败时没有反向结果；`--forward-only` 通过也不能说明梯度正常。某次 BF16／FP32 前向反向通过，只支持该保存现场，不能证明长期训练稳定。

FP32 沿用现场 TF32／矩阵计算设置，不是禁用 TF32 的严格参考。精细 attention 观察使用 MONAI 1.3.2 的原运算顺序；仍须将精细结果与未改 forward 的首次定位对照。局部 FP32 对照保留原 q／k 已发生的量化，因此不能用它证明上游线性层或整个 FP32 模型没有问题。

本地已有资源只有 JSON 报告，缺少实际 `state.pt`／`batch.pt`，且本机无 CUDA。此处的本地测试验证合成故障定位和报告逻辑；真实 UID、首个异常算子及 BF16 结果需以上服务器命令取得后再分析。

## 4. 本地测试

新增测试文件独立于原有测试与训练程序：

```bash
python -m unittest discover -s nan_diagnosis -p test_replay_localize.py -v
```

验证重点为异常层和样本对应、精细观察与原算式的输出／梯度／随机数一致性，以及前向失败、反向检查和无参数更新的报告边界。CUDA 专项需在 GPU 上执行；CPU 通过不能替代服务器 FP16 实验。

2026-10-26 本地验证：新增测试11项，10项通过、1项 CUDA 测试跳过；原诊断测试19项，18项通过、1项 CUDA 测试跳过。环境为 Python 3.10、PyTorch 2.13 CPU、MONAI 1.3.2、Lightning 2.6.6。原 `diagnose_nan.py`、`README.md`、`test_diagnose_nan.py` 已与修改前文本逐一直接对比，保持不变。
