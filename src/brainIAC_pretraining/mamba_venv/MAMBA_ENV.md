# BrainIAC pretraining 专用 Python 3.10 环境

适用于 Linux NVIDIA GPU 服务器。服务器须已安装 mamba；默认创建独立环境 `brainiac_pretrain_py310`，用于 `src/brainIAC_pretraining` 的 manifest、quality、convert、pre_process、exclude 和 train 程序。

在服务器的 BrainIAC 代码仓库根目录执行：

```bash
bash src/brainIAC_pretraining/mamba_venv/create_mamba_env.sh
mamba activate brainiac_pretrain_py310
python --version
```

也可以指定一个新的环境名：

```bash
bash src/brainIAC_pretraining/mamba_venv/create_mamba_env.sh brainiac_pretrain_py310_v2
mamba activate brainiac_pretrain_py310_v2
```

脚本拒绝覆盖同名环境。安装中途失败时保留已创建的环境；可根据错误修复后手动执行剩余命令，或换一个新名称重新创建。

环境固定 Python 3.10、PyTorch 2.5.1 / torchvision 0.20.1 的 CUDA 11.8 构建、MONAI 1.3.2、NumPy 1.26.4、Lightning 2.6.6、Lightly 1.5.26 和 setuptools 80.9.0。PyTorch 2.5.1 保留仓库自带 HD-BET 使用的旧 checkpoint 加载默认行为；NumPy < 2 避免旧 MONAI 随机增强的 seed 溢出；setuptools 80.9.0 保留旧依赖需要的 `pkg_resources`。

Python 依赖使用本目录的 `requirements-python310.txt`，不引用 embedding 的依赖表。预训练的 `pre_process.py` 仍复用仓库的预处理桥接代码与 HD-BET，因此保留 SciPy/scikit-image；MONAI ViT 需要 einops。创建时要求 pip ≥25，避免旧 pip 对 PyTorch 索引依赖名称的解析问题。

CUDA 安装命令来自 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v251)，环境创建和运行命令依据 [Mamba 文档](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)。CUDA wheel 提供运行时，服务器仍需兼容的 NVIDIA 驱动和 GPU；脚本不安装驱动。这里按 CUDA 11.8 支持的 GPU 配置，其他架构需要另选对应 PyTorch 构建。

安装末尾自动运行 `pip check` 和 `check_environment.py`：导入预训练各阶段及其依赖的预处理桥接代码与仓库自带 HD-BET，使用临时合成 NIfTI 检查预训练双增强及对比损失反向传播，并检查预处理模板与 HD-BET 权重文件是否存在。检查不读取真实 MRI，也不加载模型权重。

在分配到的 GPU 节点上再次验证：

```bash
python src/brainIAC_pretraining/mamba_venv/check_environment.py
```

只有出现 `CUDA available: True` 和 `GPU tensor operation: OK` 才表示 GPU 基础检查通过。登录节点没有可见 GPU 时会跳过 GPU 检查；完整模型训练、显存容量及真实图像预处理需要在 GPU 作业中验证。

然后可以在新环境中直接执行原来的训练命令：

```bash
python src/brainIAC_pretraining/train.py \
  --final-csv /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/processed/final_pre_process.csv \
  --input-dir /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/processed \
  --output-dir /mnt/image_test/C2D2AI/BrainMRI/BrainIAC/pretraining/training \
  --accelerator gpu --device 0
```

如果当前 shell 尚未初始化 mamba，可以先使用 `mamba run -n brainiac_pretrain_py310 python ...` 运行命令，无需激活。
