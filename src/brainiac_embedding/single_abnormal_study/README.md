# BrainIAC 单 Study：双 acquisition 拆分到 embedding

用于修正 `R01_Study_002904`：同一个 T2 series 的 AcquisitionNumber 1、2 各有 23 张图像，旧转换将它们合成了 46 层。两套 acquisition 分别生成结果，由人工检查后选择使用；它们不是两个独立患者。

## 1. 处理逻辑和与已有程序的关系

```text
单 Study DICOM
  → T1.nii.gz + T2_acq1.nii.gz + T2_acq2.nii.gz
  → 人工组成两套标准 T1.nii.gz / T2.nii.gz 输入
  → 每套独立执行 BrainIAC N4 + 刚体配准 + HD-BET
  → 原 validation transform：resize 96³ + 每模态非零体素归一化
  → backbone / brainage / mci 的多层、双 pooling embedding
```

- `convert_single_study.py`：复制自本项目 vit_survival_simple 已实现的同名脚本，无跨仓库运行依赖。选择描述含 T1/T2 的最大 SeriesNumber；T2 必须恰有 acquisition 1、2，按切片法向空间位置分别排序。保留 T1 整组。支持输入目录直属的传统单帧 DICOM；拒绝最高编号歧义、多 SeriesInstanceUID、组内重复层面和多帧图像。不将 23 层写死为通用要求。
- `preprocess_single_study.py`：直接调用父包 `preprocess_t1t2.preprocess_study()`，保留官方 N4、刚体配准及 HD-BET、现有 Windows/HD-BET bridge。读取、输出均采用 BrainIAC 原有的 `<root>/<Study_ID>/T1.nii.gz,T2.nii.gz` 层级。
- `embed_single_study.py`：复用父包的 checkpoint 加载、validation transform、层提取和 NPZ 保存函数。只处理明确指定的一个 Study，不扫描队列，不需要 cohort CSV 或 labels。

没有引入哈希校验。BrainIAC 不使用 vit_survival 的 KPSC/ANTs 预处理、179 层 prepared 或 NORM-A；不要把 vit_survival prepared 传给这些脚本。BrainIAC 归一化按当前图像计算，不拟合队列统计，此次单 Study 修正不要求重算其他患者。

## 2. 路径与环境

使用已经能运行原 BrainIAC embedding 的 Python 环境和同一套 checkpoint，保留完整仓库（特别是父级 `brainiac_embedding` 包），不能只复制本子目录。三个脚本可按绝对路径从任意工作目录执行；也可通过 `PYTHONPATH=<repo>/src` 使用模块方式。

以下是 Linux Bash 示例，请先替换路径：

```bash
BRAIN_REPO=/path/to/BrainIAC
DICOM_DIR=/path/to/dicom/R01_Study_002904
WORK_DIR=/path/to/brainiac_single_study_correction
STUDY_ID=R01_Study_002904
SINGLE_DIR="$BRAIN_REPO/src/brainiac_embedding/single_abnormal_study"
DEVICE=cuda:0
```

需使用已能运行原 BrainIAC embedding 的环境；仓库依赖见根目录 `requirements.txt`（本地若有 `requirements-embedding.txt`，可沿用其环境）。另需仓库中的：

```text
src/preprocessing/atlases/temp_head.nii.gz
src/preprocessing/hd-bet_params/0.model
src/checkpoints/BrainIAC.ckpt
src/checkpoints/brainage.ckpt
src/checkpoints/vit_mci.ckpt
```

本程序不下载模板或 encoder checkpoint。`--device` 只控制 embedding；预处理 HD-BET 沿用官方程序，根据 CUDA 可用性选择 GPU 0 或 CPU。没有 GPU 时将 `DEVICE=cpu`。

## 3. DICOM → 分 acquisition 的 raw NIfTI

```bash
python "$SINGLE_DIR/convert_single_study.py" \
  --dicom-dir "$DICOM_DIR" \
  --output-dir "$WORK_DIR/raw_split"
```

输出：`T1.nii.gz`、`T2_acq1.nii.gz`、`T2_acq2.nii.gz`、`selected_files.csv`。CSV 保存源文件路径、series UID、acquisition、InstanceNumber 和层位置。

本病例预期三个体积均为 23 层、层间距约 7 mm。程序打印尺寸和 spacing，人工检查图像、方向、覆盖和两组 T2 的质量。输出文件已存在会报错，不覆盖。

若 vit_survival 的 `raw_split` 已经正确生成，可直接复用这三个 raw NIfTI，从下一步开始；两模型的 DICOM 拆分逻辑相同。

## 4. 建立两套标准输入

与 vit_survival 一样，复制同一 raw T1，各自选择一份 T2，改名为 `T2.nii.gz`。这里比 vit_survival 的平铺示例多一层 Study_ID，以直接复用 BrainIAC 父包。

```bash
for acq in 1 2; do
  # mkdir 不加 -p：目标 Study 目录存在即停止本段，避免覆盖已有输入。
  mkdir -p "$WORK_DIR/acq${acq}/raw"
  mkdir "$WORK_DIR/acq${acq}/raw/$STUDY_ID" || break
  cp "$WORK_DIR/raw_split/T1.nii.gz" "$WORK_DIR/acq${acq}/raw/$STUDY_ID/T1.nii.gz"
  cp "$WORK_DIR/raw_split/T2_acq${acq}.nii.gz" "$WORK_DIR/acq${acq}/raw/$STUDY_ID/T2.nii.gz"
done
```

## 5. raw → BrainIAC processed

分别执行，先完成 acq1 的检查，再运行 acq2：

```bash
ACQ=1  # 第二套改成 2
python "$SINGLE_DIR/preprocess_single_study.py" \
  --study-id "$STUDY_ID" \
  --raw-root "$WORK_DIR/acq${ACQ}/raw" \
  --processed-root "$WORK_DIR/acq${ACQ}/processed" \
  --brainiac-root "$BRAIN_REPO"
```

输出为 `acq1/processed/R01_Study_002904/T1.nii.gz` 和 `T2.nii.gz`，acq2 同理。两套都重新预处理 T1/T2；不复用错误 T2 的任何 processed 或 embedding。官方配准使用随机抽样且未固定其种子，因此两次 T1 配准不保证逐体素一致。

单 Study 入口遇到已有的 `<processed-root>/<Study_ID>` 目录直接报错，包括不完整目录；不会沿用批量程序的“已有即跳过”行为。失败后检查日志，若存在该目录，改用新的 processed root。父包仍管理、清理本 Study 的 `.preprocess_tmp` 临时目录。

检查配准和去颅骨质量后再提取 embedding。

## 6. processed → 三个 checkpoint 的 embedding

下面每次处理一个 acquisition；将 `ACQ` 改为 2 后再执行一次。三个模型放在各自目录，保留原 Study_ID。

```bash
ACQ=1
python "$SINGLE_DIR/embed_single_study.py" \
  --study-id "$STUDY_ID" \
  --processed-root "$WORK_DIR/acq${ACQ}/processed" \
  --output-root "$WORK_DIR/acq${ACQ}/embeddings/brainiac_backbone" \
  --model-kind backbone \
  --checkpoint "$BRAIN_REPO/src/checkpoints/BrainIAC.ckpt" \
  --brainiac-root "$BRAIN_REPO" --device "$DEVICE"

python "$SINGLE_DIR/embed_single_study.py" \
  --study-id "$STUDY_ID" \
  --processed-root "$WORK_DIR/acq${ACQ}/processed" \
  --output-root "$WORK_DIR/acq${ACQ}/embeddings/brainage_finetuned" \
  --model-kind brainage \
  --checkpoint "$BRAIN_REPO/src/checkpoints/brainage.ckpt" \
  --brainiac-checkpoint "$BRAIN_REPO/src/checkpoints/BrainIAC.ckpt" \
  --brainiac-root "$BRAIN_REPO" --device "$DEVICE"

python "$SINGLE_DIR/embed_single_study.py" \
  --study-id "$STUDY_ID" \
  --processed-root "$WORK_DIR/acq${ACQ}/processed" \
  --output-root "$WORK_DIR/acq${ACQ}/embeddings/mci_finetuned" \
  --model-kind mci \
  --checkpoint "$BRAIN_REPO/src/checkpoints/vit_mci.ckpt" \
  --brainiac-checkpoint "$BRAIN_REPO/src/checkpoints/BrainIAC.ckpt" \
  --brainiac-root "$BRAIN_REPO" --device "$DEVICE"
```

每次输出一个 `<output-root>/R01_Study_002904.npz`，已有同名 NPZ 或 `.npz.tmp` 会报错，没有覆盖开关。两套 acquisition 共六个 NPZ。程序不自动挑选 acquisition，不替换正式队列结果。

## 7. 输出格式

与现有 BrainIAC embedding 完全一致，无新增 NPZ 键：

| NPZ 键 | shape | structured dtype 字段 |
|---|---|---|
| `block_embed` | `[2,4,768]` | `token0`、`mean_pooling`，均 float32 |
| `final_embed` | `[2,768]` | `token0`、`mean_pooling`，均 float32 |

模态轴为 T1、T2；层轴为 block 3/6/9/12；final 是最终 LayerNorm 后的表示。`token0` 是第一个空间 patch，`mean_pooling` 是排除 token0 后其余 215 个 patch 的均值。它与 vit_survival 的表示语义、维度并不相同，沿用各自原有 schema。

例如检查一个输出：

```python
import numpy as np

path = "/path/to/brainiac_single_study_correction/acq1/embeddings/brainiac_backbone/R01_Study_002904.npz"
with np.load(path, allow_pickle=False) as data:
    assert set(data.files) == {"block_embed", "final_embed"}
    for key, shape in (("block_embed", (2, 4, 768)), ("final_embed", (2, 768))):
        assert data[key].dtype.names == ("token0", "mean_pooling")
        for pooling in ("token0", "mean_pooling"):
            array = data[key][pooling]
            assert array.shape == shape and array.dtype == np.float32
            assert np.isfinite(array).all()
            print(key, pooling, array.shape)
```

## 8. 本地验证与边界

```bash
export PYTHONPATH="$BRAIN_REPO/src"
python -m unittest discover -s "$SINGLE_DIR" -p 'test_*.py' -v
python -m unittest discover -s "$BRAIN_REPO/tests" -p 'test_brainiac_embedding_*.py' -v
```

测试覆盖合成 DICOM 的 46→23+23 修正、空间顺序、spacing、缺失 acquisition、重复层面和文件保护，以及单 Study 路由、真实 validation transform、原 pooling/NPZ writer 和三个模型参数分支。测试中的昂贵预处理及 checkpoint encoder 使用替身；它们不证明真实病例配准成功或公开权重加载成功。

2026-09-29 本地验证：新增 12 项测试、原有 34 项 BrainIAC embedding 测试通过；Ruff、编译检查通过。合成拆分的 T1/T2_acq1/T2_acq2 均为 23 层、7 mm spacing，像素顺序和首末位置正确。测试环境为 Python 3.10、PyTorch 2.13.0+cpu、MONAI 1.3.2、NumPy 2.2.6、SimpleITK 2.3.1、pydicom 3.0.1；缺少的 nibabel 5.3.2 仅安装到 `E:\codex\MRI_project\.codex_tmp\brainiac_single_study_deps` 并通过测试进程的 PYTHONPATH 引入，没有修改既有环境。该环境不是服务器锁定依赖的验收。

真实 `R01_Study_002904` 的 DICOM、三个正式 checkpoint 的完整运行、配准/HD-BET 人工 QC 仍需在数据所在服务器执行。

## 9. 按 coverage 转移已有 embedding

`move_low_coverage_embeddings.py` 是独立的轻量工具，只需要 Python 标准库。
读取 `export_low_coverage_clinical.py` 输出 CSV 的 `Study_ID`、`T1_coverage_mm`、`T2_coverage_mm` 三列。
默认条件是 **T1 coverage < 101 mm 或 T2 coverage < 101 mm**；等于阈值不移动。

先预览（Linux Bash 示例；Python 环境不要求安装模型依赖）：

```bash
python "$SINGLE_DIR/move_low_coverage_embeddings.py" \
  --csv /path/to/low_coverage_clinical.csv \
  --threshold 101 \
  --embedding-dir /path/to/brainiac_embeddings \
  --transfer-dir /path/to/low_coverage_brainiac_embeddings \
  --dry-run
```

确认清单后去掉 `--dry-run`，执行相同命令即可实际移动。vit_survival 使用同一脚本，把输入和转移目录改为对应路径即可；两种模型应使用各自的转移目录。

- 输入可为单模型目录，也可为包含多个模型/归一化分支的 **embedding 根目录**。递归查找准确匹配的 `<Study_ID>.npz`，每个匹配分支都移动，目标保留相对目录结构。例如 `brainage_finetuned/A.npz` 移到 `<transfer-dir>/brainage_finetuned/A.npz`。不要把包含 prepared 等中间产物的整个项目根目录传进来。
- CSV 中的 `excluded_` 是上游增加的显示标记，匹配文件名前去掉一次。只有 coverage 满足条件才移动；例如 `excluded_R01_Study_002904` 两模态 coverage 均约 154 mm，默认不会移动。重复 ID 合并为一个筛选 ID。
- 空值、`unknown`、NaN、无穷或负数不视为有效 coverage，也不自动视为 0；另一模态有有效低值时仍移动。程序报告含未知/非法 coverage 的行数。
- 只移动 NPZ，不读取或修改其内容，不做哈希校验。vit_survival 的 embedding metadata 已在 NPZ 内；不移动队列 CSV、归一化统计、日志和临时文件，也不自动更新训练 manifest。后续训练应使用与保留样本一致的 manifest。
- 执行前检查全部已匹配目标；存在同名目标或目录冲突则停止，尚未开始任何移动。源目录和转移目录不能相同或相互嵌套。未找到 NPZ 的选中 Study 会打印 `[missing]`；不会报成已转移。重复执行只处理源中仍存在的匹配文件。
- 支持跨磁盘移动（`shutil.move`）。跨磁盘是复制后删除，不是整个批次的原子操作；中途失败时已移动文件留在目标目录，已有目标不会被自动覆盖。
- `--dry-run` 只检查并打印，不创建目录或文件。输出包括选中 Study 数、找到/缺失 Study 数及文件数；多个模型分支使文件数可能大于 Study 数。

**CSV 完整性范围：** 上游导出器默认 `--coverage-threshold-mm 100`，因此旧 CSV 通常缺少两模态均 ≥100、但至少一个 <101 的 Study。本脚本只能筛选 CSV 已有行。要完整使用默认阈值 101，请用原 `selected_series.csv` 和 labels 重新导出一个新 CSV，指定 `--coverage-threshold-mm 101`；更大的移动阈值同样需要上游导出阈值至少一样大。

本工具的专项测试（临时合成文件，不接触真实 embedding）：

```bash
export PYTHONPATH="$BRAIN_REPO/src"
python -m unittest brainiac_embedding.single_abnormal_study.test_move_low_coverage_embeddings -v
```
