# 新预训练模型的独立 embedding 导出

`embed.py` 是独立程序，直接使用 MONAI ViT、PyTorch 和 NumPy，不导入训练模块或已有 embedding 程序。依赖已包含在预训练环境中；无需导入 Lightning 或 Lightly。

## 输入与运行

在仓库根目录运行：

```bash
python src/brainIAC_pretraining/embed.py \
  --final-csv work/pretraining/processed/final_pre_process.csv \
  --input-dir work/pretraining/processed \
  --selected-csv work/pretraining/stats/selected_series.csv \
  --checkpoint /path/to/epoch198.ckpt \
  --output-dir work/pretraining/embeddings_epoch198 \
  --device cuda:0
```

CPU 使用 `--device cpu`；未指定时使用可用的 CUDA，否则使用 CPU。每次处理一个 volume，不要求 T1/T2 配对，不丢弃不足一个 batch 的样本。

- `--final-csv`：与训练相同的清单，包含 `unique_id`；只导出该清单指定的 volume，不扫描图像目录补充样本。
- `--input-dir`：预处理图像，文件名为 `<unique_id>.nii.gz`，完整保留 UID 中的点号。
- `--selected-csv`：与这批预处理结果匹配的元数据，必须包含 `unique_id,study_id,series_description`。多余记录忽略；入选 ID 的描述可以为空，但元数据缺失或重复时报错。Study 分组直接使用 `study_id`，不从文件名猜测。
- `--checkpoint`：新预训练的完整 Lightning checkpoint，严格加载 `state_dict` 内的 `backbone.*` 参数；忽略 projection、optimizer 和 scheduler。

图像执行固定的加载、单通道、trilinear resize 到 96³、非零体素逐 volume z-score。无随机增强，只生成一个固定视图。Encoder 在 eval/inference 模式下以 float32 推理，不经过 projection head，不额外做 L2 normalization。

## 输出契约

```text
embeddings_epoch198/
├── embedding_config.json
├── <study_id_A>.npz
└── <study_id_B>.npz
```

每个 NPZ 的顶层 key 是该 Study 所有入选序列的 `unique_id`。每个值是零维 NumPy structured array，包含以下字段，不使用 object 字典：

| 读取方式 | 类型／形状 | 含义 |
|---|---|---|
| `data[unique_id]["series_description"]` | Unicode 标量数组 | 原始元数据描述；`.item()` 转为 Python 字符串 |
| `data[unique_id]["block_embed"]` | float32 `(4,768)` | block 3、6、9、12，按此顺序 |
| `data[unique_id]["final_embed"]` | float32 `(768,)` | 最终 LayerNorm 后的表征 |

所有层均平均全部 216 个空间 patch tokens，包含 token 0。Block 12 在最终 LayerNorm 前，final 在其后，二者独立提取。没有 T1/T2 固定轴，也没有 `token0`／`mean_pooling` 子字段；不同序列不在导出时融合。

```python
import numpy as np

with np.load("/path/to/STUDY_0001.npz", allow_pickle=False) as data:
    for unique_id in data.files:
        record = data[unique_id]
        description = record["series_description"].item()
        blocks = record["block_embed"]  # (4, 768)
        final = record["final_embed"]  # (768,)
        print(unique_id, description, blocks.shape, final.shape)
```

## 保存、续跑与实验来源

一个 Study 全部入选序列提取成功后，先写 `<study_id>.npz.tmp`，再重命名发布正式 NPZ。读取或推理出错直接停止，失败 Study 不发布部分 NPZ；此前完成的 Study 保留。遗留 `.tmp` 不计为完成，下次计算该 Study 时覆盖临时文件。

再次运行同一命令，正式 `<study_id>.npz` 存在即跳过，不重新加载该 Study 图像，也不读取或验证已有 NPZ 内容。所有 Study 均完成时，不加载 checkpoint。重复运行前需确保旧进程已退出；同一个输出目录只运行一个进程。

`embedding_config.json` 记录 checkpoint 和输入路径、pooling、层顺序及固定变换。重跑时配置不同则要求使用新输出目录；它不能识别同一路径的 checkpoint、CSV 或图像内容已被替换。因此同一次输出固定输入和模型，更换文件内容或修改清单也应使用新目录。输入、权重和旧 embedding 不被修改。

比较 epoch 97 与 198 时分别指定对应 checkpoint 和独立输出目录。此程序仅导出特征，不训练下游模型。

## 本地验证

```bash
python -m unittest discover -s src/brainIAC_pretraining/tests -p test_embed.py -v
```

测试使用合成图像及合成 checkpoint。服务器患者图像、实际训练 checkpoint 和 CUDA 推理仍需在有对应文件与 GPU 的环境中验证。
