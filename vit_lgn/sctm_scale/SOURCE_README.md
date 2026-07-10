# LogicViT-Tiny with IWP (Input-wise Parametrization)

## 项目简介

本项目实现了基于逻辑门网络（Differentiable Logic Gate Networks）的 Vision Transformer Tiny 模型，使用输入级参数化（IWP, Input-wise Parametrization）技术。将传统 ViT 中的 MLP 前馈网络替换为可微分的逻辑门网络，支持温度计编码（Thermometer Encoding）、温度退火与软/硬评估切换，并支持两种 FFN 类型：直通型 LogicFFN 与向量化随机森林 RandomForestLogicFFN。

## 主要特性

- **逻辑门网络 FFN**：使用 LogicLayerIWP 或 RandomForestLogicFFN_IWP 替换传统 MLP，实现可微分逻辑运算
- **数据编码与 Logic 编码分离**：`--data-encoding` 控制输入数据编码（real-input / N-thresholds），`--logic-use-thermometer` 等控制 LogicFFN 内部温度计编码
- **温度退火**：训练过程中温度从 `temp-start` 退火到 `temp-end`，支持 warmup/cooldown 比例
- **软/硬评估**：训练中仅做硬前向评估（不修改权重），最终加载 best 后调用一次 `harden_model` 再评估
- **Subtrain / Fulltrain**：训练中在训练集子集（subtrain）上周期性评估；最终在全训练集（fulltrain）上取硬化前软训练精度与硬化后硬评估精度，用于泛化误差分析
- **按运行组织日志**：每次训练在 `logs/<timestamp>/` 下生成当次的 `.log`、`.csv` 与 `checkpoints/`
- **CIFAR-10/100**：完整数据加载与预处理管道

## 项目结构

```
vit_tiny_logictree_iwp/
├── README.md
├── __init__.py
├── train_logic_vit_tiny.py       # 训练脚本（参数解析、训练循环、评估、日志与 checkpoint）
├── logic_iwp.py                  # LogicLayerIWP、LogicFFN_IWP、RandomForestLogicFFN_IWP
├── logic_vit_tiny.py             # LogicViTTiny、LogicTransformerBlock、logic_vit_tiny()
├── data_pipeline.py              # 数据加载、变换、load_dataset、input_channels_of_dataset
├── vit_tiny_baseline.py          # PatchEmbedding、MultiHeadAttention、DropPath 等 ViT 组件
├── run_temp_sweep.sh             # 温度扫描示例（仅改 temp-end，多轮训练）
├── requirements.txt
├── data/                         # 数据集目录
│   └── cifar-10/                 # CIFAR-10 原始数据（torchvision 下载至此）
├── local_difflogic/              # 本地 difflogic 实现（含 CUDA 源码，可与 site-packages 版本切换）
│   ├── __init__.py
│   ├── difflogic.py              # LogicLayer、LogicLayerIWP、GroupSum、CUDA 封装
│   ├── functional.py             # bin_gate、梯度因子等
│   └── cuda/                     # CUDA 源码（需编译生成 .so/.pyd）
│       ├── bindings_iwp.cpp
│       ├── difflogic_iwp.h
│       ├── difflogic_iwp_forward_train.cu
│       ├── difflogic_iwp_forward_eval.cu
│       ├── difflogic_iwp_backward_x.cu
│       ├── difflogic_iwp_backward_w.cu
│       ├── difflogic_shared.cuh
│       └── ...
└── logs/                         # 训练日志根目录
    ├── figure.py                 # 日志可视化脚本
    ├── migrate_logs_to_run_folders.py   # 一次性迁移：将平铺的 log/csv/checkpoints 归入 run 子文件夹
    └── <timestamp>/              # 每次运行一个子文件夹，例如 20260201_132846
        ├── training_<timestamp>.log
        ├── training_<timestamp>.csv
        └── checkpoints/          # 仅当 --save-checkpoints 时存在
            ├── checkpoint_init.pt
            ├── checkpoint_step_25pct.pt
            ├── checkpoint_step_50pct.pt
            ├── checkpoint_step_75pct.pt
            ├── checkpoint_best_before_harden.pt
            └── checkpoint_best_after_harden.pt
```

## 使用方法

### 基本训练

```bash
python train_logic_vit_tiny.py \
  --dataset cifar-10 \
  --data-encoding real-input \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --num-iterations 50000 \
  --eval-freq 1000
```

### 关闭 LogicFFN 温度计编码（仅验证训练与保存）

```bash
python train_logic_vit_tiny.py \
  --data-encoding real-input \
  --no-logic-use-thermometer \
  --num-iterations 50000 \
  --save-checkpoints
```

### 完整参数示例（直通型 LogicFFN）

```bash
python train_logic_vit_tiny.py \
  --seed 42 \
  --dataset cifar-10 \
  --data-encoding real-input \
  --augment \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --num-iterations 50000 \
  --ffn-type logic \
  --logic-ffn-layers 3 \
  --logic-connections random \
  --logic-n-thresholds 31 \
  --logic-use-thermometer \
  --logic-encoding-temperature 10.0 \
  --temp-start 1.0 \
  --temp-end 0.1 \
  --temp-warmup-ratio 0.05 \
  --temp-cooldown-ratio 0.15 \
  --eval-freq 1000 \
  --ext-eval-freq 5000 \
  --save-checkpoints \
  --checkpoint-progress-ratios 0.25,0.5,0.75
```

### 温度扫描（run_temp_sweep.sh）

脚本 `run_temp_sweep.sh` 在固定其他参数下，对 `--temp-end` 做多组训练（如 0.2 到 0.7），用于超参扫描。在项目根目录执行：

```bash
./run_temp_sweep.sh
```

## 主要参数说明

**数据**
- `--dataset`：cifar-10 / cifar-100
- `--data-encoding`：输入数据编码方式。`real-input` 为原始像素；`1-thresholds`、`3-thresholds`、…、`31-thresholds` 为多阈值编码，影响输入通道数
- `--augment`：是否使用数据增强
- `--preprocess-once`：是否预处理一次（当前 load_dataset 未使用预处理缓存，仅影响预留接口）

**训练**
- `--batch-size`、`--batches-per-backward`：批大小与梯度累积步数
- `--learning-rate`、`--num-iterations`、`--weight-decay`、`--label-smoothing`
- `--valid-set-size`：从训练集中划分验证集比例
- `--eval-freq`、`--ext-eval-freq`：验证评估与 subtrain 评估间隔（步数）
- `--train-eval-subsample-size`：subtrain 评估时子采样样本数（默认 1000）
- `--no-logging`：不写 log/csv（仍会训练；若需 checkpoint 会创建 run 目录并写入 checkpoints）
- `--save-checkpoints`：是否保存 init、进度比例、best 硬化前后 checkpoint
- `--checkpoint-progress-ratios`：逗号分隔比例，如 `0.25,0.5,0.75`，在总步数对应比例处各保存一次

**ViT 骨架**
- `--img-size`、`--patch-size`、`--embed-dim`、`--depth`、`--num-heads`、`--drop-path-rate`

**LogicFFN 与温度**
- `--ffn-type`：`logic`（直通型 LogicFFN_IWP）或 `tree-logic`（RandomForestLogicFFN_IWP）
- `--logic-ffn-layers`、`--logic-connections`：仅 `--ffn-type logic` 时生效
- `--num-forest-layers`：仅 `--ffn-type tree-logic` 时生效
- 层内初始化顺序：**0) weight_init** → **1) 重尾 shift_init** → **2) 残差连接**（互不替代）。`--logic-weight-init`：权重初始化方式（`ri`=软直通 A，`gauss`=高斯随机）。`--logic-weight-init-sigma`：weight_init 尺度（ri 时 ±sigma 送入激活，SIN01 下默认 0.5 更可塑，1.0 较饱和；gauss 时标准差）。`--logic-shift-init`、`--logic-shift-init-type`、`--logic-shift-init-shift`、`--logic-shift-init-direction`：重尾初始化。`--logic-resconnection-init`、`--logic-res-connect-fraction`：残差连接（由 fraction 控制比例；resconnection-init 为 True 时残差门固定为直通 A 并关梯度，为 False 时仅改连接、权重可训练）。
- `--logic-n-thresholds`、`--logic-use-thermometer`、`--logic-encoding-temperature`：LogicFFN 内部温度计编码
- `--temp-start`、`--temp-end`、`--temp-warmup-ratio`、`--temp-cooldown-ratio`：训练温度退火

**正则与增强**
- `--weight-decay`、`--label-smoothing`、`--mixup`、`--mixup-alpha`、`--mixup-prob`、`--cutmix`、`--cutmix-prob`、`--cutmix-alpha`（CutMix 与 Mixup 独立，可单独开启）

## 训练输出

- **日志目录**：每次运行在 `logs/<timestamp>/` 下生成当次所有输出。
- **文本日志**：`training_<timestamp>.log`，含参数、每步/每次评估的简要输出。
- **CSV**：`training_<timestamp>.csv`，含 phase、step、train_loss、各 acc 百分比、temperature 等，便于绘图与分析。
- **指标含义**：
  - `valid/acc_eval`、`valid/acc_train`：验证集上硬评估与软训练精度
  - `test/acc_eval`、`test/acc_train`：测试集上同上（仅在最终评估等阶段写入）
  - `subtrain/acc_eval`、`subtrain/acc_train`：训练集子集上的硬/软精度（监控过拟合）
  - `fulltrain/acc_train`、`fulltrain/acc_eval`：最终阶段全训练集上硬化前软训练精度、硬化后硬评估精度（用于泛化误差）
- **Checkpoints**（需 `--save-checkpoints`）：在 `logs/<timestamp>/checkpoints/` 下保存 `checkpoint_init.pt`、`checkpoint_step_25pct.pt` 等及 best 硬化前后 pt，每项含 `state_dict`、`step`、`temperature`。

## 日志迁移（旧平铺结构 → 按运行子文件夹）

若之前已有平铺在 `logs/` 下的 `training_*.log`、`training_*.csv` 和 `checkpoints_*/`，可使用一次性迁移脚本归入 `logs/<timestamp>/`：

```bash
# 仅打印将要执行的操作
python logs/migrate_logs_to_run_folders.py --dry-run

# 执行迁移
python logs/migrate_logs_to_run_folders.py
```

迁移后结构：每个时间戳对应一个 `logs/<timestamp>/`，内含对应 `.log`、`.csv`，以及（若存在）`checkpoints/`（由原 `checkpoints_<timestamp>/` 内容移入）。

## 依赖

- Python 3.8+
- PyTorch 1.12+
- torchvision
- numpy
- tqdm
- matplotlib（用于 `logs/figure.py` 可视化）
- CUDA（可选，用于逻辑层加速）

安装核心依赖：

```bash
pip install -r requirements.txt
```

## 核心文件说明

### train_logic_vit_tiny.py

- `parse_args()`：所有 CLI 参数（数据、训练、ViT、LogicFFN、温度、checkpoint 等）
- `train()`：主流程（构建 run_dir、load_dataset、build_model、温度设置、训练循环、best 保存、最终硬化与 fulltrain/subtrain 评估、日志与 CSV 写入）
- `eval_on_loader()`：在指定 DataLoader 上评估（支持 `train_mode` 与 `subsample_size`）
- `run_eval()`：验证/测试集完整评估
- `evaluate_train_subset()`：subtrain 子集评估（返回 `subtrain/acc_eval`、`subtrain/acc_train`）
- `save_checkpoint()`：将 state_dict、step、temperature 写入指定路径
- `logic_layers()`、`set_model_temperature()`、`harden_model()`、`logic_eval_mode()`：逻辑层温度与硬化控制
- `reset_fixed_weights()`：每步优化后重置固定权重

### logic_iwp.py

- `LogicLayerIWP`：带温度与软/硬评估的逻辑层封装
- `LogicFFN_IWP`：多层 LogicLayerIWP，支持温度计编码
- `RandomForestLogicFFN_IWP`：向量化随机森林风格 Logic FFN（`--ffn-type tree-logic`）

### logic_vit_tiny.py

- `LogicViTTiny`、`LogicTransformerBlock`、`logic_vit_tiny()`：模型结构与工厂函数，依赖 `vit_tiny_baseline` 的 PatchEmbedding、MultiHeadAttention、DropPath

### data_pipeline.py

- `load_dataset(args)`：根据 `args.dataset` 与 `args.data_encoding` 构建 train/valid/test DataLoader 与变换
- `get_transforms(args)`：数据变换列表（含 `data_encoding` 对应的阈值变换）
- `input_channels_of_dataset(args)`、`class_count_of_dataset(args)`：输入通道数与类别数
- `PreprocessedDataset`、`preprocess_dataset()`：预留的预处理缓存接口，当前 `load_dataset` 未使用

## CUDA 扩展（可选）

逻辑层可通过 CUDA 扩展加速。编译后的模块（如 `difflogic_cuda_iwp.so`）通常由 `local_difflogic/cuda/` 下源码生成；运行时若使用 site-packages 中的 `difflogic`，则通过其导入 CUDA 扩展。项目内 `local_difflogic/` 为备用实现与 CUDA 源码所在目录，可与 site-packages 版本切换使用。

- **编译**：若有 `setup.py` 或相应构建脚本，在项目根目录执行对应 `build_ext --inplace` 等命令，生成的 `.so`/`.pyd` 一般在项目根或 build 目录。
- **推理**：硬化后推理仅需前向 CUDA 内核（如 `difflogic_iwp_forward_eval.cu`）；训练还需反向与权重梯度内核。

详细调用关系与依赖树见原 README 中“CUDA 扩展调用关系”“CUDA 模式 + 温度阈值二值化推理文件依赖树”等章节（路径中的 `vit_tiny_logic_iwp` 可视为本项目根目录 `vit_tiny_logictree_iwp`）。

## 注意事项

1. **data 目录**：当前仅使用 `data/cifar-10/`（或 `data/cifar-100/`）。`load_dataset` 不会读取其他预处理缓存目录。
2. **subtrain 返回 -1.0**：若某次 subtrain 评估得到 `-1.0`，表示该次子采样未采到有效 batch（或 loader 为空），属边界情况，可增大 `--train-eval-subsample-size` 或忽略单点。
3. **两套 difflogic**：site-packages 中的 `difflogic` 为当前默认导入；`local_difflogic/` 为本地副本与 CUDA 源码，可按需修改导入路径切换。
4. **确定性**：训练脚本会设置随机种子与 PyTorch 确定性选项，便于复现。

## 许可证

本项目基于 difflogic-light 相关思路实现，请参考相关许可证。
