"""
使用 difflogic-light 风格的训练策略训练 LogicViTTiny 模型。

本模块实现了完整的训练流程，包括：
- 命令行参数解析
- 模型构建和初始化
- 训练循环（支持梯度累积、温度退火）
- 验证和测试评估
- 日志记录（文本和 CSV 格式）
- 最佳模型保存
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import csv
import json
import torch
from torch import nn
from tqdm import tqdm

from data_pipeline import (
    class_count_of_dataset,
    load_dataset,
    num_channels_of_dataset,
)
from logic_vit_tiny import logic_vit_tiny
from vit_tiny_baseline import vit_tiny as vit_tiny_baseline

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# 自动选择计算设备（优先使用 CUDA）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description="Train LogicViTTiny with logic-style strategy.")

    # -----------------------------------------------------------------------
    # 通用：训练/数据、ViT 骨架、两种 FFN 共用的 LGN 与温度退火、正则与增强
    # -----------------------------------------------------------------------
    common = parser.add_argument_group(
        "通用（训练/数据、ViT、两种 LGN-FFN 共用）",
        "以下参数对直通型与向量化随机森林均生效，或与 FFN 类型无关。",
    )
    # ---- 训练 / 数据 ----
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--dataset", choices=["cifar-10", "cifar-100"], default="cifar-10")
    common.add_argument(
        "--data-encoding",
        choices=["real-input", "1-thresholds", "3-thresholds", "7-thresholds", "15-thresholds", "23-thresholds", "31-thresholds"],
        default="real-input",
    )
    common.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="数据增强")
    common.add_argument("--preprocess-once", action=argparse.BooleanOptionalAction, default=True, help="预处理一次，默认开启")
    common.add_argument("--batch-size", type=int, default=512)
    common.add_argument("--batches-per-backward", type=int, default=1)
    common.add_argument("--learning-rate", type=float, default=1e-3)
    common.add_argument("--num-iterations", type=int, default=180_001)
    common.add_argument("--grad-factor", type=float, default=1.0)
    common.add_argument("--valid-set-size", type=float, default=0.1)
    common.add_argument("--eval-freq", type=int, default=1_000)
    common.add_argument("--ext-eval-freq", type=int, default=5_000)
    common.add_argument("--eval-initial", action=argparse.BooleanOptionalAction, default=False)
    common.add_argument("--no-logging", action="store_true")
    common.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save checkpoints for weight comparison: init, progress ratios, best before/after harden.",
    )
    common.add_argument(
        "--checkpoint-progress-ratios",
        type=str,
        default="0.25,0.5,0.75,1",
        help="Comma-separated progress ratios (0-1] at which to save a checkpoint, e.g. 0.25,0.5,0.75,1 (1=100%%).",
    )
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--print-freq", type=int, default=100)
    # ---- ViT 骨架 ----
    common.add_argument("--img-size", type=int, default=32)
    common.add_argument("--patch-size", type=int, default=4)
    common.add_argument("--embed-dim", type=int, default=192)
    common.add_argument("--depth", type=int, default=6)
    common.add_argument("--num-heads", type=int, default=3)
    common.add_argument("--drop-path-rate", type=float, default=0.1)
    common.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["logic-ffn", "tree-logic-ffn", "baseline-mlp", "baseline-attn-only"],
        help="选择整体模型结构：logic-ffn / tree-logic-ffn / baseline-mlp / baseline-attn-only；"
             "若未指定，则根据 --ffn-type 推断为 logic-ffn 或 tree-logic-ffn。",
    )
    # ---- 两种 FFN 共用的 LGN 与温度退火 ----
    # 层内初始化执行顺序：0) weight_init 1) 重尾 2) 残差连接（互不替代）

    # 0) weight_init：创建权重时的初始化方式（IWP 下 ri=软直通 A，gauss=高斯随机）
    common.add_argument("--logic-weight-init", choices=["ri", "gauss"], default="ri",
        help="Weight init for logic layers: ri=soft pass-through A (IWP), gauss=Gaussian random.")
    common.add_argument("--logic-weight-init-sigma", type=float, default=0.5, metavar="S",
        help="Scale for weight_init: ri 时 ±S 送入激活（SIN01 下 S=1 较饱和，0.5 更可塑）；gauss 时标准差。默认 0.5。")

    # 1) 重尾初始化（shift_init）：在 weight_init 之后、残差连接之前
    common.add_argument("--logic-shift-init", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to run heavy-tailed shift_init after weight_init.")
    common.add_argument("--logic-shift-init-type", choices=["ri", "and-or", "and-or-ri", "uniform"], default="ri",
        help="Shift init type: ri (per-dim +/-), and-or, and-or-ri, or uniform over 16 patterns.")
    common.add_argument("--logic-shift-init-shift", type=float, default=1.2, metavar="S",
        help="Shift magnitude for shift_init.")
    common.add_argument("--logic-shift-init-direction", type=str, default="0101", metavar="D",
        help="4-char 0/1 string for ri shift direction (0=subtract, 1=add). Only used when type=ri.")

    # 0.5) 逻辑激活函数选择（影响 LogicLayerIWP / RandomForestLogicFFN_IWP 的 act_fn）
    common.add_argument(
        "--logic-act-fn",
        type=str,
        default="SIN01",
        choices=["SIN01", "sigmoid", "sigmoid-st", "sin-st", "linear"],
        help="Activation for logic weights: SIN01 / sigmoid / sigmoid-st / sin-st / linear.",
    )

    # 2) 残差连接初始化：改 indices + 可选冻结残差门
    common.add_argument("--logic-resconnection-init", action=argparse.BooleanOptionalAction, default=False,
        help="Residual connection init: per-layer fraction and freeze residual gates (indices a[i]=i, weights pass-through A).")
    common.add_argument(
        "--logic-res-connect-fraction",
        type=float,
        default=0,
        metavar="F",
        help="Override residual connection fraction (0-1) for all logic layers. If set, used for all layers; 0 disables. If unset, logic-resconnection-init controls per-layer fraction.",
    )

    #LGN层温度阈值编码
    common.add_argument("--logic-n-thresholds", type=int, default=1, help="Number of thresholds for thermometer encoding")
    common.add_argument("--logic-use-thermometer", action=argparse.BooleanOptionalAction, default=True, help="Use thermometer encoding")
    common.add_argument("--logic-encoding-temperature", type=float, default=10.0, help="Temperature for differentiable encoding")
    #LGN层温度退火
    common.add_argument("--temp-start", type=float, default=1.0)
    common.add_argument("--temp-end", type=float, default=0.6)
    common.add_argument("--temp-warmup-ratio", type=float, default=0.02)
    common.add_argument("--temp-cooldown-ratio", type=float, default=0.05)
    # ---- 正则与增强 ----
    common.add_argument("--weight-decay", type=float, default=0.01, help="L2 weight decay")
    common.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing coefficient")
    common.add_argument("--mixup", action=argparse.BooleanOptionalAction, default=True, help="Enable mixup augmentation")
    common.add_argument("--mixup-alpha", type=float, default=0.5, help="Beta distribution alpha for mixup")
    common.add_argument("--mixup-prob", type=float, default=1.0, help="Probability to apply mixup per batch")
    common.add_argument("--cutmix", action=argparse.BooleanOptionalAction, default=True, help="Enable CutMix augmentation (independent from mixup)")
    common.add_argument("--cutmix-prob", type=float, default=0.5, help="Probability to apply CutMix per batch")
    common.add_argument("--cutmix-alpha", type=float, default=1.0, help="Beta distribution alpha for CutMix")
    # ---- FFN 类型（直通型 / 向量化随机森林） ----
    common.add_argument("--ffn-type", choices=["logic", "tree-logic"], default="logic", help="logic=直通型 LogicFFN_IWP, tree-logic=索引融合向量化随机森林 RandomForestLogicFFN_IWP")

    # -----------------------------------------------------------------------
    # 仅直通型 LGN-FFN（--ffn-type logic）
    # -----------------------------------------------------------------------
    logic_only = parser.add_argument_group(
        "仅直通型 LGN-FFN (--ffn-type logic)",
        "下列参数仅在选用直通型 LogicFFN_IWP 时生效。",
    )
    logic_only.add_argument("--logic-ffn-layers", type=int, default=3, help="直通型 FFN 层数；仅当 --ffn-type logic 时生效")
    logic_only.add_argument("--logic-mlp-ratio", type=float, default=1.0, help="Hidden width multiplier for grouped LogicFFN reduction")
    logic_only.add_argument("--logic-connections", choices=["random", "unique"], default="random", help="LogicLayerIWP 连接模式；仅当 --ffn-type logic 时生效")
    logic_only.add_argument("--logic-connectivity", choices=["fixed", "learnable"], default="fixed", help="fixed=原随机/unique连接；learnable=T-Net候选池连接")
    logic_only.add_argument("--learnable-conn-k", type=int, default=64, help="Candidate pool size for learnable connections")
    logic_only.add_argument("--learnable-conn-use-skip-bias", action=argparse.BooleanOptionalAction, default=True, help="Use skip-distance bias in learnable connection softmax")
    logic_only.add_argument("--logic-conn-lr-multiplier", type=float, default=0.2, help="Learning-rate multiplier for conn_logits")
    logic_only.add_argument("--cross-block-logic-history", action=argparse.BooleanOptionalAction, default=False, help="Allow learnable LGN ports to select previous Transformer-block logic states")
    logic_only.add_argument("--cross-block-candidate-frac", type=float, default=0.0, help="Fraction of each candidate pool reserved for cross-block sources")
    logic_only.add_argument("--post-discretize-finetune-iters", type=int, default=0, help="Short fixed-connection gate-only finetune after connection discretization")
    logic_only.add_argument("--post-discretize-finetune-lr", type=float, default=0.0, help="LR for post-discretization finetune; 0 uses --learning-rate")

    # -----------------------------------------------------------------------
    # 仅向量化随机森林 LGN-FFN（--ffn-type tree-logic）
    # -----------------------------------------------------------------------
    forest_only = parser.add_argument_group(
        "仅向量化随机森林 LGN-FFN (--ffn-type tree-logic)",
        "下列参数仅在选用向量化随机森林 RandomForestLogicFFN_IWP 时生效。",
    )
    forest_only.add_argument("--num-forest-layers", type=int, default=2, help="森林层数；仅当 --ffn-type tree-logic 时生效")

    args = parser.parse_args()
    # -----------------------------------------------------------------------
    # model_type：显式传 baseline-* 时保留；否则由 ffn_type 推断，保证「只传 --ffn-type logic」时为 logic-ffn
    # -----------------------------------------------------------------------
    mt = getattr(args, "model_type", None)
    if mt in ("baseline-mlp", "baseline-attn-only"):
        pass  # 用户显式要 baseline，保留
    elif args.ffn_type == "logic":
        args.model_type = "logic-ffn"
    elif args.ffn_type == "tree-logic":
        args.model_type = "tree-logic-ffn"
    elif mt is None:
        args.model_type = "logic-ffn"
    return args


def set_deterministic(seed: int):
    """
    设置随机种子以确保实验的可重复性。

    Args:
        seed: 随机种子值
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    random.seed(seed)


def logic_layers(model: nn.Module) -> Iterable[nn.Module]:
    """
    获取模型中所有逻辑层模块的迭代器。

    Args:
        model: PyTorch 模型

    Yields:
        nn.Module: 具有 set_soft_eval 方法的逻辑层模块
    """
    for module in model.modules():
        if hasattr(module, "set_soft_eval"):
            yield module


def logic_ffn_modules(model: nn.Module) -> Iterable[nn.Module]:
    """
    获取模型中所有 LogicFFN_IWP 模块的迭代器，后续方便统一操作。

    Args:
        model: PyTorch 模型

    Yields:
        nn.Module: 具有 harden 和 thermometer_encode 方法的逻辑 FFN 模块
    """
    for module in model.modules():
        if hasattr(module, "harden") and hasattr(module, "thermometer_encode"):
            yield module


def harden_model(model: nn.Module, temperature: float | None = None):
    """
    硬化所有逻辑层和 FFN（推理时使用）。

    将模型从软评估模式切换到硬评估模式，使权重完全二值化。
    仅通过 logic_ffn_modules 调用 harden，避免对同一 LogicLayerIWP 硬化两次。

    Args:
        model: PyTorch 模型
        temperature: 若提供，则先对该温度调用 set_model_temperature 再硬化，保证与
            checkpoint 中保存的 temperature 一致（避免与 analyze_checkpoints 模拟硬化不一致）
    """
    if temperature is not None:
        set_model_temperature(model, temperature)
    for module in logic_ffn_modules(model):
        module.harden()


def set_model_temperature(model: nn.Module, temperature: float):
    """
    设置模型中所有支持温度控制的模块的温度参数。

    Args:
        model: PyTorch 模型
        temperature: 温度值（用于控制软硬程度，温度越低越硬）
    """
    for module in model.modules():
        if hasattr(module, "set_temperature"):
            module.set_temperature(temperature)


@contextmanager
def logic_eval_mode(model: nn.Module, train_mode: bool):
    """
    临时切换逻辑层的评估模式：
    - train_mode=True: 逻辑层软前向（训练）
    - train_mode=False: 逻辑层硬前向（推理）
    退出后恢复原状态。
    """
    modules = list(logic_layers(model))
    prev = [getattr(m, "_soft_eval", True) for m in modules]
    for module in modules:
        module.set_soft_eval(train_mode)
    try:
        yield
    finally:
        for module, state in zip(modules, prev):
            module.set_soft_eval(state)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算预测准确率。

    Args:
        logits: 模型输出的 logits 张量 [batch_size, num_classes]
        targets: 真实标签张量 [batch_size]

    Returns:
        float: 准确率（0.0 到 1.0 之间）
    """
    return (logits.argmax(-1) == targets).to(torch.float32).mean().item()


def mixup_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    """
    对一个 batch 应用 Mixup，返回混合后的图片、两组标签以及混合系数 lam。
    """
    if alpha <= 0.0:
        return images, targets, targets, 1.0
    # Beta 分布采样 lam（利用当前随机种子保证可复现）
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    targets_a = targets
    targets_b = targets[perm]
    return mixed, targets_a, targets_b, lam


def cutmix_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    """
    对一个 batch 应用 CutMix：随机裁剪粘贴区域，按面积比例混合标签。
    返回混合后的图片、两组标签以及混合系数 lam（原图保留比例，损失为 lam*L(y_a)+(1-lam)*L(y_b)）。
    """
    if alpha <= 0.0:
        return images, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(images.size(0), device=images.device)
    _, _, H, W = images.shape
    # 裁剪区域面积比例为 1-lam，边长比 sqrt(1-lam)
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(0, W, (1,), device=images.device).item()
    cy = torch.randint(0, H, (1,), device=images.device).item()
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    targets_a = targets
    targets_b = targets[perm]
    return mixed, targets_a, targets_b, lam


def save_checkpoint(
    model: nn.Module,
    path: Path,
    step: int,
    temperature: float,
) -> None:
    """保存 checkpoint：state_dict（CPU）、step、temperature。"""
    state = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "step": step,
        "temperature": temperature,
    }
    torch.save(state, path)


def eval_on_loader(
    model: nn.Module,
    loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    train_mode: bool,
    transform,
    subsample_size: Optional[int] = None,
) -> float:
    """
    在数据加载器上评估模型。

    Args:
        model: PyTorch 模型
        loader: 数据加载器（如果为 None，返回 -1.0）
        device: 计算设备
        train_mode: 是否使用训练模式（影响 dropout 和 batch norm）
        transform: 数据变换函数
        subsample_size: 子采样大小（如果指定，只评估部分数据）

    Returns:
        float: 模型在数据加载器上的准确率（如果 loader 为 None，返回 -1.0）
    """
    if loader is None:
        return -1.0
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=train_mode)
        # 训练中 eval 只做硬前向（logic_eval_mode 设 _soft_eval=False），不调用 harden_model，不改权重
        with logic_eval_mode(model, train_mode=train_mode):
            if subsample_size is not None:
                accuracies = []
                samples_processed = 0
                dataset_size = len(loader.dataset) if hasattr(loader.dataset, "__len__") else None
                batch_size = loader.batch_size or 1
                if dataset_size is not None:
                    total_batches = len(loader)
                    target_batches = min(total_batches, (subsample_size + batch_size - 1) // batch_size)
                    keep_prob = target_batches / max(total_batches, 1)
                else:
                    keep_prob = min(1.0, subsample_size / (batch_size * 1000))

                for batch_idx, (images, targets) in enumerate(loader):
                    if samples_processed >= subsample_size:
                        break
                    if random.random() > keep_prob:
                        continue
                    images = images.to(device, non_blocking=True)
                    images = transform(images)
                    targets = targets.to(device, non_blocking=True)
                    preds = model(images)
                    accuracies.append(accuracy(preds, targets))
                    samples_processed += targets.size(0)
                    if batch_idx > 0 and batch_idx % 100 == 0:
                        expected_samples = (batch_idx + 1) * batch_size * keep_prob
                        if samples_processed < expected_samples * 0.5:
                            keep_prob = min(1.0, keep_prob * 1.5)

                if not accuracies:
                    res = -1.0
                else:
                    res = float(torch.tensor(accuracies).mean().item())
            else:
                per_batch = []
                for images, targets in loader:
                    images = images.to(device, non_blocking=True)
                    images = transform(images)
                    targets = targets.to(device, non_blocking=True)
                    per_batch.append(accuracy(model(images), targets))
                res = float(torch.tensor(per_batch).mean().item())
    model.train(mode=orig_mode)
    return res


def run_eval(
    args,
    model: nn.Module,
    valid_loader,
    test_loader,
    device: torch.device,
    transform,
    metrics: Dict[str, float],
    eval_test: bool,
):
    """
    运行完整的评估流程，包括验证集和可选的测试集评估。

    Args:
        args: 命令行参数对象
        model: PyTorch 模型
        valid_loader: 验证集数据加载器
        test_loader: 测试集数据加载器
        device: 计算设备
        transform: 数据变换函数
        metrics: 指标字典（用于记录最佳值）
        eval_test: 是否评估测试集

    Returns:
        Dict[str, float]: 包含所有评估指标的字典
    """
    result = {}
    result["valid/acc_eval"] = eval_on_loader(
        model,
        valid_loader,
        device,
        train_mode=False,
        transform=transform,
    )
    result["valid/acc_train"] = eval_on_loader(
        model,
        valid_loader,
        device,
        train_mode=True,
        transform=transform,
    )

    if eval_test:
        result["test/acc_eval"] = eval_on_loader(
            model,
            test_loader,
            device,
            train_mode=False,
            transform=transform,
        )
        result["test/acc_train"] = eval_on_loader(
            model,
            test_loader,
            device,
            train_mode=True,
            transform=transform,
        )

    for key, value in result.items():
        best_key = f"{key}_best"
        metrics[best_key] = max(metrics.get(best_key, 0.0), value)

    return result


def evaluate_fulltrain(
    args,
    model: nn.Module,
    train_eval_loader,
    device: torch.device,
    transform,
    metrics: Dict[str, float],
):
    """
    在全训练集上评估模型（无随机增强，用于监控过拟合/泛化）。

    Args:
        args: 命令行参数对象
        model: PyTorch 模型
        train_eval_loader: 训练集评估用加载器（无增强，与 valid/test 同套预处理）
        device: 计算设备
        transform: 数据变换函数
        metrics: 指标字典（用于记录最佳值）

    Returns:
        Dict[str, float]: 包含 fulltrain/acc_eval、fulltrain/acc_train 的字典
    """
    train_eval = eval_on_loader(
        model,
        train_eval_loader,
        device,
        train_mode=False,
        transform=transform,
        subsample_size=None,
    )
    train_train = eval_on_loader(
        model,
        train_eval_loader,
        device,
        train_mode=True,
        transform=transform,
        subsample_size=None,
    )
    metrics["fulltrain/acc_eval"] = train_eval
    metrics["fulltrain/acc_train"] = train_train
    metrics["fulltrain/acc_eval_best"] = max(metrics.get("fulltrain/acc_eval_best", 0.0), train_eval)
    metrics["fulltrain/acc_train_best"] = max(metrics.get("fulltrain/acc_train_best", 0.0), train_train)
    return {"fulltrain/acc_eval": train_eval, "fulltrain/acc_train": train_train}


def reset_fixed_weights(model: nn.Module):
    """
    重置模型中所有固定权重（如残差连接的权重）。

    在每个优化器步骤后调用，确保固定权重保持正确的值。

    Args:
        model: PyTorch 模型
    """
    for module in model.modules():
        if hasattr(module, "reset_fixed_weights"):
            module.reset_fixed_weights()


def build_model(args):
    """
    根据命令行参数构建 LogicViTTiny 模型。

    Args:
        args: 命令行参数对象

    Returns:
        LogicViTTiny: 构建好的模型实例
    """
    in_channels = num_channels_of_dataset(args)
    num_classes = class_count_of_dataset(args)

    model_type = getattr(args, "model_type", None)
    if model_type is None:
        # 理论上 parse_args 已经填充，这里只是再次兜底
        if args.ffn_type == "logic":
            model_type = "logic-ffn"
        elif args.ffn_type == "tree-logic":
            model_type = "tree-logic-ffn"

    if model_type in {"baseline-mlp", "baseline-attn-only"}:
        attention_only = model_type == "baseline-attn-only"
        model = vit_tiny_baseline(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=4.0,
            drop_path_rate=args.drop_path_rate,
            attention_only=attention_only,
        )
    elif model_type in {"logic-ffn", "tree-logic-ffn"}:
        ffn_type = "logic" if model_type == "logic-ffn" else "tree-logic"
        model = logic_vit_tiny(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            drop_path_rate=args.drop_path_rate,
            logic_ffn_layers=args.logic_ffn_layers,
            logic_hidden_multiplier=getattr(args, "logic_mlp_ratio", 1.0),
            logic_grad_factor=args.grad_factor,
            logic_resconnection_init=args.logic_resconnection_init,
            logic_weight_init=args.logic_weight_init,
            logic_weight_init_sigma=args.logic_weight_init_sigma,
            logic_res_connect_fraction=args.logic_res_connect_fraction,
            logic_shift_init=args.logic_shift_init,
            logic_shift_init_type=args.logic_shift_init_type,
            logic_shift_init_shift=args.logic_shift_init_shift,
            logic_shift_init_direction=args.logic_shift_init_direction,
            logic_connections=args.logic_connections,
            logic_n_thresholds=args.logic_n_thresholds,
            logic_use_thermometer=args.logic_use_thermometer,
            logic_encoding_temperature=args.logic_encoding_temperature,
            logic_act_fn=args.logic_act_fn,
            ffn_type=ffn_type,
            num_forest_layers=args.num_forest_layers,
            logic_connectivity=args.logic_connectivity,
            learnable_conn_k=args.learnable_conn_k,
            learnable_conn_use_skip_bias=args.learnable_conn_use_skip_bias,
            cross_block_logic_history=getattr(args, "cross_block_logic_history", False),
            cross_block_candidate_frac=getattr(args, "cross_block_candidate_frac", 0.0),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def get_temperature(step: int, total_steps: int, args) -> float:
    """
    根据训练步数计算当前温度值（支持预热、退火和冷却三个阶段）。

    温度退火策略：
    - 预热阶段：保持初始温度
    - 退火阶段：从初始温度指数衰减到最终温度
    - 冷却阶段：保持最终温度

    Args:
        step: 当前训练步数
        total_steps: 总训练步数
        args: 命令行参数对象（包含温度相关参数）

    Returns:
        float: 当前温度值
    """
    temp_start = max(float(args.temp_start), 1e-3)
    temp_end = max(float(args.temp_end), 1e-3)
    warmup_steps = int(total_steps * max(args.temp_warmup_ratio, 0.0))
    cooldown_steps = int(total_steps * max(args.temp_cooldown_ratio, 0.0))
    anneal_steps = max(total_steps - warmup_steps - cooldown_steps, 1)

    if step < warmup_steps:
        return temp_start
    if step >= total_steps - cooldown_steps:
        return temp_end

    progress = (step - warmup_steps) / anneal_steps
    ratio = (temp_end / temp_start) ** progress
    return temp_start * ratio


def get_connection_schedule(step: int, total_steps: int, args) -> tuple[float, float, bool]:
    progress = float(step) / max(float(total_steps), 1.0)
    if progress < 0.05:
        tau_conn = 2.0
        freeze = True
    elif progress < 0.80:
        local = (progress - 0.05) / 0.75
        tau_conn = 2.0 + (0.3 - 2.0) * local
        freeze = False
    else:
        local = (progress - 0.80) / 0.20
        tau_conn = 0.3 + (0.1 - 0.3) * local
        freeze = False
    beta_skip = 0.5 if getattr(args, "learnable_conn_use_skip_bias", True) else 0.0
    return float(tau_conn), float(beta_skip), bool(freeze)


def set_connection_schedule(model: nn.Module, tau_conn: float, beta_skip: float, freeze: bool) -> None:
    for module in model.modules():
        if hasattr(module, "set_connection_schedule"):
            module.set_connection_schedule(tau_conn=tau_conn, beta_skip=beta_skip, freeze=freeze)


def discretize_learnable_connections(model: nn.Module, beta_skip: float | None = None) -> None:
    for module in model.modules():
        if hasattr(module, "discretize_connections"):
            module.discretize_connections(beta_skip=beta_skip)


def connection_stats(model: nn.Module) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    count = 0
    for module in model.modules():
        if hasattr(module, "connection_stats"):
            stats = module.connection_stats()
            if not stats:
                continue
            count += 1
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value)
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    conn_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "conn_logits" in name:
            conn_params.append(param)
        else:
            other_params.append(param)
    groups = [{"params": other_params, "lr": args.learning_rate, "weight_decay": args.weight_decay}]
    if conn_params:
        groups.append(
            {
                "params": conn_params,
                "lr": args.learning_rate * float(args.logic_conn_lr_multiplier),
                "weight_decay": args.weight_decay,
            }
        )
    return torch.optim.AdamW(groups)


def configure_gate_only_trainable(model: nn.Module) -> int:
    for param in model.parameters():
        param.requires_grad_(False)
    trainable = 0
    for module in model.modules():
        weights = getattr(module, "weights", None)
        if isinstance(weights, nn.Parameter):
            weights.requires_grad_(True)
            trainable += int(weights.numel())
    return trainable


def finetune_fixed_connections(
    model: nn.Module,
    train_loader,
    transform,
    args,
    criterion: nn.Module,
    append_log,
) -> None:
    iters = int(getattr(args, "post_discretize_finetune_iters", 0))
    if iters <= 0:
        return
    _tau_conn, beta_skip, _freeze = get_connection_schedule(args.num_iterations, args.num_iterations, args)
    discretize_learnable_connections(model, beta_skip=beta_skip)
    trainable = configure_gate_only_trainable(model)
    lr = float(args.post_discretize_finetune_lr) if args.post_discretize_finetune_lr > 0 else float(args.learning_rate)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        append_log("Post-discretize finetune skipped: no gate parameters are trainable.")
        return
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    iterator = iter(train_loader)
    set_model_temperature(model, args.temp_end)
    set_connection_schedule(model, tau_conn=0.1, beta_skip=beta_skip, freeze=True)
    model.train()
    msg = f"Post-discretize gate-only finetune: iters={iters}, trainable_params={trainable}, lr={lr}"
    print(msg)
    append_log(msg)
    for step in range(iters):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, targets = next(iterator)
        images = images.to(DEVICE, non_blocking=True)
        images = transform(images)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        reset_fixed_weights(model)
        if (step == 0 or step == iters - 1 or (step + 1) % max(1, iters // 5) == 0) and not args.no_logging:
            append_log(f"[PostDiscFT {step + 1}/{iters}] loss={loss.item():.6f}")
    for param in model.parameters():
        param.requires_grad_(True)


def train():
    """
    主训练函数。

    执行完整的训练流程：
    1. 解析命令行参数并设置随机种子
    2. 初始化日志文件
    3. 加载数据集和构建模型
    4. 设置优化器和损失函数
    5. 执行训练循环（包括温度退火）
    6. 定期评估模型并保存最佳状态
    7. 最终评估并输出结果
    """
    # 1. 解析命令行参数，并设置随机种子（保证实验可复现）
    args = parse_args()
    d = getattr(args, "logic_shift_init_direction", "0101")
    if len(d) != 4 or not all(c in "01" for c in d):
        raise ValueError("--logic-shift-init-direction must be a 4-char string of 0s and 1s")
    set_deterministic(args.seed)

    # 2. 准备日志目录与日志文件路径（每次运行一个子文件夹：logs/<timestamp>/）
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_txt_path = run_dir / f"training_{timestamp}.log"
    log_csv_path = run_dir / f"training_{timestamp}.csv"
    checkpoint_dir: Optional[Path] = None
    if getattr(args, "save_checkpoints", False):
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_headers = [
        "phase",
        "step",
        "train_loss",
        "train_acc_percent",
        "valid_acc_eval_percent",
        "valid_acc_train_percent",
        "test_acc_eval_percent",
        "test_acc_train_percent",
        "fulltrain_acc_eval_percent",
        "fulltrain_acc_train_percent",
        "temperature",
        "tau_conn",
        "beta_skip",
        "conn_entropy",
        "conn_avg_skip_distance",
        "conn_pct_l1",
        "conn_pct_l2",
        "conn_pct_older",
        "conn_pct_input",
        "conn_pct_cross_block",
        "conn_dead_gate_ratio",
        "conn_effective_depth",
        "conn_fanout_max",
        "conn_fanout_p95",
    ]

    # 如未关闭日志，则写入日志头信息与 CSV 表头
    if not args.no_logging:
        with open(log_txt_path, "w", encoding="utf-8") as f:
            f.write("LogicViTTiny Training Log\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str))
            f.write("\n")
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()

    def append_log(message: str):
        if args.no_logging:
            return
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def append_csv_row(row_data: Dict[str, object]):
        if args.no_logging:
            return
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(row_data)

    def fmt_pct(value: float) -> str:
        if value is None or value < 0:
            return ""
        return f"{value * 100:.4f}"

    # 3. 加载数据集（train/valid/test/train_eval）与数据增强；构建模型并放到指定设备
    train_loader, valid_loader, test_loader, train_eval_loader, _, transform = load_dataset(args)
    model = build_model(args).to(DEVICE)
    # 初始化逻辑层温度（温度退火的起点）
    set_model_temperature(model, args.temp_start)
    tau_conn0, beta_skip0, freeze_conn0 = get_connection_schedule(0, args.num_iterations, args)
    set_connection_schedule(model, tau_conn0, beta_skip0, freeze_conn0)
    if checkpoint_dir is not None:
        save_checkpoint(model, checkpoint_dir / "checkpoint_init.pt", step=0, temperature=args.temp_start)

    # 4. 构建优化器与损失函数（支持 label smoothing）
    optimizer = build_optimizer(model, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0)

    # 5. 初始化用于追踪最优验证/测试/训练精度的指标字典
    metrics: Dict[str, float] = {
        "valid/acc_eval_best": 0.0,
        "valid/acc_train_best": 0.0,
        "test/acc_eval_best": 0.0,
        "test/acc_train_best": 0.0,
        "fulltrain/acc_eval_best": 0.0,
        "fulltrain/acc_train_best": 0.0,
    }
    best_state = None
    best_step = -1
    best_temp_record = args.temp_start
    best_tau_conn = tau_conn0
    best_beta_skip = beta_skip0
    best_freeze_conn = freeze_conn0
    best_valid_acc = -float("inf")

    # 如需要，先对“未训练模型”在验证/测试集上做一次初始评估，作为基线参考
    if args.eval_initial and not args.no_logging:
        eval_metrics = run_eval(args, model, valid_loader, test_loader, DEVICE, transform, metrics, eval_test=False)
        msg = f"[Eval@start] {eval_metrics}"
        print(msg)
        append_log(msg)
        append_csv_row(
            {
                "phase": "eval_start",
                "step": 0,
                "valid_acc_eval_percent": fmt_pct(eval_metrics["valid/acc_eval"]),
                "valid_acc_train_percent": fmt_pct(eval_metrics["valid/acc_train"]),
                "test_acc_eval_percent": fmt_pct(eval_metrics.get("test/acc_eval", -1.0)),
                "test_acc_train_percent": fmt_pct(eval_metrics.get("test/acc_train", -1.0)),
            }
        )

    # 6. 创建训练数据迭代器，并清空一次梯度
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    # 7. 计算总训练步数
    total_steps = args.num_iterations
    checkpoint_progress_ratios: list[float] = []
    checkpoint_target_steps: list[int] = []
    checkpoint_saved_ratio_indices: set[int] = set()
    if checkpoint_dir is not None and getattr(args, "checkpoint_progress_ratios", ""):
        raw = [x.strip() for x in args.checkpoint_progress_ratios.split(",") if x.strip()]
        for s in raw:
            r = float(s)
            if 0 < r <= 1:
                checkpoint_progress_ratios.append(r)
        checkpoint_progress_ratios = sorted(set(checkpoint_progress_ratios))
        # 100% 对应最后一步 total_steps-1（循环为 range(total_steps)），其余为 int(r * total_steps)
        checkpoint_target_steps = [
            total_steps - 1 if r >= 1.0 else int(r * total_steps)
            for r in checkpoint_progress_ratios
        ]
    # tqdm 进度条，显示训练过程
    progress = tqdm(range(total_steps), desc="train", disable=args.no_logging)
    for step in progress:
        # 7.1 根据当前 step 更新温度，并应用到所有逻辑层
        current_temp = get_temperature(step, args.num_iterations, args)
        set_model_temperature(model, current_temp)
        tau_conn, beta_skip, freeze_conn = get_connection_schedule(step, args.num_iterations, args)
        set_connection_schedule(model, tau_conn, beta_skip, freeze_conn)
        # 切换到训练模式（启用 Dropout 等）
        model.train()
        loss_accum = 0.0
        acc_accum = 0.0

        # 7.3 梯度累积：一个“大步”内可以包含多个 micro-batch
        for micro in range(args.batches_per_backward):
            try:
                images, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, targets = next(train_iter)

            images = images.to(DEVICE, non_blocking=True)
            images = transform(images)
            targets = targets.to(DEVICE, non_blocking=True)

            # 可选 CutMix / Mixup（仅训练阶段，评估时不使用；CutMix 与 Mixup 独立）
            use_cutmix = args.cutmix and (random.random() < args.cutmix_prob)
            use_mixup = (not use_cutmix) and args.mixup and (random.random() < args.mixup_prob)
            if use_cutmix:
                images, targets_a, targets_b, lam = cutmix_batch(images, targets, args.cutmix_alpha)
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
                acc_batch = (outputs.argmax(-1) == targets_a).float().mean().item()
            elif use_mixup:
                images, targets_a, targets_b, lam = mixup_batch(images, targets, args.mixup_alpha)
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
                acc_batch = (outputs.argmax(-1) == targets_a).float().mean().item()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                acc_batch = (outputs.argmax(-1) == targets).float().mean().item()

            (loss / args.batches_per_backward).backward()
            loss_accum += loss.item()
            acc_accum += acc_batch

            # 在累积到指定 micro-batch 数量后，执行一次参数更新，并重置逻辑层的固定权重
            if (micro + 1) == args.batches_per_backward:
                optimizer.step()
                reset_fixed_weights(model)

        # 清空梯度，为下一步训练做准备，并对多个 micro-batch 的损失/精度取平均
        optimizer.zero_grad(set_to_none=True)
        loss_accum /= args.batches_per_backward
        acc_accum /= args.batches_per_backward

        # 7.4 训练阶段的日志打印与 CSV 记录（按 print_freq 间隔触发）
        if not args.no_logging and (step % args.print_freq == 0 or step == args.num_iterations - 1):
            conn_metrics = connection_stats(model)
            msg = (
                f"[Iter {step:06d}] loss={loss_accum:.4f} train_acc={acc_accum * 100:.2f}% "
                f"temp={current_temp:.3f} tau_conn={tau_conn:.3f} beta_skip={beta_skip:.3f}"
            )
            if conn_metrics:
                msg += (
                    f" conn_entropy={conn_metrics.get('conn_entropy', 0.0):.3f} "
                    f"avg_skip={conn_metrics.get('conn_avg_skip_distance', 0.0):.2f}"
                )
            progress.write(msg)
            append_log(msg)
            row = {
                "phase": "train",
                "step": step,
                "train_loss": f"{loss_accum:.6f}",
                "train_acc_percent": f"{acc_accum * 100:.4f}",
                "temperature": f"{current_temp:.4f}",
                "tau_conn": f"{tau_conn:.4f}",
                "beta_skip": f"{beta_skip:.4f}",
            }
            row.update({key: f"{value:.6f}" for key, value in conn_metrics.items()})
            append_csv_row(row)

        # 7.5 按 eval_freq 在验证集上评估（同时记录软/硬两种精度），并更新最佳模型
        should_eval = (step % args.eval_freq == 0) and (step != 0 or args.eval_initial)
        if not args.no_logging and should_eval:
            eval_metrics = run_eval(args, model, valid_loader, test_loader, DEVICE, transform, metrics, eval_test=False)
            val_acc = eval_metrics["valid/acc_eval"]
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step
                best_temp_record = current_temp
                best_tau_conn = tau_conn
                best_beta_skip = beta_skip
                best_freeze_conn = freeze_conn
            msg = f"[Eval@{step:06d}] {eval_metrics}"
            progress.write(msg)
            append_log(msg)
            append_csv_row(
                {
                    "phase": "eval",
                    "step": step,
                    "valid_acc_eval_percent": fmt_pct(eval_metrics["valid/acc_eval"]),
                    "valid_acc_train_percent": fmt_pct(eval_metrics["valid/acc_train"]),
                }
            )

        # 7.6 按 ext_eval_freq 在 fulltrain（全训练集、无增强）上做评估，用于监控是否过拟合
        should_ext_eval = (step % args.ext_eval_freq == 0) and (step != 0 or args.eval_initial)
        if not args.no_logging and should_ext_eval:
            fulltrain_metrics = evaluate_fulltrain(args, model, train_eval_loader, DEVICE, transform, metrics)
            msg = f"[FulltrainEval@{step:06d}] {fulltrain_metrics}"
            progress.write(msg)
            append_log(msg)
            append_csv_row(
                {
                    "phase": "fulltrain",
                    "step": step,
                    "fulltrain_acc_eval_percent": fmt_pct(fulltrain_metrics["fulltrain/acc_eval"]),
                    "fulltrain_acc_train_percent": fmt_pct(fulltrain_metrics["fulltrain/acc_train"]),
                }
            )

        # 7.7 按进度比例保存 checkpoint（每比例只保存一次）
        if checkpoint_dir is not None and checkpoint_target_steps:
            for i in range(len(checkpoint_target_steps)):
                if step >= checkpoint_target_steps[i] and i not in checkpoint_saved_ratio_indices:
                    pct = int(checkpoint_progress_ratios[i] * 100)
                    save_checkpoint(
                        model,
                        checkpoint_dir / f"checkpoint_step_{pct}pct.pt",
                        step=step,
                        temperature=current_temp,
                    )
                    checkpoint_saved_ratio_indices.add(i)

    # 8. 训练完成后，如果有最佳模型状态，则加载该状态与对应的温度
    if best_state is not None:
        model.load_state_dict(best_state)
        set_model_temperature(model, best_temp_record)
        set_connection_schedule(model, best_tau_conn, best_beta_skip, best_freeze_conn)
        final_step = best_step
    else:
        final_step = args.num_iterations - 1
        best_tau_conn, best_beta_skip, best_freeze_conn = get_connection_schedule(final_step, args.num_iterations, args)
        set_connection_schedule(model, best_tau_conn, best_beta_skip, best_freeze_conn)

    if getattr(args, "logic_connectivity", "fixed") == "learnable":
        finetune_fixed_connections(model, train_loader, transform, args, criterion, append_log)

    # 9. 最终评估：先报验证集 best（未硬化），再显式 Harden model，然后做 Harden fulltrain + 测试集，最后 Best metrics
    if not args.no_logging:
        if getattr(args, "save_checkpoints", False) and best_state is not None:
            save_checkpoint(
                model,
                checkpoint_dir / "checkpoint_best_before_harden.pt",
                step=final_step,
                temperature=best_temp_record,
            )
        # 9.1 Final evaluation：只报告验证集 best（未硬化）
        eval_unhardened = run_eval(args, model, valid_loader, test_loader, DEVICE, transform, metrics, eval_test=False)
        msg_final = f"Final evaluation (best step {final_step}): valid/acc_eval={eval_unhardened['valid/acc_eval']:.4f}, valid/acc_train={eval_unhardened['valid/acc_train']:.4f}"
        print(msg_final)
        append_log(msg_final)
        append_csv_row(
            {
                "phase": "final_eval",
                "step": args.num_iterations,
                "valid_acc_eval_percent": fmt_pct(eval_unhardened["valid/acc_eval"]),
                "valid_acc_train_percent": fmt_pct(eval_unhardened["valid/acc_train"]),
            }
        )
        if getattr(args, "logic_connectivity", "fixed") == "learnable":
            disc_model = copy.deepcopy(model).to(DEVICE)
            discretize_learnable_connections(disc_model, beta_skip=best_beta_skip)
            disc_eval = run_eval(args, disc_model, valid_loader, test_loader, DEVICE, transform, metrics, eval_test=False)
            disc_stats = connection_stats(disc_model)
            msg_disc = (
                f"Final discretized-connection evaluation: "
                f"valid/acc_eval={disc_eval['valid/acc_eval']:.4f}, "
                f"valid/acc_train={disc_eval['valid/acc_train']:.4f}, stats={disc_stats}"
            )
            print(msg_disc)
            append_log(msg_disc)
            row = {
                "phase": "final_disc_conn_eval",
                "step": args.num_iterations,
                "valid_acc_eval_percent": fmt_pct(disc_eval["valid/acc_eval"]),
                "valid_acc_train_percent": fmt_pct(disc_eval["valid/acc_train"]),
            }
            row.update({key: f"{value:.6f}" for key, value in disc_stats.items()})
            append_csv_row(row)
        # 9.2 显式打印 Harden model（显式传入 best_temp_record，与 checkpoint 中保存的 temperature 一致）
        msg_harden = "Harden model"
        print(msg_harden)
        append_log(msg_harden)
        harden_model(model, temperature=best_temp_record)
        conn_after_harden = connection_stats(model)
        if conn_after_harden:
            msg_conn = f"Connection stats after discretization: {conn_after_harden}"
            print(msg_conn)
            append_log(msg_conn)
        if getattr(args, "save_checkpoints", False) and best_state is not None:
            save_checkpoint(
                model,
                checkpoint_dir / "checkpoint_best_after_harden.pt",
                step=final_step,
                temperature=best_temp_record,
            )
        # 9.3 Harden fulltrain evaluation + 测试集
        harden_fulltrain_metrics = evaluate_fulltrain(args, model, train_eval_loader, DEVICE, transform, metrics)
        final_eval = run_eval(args, model, valid_loader, test_loader, DEVICE, transform, metrics, eval_test=True)
        msg_fulltrain = f"Harden fulltrain evaluation: fulltrain/acc_train={harden_fulltrain_metrics['fulltrain/acc_train']:.4f}, fulltrain/acc_eval={harden_fulltrain_metrics['fulltrain/acc_eval']:.4f}"
        msg_test = f"Test set: test/acc_eval={final_eval['test/acc_eval']:.4f}, test/acc_train={final_eval['test/acc_train']:.4f}"
        print(msg_fulltrain)
        print(msg_test)
        append_log(msg_fulltrain)
        append_log(msg_test)
        append_csv_row(
            {
                "phase": "final_fulltrain",
                "step": args.num_iterations,
                "fulltrain_acc_eval_percent": fmt_pct(harden_fulltrain_metrics["fulltrain/acc_eval"]),
                "fulltrain_acc_train_percent": fmt_pct(harden_fulltrain_metrics["fulltrain/acc_train"]),
            }
        )
        append_csv_row(
            {
                "phase": "final_test",
                "step": args.num_iterations,
                "test_acc_eval_percent": fmt_pct(final_eval["test/acc_eval"]),
                "test_acc_train_percent": fmt_pct(final_eval["test/acc_train"]),
            }
        )
        # 9.4 Best metrics
        msg_best = f"Best metrics: { {k: v for k, v in metrics.items() if k.endswith('_best')} }"
        print(msg_best)
        append_log(msg_best)


if __name__ == "__main__":
    train()
