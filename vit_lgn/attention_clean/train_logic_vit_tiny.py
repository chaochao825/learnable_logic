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
import math
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
from torch.nn.parameter import UninitializedParameter
from tqdm import tqdm

from data_pipeline import (
    class_count_of_dataset,
    load_dataset,
    num_channels_of_dataset,
)
from vit_tiny_attention_logic import vit_tiny as logic_vit_tiny

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
    parser = argparse.ArgumentParser(description="Train logic ViT with AttentionLogic transformer blocks.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["cifar-10", "cifar-100"], default="cifar-10")
    parser.add_argument(
        "--data-encoding",
        choices=["real-input", "1-thresholds", "3-thresholds", "7-thresholds", "15-thresholds", "23-thresholds", "31-thresholds"],
        default="real-input",
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="数据增强")
    parser.add_argument("--preprocess-once", action=argparse.BooleanOptionalAction, default=True, help="预处理一次，默认开启")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches-per-backward", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-4, help="Cosine annealing minimum learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio of total iterations in [0, 1)")
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "warmup-cosine", "warmup-stay-cosine"],
        default="constant",
        help="Learning rate schedule type: constant keeps --learning-rate unchanged",
    )
    parser.add_argument(
        "--stay-ratio",
        type=float,
        default=0.5,
        help="For warmup-stay-cosine: ratio where cosine decay starts (in [warmup-ratio, 1])",
    )
    parser.add_argument(
        "--last-learning-rate",
        type=float,
        default=None,
        help="For warmup-stay-cosine: final learning rate of cosine decay (default: min-learning-rate)",
    )
    parser.add_argument("--num-iterations", type=int, default=5_0001)
    parser.add_argument("--valid-set-size", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=1_000)
    parser.add_argument("--ext-eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-initial", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-logging", action="store_true")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-progress-ratios",
        type=str,
        default="0.25,0.5,0.75,1",
        help="Comma-separated progress ratios (0-1] for checkpoint saving.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=100)

    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attention-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--attention-k", type=int, default=15, help="Top-k row selection size in logic attention")
    parser.add_argument(
        "--topk-impl",
        choices=["winner-tree", "torch-topk"],
        default="winner-tree",
        help="Top-k implementation in attention logic (ties may pick different indices)",
    )
    parser.add_argument(
        "--topk-forward-mode",
        choices=["topk", "random-k", "fixed-random-k"],
        default="topk",
        help="Forward selector mode in attention logic: score-based top-k, dynamic random-k, or fixed random-k per block/query-row",
    )
    parser.add_argument(
        "--topk-surrogate-mode",
        choices=["kth", "random-kth", "soft-rank", "sigmoid-topk", "subset-gibbs"],
        default="kth",
        help="Backward surrogate used by top-k selection",
    )
    parser.add_argument(
        "--topk-surrogate-proxy-source",
        choices=["encoded", "soft-thermometer"],
        default="encoded",
        help="Proxy score source used by kth top-k backward surrogate",
    )
    parser.add_argument(
        "--topk-kth-detach-value",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether kth surrogate detaches the kth threshold value",
    )
    parser.add_argument(
        "--topk-kth-use-midpoint-theta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether kth surrogate uses the arithmetic mean of the kth and (k+1)th scores as theta",
    )
    parser.add_argument(
        "--topk-kth-normalize-soft-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether kth surrogate normalizes soft_mask so that its sum is approximately k",
    )
    parser.add_argument(
        "--fast-vote",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SelectedColumnMajorityCudaMatmul for hard majority forward",
    )
    parser.add_argument("--validate-input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-thermometer-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-thresholds", type=int, default=3)
    parser.add_argument(
        "--v-n-thresholds",
        type=int,
        default=None,
        help="Override threshold count used only for V thermometer encoding/decoding (default: same as --n-thresholds)",
    )
    parser.add_argument("--encoding-scale", type=float, default=10.0)
    parser.add_argument("--apply-sigmoid-before-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--thermometer-decode-use-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether thermometer decode uses learnable weights (disable for fixed unweighted decode)",
    )
    parser.add_argument(
        "--boundary-surrogate-temp-max",
        type=float,
        default=1.0,
        help="Top-k kth surrogate temperature max value at beginning",
    )
    parser.add_argument(
        "--boundary-surrogate-temp-min",
        type=float,
        default=1.0,
        help="Top-k kth surrogate temperature min value at end",
    )
    parser.add_argument(
        "--boundary-surrogate-temp-start-ratio",
        type=float,
        default=0.1,
        help="Keep boundary-surrogate max temperature until this training progress ratio",
    )
    parser.add_argument(
        "--boundary-surrogate-temp-end-ratio",
        type=float,
        default=0.8,
        help="Reach boundary-surrogate min temperature by this training progress ratio and keep constant afterwards",
    )
    parser.add_argument(
        "--majority-train-temperature",
        type=float,
        default=0.125,
        help="Temperature for SelectorMaskSelectedColumnMajority soft-train branch",
    )
    parser.add_argument(
        "--majority-surrogate-mode",
        choices=["count", "fraction"],
        default="count",
        help="Backward surrogate space for selector majority: raw count or normalized selected fraction",
    )
    parser.add_argument(
        "--majority-k-root-degree",
        type=int,
        default=2,
        help="Use k^(1/n) in the majority fraction surrogate denominator scaling; e.g. 2=sqrt(k), 3=cuberoot(k)",
    )
    parser.add_argument(
        "--majority-train-temp-max",
        type=float,
        default=0.5,
        help="Majority soft temperature max value at beginning (default: --majority-train-temperature)",
    )
    parser.add_argument(
        "--majority-train-temp-min",
        type=float,
        default=0.1,
        help="Majority soft temperature min value at end (default: --majority-train-temperature)",
    )
    parser.add_argument(
        "--majority-train-temp-start-ratio",
        type=float,
        default=0.1,
        help="Keep majority max temperature until this training progress ratio",
    )
    parser.add_argument(
        "--majority-train-temp-end-ratio",
        type=float,
        default=0.8,
        help="Reach majority min temperature by this training progress ratio and keep constant afterwards",
    )

    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing coefficient")
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, default=False, help="Enable mixup augmentation")
    parser.add_argument("--mixup-alpha", type=float, default=0.5, help="Beta distribution alpha for mixup")
    parser.add_argument("--mixup-prob", type=float, default=1.0, help="Probability to apply mixup per batch")
    parser.add_argument("--cutmix", action=argparse.BooleanOptionalAction, default=False, help="Enable CutMix augmentation")
    parser.add_argument("--cutmix-prob", type=float, default=0.5, help="Probability to apply CutMix per batch")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0, help="Beta distribution alpha for CutMix")

    args = parser.parse_args()

    if args.majority_train_temp_max is None:
        args.majority_train_temp_max = args.majority_train_temperature
    if args.majority_train_temp_min is None:
        args.majority_train_temp_min = args.majority_train_temperature

    if args.topk_surrogate_proxy_source == "soft-thermometer" and not args.use_thermometer_encoding:
        raise ValueError("--topk-surrogate-proxy-source soft-thermometer requires --use-thermometer-encoding")

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
    state_dict = {}
    for key, value in model.state_dict().items():
        if isinstance(value, UninitializedParameter):
            continue
        state_dict[key] = value.detach().cpu().clone()

    state = {
        "state_dict": state_dict,
        "step": step,
        "temperature": temperature,
    }
    torch.save(state, path)


def initialize_lazy_modules(
    model: nn.Module,
    loader,
    device: torch.device,
    transform,
) -> None:
    """用一个真实 batch 跑一次前向，确保 LazyModule 参数完成初始化。"""
    for parameter in model.parameters():
        if isinstance(parameter, UninitializedParameter):
            break
    else:
        return

    images, _ = next(iter(loader))
    images = images.to(device, non_blocking=True)
    images = transform(images)

    original_mode = model.training
    with torch.no_grad():
        model.eval()
        model(images)
    model.train(original_mode)


def eval_on_loader(
    model: nn.Module,
    loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    train_mode: bool,
    transform,
    use_inference_mode: bool = False,
    desc: Optional[str] = None,
    show_progress: bool = False,
) -> float:
    """
    在数据加载器上评估模型。

    Args:
        model: PyTorch 模型
        loader: 数据加载器（如果为 None，返回 -1.0）
        device: 计算设备
        train_mode: 是否使用训练模式（影响 dropout 和 batch norm）
        transform: 数据变换函数

    Returns:
        float: 模型在数据加载器上的准确率（如果 loader 为 None，返回 -1.0）
    """
    if loader is None:
        return -1.0
    orig_mode = model.training
    grad_context = torch.inference_mode if use_inference_mode else torch.no_grad
    with grad_context():
        model.train(mode=train_mode)
        # packed attention 通过 model.train(mode=...) 在训练/推理前向之间切换。
        # use_inference_mode=True 时，前向 additionally 使用 inference_mode。
        with logic_eval_mode(model, train_mode=train_mode):
            iterator = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
            per_batch = []
            for images, targets in iterator:
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
    include_inference_eval: bool = False,
    include_inference_test: bool = False,
    show_progress: bool = False,
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
        use_inference_mode=False,
        desc="valid/eval",
        show_progress=show_progress,
    )
    if include_inference_eval:
        result["valid/acc_inference"] = eval_on_loader(
            model,
            valid_loader,
            device,
            train_mode=False,
            transform=transform,
            use_inference_mode=True,
            desc="valid/inference",
            show_progress=show_progress,
        )
    result["valid/acc_train"] = eval_on_loader(
        model,
        valid_loader,
        device,
        train_mode=True,
        transform=transform,
        use_inference_mode=False,
        desc="valid/train",
        show_progress=show_progress,
    )

    if eval_test:
        result["test/acc_eval"] = eval_on_loader(
            model,
            test_loader,
            device,
            train_mode=False,
            transform=transform,
            use_inference_mode=False,
            desc="test/eval",
            show_progress=show_progress,
        )
        if include_inference_test:
            result["test/acc_inference"] = eval_on_loader(
                model,
                test_loader,
                device,
                train_mode=False,
                transform=transform,
                use_inference_mode=True,
                desc="test/inference",
                show_progress=show_progress,
            )
        result["test/acc_train"] = eval_on_loader(
            model,
            test_loader,
            device,
            train_mode=True,
            transform=transform,
            use_inference_mode=False,
            desc="test/train",
            show_progress=show_progress,
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
        use_inference_mode=False,
        desc="fulltrain/eval",
        show_progress=not args.no_logging,
    )
    train_train = eval_on_loader(
        model,
        train_eval_loader,
        device,
        train_mode=True,
        transform=transform,
        use_inference_mode=False,
        desc="fulltrain/train",
        show_progress=not args.no_logging,
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

    return logic_vit_tiny(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        attention_only=args.attention_only,
        attention_k=args.attention_k,
        topk_impl=args.topk_impl,
        topk_forward_mode=args.topk_forward_mode,
        topk_surrogate_mode=args.topk_surrogate_mode,
        topk_surrogate_proxy_source=args.topk_surrogate_proxy_source,
        topk_kth_detach_value=args.topk_kth_detach_value,
        topk_kth_use_midpoint_theta=args.topk_kth_use_midpoint_theta,
        topk_kth_normalize_soft_mask=args.topk_kth_normalize_soft_mask,
        validate_input=args.validate_input,
        use_thermometer_encoding=args.use_thermometer_encoding,
        n_thresholds=args.n_thresholds,
        v_n_thresholds=args.v_n_thresholds,
        encoding_scale=args.encoding_scale,
        apply_sigmoid_before_encoding=args.apply_sigmoid_before_encoding,
        decode_output=args.decode_output,
        thermometer_decode_use_weights=args.thermometer_decode_use_weights,
        boundary_surrogate_temperature=args.boundary_surrogate_temp_max,
        majority_train_temperature=args.majority_train_temp_max,
        fast_vote=args.fast_vote,
        majority_surrogate_mode=args.majority_surrogate_mode,
        majority_k_root_degree=args.majority_k_root_degree,
    )


def _validate_temperature_schedule(
    name: str,
    max_temp: float,
    min_temp: float,
    start_ratio: float,
    end_ratio: float,
) -> None:
    if max_temp <= 0:
        raise ValueError(f"--{name}-max 必须大于 0")
    if min_temp <= 0:
        raise ValueError(f"--{name}-min 必须大于 0")
    if min_temp > max_temp:
        raise ValueError(f"--{name}-min 必须小于等于 --{name}-max")
    if not (0.0 <= start_ratio <= end_ratio <= 1.0):
        raise ValueError(f"--{name}-start-ratio 和 --{name}-end-ratio 必须满足 0 <= start <= end <= 1")


def compute_piecewise_temperature(
    step: int,
    total_steps: int,
    max_temp: float,
    min_temp: float,
    start_ratio: float,
    end_ratio: float,
) -> float:
    if total_steps <= 1:
        return max_temp

    progress = float(step) / float(total_steps - 1)
    if progress <= start_ratio:
        return max_temp
    if progress >= end_ratio:
        return min_temp

    if end_ratio <= start_ratio:
        return min_temp

    decay_progress = (progress - start_ratio) / (end_ratio - start_ratio)
    return max_temp + (min_temp - max_temp) * decay_progress


def set_attention_soft_train_temperatures(
    model: nn.Module,
    boundary_surrogate_temperature: float,
    majority_temperature: float,
) -> None:
    boundary_surrogate_temperature = float(boundary_surrogate_temperature)
    majority_temperature = float(majority_temperature)

    for module in model.modules():
        if hasattr(module, "boundary_surrogate_temperature"):
            module.boundary_surrogate_temperature = boundary_surrogate_temperature
        if hasattr(module, "train_temperature"):
            module.train_temperature = majority_temperature
        if hasattr(module, "majority_train_temperature"):
            module.majority_train_temperature = majority_temperature


def compute_warmup_cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_ratio: float,
) -> float:
    """按 step 计算 warmup + cosine 学习率。"""
    if total_steps <= 0:
        return base_lr

    warmup_ratio = max(0.0, min(warmup_ratio, 0.999999))
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        # Warmup 从 min_lr 线性升到 base_lr，避免起始学习率过小。
        if warmup_steps == 1:
            return base_lr
        warmup_progress = float(step) / float(warmup_steps - 1)
        return min_lr + (base_lr - min_lr) * warmup_progress

    if total_steps - warmup_steps <= 1:
        return min_lr

    cosine_step = step - warmup_steps
    cosine_total = total_steps - warmup_steps - 1
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_step / max(1, cosine_total)))
    return min_lr + (base_lr - min_lr) * cosine


def compute_warmup_stay_cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_ratio: float,
    stay_ratio: float,
    last_lr: float,
) -> float:
    """按 step 计算 warmup + stay + cosine 学习率。"""
    if total_steps <= 0:
        return base_lr

    step = max(0, min(step, total_steps - 1))
    warmup_ratio = max(0.0, min(warmup_ratio, 1.0))
    stay_ratio = max(warmup_ratio, min(stay_ratio, 1.0))

    warmup_steps = int(total_steps * warmup_ratio)
    stay_steps = int(total_steps * stay_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps))
    stay_steps = max(warmup_steps, min(stay_steps, total_steps))

    if warmup_steps > 0 and step < warmup_steps:
        if warmup_steps == 1:
            return base_lr
        warmup_progress = float(step) / float(warmup_steps - 1)
        return min_lr + (base_lr - min_lr) * warmup_progress

    if step < stay_steps:
        return base_lr

    if total_steps - stay_steps <= 1:
        return last_lr

    cosine_step = step - stay_steps
    cosine_total = total_steps - stay_steps - 1
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_step / max(1, cosine_total)))
    return last_lr + (base_lr - last_lr) * cosine


def compute_learning_rate(step: int, total_steps: int, args: argparse.Namespace) -> float:
    """按命令行设置选择学习率调度策略。"""
    if args.lr_schedule == "constant":
        return args.learning_rate

    if args.lr_schedule == "warmup-stay-cosine":
        last_lr = args.last_learning_rate if args.last_learning_rate is not None else args.min_learning_rate
        return compute_warmup_stay_cosine_lr(
            step=step,
            total_steps=total_steps,
            base_lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            warmup_ratio=args.warmup_ratio,
            stay_ratio=args.stay_ratio,
            last_lr=last_lr,
        )

    return compute_warmup_cosine_lr(
        step=step,
        total_steps=total_steps,
        base_lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_ratio=args.warmup_ratio,
    )


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
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio 必须在 [0, 1] 区间内")
    if args.lr_schedule == "warmup-stay-cosine":
        if not args.warmup_ratio <= args.stay_ratio <= 1.0:
            raise ValueError("在 --lr-schedule warmup-stay-cosine 下，--stay-ratio 必须满足 warmup-ratio <= stay-ratio <= 1")
    if args.majority_train_temperature <= 0:
        raise ValueError("--majority-train-temperature 必须大于 0")
    if args.v_n_thresholds is not None and args.v_n_thresholds <= 0:
        raise ValueError("--v-n-thresholds 必须大于 0")
    _validate_temperature_schedule(
        name="boundary-surrogate-temp",
        max_temp=args.boundary_surrogate_temp_max,
        min_temp=args.boundary_surrogate_temp_min,
        start_ratio=args.boundary_surrogate_temp_start_ratio,
        end_ratio=args.boundary_surrogate_temp_end_ratio,
    )
    _validate_temperature_schedule(
        name="majority-train-temp",
        max_temp=args.majority_train_temp_max,
        min_temp=args.majority_train_temp_min,
        start_ratio=args.majority_train_temp_start_ratio,
        end_ratio=args.majority_train_temp_end_ratio,
    )
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
        "learning_rate",
        "boundary_surrogate_temperature",
        "majority_train_temperature",
        "train_loss",
        "train_acc_percent",
        "valid_acc_eval_percent",
        "valid_acc_inference_percent",
        "valid_acc_train_percent",
        "test_acc_eval_percent",
        "test_acc_inference_percent",
        "test_acc_train_percent",
        "fulltrain_acc_eval_percent",
        "fulltrain_acc_train_percent",
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
    set_attention_soft_train_temperatures(
        model,
        boundary_surrogate_temperature=args.boundary_surrogate_temp_max,
        majority_temperature=args.majority_train_temp_max,
    )
    initialize_lazy_modules(model, train_loader, DEVICE, transform)
    if checkpoint_dir is not None:
        save_checkpoint(model, checkpoint_dir / "checkpoint_init.pt", step=0, temperature=1.0)

    # 4. 构建优化器与损失函数（支持 label smoothing）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
    best_valid_acc = -float("inf")

    # 如需要，先对“未训练模型”在验证/测试集上做一次初始评估，作为基线参考
    if args.eval_initial and not args.no_logging:
        eval_metrics = run_eval(
            args,
            model,
            valid_loader,
            test_loader,
            DEVICE,
            transform,
            metrics,
            eval_test=True,
            include_inference_eval=True,
            include_inference_test=True,
            show_progress=True,
        )
        msg = f"[Eval@start] {eval_metrics}"
        print(msg)
        append_log(msg)
        append_csv_row(
            {
                "phase": "eval_start",
                "step": 0,
                "valid_acc_eval_percent": fmt_pct(eval_metrics["valid/acc_eval"]),
                "valid_acc_inference_percent": fmt_pct(eval_metrics.get("valid/acc_inference")),
                "valid_acc_train_percent": fmt_pct(eval_metrics["valid/acc_train"]),
                "test_acc_eval_percent": fmt_pct(eval_metrics["test/acc_eval"]),
                "test_acc_inference_percent": fmt_pct(eval_metrics.get("test/acc_inference")),
                "test_acc_train_percent": fmt_pct(eval_metrics["test/acc_train"]),
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
        current_majority_train_temperature = compute_piecewise_temperature(
            step=step,
            total_steps=total_steps,
            max_temp=args.majority_train_temp_max,
            min_temp=args.majority_train_temp_min,
            start_ratio=args.majority_train_temp_start_ratio,
            end_ratio=args.majority_train_temp_end_ratio,
        )
        current_boundary_surrogate_temperature = compute_piecewise_temperature(
            step=step,
            total_steps=total_steps,
            max_temp=args.boundary_surrogate_temp_max,
            min_temp=args.boundary_surrogate_temp_min,
            start_ratio=args.boundary_surrogate_temp_start_ratio,
            end_ratio=args.boundary_surrogate_temp_end_ratio,
        )
        set_attention_soft_train_temperatures(
            model,
            boundary_surrogate_temperature=current_boundary_surrogate_temperature,
            majority_temperature=current_majority_train_temperature,
        )

        current_lr = compute_learning_rate(step=step, total_steps=total_steps, args=args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

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
            msg = (
                f"[Iter {step:06d}] loss={loss_accum:.4f} train_acc={acc_accum * 100:.2f}% "
                f"lr={current_lr:.6e} "
                f"boundary_t={current_boundary_surrogate_temperature:.6f} "
                f"majority_t={current_majority_train_temperature:.6f}"
            )
            progress.write(msg)
            append_log(msg)
            append_csv_row(
                {
                    "phase": "train",
                    "step": step,
                    "learning_rate": f"{current_lr:.8e}",
                    "boundary_surrogate_temperature": f"{current_boundary_surrogate_temperature:.8f}",
                    "majority_train_temperature": f"{current_majority_train_temperature:.8f}",
                    "train_loss": f"{loss_accum:.6f}",
                    "train_acc_percent": f"{acc_accum * 100:.4f}",
                }
            )

        # 7.5 按 eval_freq 在验证集上评估（同时记录软/硬两种精度），并更新最佳模型
        should_eval = (step % args.eval_freq == 0) and (step != 0 or args.eval_initial)
        if not args.no_logging and should_eval:
            progress.write(f"[Eval@{step:06d}] starting validation on full validation set")
            eval_metrics = run_eval(
                args,
                model,
                valid_loader,
                test_loader,
                DEVICE,
                transform,
                metrics,
                eval_test=False,
                include_inference_eval=True,
                show_progress=True,
            )
            val_acc = eval_metrics["valid/acc_eval"]
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step
            msg = f"[Eval@{step:06d}] {eval_metrics}"
            progress.write(msg)
            append_log(msg)
            append_csv_row(
                {
                    "phase": "eval",
                    "step": step,
                    "valid_acc_eval_percent": fmt_pct(eval_metrics["valid/acc_eval"]),
                    "valid_acc_inference_percent": fmt_pct(eval_metrics.get("valid/acc_inference")),
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
                        temperature=1.0,
                    )
                    checkpoint_saved_ratio_indices.add(i)

    # 8. 训练完成后，如果有最佳模型状态，则加载该状态与对应的温度
    if best_state is not None:
        model.load_state_dict(best_state)
        final_step = best_step
    else:
        final_step = args.num_iterations - 1

    # 9. 最终评估：报告最佳验证结果，并在 fulltrain/test 上给出最终评估
    if not args.no_logging:
        if getattr(args, "save_checkpoints", False) and best_state is not None:
            save_checkpoint(
                model,
                checkpoint_dir / "checkpoint_best.pt",
                step=final_step,
                temperature=1.0,
            )
        eval_unhardened = run_eval(
            args,
            model,
            valid_loader,
            test_loader,
            DEVICE,
            transform,
            metrics,
            eval_test=False,
            include_inference_eval=True,
            show_progress=True,
        )
        msg_final = (
            f"Final evaluation (best step {final_step}): valid/acc_eval={eval_unhardened['valid/acc_eval']:.4f}, "
            f"valid/acc_inference={eval_unhardened['valid/acc_inference']:.4f}, "
            f"valid/acc_train={eval_unhardened['valid/acc_train']:.4f}"
        )
        print(msg_final)
        append_log(msg_final)
        append_csv_row(
            {
                "phase": "final_eval",
                "step": args.num_iterations,
                "valid_acc_eval_percent": fmt_pct(eval_unhardened["valid/acc_eval"]),
                "valid_acc_inference_percent": fmt_pct(eval_unhardened.get("valid/acc_inference")),
                "valid_acc_train_percent": fmt_pct(eval_unhardened["valid/acc_train"]),
            }
        )
        harden_fulltrain_metrics = evaluate_fulltrain(args, model, train_eval_loader, DEVICE, transform, metrics)
        final_eval = run_eval(
            args,
            model,
            valid_loader,
            test_loader,
            DEVICE,
            transform,
            metrics,
            eval_test=True,
            include_inference_eval=True,
            include_inference_test=True,
            show_progress=True,
        )
        msg_fulltrain = f"Final fulltrain evaluation: fulltrain/acc_train={harden_fulltrain_metrics['fulltrain/acc_train']:.4f}, fulltrain/acc_eval={harden_fulltrain_metrics['fulltrain/acc_eval']:.4f}"
        msg_test = (
            f"Test set: test/acc_eval={final_eval['test/acc_eval']:.4f}, "
            f"test/acc_inference={final_eval['test/acc_inference']:.4f}, "
            f"test/acc_train={final_eval['test/acc_train']:.4f}"
        )
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
                "test_acc_inference_percent": fmt_pct(final_eval.get("test/acc_inference")),
                "test_acc_train_percent": fmt_pct(final_eval["test/acc_train"]),
            }
        )
        msg_best = f"Best metrics: { {k: v for k, v in metrics.items() if k.endswith('_best')} }"
        print(msg_best)
        append_log(msg_best)


if __name__ == "__main__":
    train()
