from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from data_pipeline import load_dataset
from train_logic_vit_tiny import (
    build_optimizer,
    build_model,
    configure_gate_only_trainable,
    connection_stats,
    discretize_learnable_connections,
    get_temperature,
    get_connection_schedule,
    harden_model,
    logic_layers,
    reset_fixed_weights,
    set_deterministic,
    set_connection_schedule,
    set_model_temperature,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-budget ViT-LGN learnable-connection experiment."
    )
    parser.add_argument("--dataset", choices=["cifar-10", "cifar-100"], default="cifar-10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-encoding", default="real-input")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--valid-set-size", type=float, default=0.1)

    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--ffn-type", choices=["logic", "tree-logic"], default="logic")
    parser.add_argument("--logic-ffn-layers", type=int, default=2)
    parser.add_argument("--logic-mlp-ratio", type=float, default=1.0)
    parser.add_argument("--num-forest-layers", type=int, default=2)
    parser.add_argument("--logic-connections", choices=["random", "unique"], default="random")
    parser.add_argument("--logic-n-thresholds", type=int, default=7)
    parser.add_argument("--logic-use-thermometer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logic-encoding-temperature", type=float, default=10.0)
    parser.add_argument("--logic-act-fn", default="SIN01")
    parser.add_argument("--logic-weight-init", choices=["ri", "gauss"], default="ri")
    parser.add_argument("--logic-weight-init-sigma", type=float, default=0.5)
    parser.add_argument("--logic-shift-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logic-shift-init-type", default="ri")
    parser.add_argument("--logic-shift-init-shift", type=float, default=1.2)
    parser.add_argument("--logic-shift-init-direction", default="0101")
    parser.add_argument("--logic-resconnection-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logic-res-connect-fraction", type=float, default=0.0)
    parser.add_argument("--grad-factor", type=float, default=1.0)
    parser.add_argument("--logic-connectivity", choices=["fixed", "learnable"], default="fixed")
    parser.add_argument("--learnable-conn-k", type=int, default=64)
    parser.add_argument("--learnable-conn-use-skip-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logic-conn-lr-multiplier", type=float, default=0.2)
    parser.add_argument("--cross-block-logic-history", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cross-block-candidate-frac", type=float, default=0.0)
    parser.add_argument("--post-discretize-finetune-iters", type=int, default=0)
    parser.add_argument("--post-discretize-finetune-lr", type=float, default=0.0)
    parser.add_argument("--teacher-iters", type=int, default=300)
    parser.add_argument("--student-iters", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--temp-start", type=float, default=1.0)
    parser.add_argument("--temp-end", type=float, default=0.3)
    parser.add_argument("--temp-warmup-ratio", type=float, default=0.02)
    parser.add_argument("--temp-cooldown-ratio", type=float, default=0.05)

    parser.add_argument("--distill-alpha", type=float, default=1.0)
    parser.add_argument("--distill-tau", type=float, default=2.0)
    parser.add_argument(
        "--student-scope",
        choices=["logic", "logic_decode", "logic_decode_head", "head", "all"],
        default="logic_decode_head",
    )
    parser.add_argument("--student-hard-thermometer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--eval-max-batches", type=int, default=20, help="-1 evaluates the full selected split.")
    parser.add_argument("--out-dir", type=str, default="runs/goal6plus_vit")
    return parser.parse_args()


def make_train_args(args: argparse.Namespace) -> SimpleNamespace:
    model_type = "logic-ffn" if args.ffn_type == "logic" else "tree-logic-ffn"
    return SimpleNamespace(
        seed=args.seed,
        dataset=args.dataset,
        data_encoding=args.data_encoding,
        augment=args.augment,
        preprocess_once=True,
        batch_size=args.batch_size,
        batches_per_backward=1,
        learning_rate=args.learning_rate,
        num_iterations=args.teacher_iters,
        grad_factor=args.grad_factor,
        valid_set_size=args.valid_set_size,
        eval_freq=1000,
        ext_eval_freq=5000,
        eval_initial=False,
        no_logging=True,
        save_checkpoints=False,
        checkpoint_progress_ratios="",
        num_workers=args.num_workers,
        print_freq=100,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        drop_path_rate=args.drop_path_rate,
        model_type=model_type,
        logic_weight_init=args.logic_weight_init,
        logic_weight_init_sigma=args.logic_weight_init_sigma,
        logic_shift_init=args.logic_shift_init,
        logic_shift_init_type=args.logic_shift_init_type,
        logic_shift_init_shift=args.logic_shift_init_shift,
        logic_shift_init_direction=args.logic_shift_init_direction,
        logic_act_fn=args.logic_act_fn,
        logic_resconnection_init=args.logic_resconnection_init,
        logic_res_connect_fraction=args.logic_res_connect_fraction,
        logic_n_thresholds=args.logic_n_thresholds,
        logic_use_thermometer=args.logic_use_thermometer,
        logic_encoding_temperature=args.logic_encoding_temperature,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        temp_warmup_ratio=args.temp_warmup_ratio,
        temp_cooldown_ratio=args.temp_cooldown_ratio,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup=False,
        mixup_alpha=0.0,
        mixup_prob=0.0,
        cutmix=False,
        cutmix_prob=0.0,
        cutmix_alpha=0.0,
        ffn_type=args.ffn_type,
        logic_ffn_layers=args.logic_ffn_layers,
        logic_mlp_ratio=getattr(args, "logic_mlp_ratio", 1.0),
        logic_connections=args.logic_connections,
        logic_connectivity=args.logic_connectivity,
        learnable_conn_k=args.learnable_conn_k,
        learnable_conn_use_skip_bias=args.learnable_conn_use_skip_bias,
        cross_block_logic_history=getattr(args, "cross_block_logic_history", False),
        cross_block_candidate_frac=getattr(args, "cross_block_candidate_frac", 0.0),
        logic_conn_lr_multiplier=args.logic_conn_lr_multiplier,
        post_discretize_finetune_iters=args.post_discretize_finetune_iters,
        post_discretize_finetune_lr=args.post_discretize_finetune_lr,
        num_forest_layers=args.num_forest_layers,
    )


def set_logic_train_hard_thermometer(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "set_train_hard_thermometer"):
            module.set_train_hard_thermometer(enabled)


def set_logic_soft_eval(model: nn.Module, enabled: bool) -> None:
    for module in logic_layers(model):
        module.set_soft_eval(enabled)


def disable_stochastic_modules(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout) or module.__class__.__name__ == "DropPath":
            module.eval()


def prepare_forward(model: nn.Module, mode: str) -> None:
    set_logic_train_hard_thermometer(model, False)
    if mode == "relaxed_trainpath":
        model.train()
        disable_stochastic_modules(model)
        set_logic_soft_eval(model, True)
    elif mode == "relaxed_eval":
        model.eval()
        set_logic_soft_eval(model, True)
    elif mode == "hard_forward":
        model.eval()
        set_logic_soft_eval(model, False)
    else:
        raise ValueError(f"Unknown forward mode: {mode}")


def preprocess_batch(images: torch.Tensor, transform) -> torch.Tensor:
    return transform(images.to(DEVICE, non_blocking=True))


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_teacher(model: nn.Module, train_loader, transform, args, train_args) -> float:
    model.train()
    optimizer = build_optimizer(model, train_args)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0
    )
    iterator = iter(train_loader)
    start = time.time()
    for step in range(args.teacher_iters):
        temp = get_temperature(step, max(args.teacher_iters, 1), train_args)
        set_model_temperature(model, temp)
        tau_conn, beta_skip, freeze_conn = get_connection_schedule(step, max(args.teacher_iters, 1), train_args)
        set_connection_schedule(model, tau_conn, beta_skip, freeze_conn)
        model.train()
        (images, targets), iterator = next_batch(train_loader, iterator)
        images = preprocess_batch(images, transform)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        reset_fixed_weights(model)
    set_model_temperature(model, args.temp_end)
    return time.time() - start


def finetune_teacher_fixed_connections(
    model: nn.Module,
    train_loader,
    transform,
    args,
    train_args,
) -> tuple[float, int]:
    iters = int(getattr(args, "post_discretize_finetune_iters", 0))
    if iters <= 0:
        return 0.0, 0
    _tau_conn, beta_skip, _freeze_conn = get_connection_schedule(args.teacher_iters, max(args.teacher_iters, 1), train_args)
    discretize_learnable_connections(model, beta_skip=beta_skip)
    set_connection_schedule(model, tau_conn=0.1, beta_skip=beta_skip, freeze=True)
    trainable = configure_gate_only_trainable(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0, 0
    lr = float(args.post_discretize_finetune_lr) if args.post_discretize_finetune_lr > 0 else float(args.learning_rate)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0
    )
    iterator = iter(train_loader)
    start = time.time()
    model.train()
    set_model_temperature(model, args.temp_end)
    for _step in range(iters):
        (images, targets), iterator = next_batch(train_loader, iterator)
        images = preprocess_batch(images, transform)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        reset_fixed_weights(model)
    for param in model.parameters():
        param.requires_grad_(True)
    set_connection_schedule(model, tau_conn=0.1, beta_skip=beta_skip, freeze=True)
    return time.time() - start, trainable


def configure_student_trainable(model: nn.Module, scope: str) -> int:
    if scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        if scope in {"logic", "logic_decode", "logic_decode_head"}:
            for layer in logic_layers(model):
                if hasattr(layer, "weights"):
                    layer.weights.requires_grad_(True)
        if scope in {"logic_decode", "logic_decode_head"}:
            for module in model.modules():
                if hasattr(module, "decode_weights"):
                    module.decode_weights.requires_grad_(True)
        if scope in {"logic_decode_head", "head"} and hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_connection_logits(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "conn_logits" in name:
            param.requires_grad_(False)


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, targets)
    if alpha <= 0.0:
        return ce
    teacher_prob = F.softmax(teacher_logits / tau, dim=-1)
    student_log_prob = F.log_softmax(student_logits / tau, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (tau * tau)
    return ce + alpha * kl


def train_student(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    transform,
    args,
    train_args,
) -> tuple[float, int]:
    trainable = configure_student_trainable(student, args.student_scope)
    conn_params = []
    other_params = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if "conn_logits" in name:
            conn_params.append(param)
        else:
            other_params.append(param)
    groups = [{"params": other_params, "lr": args.student_lr, "weight_decay": args.weight_decay}]
    if conn_params:
        groups.append(
            {
                "params": conn_params,
                "lr": args.student_lr * args.logic_conn_lr_multiplier,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(groups)
    iterator = iter(train_loader)
    start = time.time()
    for step in range(args.student_iters):
        temp = get_temperature(step, max(args.student_iters, 1), train_args)
        set_model_temperature(student, temp)
        set_model_temperature(teacher, temp)
        tau_conn, beta_skip, freeze_conn = get_connection_schedule(step, max(args.student_iters, 1), train_args)
        set_connection_schedule(student, tau_conn, beta_skip, freeze_conn)
        if args.student_scope != "all":
            freeze_connection_logits(student)
        set_connection_schedule(teacher, tau_conn, beta_skip, True)

        (images, targets), iterator = next_batch(train_loader, iterator)
        images = preprocess_batch(images, transform)
        targets = targets.to(DEVICE, non_blocking=True)

        prepare_forward(teacher, "relaxed_trainpath")
        with torch.no_grad():
            teacher_logits = teacher(images)

        student.train()
        disable_stochastic_modules(student)
        set_logic_soft_eval(student, False)
        set_logic_train_hard_thermometer(student, args.student_hard_thermometer)

        optimizer.zero_grad(set_to_none=True)
        logits = student(images)
        loss = distill_loss(logits, teacher_logits, targets, args.distill_alpha, args.distill_tau)
        loss.backward()
        optimizer.step()
        reset_fixed_weights(student)
    set_model_temperature(student, args.temp_end)
    set_logic_train_hard_thermometer(student, False)
    return time.time() - start, trainable


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    transform,
    mode: str,
    max_batches: int | None,
) -> tuple[float, float]:
    prepare_forward(model, mode)
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = preprocess_batch(images, transform)
        targets = targets.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        loss_sum += float(loss.item())
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total += int(targets.numel())
    return correct / max(total, 1), loss_sum / max(total, 1)


def evaluate_permanent_hard(
    model: nn.Module,
    loader,
    transform,
    temperature: float,
    max_batches: int | None,
) -> tuple[float, float]:
    hard_model = copy.deepcopy(model).to(DEVICE)
    harden_model(hard_model, temperature=temperature)
    return evaluate(hard_model, loader, transform, "hard_forward", max_batches)


def evaluate_permanent_hard_with_stats(
    model: nn.Module,
    loader,
    transform,
    temperature: float,
    max_batches: int | None,
    topk: int,
) -> tuple[float, float, dict[str, float]]:
    hard_model = copy.deepcopy(model).to(DEVICE)
    harden_model(hard_model, temperature=temperature)
    acc, loss = evaluate(hard_model, loader, transform, "hard_forward", max_batches)
    return acc, loss, model_stats(hard_model, topk)


def gate_stats(model: nn.Module) -> dict[str, float]:
    total = 0
    constant = 0
    fanout_max = 0
    depth = 0
    with torch.no_grad():
        for layer in logic_layers(model):
            weights = getattr(layer, "weights", None)
            indices = getattr(layer, "indices", None)
            if weights is None:
                continue
            depth += 1
            temp = max(float(getattr(layer, "temperature", 1.0)), 1e-4)
            bits = (layer.act_fn(weights / temp) >= 0.5)
            total += int(bits.shape[0])
            constant += int((bits.amin(dim=1) == bits.amax(dim=1)).sum().item())
            if indices is not None:
                in_dim = int(getattr(layer, "in_dim", 0))
                if in_dim > 0:
                    both = torch.cat([indices[0].detach().cpu(), indices[1].detach().cpu()])
                    counts = torch.bincount(both, minlength=in_dim)
                    fanout_max = max(fanout_max, int(counts.max().item()))
    return {
        "gate_count": float(total),
        "depth": float(depth),
        "fanout_max": float(fanout_max),
        "unused_gate_ratio": float(constant / total) if total else 0.0,
        "dead_gate_ratio": float(constant / total) if total else 0.0,
    }


def head_stats(model: nn.Module, topk: int) -> dict[str, float]:
    head = getattr(model, "head", None)
    if hasattr(head, "effective_weight"):
        weight = head.effective_weight()
    else:
        weight = getattr(head, "weight", None)
    if weight is None:
        return {}
    with torch.no_grad():
        w = weight.detach().float().cpu()
        total = max(int(w.numel()), 1)
        abs_w = w.abs()
        near_zero = abs_w < 1e-4
        stats = {
            "head_negative_ratio": float((w < 0).sum().item() / total),
            "head_positive_ratio": float((w > 0).sum().item() / total),
            "head_near_zero_ratio": float(near_zero.sum().item() / total),
            "head_mean_abs_weight": float(abs_w.mean().item()),
        }
        if hasattr(head, "counter_scale") and bool(getattr(head, "scaled_counter_flag", torch.tensor(False)).item()):
            scale = head.counter_scale.detach().float().cpu()
            stats["head_counter_scale_mean"] = float(scale.mean().item())
            stats["head_counter_scale_min"] = float(scale.min().item())
            stats["head_counter_scale_max"] = float(scale.max().item())
        if topk > 0 and topk < w.shape[1]:
            topk_res = abs_w.topk(topk, dim=1)
            topk_abs = topk_res.values.sum(dim=1)
            denom = abs_w.sum(dim=1).clamp_min(1e-12)
            stats["head_topk_abs_mass_ratio"] = float((topk_abs / denom).mean().item())
            selected = torch.zeros_like(w, dtype=torch.bool)
            selected.scatter_(1, topk_res.indices, True)
            per_feature_classes = selected.sum(dim=0).float()
            used = per_feature_classes > 0
            stats["head_topk_used_feature_ratio"] = float(used.float().mean().item())
            stats["head_topk_mean_classes_per_used_feature"] = float(
                per_feature_classes[used].mean().item() if used.any() else 0.0
            )
            pos_selected = selected & (w > 0)
            neg_selected = selected & (w < 0)
            stats["head_topk_pos_share_ratio"] = float((pos_selected.sum(dim=0) > 1).float().mean().item())
            stats["head_topk_neg_share_ratio"] = float((neg_selected.sum(dim=0) > 1).float().mean().item())
        return stats


def model_stats(model: nn.Module, topk: int) -> dict[str, float]:
    stats = gate_stats(model)
    stats.update(head_stats(model, topk))
    stats.update(connection_stats(model))
    return stats


def prefixed_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in stats.items()}


def write_metrics(out_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out_root = Path(args.out_dir)
    out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_args = make_train_args(args)
    train_loader, valid_loader, test_loader, _train_eval_loader, _num_batches, transform = load_dataset(train_args)
    eval_loader = test_loader if args.eval_split == "test" else (valid_loader if valid_loader is not None else test_loader)
    eval_max_batches = None if args.eval_max_batches < 0 else args.eval_max_batches

    teacher = build_model(train_args).to(DEVICE)
    teacher_time = train_teacher(teacher, train_loader, transform, args, train_args)
    post_disc_ft_time, post_disc_ft_trainable = finetune_teacher_fixed_connections(
        teacher, train_loader, transform, args, train_args
    )

    student = copy.deepcopy(teacher).to(DEVICE)
    student_time, trainable_params = train_student(
        teacher, student, train_loader, transform, args, train_args
    )
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available()
        else 0.0
    )

    head_diag_topk = 32
    teacher_stats = model_stats(teacher, head_diag_topk)
    student_stats = model_stats(student, head_diag_topk)
    rows: list[dict[str, object]] = []
    eval_specs = [
        ("teacher_relaxed_trainpath", teacher, "relaxed_trainpath", False),
        ("teacher_relaxed_eval", teacher, "relaxed_eval", False),
        ("teacher_hard_forward", teacher, "hard_forward", False),
        ("teacher_permanent_hard", teacher, "hard_forward", True),
        ("student_relaxed_trainpath", student, "relaxed_trainpath", False),
        ("student_hard_forward", student, "hard_forward", False),
        ("student_permanent_hard", student, "hard_forward", True),
    ]
    for method, model, mode, permanent in eval_specs:
        if permanent:
            acc, loss, current_stats = evaluate_permanent_hard_with_stats(
                model, eval_loader, transform, args.temp_end, eval_max_batches, head_diag_topk
            )
        else:
            current_stats = model_stats(model, head_diag_topk)
            acc, loss = evaluate(model, eval_loader, transform, mode, eval_max_batches)
        row = {
            "method": method,
            "dataset": args.dataset,
            "eval_split": args.eval_split,
            "eval_max_batches": args.eval_max_batches,
            "seed": args.seed,
            "acc": acc,
            "loss": loss,
            "teacher_train_time": teacher_time,
            "student_train_time": student_time,
            "post_disc_ft_time": post_disc_ft_time,
            "post_disc_ft_trainable_params": post_disc_ft_trainable,
            "peak_memory_mb": peak_memory_mb,
            "teacher_iters": args.teacher_iters,
            "student_iters": args.student_iters,
            "student_trainable_params": trainable_params,
            **current_stats,
        }
        rows.append(row)

    by_method = {str(r["method"]): r for r in rows}
    teacher_soft = float(by_method["teacher_relaxed_trainpath"]["acc"])
    teacher_hard = float(by_method["teacher_permanent_hard"]["acc"])
    student_hard = float(by_method["student_permanent_hard"]["acc"])
    summary = {
        "out_dir": str(out_dir),
        "eval_split": args.eval_split,
        "eval_max_batches": args.eval_max_batches,
        "teacher_soft_acc": teacher_soft,
        "teacher_hard_acc": teacher_hard,
        "teacher_acc_gap": abs(teacher_soft - teacher_hard),
        "teacher_soft_to_discrete_drop": teacher_soft - teacher_hard,
        "student_hard_acc": student_hard,
        "student_delta_vs_teacher_hard": student_hard - teacher_hard,
        "student_gap_vs_teacher_soft": abs(teacher_soft - student_hard),
        "teacher_train_time": teacher_time,
        "student_train_time": student_time,
        "post_disc_ft_time": post_disc_ft_time,
        "post_disc_ft_trainable_params": post_disc_ft_trainable,
        "peak_memory_mb": peak_memory_mb,
        "student_trainable_params": trainable_params,
        **prefixed_stats("teacher_", teacher_stats),
        **prefixed_stats("student_", student_stats),
    }
    write_metrics(out_dir, rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
