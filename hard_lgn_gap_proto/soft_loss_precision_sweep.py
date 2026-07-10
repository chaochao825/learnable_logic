#!/usr/bin/env python3
"""Precision-first sweeps for diagnosing high soft loss in LightLogic/LUT LGNs.

This runner intentionally avoids ABC and synthesis.  It keeps the logic
backbone comparable while varying output heads, input threshold expansion, and
training mode so we can identify which component is limiting the continuous
teacher before spending effort on soft-to-hard conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from hard_lgn_benchmark import GroupSum, load_dataset, make_loader, set_seed
from lightlogic_b_lut_minimization import (
    BInputLutLayer,
    effective_width,
    fanout_max_model,
    parameter_count_model,
    temperature_at,
    unused_gate_ratio_model,
)
from lightlogic_b_lut_minimization import build_architecture as build_b_lut_architecture


HEAD_TYPES = [
    "groupsum",
    "affine_groupsum",
    "final_linear",
    "concat_linear",
    "concat_mlp",
    "input_linear",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def make_layers(
    arch: list[dict[str, torch.Tensor | int | str]],
    estimator: str,
) -> list[BInputLutLayer]:
    layers: list[BInputLutLayer] = []
    for spec in arch:
        layers.append(
            BInputLutLayer(
                int(spec["in_dim"]),
                int(spec["out_dim"]),
                int(spec["arity"]),
                spec["indices"],  # type: ignore[arg-type]
                spec["raw"],  # type: ignore[arg-type]
                estimator=estimator,
            )
        )
    return layers


class PrecisionLutNet(nn.Module):
    def __init__(
        self,
        layers: list[BInputLutLayer],
        input_dim: int,
        num_classes: int,
        head_type: str,
        group_tau: float,
        include_input_in_concat: bool,
        mlp_hidden: int,
        head_norm: bool,
    ) -> None:
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(head_type)
        self.layers = nn.ModuleList(layers)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.head_type = head_type
        self.include_input_in_concat = bool(include_input_in_concat)
        self.group_sum = GroupSum(num_classes, tau=group_tau)

        final_dim = int(layers[-1].out_dim) if layers else int(input_dim)
        concat_dim = sum(int(layer.out_dim) for layer in layers)
        if include_input_in_concat:
            concat_dim += int(input_dim)

        if head_type == "affine_groupsum":
            self.class_scale = nn.Parameter(torch.ones(num_classes))
            self.class_bias = nn.Parameter(torch.zeros(num_classes))
        elif head_type == "final_linear":
            self.head = self._make_linear_head(final_dim, num_classes, head_norm)
        elif head_type == "concat_linear":
            self.head = self._make_linear_head(concat_dim, num_classes, head_norm)
        elif head_type == "concat_mlp":
            hidden = max(int(mlp_hidden), num_classes)
            prefix: list[nn.Module] = [nn.LayerNorm(concat_dim)] if head_norm else []
            self.head = nn.Sequential(
                *prefix,
                nn.Linear(concat_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_classes),
            )
        elif head_type == "input_linear":
            self.head = self._make_linear_head(input_dim, num_classes, head_norm)

    @staticmethod
    def _make_linear_head(in_dim: int, num_classes: int, head_norm: bool) -> nn.Module:
        if head_norm:
            return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, num_classes))
        return nn.Linear(in_dim, num_classes)

    def _logic_features(
        self,
        x: torch.Tensor,
        mode: str,
        temperature: float,
        gumbel_tau: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, mode=mode, temperature=temperature, gumbel_tau=gumbel_tau)
            features.append(x)
        return x, features

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "continuous",
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
    ) -> torch.Tensor:
        x0 = x
        final, features = self._logic_features(x, mode, temperature, gumbel_tau)
        if self.head_type == "groupsum":
            return self.group_sum(final)
        if self.head_type == "affine_groupsum":
            logits = self.group_sum(final)
            return logits * self.class_scale.view(1, -1) + self.class_bias.view(1, -1)
        if self.head_type == "final_linear":
            return self.head(final)
        if self.head_type == "input_linear":
            return self.head(x0)
        if self.head_type in {"concat_linear", "concat_mlp"}:
            values = features
            if self.include_input_in_concat:
                values = [x0, *features]
            return self.head(torch.cat(values, dim=-1))
        raise ValueError(self.head_type)


@torch.no_grad()
def evaluate_model(
    model: PrecisionLutNet,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    temperature: float,
    gumbel_tau: float,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits = model(xb, mode=mode, temperature=temperature, gumbel_tau=gumbel_tau)
        loss_sum += float(F.cross_entropy(logits, yb, reduction="sum").detach().cpu().item())
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    if was_training:
        model.train()
    return correct / max(total, 1), loss_sum / max(total, 1)


def train_one(
    dataset_name: str,
    seed: int,
    threshold_levels: int,
    b: int,
    head_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    dataset_args = deepcopy(args)
    dataset_args.threshold_levels = threshold_levels
    dataset = load_dataset(dataset_name, seed, dataset_args)
    width = effective_width(dataset.input_dim, dataset.num_classes, b, args.width)
    arch = build_b_lut_architecture(
        dataset.input_dim,
        width,
        args.layers,
        b,
        seed,
        args.estimator,
        args.init,
        args.init_strength,
    )

    set_seed(seed + 100003 + b * 1009 + HEAD_TYPES.index(head_type) * 9176 + threshold_levels * 37)
    model = PrecisionLutNet(
        make_layers(arch, args.estimator),
        dataset.input_dim,
        dataset.num_classes,
        head_type,
        args.group_tau,
        args.include_input_in_concat,
        args.mlp_hidden,
        args.head_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * args.lr_floor)
        if args.cosine_lr
        else None
    )
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed + 30013 + b * 101)

    best_soft_loss = math.inf
    best_soft_acc = math.nan
    best_epoch = -1
    epoch_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    final_train_loss = math.nan
    for epoch in range(1, args.epochs + 1):
        temp = temperature_at(epoch - 1, args.epochs, args.temp_start, args.temp_end) if args.anneal_train else args.temp_eval
        gumbel_tau = temperature_at(epoch - 1, args.epochs, args.gumbel_temp_start, args.gumbel_temp_end)
        train_mode = args.warmup_mode if epoch <= args.warmup_epochs else args.train_forward_mode
        model.train()
        loss_sum = 0.0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mode=train_mode, temperature=temp, gumbel_tau=gumbel_tau)
            loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            if args.entropy_weight:
                ent = torch.stack([layer.entropy(temp) for layer in model.layers]).mean()
                loss = loss + args.entropy_weight * ent
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item()) * int(yb.numel())
            total += int(yb.numel())
        if scheduler is not None:
            scheduler.step()
        final_train_loss = loss_sum / max(total, 1)

        if args.eval_every > 0 and (epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0):
            soft_acc, soft_loss = evaluate_model(
                model,
                dataset.x_test,
                dataset.y_test,
                args.eval_batch_size,
                device,
                "continuous",
                args.temp_eval,
                args.gumbel_temp_end,
            )
            if soft_loss < best_soft_loss:
                best_soft_loss = soft_loss
                best_soft_acc = soft_acc
                best_epoch = epoch
            epoch_rows.append(
                {
                    "dataset": dataset.name,
                    "seed": seed,
                    "threshold_levels": threshold_levels,
                    "b": b,
                    "head_type": head_type,
                    "epoch": epoch,
                    "train_mode": train_mode,
                    "train_loss": final_train_loss,
                    "soft_acc": soft_acc,
                    "soft_loss": soft_loss,
                    "temperature": temp,
                    "gumbel_tau": gumbel_tau,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed": time.perf_counter() - started,
                }
            )

    train_time = time.perf_counter() - started
    soft_acc, soft_loss = evaluate_model(
        model,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "continuous",
        args.temp_eval,
        args.gumbel_temp_end,
    )
    hard_acc, hard_loss = evaluate_model(
        model,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "hard",
        1.0,
        args.gumbel_temp_end,
    )
    if soft_loss < best_soft_loss:
        best_soft_loss = soft_loss
        best_soft_acc = soft_acc
        best_epoch = args.epochs
    row = {
        "dataset": dataset.name,
        "source_dataset_arg": dataset_name,
        "seed": seed,
        "threshold_levels": threshold_levels,
        "input_dim": dataset.input_dim,
        "num_classes": dataset.num_classes,
        "b": b,
        "head_type": head_type,
        "width": width,
        "layers": args.layers,
        "epochs": args.epochs,
        "train_forward_mode": args.train_forward_mode,
        "warmup_epochs": args.warmup_epochs,
        "warmup_mode": args.warmup_mode,
        "lr": args.lr,
        "cosine_lr": bool(args.cosine_lr),
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "head_norm": bool(args.head_norm),
        "include_input_in_concat": bool(args.include_input_in_concat),
        "estimator": args.estimator,
        "init": args.init,
        "init_strength": args.init_strength,
        "soft_acc": soft_acc,
        "soft_loss": soft_loss,
        "hard_acc": hard_acc,
        "hard_loss": hard_loss,
        "acc_gap": abs(soft_acc - hard_acc),
        "loss_gap": abs(soft_loss - hard_loss),
        "best_soft_acc": best_soft_acc,
        "best_soft_loss": best_soft_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "train_time": train_time,
        "parameter_count": parameter_count_model(model),
        "logic_parameter_count": sum(int(layer.raw.numel()) for layer in model.layers),
        "head_parameter_count": parameter_count_model(model) - sum(int(layer.raw.numel()) for layer in model.layers),
        "unused_gate_ratio": unused_gate_ratio_model(model, dataset.x_train, args.eval_batch_size, device),
        "fanout_max": fanout_max_model(list(model.layers)),
    }
    return row, epoch_rows


def rank_key(row: dict[str, object]) -> tuple[float, float]:
    return (safe_float(row.get("soft_loss")), -safe_float(row.get("soft_acc")))


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Soft-Loss Precision Sweep",
        "",
        "This diagnostic prioritizes continuous loss/accuracy and intentionally ignores synthesis cost.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(vars(args), indent=2, sort_keys=True),
        "```",
        "",
        "## Best Rows By Dataset",
        "",
        "| dataset | best_head | seed | threshold_levels | b | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | params |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        group = [row for row in rows if row["dataset"] == dataset]
        best = min(group, key=rank_key)
        lines.append(
            "| {dataset} | {head_type} | {seed} | {threshold_levels} | {b} | {width} | {soft_acc:.6g} | {soft_loss:.6g} | {hard_acc:.6g} | {hard_loss:.6g} | {acc_gap:.6g} | {parameter_count} |".format(
                **best
            )
        )
    lines.extend(
        [
            "",
            "## Component Deltas Vs GroupSum",
            "",
            "| dataset | seed | threshold_levels | b | width | head | soft_loss_delta | soft_acc_delta |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    keys = sorted({(row["dataset"], row["seed"], row["threshold_levels"], row["b"], row["width"]) for row in rows})
    for key in keys:
        group = [
            row
            for row in rows
            if (row["dataset"], row["seed"], row["threshold_levels"], row["b"], row["width"]) == key
        ]
        base = next((row for row in group if row["head_type"] == "groupsum"), None)
        if base is None:
            continue
        for row in sorted(group, key=lambda item: str(item["head_type"])):
            if row is base:
                continue
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {:.6g} | {:.6g} |".format(
                    row["dataset"],
                    row["seed"],
                    row["threshold_levels"],
                    row["b"],
                    row["width"],
                    row["head_type"],
                    safe_float(row["soft_loss"]) - safe_float(base["soft_loss"]),
                    safe_float(row["soft_acc"]) - safe_float(base["soft_acc"]),
                )
            )
    lines.extend(
        [
            "",
            "## All Rows",
            "",
            "| dataset | seed | threshold_levels | b | head | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | train_time | params | unused_gate_ratio |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["dataset"]), rank_key(item), str(item["head_type"]))):
        lines.append(
            "| {dataset} | {seed} | {threshold_levels} | {b} | {head_type} | {width} | {soft_acc:.6g} | {soft_loss:.6g} | {hard_acc:.6g} | {hard_loss:.6g} | {acc_gap:.6g} | {train_time:.6g} | {parameter_count} | {unused_gate_ratio:.6g} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["digits", "binarized_mnist"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--threshold-levels-list", nargs="+", type=int, default=[1])
    parser.add_argument("--b-values", nargs="+", type=int, default=[4])
    parser.add_argument("--head-types", nargs="+", choices=HEAD_TYPES, default=["groupsum", "affine_groupsum", "final_linear", "concat_linear", "concat_mlp"])
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr-floor", type=float, default=0.05)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--estimator", choices=["sigmoid", "sinusoidal"], default="sinusoidal")
    parser.add_argument("--init", choices=["residual", "and_or", "xor", "random", "uniform"], default="residual")
    parser.add_argument("--init-strength", type=float, default=0.98)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.2)
    parser.add_argument("--temp-eval", type=float, default=1.0)
    parser.add_argument("--gumbel-temp-start", type=float, default=1.5)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.3)
    parser.add_argument("--anneal-train", action="store_true")
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--train-forward-mode", choices=["continuous", "st", "gumbel_st"], default="continuous")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-mode", choices=["continuous", "st", "gumbel_st"], default="continuous")
    parser.add_argument("--include-input-in-concat", action="store_true")
    parser.add_argument("--mlp-hidden", type=int, default=1024)
    parser.add_argument("--head-norm", action="store_true")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=12000)
    parser.add_argument("--image-max-test", type=int, default=3000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/soft_loss_precision_sweep_v1")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets = ["digits"]
        args.seeds = [0]
        args.threshold_levels_list = [1]
        args.b_values = [3]
        args.head_types = ["groupsum", "affine_groupsum", "final_linear"]
        args.width = min(args.width, 160)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 5)
        args.image_max_train = min(args.image_max_train, 1000)
        args.image_max_test = min(args.image_max_test, 500)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    print(f"device={device} out_dir={out_dir}", flush=True)
    for seed in args.seeds:
        for dataset_name in args.datasets:
            for threshold_levels in args.threshold_levels_list:
                for b in args.b_values:
                    for head_type in args.head_types:
                        print(
                            f"dataset={dataset_name} seed={seed} threshold={threshold_levels} b={b} head={head_type}",
                            flush=True,
                        )
                        row, history = train_one(dataset_name, seed, threshold_levels, b, head_type, args, device)
                        rows.append(row)
                        epoch_rows.extend(history)
                        write_csv(out_dir / "results.partial.csv", rows)
                        if epoch_rows:
                            write_csv(out_dir / "epoch_log.partial.csv", epoch_rows)
                        print(
                            "  soft_acc={:.4f} soft_loss={:.4f} hard_acc={:.4f} gap={:.4f} params={}".format(
                                row["soft_acc"],
                                row["soft_loss"],
                                row["hard_acc"],
                                row["acc_gap"],
                                row["parameter_count"],
                            ),
                            flush=True,
                        )

    write_csv(out_dir / "results.csv", rows)
    if epoch_rows:
        write_csv(out_dir / "epoch_log.csv", epoch_rows)
    write_markdown(out_dir / "soft_loss_precision_sweep.md", rows, args)
    print(out_dir / "results.csv")
    print(out_dir / "soft_loss_precision_sweep.md")


if __name__ == "__main__":
    main()
