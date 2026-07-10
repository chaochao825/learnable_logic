#!/usr/bin/env python3
"""K-gate truth-table expansion for trained LightLogic gates.

Goal 3 prototype:

1. Train a continuous LightLogic/IWP teacher.
2. For every two-input gate, read its continuous truth table q for
   input patterns 00, 01, 10, 11.
3. Quantize q with r = floor(K*q + 0.5), so r/K is realizable by K hard
   truth tables whose popcount matches r for each input pattern.
4. Feed popcount/K through a fixed threshold to recover one bit per original
   neuron, preserving the original network wiring.
5. Report local truth-table error and full-network discrete accuracy/gap.

The implementation uses the algebraically equivalent thresholded r/K truth
table during inference instead of materializing all K parallel gates.  The
reported expanded_gate_count still counts K hard gates per original gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from hard_lgn_benchmark import DatasetBundle, GroupSum, fanout_max, load_dataset, make_loader, set_seed
from lightlogic_experiments import (
    LightLogicLayer,
    LightNet,
    build_light_architecture,
    evaluate,
    make_light_net,
    temperature_at,
    unused_gate_ratio,
)


@dataclass
class KExpansionRow:
    dataset: str
    seed: int
    k: int
    calibration_mode: str
    threshold: float
    temp_eval: float
    estimator: str
    init: str
    teacher_acc: float
    expanded_acc: float
    acc_gap: float
    teacher_loss: float
    expanded_loss: float
    loss_gap: float
    local_mae: float
    local_mse: float
    local_max_abs_error: float
    local_binary_flip_ratio: float
    original_gate_count: int
    expanded_gate_count: int
    gate_count_multiplier: int
    depth: int
    fanout_max: int
    unused_gate_ratio: float
    train_time: float


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


class KExpandedThresholdLayer(nn.Module):
    def __init__(self, source: LightLogicLayer, k: int, threshold: float, teacher_temperature: float) -> None:
        super().__init__()
        self.in_dim = source.in_dim
        self.out_dim = source.out_dim
        self.k = int(k)
        self.threshold = float(threshold)
        self.register_buffer("indices_0", source.indices_0.detach().cpu().clone().long())
        self.register_buffer("indices_1", source.indices_1.detach().cpu().clone().long())

        self.teacher_temperature = float(teacher_temperature)
        q = source.probs(self.teacher_temperature).detach().cpu().clamp(0.0, 1.0)
        r = torch.floor(q * self.k + 0.5).clamp(0, self.k)
        qk = r / float(self.k)
        self.register_buffer("q", q)
        self.register_buffer("r", r.to(torch.int16))
        self.register_buffer("qk", qk)
        self.register_buffer("threshold_value", torch.full((source.out_dim,), float(threshold), dtype=torch.float32))

    def set_thresholds(self, thresholds: torch.Tensor | float) -> None:
        if isinstance(thresholds, float):
            value = torch.full((self.out_dim,), thresholds, dtype=torch.float32)
        else:
            value = thresholds.detach().cpu().float().reshape(-1)
            if value.numel() == 1:
                value = value.repeat(self.out_dim)
            if value.numel() != self.out_dim:
                raise ValueError((value.shape, self.out_dim))
        self.threshold_value.copy_(value.clamp(0.0, 1.0))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]
        p = self.qk.to(device=x.device, dtype=x.dtype)
        return (
            p[:, 0].view(1, -1) * (1 - a) * (1 - b)
            + p[:, 1].view(1, -1) * (1 - a) * b
            + p[:, 2].view(1, -1) * a * (1 - b)
            + p[:, 3].view(1, -1) * a * b
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = self.threshold_value.to(device=x.device, dtype=x.dtype).view(1, -1)
        return (self.scores(x) >= threshold).to(dtype=x.dtype)

    def local_error(self) -> dict[str, float]:
        diff = (self.q - self.qk).abs()
        return {
            "mae": float(diff.mean().item()),
            "mse": float(diff.square().mean().item()),
            "max_abs_error": float(diff.max().item()),
            "binary_flip_ratio": float(((self.q >= 0.5) != (self.qk >= self.threshold_value.view(-1, 1))).float().mean().item()),
        }


def train_teacher(
    dataset: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[LightNet, float]:
    arch = build_light_architecture(
        dataset.input_dim,
        args.width,
        args.layers,
        seed,
        args.estimator,
        args.init,
        args.init_strength,
    )
    model = make_light_net(arch, dataset.num_classes, args.group_tau, args.estimator, "iwp").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed + 20001)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        temp = temperature_at(epoch, args.epochs, args.temp_start, args.temp_end) if args.anneal_train else args.temp_eval
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mode="continuous", temperature=temp)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
    return model, time.perf_counter() - started


def make_k_model(
    teacher: LightNet,
    num_classes: int,
    group_tau: float,
    k: int,
    threshold: float,
    teacher_temperature: float,
) -> LightNet:
    layers = []
    for layer in teacher.layers:
        if not isinstance(layer, LightLogicLayer):
            raise TypeError(f"K expansion currently expects LightLogicLayer, got {type(layer)}")
        layers.append(KExpandedThresholdLayer(layer, k, threshold, teacher_temperature))
    return LightNet(layers, num_classes=num_classes, group_tau=group_tau)


@torch.no_grad()
def calibrate_thresholds(
    teacher: LightNet,
    expanded: LightNet,
    x_cal: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    grid_size: int,
    teacher_temperature: float,
) -> None:
    if mode == "fixed":
        return
    if grid_size < 2:
        raise ValueError("--calibration-grid-size must be >= 2")
    teacher_layers = teacher.light_layers()
    expanded_layers = [layer for layer in expanded.layers if isinstance(layer, KExpandedThresholdLayer)]
    if len(teacher_layers) != len(expanded_layers):
        raise ValueError((len(teacher_layers), len(expanded_layers)))
    thresholds = torch.linspace(0.0, 1.0, grid_size, device=device)
    current_chunks = []
    for start in range(0, x_cal.shape[0], batch_size):
        current_chunks.append(x_cal[start : start + batch_size].to(device))

    for teacher_layer, expanded_layer in zip(teacher_layers, expanded_layers, strict=True):
        score_chunks = []
        target_chunks = []
        for xb in current_chunks:
            score = expanded_layer.scores(xb)
            target = (teacher_layer(xb, mode="continuous", temperature=teacher_temperature) >= 0.5).to(score.dtype)
            score_chunks.append(score)
            target_chunks.append(target)
        scores = torch.cat(score_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)

        if mode == "per_layer":
            losses = [((scores >= t).to(targets.dtype) != targets).float().mean() for t in thresholds]
            best_t = thresholds[int(torch.stack(losses).argmin().item())].detach().cpu()
            expanded_layer.set_thresholds(float(best_t.item()))
        elif mode == "per_neuron":
            chosen = []
            for neuron in range(scores.shape[1]):
                losses = [
                    ((scores[:, neuron] >= t).to(targets.dtype) != targets[:, neuron]).float().mean()
                    for t in thresholds
                ]
                chosen.append(thresholds[int(torch.stack(losses).argmin().item())])
            expanded_layer.set_thresholds(torch.stack(chosen).detach().cpu())
        else:
            raise ValueError(mode)

        next_chunks = []
        for xb in current_chunks:
            next_chunks.append(expanded_layer(xb))
        current_chunks = next_chunks


def local_errors(model: LightNet) -> dict[str, float]:
    errors = [layer.local_error() for layer in model.layers if isinstance(layer, KExpandedThresholdLayer)]
    if not errors:
        return {"mae": math.nan, "mse": math.nan, "max_abs_error": math.nan, "binary_flip_ratio": math.nan}
    return {
        "mae": sum(item["mae"] for item in errors) / len(errors),
        "mse": sum(item["mse"] for item in errors) / len(errors),
        "max_abs_error": max(item["max_abs_error"] for item in errors),
        "binary_flip_ratio": sum(item["binary_flip_ratio"] for item in errors) / len(errors),
    }


def original_gate_count(model: LightNet) -> int:
    return sum(int(layer.out_dim) for layer in model.layers)  # type: ignore[attr-defined]


def run_dataset(dataset_name: str, seed: int, args: argparse.Namespace, device: torch.device) -> list[KExpansionRow]:
    dataset = load_dataset(dataset_name, seed, args)
    if args.width % dataset.num_classes != 0:
        raise ValueError(f"width={args.width} must be divisible by num_classes={dataset.num_classes}")
    if args.width * 2 < dataset.input_dim:
        raise ValueError(f"width={args.width} too small for input_dim={dataset.input_dim}")
    teacher, train_time = train_teacher(dataset, args, device, seed)
    teacher_acc, teacher_loss = evaluate(
        teacher,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "continuous",
        args.temp_eval,
    )
    rows: list[KExpansionRow] = []
    base_gates = original_gate_count(teacher)
    for k in args.k_values:
        expanded = make_k_model(teacher, dataset.num_classes, args.group_tau, k, args.threshold, args.temp_eval).to(device)
        calibrate_thresholds(
            teacher,
            expanded,
            dataset.x_train,
            args.eval_batch_size,
            device,
            args.calibration_mode,
            args.calibration_grid_size,
            args.temp_eval,
        )
        expanded_acc, expanded_loss = evaluate(
            expanded,
            dataset.x_test,
            dataset.y_test,
            args.eval_batch_size,
            device,
            "hard",
            1.0,
        )
        err = local_errors(expanded)
        rows.append(
            KExpansionRow(
                dataset=dataset.name,
                seed=seed,
                k=int(k),
                calibration_mode=args.calibration_mode,
                threshold=float(args.threshold),
                temp_eval=float(args.temp_eval),
                estimator=args.estimator,
                init=args.init,
                teacher_acc=teacher_acc,
                expanded_acc=expanded_acc,
                acc_gap=abs(teacher_acc - expanded_acc),
                teacher_loss=teacher_loss,
                expanded_loss=expanded_loss,
                loss_gap=abs(teacher_loss - expanded_loss),
                local_mae=err["mae"],
                local_mse=err["mse"],
                local_max_abs_error=err["max_abs_error"],
                local_binary_flip_ratio=err["binary_flip_ratio"],
                original_gate_count=base_gates,
                expanded_gate_count=base_gates * int(k),
                gate_count_multiplier=int(k),
                depth=len(teacher.layers),
                fanout_max=fanout_max(expanded.layers),
                unused_gate_ratio=unused_gate_ratio(expanded, dataset.x_train, args.eval_batch_size, device),
                train_time=train_time,
            )
        )
    return rows


def markdown_report(rows: list[KExpansionRow], args: argparse.Namespace) -> str:
    lines = [
        "# LightLogic K-Gate Expansion Report",
        "",
        "This is Goal 3 evidence: each trained LightLogic gate is expanded to K hard gates whose popcount truth table approximates the continuous truth table, then thresholded back to one bit for the next layer.",
        "",
        "| dataset | seed | K | teacher_acc | expanded_acc | acc_gap | local_mae | local_mse | flip_ratio | expanded_gates | unused_gate_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {seed} | {k} | {teacher_acc:.6g} | {expanded_acc:.6g} | {acc_gap:.6g} | {local_mae:.6g} | {local_mse:.6g} | {local_binary_flip_ratio:.6g} | {expanded_gate_count} | {unused_gate_ratio:.6g} |".format(
                **asdict(row)
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            f"- K values: {', '.join(str(k) for k in args.k_values)}",
            f"- Threshold: {args.threshold}",
            f"- Calibration mode: {args.calibration_mode}",
            "- `local_*` errors compare q against r/K before thresholding.",
            "- `expanded_acc` evaluates the thresholded hard network. This first version intentionally preserves one output bit per original neuron.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calibration-mode", choices=["fixed", "per_layer", "per_neuron"], default="fixed")
    parser.add_argument("--calibration-grid-size", type=int, default=101)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--estimator", choices=["sigmoid", "sinusoidal"], default="sinusoidal")
    parser.add_argument("--init", choices=["residual", "and_or", "xor", "random", "uniform"], default="residual")
    parser.add_argument("--init-strength", type=float, default=0.98)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.1)
    parser.add_argument("--temp-eval", type=float, default=1.0)
    parser.add_argument("--anneal-train", action="store_true")
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/lightlogic_k_expansion_v1")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets = ["parity6", "majority7", "random_sparse8"]
        args.seeds = [0]
        args.k_values = [2, 4, 8]
        args.width = min(args.width, 32)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 12)
    if any(k <= 0 for k in args.k_values):
        raise ValueError("--k-values must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1]")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[KExpansionRow] = []
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            rows = run_dataset(dataset_name, seed, args, device)
            all_rows.extend(rows)
            for row in rows:
                print(
                    "dataset={} seed={} K={} teacher_acc={:.4f} expanded_acc={:.4f} gap={:.4f} local_mae={:.4f}".format(
                        row.dataset,
                        row.seed,
                        row.k,
                        row.teacher_acc,
                        row.expanded_acc,
                        row.acc_gap,
                        row.local_mae,
                    ),
                    flush=True,
                )
            write_csv(out_dir / "k_expansion_results.partial.csv", [asdict(row) for row in all_rows])
    write_csv(out_dir / "k_expansion_results.csv", [asdict(row) for row in all_rows])
    (out_dir / "k_expansion_summary.md").write_text(markdown_report(all_rows, args), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'k_expansion_results.csv'}")


if __name__ == "__main__":
    main()
