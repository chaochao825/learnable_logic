#!/usr/bin/env python3
"""Full K-popcount propagation for LightLogic gate expansion.

This script tests the structural alternative to the earlier K-expanded
student:

    old path:  K hard gates -> popcount/K -> threshold -> 1-bit wire
    new path:  K hard gates -> popcount/K -> multi-level wire

The new path keeps the quantized multi-level value in hidden layers and feeds
it into the next gate through the same multilinear extension used by
LightLogic.  It is not the same cost model as a pure one-bit LGN.  It is a
K-bit / popcount or small-LUT deployable network, and is meant to isolate
whether per-layer 1-bit thresholding is the source of accuracy loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from hard_lgn_benchmark import DatasetBundle, fanout_max, load_dataset, make_loader, set_seed
from lightlogic_experiments import LightLogicLayer, LightNet, evaluate, unused_gate_ratio
from lightlogic_k_expansion import (
    KExpandedThresholdLayer,
    calibrate_thresholds,
    local_errors,
    make_k_model,
    original_gate_count,
    train_teacher,
)


@dataclass
class FullKRow:
    dataset: str
    seed: int
    k: int
    propagation: str
    final_readout: str
    calibration_mode: str
    teacher_acc: float
    candidate_acc: float
    acc_gap_vs_teacher: float
    teacher_loss: float
    candidate_loss: float
    loss_gap_vs_teacher: float
    threshold_each_layer_acc: float
    threshold_each_layer_loss: float
    delta_acc_vs_threshold_each_layer: float
    local_mae: float
    local_mse: float
    local_max_abs_error: float
    layer_alignment_mae: float
    final_layer_alignment_mae: float
    original_gate_count: int
    expanded_gate_count: int
    gate_count_multiplier: int
    representation_levels: int
    representation_bits: int
    depth: int
    fanout_max: int
    unused_gate_ratio: float
    train_time: float
    eval_time: float


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


class KExpandedPopcountLayer(nn.Module):
    """Quantized K-popcount layer that can preserve multi-level outputs."""

    def __init__(
        self,
        source: LightLogicLayer,
        k: int,
        teacher_temperature: float,
        output_mode: str = "popcount",
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_dim = source.in_dim
        self.out_dim = source.out_dim
        self.k = int(k)
        self.teacher_temperature = float(teacher_temperature)
        self.output_mode = output_mode
        self.register_buffer("indices_0", source.indices_0.detach().cpu().clone().long())
        self.register_buffer("indices_1", source.indices_1.detach().cpu().clone().long())
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
        score = self.scores(x)
        if self.output_mode == "popcount":
            return score
        if self.output_mode == "threshold":
            threshold = self.threshold_value.to(device=x.device, dtype=x.dtype).view(1, -1)
            return (score >= threshold).to(dtype=x.dtype)
        raise ValueError(self.output_mode)

    def local_error(self) -> dict[str, float]:
        diff = self.q - self.qk
        return {
            "mae": float(diff.abs().mean().item()),
            "mse": float(diff.square().mean().item()),
            "max_abs_error": float(diff.abs().max().item()),
        }


def make_full_k_model(
    teacher: LightNet,
    num_classes: int,
    group_tau: float,
    k: int,
    threshold: float,
    teacher_temperature: float,
    final_readout: str,
) -> LightNet:
    layers: list[nn.Module] = []
    teacher_layers = list(teacher.layers)
    for layer_id, layer in enumerate(teacher_layers):
        if not isinstance(layer, LightLogicLayer):
            raise TypeError(f"Full K expansion expects LightLogicLayer, got {type(layer)}")
        is_last = layer_id == len(teacher_layers) - 1
        output_mode = "threshold" if final_readout == "final_threshold" and is_last else "popcount"
        layers.append(KExpandedPopcountLayer(layer, k, teacher_temperature, output_mode, threshold))
    return LightNet(layers, num_classes=num_classes, group_tau=group_tau)


def local_errors_full(model: LightNet) -> dict[str, float]:
    errors = [layer.local_error() for layer in model.layers if isinstance(layer, KExpandedPopcountLayer)]
    if not errors:
        return {"mae": math.nan, "mse": math.nan, "max_abs_error": math.nan}
    return {
        "mae": sum(item["mae"] for item in errors) / len(errors),
        "mse": sum(item["mse"] for item in errors) / len(errors),
        "max_abs_error": max(item["max_abs_error"] for item in errors),
    }


@torch.no_grad()
def layer_alignment(
    teacher: LightNet,
    candidate: LightNet,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    teacher_temperature: float,
) -> tuple[float, float]:
    teacher_layers = list(teacher.layers)
    candidate_layers = list(candidate.layers)
    if len(teacher_layers) != len(candidate_layers):
        raise ValueError((len(teacher_layers), len(candidate_layers)))
    total_abs = 0.0
    total_count = 0
    final_sum = 0.0
    final_count = 0
    was_teacher_training = teacher.training
    was_candidate_training = candidate.training
    teacher.eval()
    candidate.eval()
    for start in range(0, x.shape[0], batch_size):
        xt = x[start : start + batch_size].to(device)
        xc = xt
        for layer_id, (teacher_layer, candidate_layer) in enumerate(zip(teacher_layers, candidate_layers, strict=True)):
            if isinstance(teacher_layer, LightLogicLayer):
                xt = teacher_layer(xt, mode="continuous", temperature=teacher_temperature)
            else:
                xt = teacher_layer(xt)
            xc = candidate_layer(xc)
            diff = (xt - xc).abs()
            total_abs += float(diff.sum().detach().cpu().item())
            total_count += int(diff.numel())
            if layer_id == len(teacher_layers) - 1:
                final_sum += float(diff.sum().detach().cpu().item())
                final_count += int(diff.numel())
    if was_teacher_training:
        teacher.train()
    if was_candidate_training:
        candidate.train()
    return total_abs / max(total_count, 1), final_sum / max(final_count, 1)


@torch.no_grad()
def calibrate_final_threshold(
    teacher: LightNet,
    expanded: LightNet,
    x_cal: torch.Tensor,
    batch_size: int,
    device: torch.device,
    grid_size: int,
    teacher_temperature: float,
) -> None:
    layers = list(expanded.layers)
    if not layers or not isinstance(layers[-1], KExpandedPopcountLayer):
        return
    last = layers[-1]
    if last.output_mode != "threshold":
        return
    if grid_size < 2:
        raise ValueError("--calibration-grid-size must be >= 2")

    thresholds = torch.linspace(0.0, 1.0, grid_size, device=device)
    scores_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []
    teacher_layers = list(teacher.layers)
    for start in range(0, x_cal.shape[0], batch_size):
        xt = x_cal[start : start + batch_size].to(device)
        xc = xt
        for layer_id, (teacher_layer, candidate_layer) in enumerate(zip(teacher_layers, layers, strict=True)):
            if isinstance(teacher_layer, LightLogicLayer):
                xt = teacher_layer(xt, mode="continuous", temperature=teacher_temperature)
            else:
                xt = teacher_layer(xt)
            if layer_id == len(layers) - 1:
                scores_chunks.append(candidate_layer.scores(xc))
                targets_chunks.append((xt >= 0.5).to(xt.dtype))
                break
            xc = candidate_layer(xc)
    scores = torch.cat(scores_chunks, dim=0)
    targets = torch.cat(targets_chunks, dim=0)
    chosen = []
    for neuron in range(scores.shape[1]):
        losses = [((scores[:, neuron] >= t).to(targets.dtype) != targets[:, neuron]).float().mean() for t in thresholds]
        chosen.append(thresholds[int(torch.stack(losses).argmin().item())])
    last.set_thresholds(torch.stack(chosen).detach().cpu())


def run_dataset(dataset_name: str, seed: int, args: argparse.Namespace, device: torch.device) -> list[FullKRow]:
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

    rows: list[FullKRow] = []
    base_gates = original_gate_count(teacher)
    for k in args.k_values:
        eval_start = time.perf_counter()
        threshold_model = make_k_model(teacher, dataset.num_classes, args.group_tau, k, args.threshold, args.temp_eval).to(device)
        calibrate_thresholds(
            teacher,
            threshold_model,
            dataset.x_train,
            args.eval_batch_size,
            device,
            args.calibration_mode,
            args.calibration_grid_size,
            args.temp_eval,
        )
        threshold_acc, threshold_loss = evaluate(
            threshold_model,
            dataset.x_test,
            dataset.y_test,
            args.eval_batch_size,
            device,
            "hard",
            1.0,
        )
        threshold_err = local_errors(threshold_model)

        for final_readout in args.final_readouts:
            full_model = make_full_k_model(
                teacher,
                dataset.num_classes,
                args.group_tau,
                k,
                args.threshold,
                args.temp_eval,
                final_readout,
            ).to(device)
            if final_readout == "final_threshold" and args.calibration_mode != "fixed":
                calibrate_final_threshold(
                    teacher,
                    full_model,
                    dataset.x_train,
                    args.eval_batch_size,
                    device,
                    args.calibration_grid_size,
                    args.temp_eval,
                )
            candidate_acc, candidate_loss = evaluate(
                full_model,
                dataset.x_test,
                dataset.y_test,
                args.eval_batch_size,
                device,
                "hard",
                1.0,
            )
            align_mae, final_align_mae = layer_alignment(
                teacher,
                full_model,
                dataset.x_test,
                args.eval_batch_size,
                device,
                args.temp_eval,
            )
            full_err = local_errors_full(full_model)
            if not math.isfinite(full_err["mae"]):
                full_err = {
                    "mae": threshold_err["mae"],
                    "mse": threshold_err["mse"],
                    "max_abs_error": threshold_err["max_abs_error"],
                }
            rows.append(
                FullKRow(
                    dataset=dataset.name,
                    seed=seed,
                    k=int(k),
                    propagation="full_popcount",
                    final_readout=final_readout,
                    calibration_mode=args.calibration_mode,
                    teacher_acc=teacher_acc,
                    candidate_acc=candidate_acc,
                    acc_gap_vs_teacher=abs(teacher_acc - candidate_acc),
                    teacher_loss=teacher_loss,
                    candidate_loss=candidate_loss,
                    loss_gap_vs_teacher=abs(teacher_loss - candidate_loss),
                    threshold_each_layer_acc=threshold_acc,
                    threshold_each_layer_loss=threshold_loss,
                    delta_acc_vs_threshold_each_layer=candidate_acc - threshold_acc,
                    local_mae=full_err["mae"],
                    local_mse=full_err["mse"],
                    local_max_abs_error=full_err["max_abs_error"],
                    layer_alignment_mae=align_mae,
                    final_layer_alignment_mae=final_align_mae,
                    original_gate_count=base_gates,
                    expanded_gate_count=base_gates * int(k),
                    gate_count_multiplier=int(k),
                    representation_levels=int(k) + 1,
                    representation_bits=int(math.ceil(math.log2(int(k) + 1))),
                    depth=len(teacher.layers),
                    fanout_max=fanout_max(full_model.layers),
                    unused_gate_ratio=unused_gate_ratio(full_model, dataset.x_train, args.eval_batch_size, device),
                    train_time=train_time,
                    eval_time=time.perf_counter() - eval_start,
                )
            )
    return rows


def markdown_report(rows: list[FullKRow], args: argparse.Namespace) -> str:
    lines = [
        "# LightLogic Full K-Popcount Expansion Report",
        "",
        "This experiment keeps K-expanded popcount values as multi-level hidden wires instead of thresholding every layer back to one bit.",
        "",
        "Cost interpretation: `threshold_each_layer` is a pure 1-bit LGN replacement; `full_popcount` is a K-bit/popcount or local-LUT network with K+1 activation levels per hidden wire.",
        "",
        "| dataset | seed | K | final_readout | teacher_acc | full_acc | threshold_each_layer_acc | delta | gap_vs_teacher | layer_mae | final_mae | bits/wire | gates |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        data = asdict(row)
        lines.append(
            "| {dataset} | {seed} | {k} | {final_readout} | {teacher_acc:.6g} | {candidate_acc:.6g} | {threshold_each_layer_acc:.6g} | {delta_acc_vs_threshold_each_layer:.6g} | {acc_gap_vs_teacher:.6g} | {layer_alignment_mae:.6g} | {final_layer_alignment_mae:.6g} | {representation_bits} | {expanded_gate_count} |".format(
                **data
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `full_acc` evaluates the quantized multi-level network, not a pure one-bit hard network.",
            "- `final_readout=popcount` sends multi-level final layer outputs directly to GroupSum.",
            "- `final_readout=final_threshold` keeps hidden layers multi-level and thresholds only the last layer before GroupSum.",
            "- `threshold_each_layer_acc` is the old K-expanded path with a 1-bit conversion after every layer.",
            "- A positive `delta` isolates the benefit of preserving multi-level information across depth.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--final-readouts", nargs="+", choices=["popcount", "final_threshold"], default=["popcount", "final_threshold"])
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
    parser.add_argument("--out-dir", default="runs/lightlogic_full_k_expansion_v1")
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
        args.image_max_train = min(args.image_max_train, 1000)
        args.image_max_test = min(args.image_max_test, 300)
    if any(k <= 0 for k in args.k_values):
        raise ValueError("--k-values must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1]")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[FullKRow] = []
    started = time.perf_counter()
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            rows = run_dataset(dataset_name, seed, args, device)
            all_rows.extend(rows)
            for row in rows:
                print(
                    "dataset={} seed={} K={} readout={} teacher_acc={:.4f} full_acc={:.4f} threshold_acc={:.4f} delta={:+.4f} gap={:.4f}".format(
                        row.dataset,
                        row.seed,
                        row.k,
                        row.final_readout,
                        row.teacher_acc,
                        row.candidate_acc,
                        row.threshold_each_layer_acc,
                        row.delta_acc_vs_threshold_each_layer,
                        row.acc_gap_vs_teacher,
                    ),
                    flush=True,
                )
            write_csv(out_dir / "full_k_expansion_results.partial.csv", [asdict(row) for row in all_rows])

    write_csv(out_dir / "full_k_expansion_results.csv", [asdict(row) for row in all_rows])
    (out_dir / "full_k_expansion_summary.md").write_text(markdown_report(all_rows, args), encoding="utf-8")
    config = vars(args).copy()
    config["elapsed_seconds"] = time.perf_counter() - started
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'full_k_expansion_results.csv'}")


if __name__ == "__main__":
    main()
