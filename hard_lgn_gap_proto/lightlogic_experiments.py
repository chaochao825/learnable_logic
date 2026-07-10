#!/usr/bin/env python3
"""LightLogic-first baseline and hardening sweeps.

This runner is intentionally separate from hard_lgn_benchmark.py.  Its first
job is Goal 0 from the current research plan: train an input-wise
parametrized LightLogic network, discretize it by rounding the learned
truth-table outputs, and report the continuous-vs-discrete gap under the same
wiring, budget, and seeds.  Optional methods add annealing, entropy, and
straight-through/Gumbel training without changing the baseline semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hard_lgn_benchmark import (
    GATE_TRUTH,
    DatasetBundle,
    GroupSum,
    FrozenHardLogicLayer,
    SoftLogicLayer,
    fanout_max,
    gate_count,
    hard_gate,
    load_dataset,
    make_loader,
    random_connections,
    set_seed,
    weighted_gate,
)


@dataclass
class LightRow:
    method: str
    dataset: str
    seed: int
    estimator: str
    init: str
    continuous_acc: float
    discrete_acc: float
    acc_gap: float
    continuous_loss: float
    discrete_loss: float
    loss_gap: float
    gate_utilization: float
    unused_gate_ratio: float
    gate_count: int
    inference_gate_count: int
    parameter_count: int
    depth: int
    fanout_max: int
    train_time: float
    epochs_to_target: int
    time_to_target: float
    final_temperature: float
    entropy_weight: float
    train_forward_mode: str


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def temperature_at(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    t = epoch / float(epochs - 1)
    if start <= 0 or end <= 0:
        return end + (start - end) * (1.0 - t)
    return float(start * ((end / start) ** t))


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-8, 1 - 1e-8)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def inverse_estimator(values: torch.Tensor, estimator: str) -> torch.Tensor:
    values = values.clamp(1e-4, 1 - 1e-4)
    if estimator == "sigmoid":
        return torch.logit(values)
    if estimator == "sinusoidal":
        return torch.asin((2 * values - 1).clamp(-0.9999, 0.9999))
    raise ValueError(estimator)


def gate_truth_for_name(name: str) -> torch.Tensor:
    if name == "a":
        return torch.tensor([0.0, 0.0, 1.0, 1.0])
    if name == "b":
        return torch.tensor([0.0, 1.0, 0.0, 1.0])
    if name == "and":
        return torch.tensor([0.0, 0.0, 0.0, 1.0])
    if name == "or":
        return torch.tensor([0.0, 1.0, 1.0, 1.0])
    if name == "xor":
        return torch.tensor([0.0, 1.0, 1.0, 0.0])
    raise ValueError(name)


class LightLogicLayer(nn.Module):
    """Input-wise parametrized two-input logic layer.

    Each neuron has four bounded parameters corresponding to the truth-table
    outputs for input patterns 00, 01, 10, and 11.  The continuous forward is
    multilinear interpolation over soft inputs.  The hard forward rounds each
    truth-table output to 0/1 and evaluates the resulting Boolean gate.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        indices_0: torch.Tensor,
        indices_1: torch.Tensor,
        init_raw: torch.Tensor,
        estimator: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.estimator = estimator
        self.register_buffer("indices_0", indices_0.clone().long())
        self.register_buffer("indices_1", indices_1.clone().long())
        self.raw = nn.Parameter(init_raw.clone().float())

    def probs(self, temperature: float = 1.0) -> torch.Tensor:
        scaled = self.raw / max(float(temperature), 1e-6)
        if self.estimator == "sigmoid":
            return torch.sigmoid(scaled)
        if self.estimator == "sinusoidal":
            return 0.5 + 0.5 * torch.sin(scaled)
        raise ValueError(self.estimator)

    def rounded_truth(self) -> torch.Tensor:
        return (self.probs(1.0) >= 0.5).to(torch.float32)

    def op_ids(self) -> torch.Tensor:
        truth = self.rounded_truth().detach().cpu().to(torch.long)
        powers = torch.tensor([8, 4, 2, 1], dtype=torch.long)
        return (truth * powers.view(1, 4)).sum(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "continuous",
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
    ) -> torch.Tensor:
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]
        p = self.probs(temperature).to(dtype=x.dtype)
        if mode == "hard":
            p = (p >= 0.5).to(dtype=x.dtype)
        elif mode == "st":
            hard = (p >= 0.5).to(dtype=x.dtype)
            p = hard.detach() - p.detach() + p
        elif mode == "gumbel_st":
            logits = torch.stack([(1 - p).clamp_min(1e-8).log(), p.clamp_min(1e-8).log()], dim=-1)
            sample = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)[..., 1]
            p = sample.to(dtype=x.dtype)
        elif mode != "continuous":
            raise ValueError(mode)
        return (
            p[:, 0].view(1, -1) * (1 - a) * (1 - b)
            + p[:, 1].view(1, -1) * (1 - a) * b
            + p[:, 2].view(1, -1) * a * (1 - b)
            + p[:, 3].view(1, -1) * a * b
        )

    def entropy(self, temperature: float = 1.0) -> torch.Tensor:
        return bernoulli_entropy(self.probs(temperature))


class LightHardLayer(nn.Module):
    def __init__(self, source: LightLogicLayer) -> None:
        super().__init__()
        self.in_dim = source.in_dim
        self.out_dim = source.out_dim
        self.register_buffer("indices_0", source.indices_0.detach().cpu().clone().long())
        self.register_buffer("indices_1", source.indices_1.detach().cpu().clone().long())
        self.register_buffer("truth", source.rounded_truth().detach().cpu().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]
        p = self.truth.to(device=x.device, dtype=x.dtype)
        return (
            p[:, 0].view(1, -1) * (1 - a) * (1 - b)
            + p[:, 1].view(1, -1) * (1 - a) * b
            + p[:, 2].view(1, -1) * a * (1 - b)
            + p[:, 3].view(1, -1) * a * b
        )


class LightNet(nn.Module):
    def __init__(self, layers: Iterable[nn.Module], num_classes: int, group_tau: float = 1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        self.group_sum = GroupSum(num_classes, tau=group_tau)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "continuous",
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
    ) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, LightLogicLayer):
                x = layer(x, mode=mode, temperature=temperature, gumbel_tau=gumbel_tau)
            elif isinstance(layer, SoftLogicLayer):
                layer_mode = {
                    "continuous": "soft",
                    "hard": "hard",
                    "st": "gumbel",
                    "gumbel_st": "gumbel",
                }[mode]
                x = layer(x, mode=layer_mode, tau=temperature, gumbel_hard=mode in {"st", "gumbel_st"})
            else:
                x = layer(x)
        return self.group_sum(x)

    def light_layers(self) -> list[LightLogicLayer]:
        return [layer for layer in self.layers if isinstance(layer, LightLogicLayer)]

    def soft_layers(self) -> list[SoftLogicLayer]:
        return [layer for layer in self.layers if isinstance(layer, SoftLogicLayer)]


def init_light_raw(
    out_dim: int,
    generator: torch.Generator,
    estimator: str,
    init: str,
    strength: float,
) -> torch.Tensor:
    if init == "random":
        return torch.randn(out_dim, 4, generator=generator) * 0.1
    if init == "uniform":
        return inverse_estimator(torch.full((out_dim, 4), 0.5), estimator)

    templates = []
    if init == "residual":
        templates = [gate_truth_for_name("a"), gate_truth_for_name("b")]
    elif init == "and_or":
        templates = [gate_truth_for_name("and"), gate_truth_for_name("or")]
    elif init == "xor":
        templates = [gate_truth_for_name("xor")]
    else:
        raise ValueError(init)

    choice = torch.randint(len(templates), (out_dim,), generator=generator)
    truth = torch.stack([templates[int(i)] for i in choice], dim=0)
    lo = max(1e-4, min(0.49, 1.0 - strength))
    hi = min(1.0 - 1e-4, max(0.51, strength))
    target = torch.where(truth > 0.5, torch.full_like(truth, hi), torch.full_like(truth, lo))
    return inverse_estimator(target, estimator)


def build_light_architecture(
    input_dim: int,
    width: int,
    depth: int,
    seed: int,
    estimator: str,
    init: str,
    init_strength: float,
) -> list[dict[str, torch.Tensor | int | str]]:
    generator = torch.Generator().manual_seed(seed)
    arch: list[dict[str, torch.Tensor | int | str]] = []
    in_dim = input_dim
    for _ in range(depth):
        idx0, idx1 = random_connections(in_dim, width, generator)
        raw = init_light_raw(width, generator, estimator, init, init_strength)
        op_logits = torch.randn(width, 16, generator=generator) * 0.1
        arch.append({"in_dim": in_dim, "out_dim": width, "idx0": idx0, "idx1": idx1, "raw": raw, "op_logits": op_logits})
        in_dim = width
    return arch


def make_light_net(
    arch: list[dict[str, torch.Tensor | int | str]],
    num_classes: int,
    group_tau: float,
    estimator: str,
    parametrization: str,
) -> LightNet:
    layers: list[nn.Module] = []
    for spec in arch:
        if parametrization == "iwp":
            layers.append(
                LightLogicLayer(
                    int(spec["in_dim"]),
                    int(spec["out_dim"]),
                    spec["idx0"],  # type: ignore[arg-type]
                    spec["idx1"],  # type: ignore[arg-type]
                    spec["raw"],  # type: ignore[arg-type]
                    estimator=estimator,
                )
            )
        elif parametrization == "op":
            layers.append(
                SoftLogicLayer(
                    int(spec["in_dim"]),
                    int(spec["out_dim"]),
                    spec["idx0"],  # type: ignore[arg-type]
                    spec["idx1"],  # type: ignore[arg-type]
                    spec["op_logits"],  # type: ignore[arg-type]
                )
            )
        else:
            raise ValueError(parametrization)
    return LightNet(layers, num_classes=num_classes, group_tau=group_tau)


def make_hard_net(model: LightNet, num_classes: int, group_tau: float) -> LightNet:
    hard_layers: list[nn.Module] = []
    for layer in model.layers:
        if isinstance(layer, LightLogicLayer):
            hard_layers.append(LightHardLayer(layer))
        elif isinstance(layer, SoftLogicLayer):
            hard_layers.append(
                FrozenHardLogicLayer(
                    layer.in_dim,
                    layer.out_dim,
                    layer.indices_0.detach().cpu(),
                    layer.indices_1.detach().cpu(),
                    layer.hard_ops_argmax(),
                )
            )
        else:
            raise TypeError(type(layer))
    return LightNet(hard_layers, num_classes=num_classes, group_tau=group_tau)


@torch.no_grad()
def evaluate(
    model: LightNet,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    temperature: float = 1.0,
    gumbel_tau: float = 1.0,
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


@torch.no_grad()
def unused_gate_ratio(model: LightNet, x: torch.Tensor, batch_size: int, device: torch.device) -> float:
    layers = list(model.layers)
    inactive = 0
    total = 0
    for layer_id, layer in enumerate(layers):
        del layer_id
    outputs = [torch.empty((x.shape[0], int(layer.out_dim)), dtype=torch.float32) for layer in layers]  # type: ignore[attr-defined]
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        for i, layer in enumerate(layers):
            if isinstance(layer, LightLogicLayer):
                xb = layer(xb, mode="hard")
            elif isinstance(layer, SoftLogicLayer):
                xb = layer(xb, mode="hard")
            else:
                xb = layer(xb)
            outputs[i][start : start + xb.shape[0]] = xb.detach().cpu()
    for y in outputs:
        total += y.shape[1]
        inactive += int(((y.max(dim=0).values - y.min(dim=0).values) < 1e-6).sum().item())
    return inactive / max(total, 1)


def parameter_count(model: LightNet) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def method_config(method: str) -> tuple[str, str]:
    configs = {
        "light_iwp": ("iwp", "continuous"),
        "light_iwp_anneal": ("iwp", "continuous"),
        "light_iwp_st": ("iwp", "st"),
        "light_iwp_gumbel_st": ("iwp", "gumbel_st"),
        "dlgn_op": ("op", "continuous"),
        "dlgn_op_anneal": ("op", "continuous"),
        "dlgn_op_gumbel_st": ("op", "gumbel_st"),
    }
    if method not in configs:
        raise ValueError(f"Unknown method: {method}")
    return configs[method]


def train_one(
    method: str,
    dataset: DatasetBundle,
    arch: list[dict[str, torch.Tensor | int | str]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[LightRow, list[dict[str, object]]]:
    parametrization, train_forward_mode = method_config(method)
    model = make_light_net(arch, dataset.num_classes, args.group_tau, args.estimator, parametrization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed + 10001)
    target = args.target_acc_override if args.target_acc_override is not None else dataset.target_acc
    epoch_rows: list[dict[str, object]] = []
    epochs_to_target = -1
    time_to_target = -1.0
    start = time.perf_counter()

    for epoch in range(args.epochs):
        if method.endswith("_anneal") or args.force_anneal:
            temp = temperature_at(epoch, args.epochs, args.temp_start, args.temp_end)
            entropy_weight = args.entropy_weight
        else:
            temp = args.temp_eval
            entropy_weight = 0.0
        gumbel_tau = temperature_at(epoch, args.epochs, args.gumbel_temp_start, args.gumbel_temp_end)
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mode=train_forward_mode, temperature=temp, gumbel_tau=gumbel_tau)
            loss = F.cross_entropy(logits, yb)
            if entropy_weight:
                ent_terms = [layer.entropy(temp) for layer in model.light_layers()]
                ent_terms.extend(layer.entropy(temp) for layer in model.soft_layers())
                if ent_terms:
                    loss = loss + entropy_weight * torch.stack(ent_terms).mean()
            loss.backward()
            optimizer.step()

        cont_acc, cont_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "continuous", args.temp_eval)
        hard_model = make_hard_net(model, dataset.num_classes, args.group_tau).to(device)
        disc_acc, disc_loss = evaluate(hard_model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
        elapsed = time.perf_counter() - start
        if epochs_to_target < 0 and disc_acc >= target:
            epochs_to_target = epoch + 1
            time_to_target = elapsed
        epoch_rows.append(
            {
                "method": method,
                "dataset": dataset.name,
                "seed": seed,
                "epoch": epoch + 1,
                "temperature": temp,
                "gumbel_tau": gumbel_tau,
                "continuous_acc": cont_acc,
                "discrete_acc": disc_acc,
                "acc_gap": abs(cont_acc - disc_acc),
                "continuous_loss": cont_loss,
                "discrete_loss": disc_loss,
                "elapsed_time": elapsed,
            }
        )

    train_time = time.perf_counter() - start
    cont_acc, cont_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "continuous", args.temp_eval)
    hard_model = make_hard_net(model, dataset.num_classes, args.group_tau).to(device)
    disc_acc, disc_loss = evaluate(hard_model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
    hard_unused = unused_gate_ratio(hard_model, dataset.x_train, args.eval_batch_size, device)
    row = LightRow(
        method=method,
        dataset=dataset.name,
        seed=seed,
        estimator=args.estimator,
        init=args.init,
        continuous_acc=cont_acc,
        discrete_acc=disc_acc,
        acc_gap=abs(cont_acc - disc_acc),
        continuous_loss=cont_loss,
        discrete_loss=disc_loss,
        loss_gap=abs(cont_loss - disc_loss),
        gate_utilization=1.0 - hard_unused,
        unused_gate_ratio=hard_unused,
        gate_count=gate_count(hard_model.layers),
        inference_gate_count=gate_count(hard_model.layers),
        parameter_count=parameter_count(model),
        depth=len(model.layers),
        fanout_max=fanout_max(hard_model.layers),
        train_time=train_time,
        epochs_to_target=epochs_to_target,
        time_to_target=time_to_target,
        final_temperature=args.temp_eval,
        entropy_weight=args.entropy_weight if method.endswith("_anneal") or args.force_anneal else 0.0,
        train_forward_mode=train_forward_mode,
    )
    return row, epoch_rows


def markdown_report(rows: list[LightRow], args: argparse.Namespace, elapsed: float) -> str:
    lines = [
        "# LightLogic Baseline Report",
        "",
        "This report is LightLogic-first. Goal 0 rows use input-wise parametrization and round truth-table outputs for discrete inference.",
        "",
        f"- datasets: {', '.join(args.datasets)}",
        f"- methods: {', '.join(args.methods)}",
        f"- seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- width/depth/epochs: {args.width}/{args.layers}/{args.epochs}",
        f"- elapsed_seconds: {elapsed:.6g}",
        "",
        "| method | dataset | seed | continuous_acc | discrete_acc | acc_gap | gate_utilization | gates | params | train_time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {dataset} | {seed} | {continuous_acc:.6g} | {discrete_acc:.6g} | {acc_gap:.6g} | {gate_utilization:.6g} | {gate_count} | {parameter_count} | {train_time:.6g} |".format(
                **asdict(row)
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `continuous_acc` evaluates the trained relaxed model.",
            "- `discrete_acc` evaluates the rounded hard network under the same fixed wiring.",
            "- `acc_gap = abs(continuous_acc - discrete_acc)`.",
            "- `gate_utilization = 1 - unused_gate_ratio`, measured from hard layer activity on the training set.",
        ]
    )
    return "\n".join(lines) + "\n"


def accuracy_threshold(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("accuracy threshold must be in [0,1]")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument("--methods", nargs="+", default=["light_iwp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--estimator", choices=["sigmoid", "sinusoidal"], default="sinusoidal")
    parser.add_argument("--init", choices=["residual", "and_or", "xor", "random", "uniform"], default="residual")
    parser.add_argument("--init-strength", type=float, default=0.98)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.1)
    parser.add_argument("--temp-eval", type=float, default=1.0)
    parser.add_argument("--gumbel-temp-start", type=float, default=1.5)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--force-anneal", action="store_true")
    parser.add_argument("--target-acc-override", type=accuracy_threshold)
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/lightlogic_goal0_v1")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets = ["parity6", "majority7", "random_sparse8"]
        args.seeds = [0]
        args.width = min(args.width, 32)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 12)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[LightRow] = []
    epoch_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            dataset = load_dataset(dataset_name, seed, args)
            if args.width % dataset.num_classes != 0:
                raise ValueError(f"width={args.width} must be divisible by num_classes={dataset.num_classes}")
            if args.width * 2 < dataset.input_dim:
                raise ValueError(f"width={args.width} too small for input_dim={dataset.input_dim}")
            arch = build_light_architecture(
                dataset.input_dim,
                args.width,
                args.layers,
                seed,
                args.estimator,
                args.init,
                args.init_strength,
            )
            print(f"dataset={dataset.name} seed={seed} train={len(dataset.y_train)} test={len(dataset.y_test)}")
            for method in args.methods:
                print(f"  method={method}", flush=True)
                row, epochs = train_one(method, dataset, arch, args, device, seed)
                rows.append(row)
                epoch_rows.extend(epochs)
                print(
                    "    continuous_acc={:.4f} discrete_acc={:.4f} gap={:.4f} util={:.4f} params={} time={:.2f}s".format(
                        row.continuous_acc,
                        row.discrete_acc,
                        row.acc_gap,
                        row.gate_utilization,
                        row.parameter_count,
                        row.train_time,
                    ),
                    flush=True,
                )
                write_csv(out_dir / "lightlogic_results.partial.csv", [asdict(r) for r in rows])
                write_csv(out_dir / "lightlogic_per_epoch.partial.csv", epoch_rows)
    elapsed = time.perf_counter() - started
    write_csv(out_dir / "lightlogic_results.csv", [asdict(r) for r in rows])
    write_csv(out_dir / "lightlogic_per_epoch.csv", epoch_rows)
    (out_dir / "lightlogic_summary.md").write_text(markdown_report(rows, args, elapsed), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'lightlogic_results.csv'}")
    print(f"wrote {out_dir / 'lightlogic_summary.md'}")


if __name__ == "__main__":
    main()
