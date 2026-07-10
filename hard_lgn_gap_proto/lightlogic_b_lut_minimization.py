#!/usr/bin/env python3
"""Train real b-input LightLogic LUT networks and minimize them with ABC.

This closes the main remaining Goal 7/8 boundary by training a network whose
local modules are directly b-input truth tables, not post-hoc cones extracted
from a two-input network. The runner then exports the hard discrete network to
BLIF, minimizes it with ABC, and compares continuous, raw discrete, source
BLIF, and optimized BLIF behavior plus structural cost.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from evaluate_abc_blif import evaluate_blif
from hard_lgn_benchmark import GroupSum, fanout_max, load_dataset, make_loader, set_seed
from lut_minimization import (
    LutMinimizer,
    blif_names_for_lut,
    pattern_for_index,
    raw_lut_estimate,
    run_abc_blif,
    run_self_test,
    write_csv,
)


DEFAULT_B_VALUES = [3, 4]


def to_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def temperature_at(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    t = epoch / float(epochs - 1)
    if start <= 0 or end <= 0:
        return end + (start - end) * (1.0 - t)
    return float(start * ((end / start) ** t))


def inverse_estimator(values: torch.Tensor, estimator: str) -> torch.Tensor:
    values = values.clamp(1e-4, 1 - 1e-4)
    if estimator == "sigmoid":
        return torch.logit(values)
    if estimator == "sinusoidal":
        return torch.asin((2 * values - 1).clamp(-0.9999, 0.9999))
    raise ValueError(estimator)


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-8, 1 - 1e-8)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def truth_patterns(b: int) -> torch.Tensor:
    return torch.tensor(
        [[float(bit) for bit in pattern_for_index(idx, b)] for idx in range(1 << b)],
        dtype=torch.float32,
    )


def projection_truth(patterns: torch.Tensor, input_id: int) -> torch.Tensor:
    return patterns[:, input_id].clone()


def and_truth(patterns: torch.Tensor) -> torch.Tensor:
    return patterns.prod(dim=1)


def or_truth(patterns: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - patterns).prod(dim=1)


def xor_truth(patterns: torch.Tensor) -> torch.Tensor:
    return (patterns.sum(dim=1).remainder(2.0) > 0.5).to(torch.float32)


def init_truth_templates(b: int, init: str, generator: torch.Generator, out_dim: int) -> torch.Tensor:
    patterns = truth_patterns(b)
    if init == "random":
        return torch.rand(out_dim, 1 << b, generator=generator)
    if init == "uniform":
        return torch.full((out_dim, 1 << b), 0.5)
    if init == "residual":
        input_choices = torch.randint(b, (out_dim,), generator=generator)
        return torch.stack([projection_truth(patterns, int(i)) for i in input_choices.tolist()], dim=0)
    if init == "and_or":
        choices = torch.randint(2, (out_dim,), generator=generator)
        and_vec = and_truth(patterns)
        or_vec = or_truth(patterns)
        return torch.stack([and_vec if int(i) == 0 else or_vec for i in choices.tolist()], dim=0)
    if init == "xor":
        parity = xor_truth(patterns)
        return parity.view(1, -1).repeat(out_dim, 1)
    raise ValueError(init)


def init_lut_raw(
    b: int,
    out_dim: int,
    generator: torch.Generator,
    estimator: str,
    init: str,
    strength: float,
) -> torch.Tensor:
    if init == "random":
        return torch.randn(out_dim, 1 << b, generator=generator) * 0.1
    target = init_truth_templates(b, init, generator, out_dim)
    lo = max(1e-4, min(0.49, 1.0 - strength))
    hi = min(1.0 - 1e-4, max(0.51, strength))
    probs = torch.where(target > 0.5, torch.full_like(target, hi), torch.full_like(target, lo))
    return inverse_estimator(probs, estimator)


def random_k_connections(in_dim: int, out_dim: int, arity: int, generator: torch.Generator) -> torch.Tensor:
    if out_dim * arity < in_dim:
        raise ValueError(f"out_dim={out_dim} arity={arity} cannot cover in_dim={in_dim}")
    c = torch.randperm(out_dim * arity, generator=generator) % in_dim
    c = torch.randperm(in_dim, generator=generator)[c]
    return c.reshape(arity, out_dim).transpose(0, 1).contiguous().long()


class BInputLutLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        arity: int,
        input_indices: torch.Tensor,
        init_raw: torch.Tensor,
        estimator: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if input_indices.shape != (out_dim, arity):
            raise ValueError((input_indices.shape, out_dim, arity))
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.arity = int(arity)
        self.estimator = estimator
        self.register_buffer("input_indices", input_indices.clone().long())
        self.register_buffer("pattern_bits", truth_patterns(arity))
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

    def entropy(self, temperature: float = 1.0) -> torch.Tensor:
        return bernoulli_entropy(self.probs(temperature))

    def truth_bits(self, gate_id: int) -> str:
        truth = self.rounded_truth().detach().cpu()
        return "".join("1" if float(truth[gate_id, idx].item()) >= 0.5 else "0" for idx in range(truth.shape[1]))

    def _active_probs(self, mode: str, temperature: float, gumbel_tau: float, dtype: torch.dtype) -> torch.Tensor:
        p = self.probs(temperature).to(dtype=dtype)
        if mode == "hard":
            return (p >= 0.5).to(dtype=dtype)
        if mode == "st":
            hard = (p >= 0.5).to(dtype=dtype)
            return hard.detach() - p.detach() + p
        if mode == "gumbel_st":
            logits = torch.stack([(1 - p).clamp_min(1e-8).log(), p.clamp_min(1e-8).log()], dim=-1)
            return F.gumbel_softmax(logits, tau=gumbel_tau, hard=True, dim=-1)[..., 1].to(dtype=dtype)
        if mode == "continuous":
            return p
        raise ValueError(mode)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "continuous",
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
    ) -> torch.Tensor:
        gathered = x[..., self.input_indices]
        patterns = self.pattern_bits.to(device=x.device, dtype=x.dtype).view(1, 1, 1 << self.arity, self.arity)
        terms = torch.where(patterns > 0.5, gathered.unsqueeze(-2), 1.0 - gathered.unsqueeze(-2))
        basis = terms.prod(dim=-1)
        probs = self._active_probs(mode, temperature, gumbel_tau, x.dtype)
        return (basis * probs.unsqueeze(0)).sum(dim=-1)


class BInputLutNet(nn.Module):
    def __init__(self, layers: list[BInputLutLayer], num_classes: int, group_tau: float = 1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(num_classes, tau=group_tau)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "continuous",
        temperature: float = 1.0,
        gumbel_tau: float = 1.0,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mode=mode, temperature=temperature, gumbel_tau=gumbel_tau)
        return self.group_sum(x)


def effective_width(input_dim: int, num_classes: int, arity: int, base_width: int) -> int:
    width = max(int(base_width), math.ceil(input_dim / max(arity, 1)))
    if width % num_classes != 0:
        width = math.ceil(width / num_classes) * num_classes
    if width * arity < input_dim:
        width = math.ceil(input_dim / arity / num_classes) * num_classes
    return width


def effective_two_input_width(input_dim: int, num_classes: int, base_width: int) -> int:
    width = max(int(base_width), math.ceil(input_dim / 2))
    if width % num_classes != 0:
        width = math.ceil(width / num_classes) * num_classes
    if width * 2 < input_dim:
        width = math.ceil(input_dim / 2 / num_classes) * num_classes
    return width


def build_architecture(
    input_dim: int,
    width: int,
    depth: int,
    arity: int,
    seed: int,
    estimator: str,
    init: str,
    init_strength: float,
) -> list[dict[str, torch.Tensor | int | str]]:
    generator = torch.Generator().manual_seed(seed)
    arch: list[dict[str, torch.Tensor | int | str]] = []
    in_dim = input_dim
    for _ in range(depth):
        indices = random_k_connections(in_dim, width, arity, generator)
        raw = init_lut_raw(arity, width, generator, estimator, init, init_strength)
        arch.append({"in_dim": in_dim, "out_dim": width, "arity": arity, "indices": indices, "raw": raw})
        in_dim = width
    return arch


def make_model(
    arch: list[dict[str, torch.Tensor | int | str]],
    num_classes: int,
    group_tau: float,
    estimator: str,
) -> BInputLutNet:
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
    return BInputLutNet(layers, num_classes=num_classes, group_tau=group_tau)


@torch.no_grad()
def evaluate_model(
    model: BInputLutNet,
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
def unused_gate_ratio_model(model: BInputLutNet, x: torch.Tensor, batch_size: int, device: torch.device) -> float:
    layers = list(model.layers)
    outputs = [torch.empty((x.shape[0], int(layer.out_dim)), dtype=torch.float32) for layer in layers]
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        for layer_id, layer in enumerate(layers):
            xb = layer(xb, mode="hard")
            outputs[layer_id][start : start + xb.shape[0]] = xb.detach().cpu()
    total = 0
    inactive = 0
    for values in outputs:
        total += values.shape[1]
        inactive += int(((values.max(dim=0).values - values.min(dim=0).values) < 1e-6).sum().item())
    return inactive / max(total, 1)


def fanout_max_model(layers: list[BInputLutLayer]) -> int:
    max_fanout = 0
    for layer in layers:
        counts = torch.bincount(layer.input_indices.detach().cpu().reshape(-1), minlength=layer.in_dim)
        max_fanout = max(max_fanout, int(counts.max().item()))
    return max_fanout


def parameter_count_model(model: nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def safe_stem(dataset: str, seed: int, b: int) -> str:
    return f"{dataset}__seed{seed}__b{b}"


def layer_truth_bits(layer: BInputLutLayer, gate_id: int) -> str:
    return layer.truth_bits(gate_id)


def collect_lut_instances(
    model: BInputLutNet,
    dataset_name: str,
    seed: int,
    b: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_id, layer in enumerate(model.layers):
        indices = layer.input_indices.detach().cpu().tolist()
        for gate_id in range(layer.out_dim):
            truth_bits = layer_truth_bits(layer, gate_id)
            estimate = raw_lut_estimate(b, truth_bits)
            row = {
                "dataset": dataset_name,
                "seed": seed,
                "b": b,
                "layer": layer_id,
                "gate": gate_id,
                "truth_bits": truth_bits,
                "raw_lut_bits": estimate.raw_lut_bits,
                "raw_mux2_count": estimate.raw_mux2_count,
                "raw_sop_literals": estimate.raw_sop_literals,
                "raw_sop_not_estimate": estimate.raw_sop_not_estimate,
                "on_set_size": estimate.on_set_size,
            }
            for input_id, input_index in enumerate(indices[gate_id]):
                row[f"input_{input_id}"] = int(input_index)
            rows.append(row)
    return rows


def write_network_blif(model: BInputLutNet, input_dim: int, path: Path, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_names = [f"i{i}" for i in range(input_dim)]
    outputs = [f"l{len(model.layers) - 1}_g{idx}" for idx in range(model.layers[-1].out_dim)] if model.layers else prev_names
    lines = [
        f".model {model_name}",
        ".inputs " + " ".join(prev_names),
        ".outputs " + " ".join(outputs),
    ]
    for layer_id, layer in enumerate(model.layers):
        next_names = []
        indices = layer.input_indices.detach().cpu().tolist()
        for gate_id in range(layer.out_dim):
            output_name = f"l{layer_id}_g{gate_id}"
            next_names.append(output_name)
            input_names = [prev_names[int(idx)] for idx in indices[gate_id]]
            lines.extend(blif_names_for_lut(input_names, output_name, layer_truth_bits(layer, gate_id)))
        prev_names = next_names
    lines.append(".end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_blif_safe(path: Path, dataset, group_tau: float) -> tuple[str, float, float, dict[str, object], str]:
    try:
        acc, loss, stats = evaluate_blif(path, dataset.x_test, dataset.y_test, dataset.num_classes, group_tau)
        return "ok", acc, loss, dict(stats), ""
    except Exception as exc:  # pragma: no cover - operational path.
        return "exception", math.nan, math.nan, {}, repr(exc)


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["b"])), []).append(row)
    for key in sorted(grouped):
        dataset, b = key
        group = grouped[key]
        summary.append(
            {
                "dataset": dataset,
                "b": b,
                "rows": len(group),
                "continuous_acc_mean": sum(float(row["continuous_acc"]) for row in group) / len(group),
                "discrete_acc_mean": sum(float(row["discrete_acc"]) for row in group) / len(group),
                "minimized_blif_acc_mean": sum(float(row["minimized_blif_acc"]) for row in group) / len(group),
                "abc_and_count_mean": sum(float(row["abc_and_count"]) for row in group) / len(group),
                "abc_level_mean": sum(float(row["abc_level"]) for row in group) / len(group),
                "total_abc_seconds_mean": sum(float(row["total_abc_seconds"]) for row in group) / len(group),
                "raw_lut_gate_estimate_mean": sum(float(row["raw_lut_gate_estimate"]) for row in group) / len(group),
                "best_seed": max(group, key=lambda item: float(item["discrete_acc"]))["seed"],
                "best_discrete_acc": max(float(row["discrete_acc"]) for row in group),
                "best_minimized_blif_acc": max(float(row["minimized_blif_acc"]) for row in group),
                "best_abc_and_count": min(float(row["abc_and_count"]) for row in group),
            }
        )
    return summary


def write_markdown_report(path: Path, rows: list[dict[str, object]], summary_rows: list[dict[str, object]], unique_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    ok_unique = sum(
        1
        for row in unique_rows
        if row.get("abc_status") == "ok"
        and str(row.get("source_equivalent")) == "True"
        and str(row.get("equivalent")) == "True"
    )
    accuracy_preserved = sum(1 for row in rows if bool(row.get("accuracy_preserved")))
    lines = [
        "# Real b-Input LUT Network Minimization Report",
        "",
        "This run trains true b-input local LUT networks, then exports and minimizes the hard discrete network with ABC.",
        "",
        f"- datasets: {', '.join(args.datasets)}",
        f"- seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- b values: {', '.join(str(b) for b in args.b_values)}",
        f"- rows: {len(rows)}",
        f"- unique LUTs equivalent: {ok_unique}/{len(unique_rows)}",
        f"- network rows with preserved source/optimized accuracy at tolerance {args.accuracy_tolerance}: {accuracy_preserved}/{len(rows)}",
        "",
        "## Summary By Dataset/B",
        "",
        "| dataset | b | rows | continuous_acc_mean | discrete_acc_mean | minimized_blif_acc_mean | abc_and_count_mean | abc_level_mean | total_abc_seconds_mean | raw_lut_gate_estimate_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {b} | {rows} | {continuous_acc_mean:.6g} | {discrete_acc_mean:.6g} | {minimized_blif_acc_mean:.6g} | {abc_and_count_mean:.6g} | {abc_level_mean:.6g} | {total_abc_seconds_mean:.6g} | {raw_lut_gate_estimate_mean:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## All Rows",
            "",
            "| dataset | seed | b | width | continuous_acc | discrete_acc | minimized_blif_acc | raw_lut_gate_estimate | abc_and_count | abc_level | total_abc_seconds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["seed"]),
                    str(row["b"]),
                    str(row["width"]),
                    f"{to_float(row['continuous_acc']):.6g}",
                    f"{to_float(row['discrete_acc']):.6g}",
                    f"{to_float(row['minimized_blif_acc']):.6g}",
                    str(row["raw_lut_gate_estimate"]),
                    str(row["abc_and_count"]),
                    str(row["abc_level"]),
                    f"{to_float(row['total_abc_seconds']):.6g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `raw_lut_gate_estimate` is the sum of `2^b - 1` mux-tree estimates over all trained local LUTs.",
            "- `abc_and_count` and `abc_level` come from ABC after `read_blif; strash; dc2; print_stats`.",
            "- `xag_*` counts come from a canonical local ANF/XAG decomposition of each LUT truth table.",
            "- `discrete_acc` is the rounded truth-table network accuracy before BLIF export.",
            "- `minimized_blif_acc` is the post-ABC optimized BLIF accuracy on the same test split.",
            "- `xor_count` is the local XAG XOR count per LUT, not a global network-level XAG optimization count.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset_b(
    dataset_name: str,
    seed: int,
    b: int,
    args: argparse.Namespace,
    device: torch.device,
    minimizer: LutMinimizer,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    dataset = load_dataset(dataset_name, seed, args)
    teacher = None
    teacher_acc = math.nan
    teacher_loss = math.nan
    teacher_train_time = 0.0
    if args.distill_weight > 0.0:
        from lightlogic_k_expansion import train_teacher
        from lightlogic_experiments import evaluate as evaluate_teacher_model

        teacher_args = deepcopy(args)
        teacher_base_width = args.teacher_width if args.teacher_width > 0 else args.width
        teacher_args.width = effective_two_input_width(dataset.input_dim, dataset.num_classes, teacher_base_width)
        teacher_args.layers = args.teacher_layers if args.teacher_layers > 0 else args.layers
        teacher_args.init = args.teacher_init
        teacher_args.init_strength = args.teacher_init_strength
        teacher_args.epochs = args.teacher_epochs if args.teacher_epochs > 0 else args.epochs
        teacher, teacher_train_time = train_teacher(dataset, teacher_args, device, seed)
        teacher_acc, teacher_loss = evaluate_teacher_model(
            teacher,
            dataset.x_test,
            dataset.y_test,
            args.eval_batch_size,
            device,
            "continuous",
            args.teacher_temp_eval,
        )
    width = effective_width(dataset.input_dim, dataset.num_classes, b, args.width)
    arch = build_architecture(
        dataset.input_dim,
        width,
        args.layers,
        b,
        seed,
        args.estimator,
        args.init,
        args.init_strength,
    )
    model = make_model(arch, dataset.num_classes, args.group_tau, args.estimator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed + 31001 + b * 101)

    started = time.perf_counter()
    for epoch in range(args.epochs):
        temp = temperature_at(epoch, args.epochs, args.temp_start, args.temp_end) if args.anneal_train else args.temp_eval
        gumbel_tau = temperature_at(epoch, args.epochs, args.gumbel_temp_start, args.gumbel_temp_end)
        train_mode = args.warmup_mode if epoch < args.warmup_epochs else args.train_forward_mode
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mode=train_mode, temperature=temp, gumbel_tau=gumbel_tau)
            loss = F.cross_entropy(logits, yb)
            if teacher is not None and args.distill_weight > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(xb, mode="continuous", temperature=args.teacher_temp_eval)
                tau = max(float(args.distill_tau), 1e-6)
                student_log_probs = F.log_softmax(logits / tau, dim=1)
                teacher_probs = F.softmax(teacher_logits / tau, dim=1)
                distill = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (tau * tau)
                loss = loss + args.distill_weight * distill
            if args.entropy_weight:
                ent = torch.stack([layer.entropy(temp) for layer in model.layers]).mean()
                loss = loss + args.entropy_weight * ent
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - started

    continuous_acc, continuous_loss = evaluate_model(
        model,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "continuous",
        args.temp_eval,
        args.gumbel_temp_end,
    )
    discrete_acc, discrete_loss = evaluate_model(
        model,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "hard",
        1.0,
        args.gumbel_temp_end,
    )

    instance_rows = collect_lut_instances(model, dataset.name, seed, b)
    per_instance_abc_seconds_estimate = 0.0
    unique_abc_seconds = 0.0
    per_instance_abc_and = 0
    per_instance_abc_level_sum = 0
    per_instance_xag_and = 0
    per_instance_xag_xor = 0
    per_instance_xag_not = 0
    per_instance_xag_level_sum = 0
    all_unique_equivalent = True
    for row in instance_rows:
        cache_key = (int(row["b"]), str(row["truth_bits"]))
        was_cached = cache_key in minimizer.cache
        result = minimizer.minimize(int(row["b"]), str(row["truth_bits"]))
        if not was_cached:
            unique_abc_seconds += result.abc_seconds
        row.update(
            {
                "truth_hash": result.truth_hash,
                "abc_status": result.abc_status,
                "abc_and_count": result.abc_and_count,
                "abc_level": result.abc_level,
                "abc_seconds": result.abc_seconds,
                "source_equivalent": result.source_equivalent,
                "equivalent": result.equivalent,
                "xag_and_count": result.xag_and_count,
                "xor_count": result.xor_count,
                "xag_not_count": result.xag_not_count,
                "xag_level_estimate": result.xag_level_estimate,
                "xag_backend": result.xag_backend,
                "xor_count_status": result.xor_count_status,
                "unique_raw_blif_path": result.raw_blif_path,
                "unique_optimized_blif_path": result.optimized_blif_path,
            }
        )
        per_instance_abc_seconds_estimate += result.abc_seconds
        per_instance_abc_and += result.abc_and_count
        per_instance_abc_level_sum += result.abc_level
        per_instance_xag_and += result.xag_and_count
        per_instance_xag_xor += result.xor_count
        per_instance_xag_not += result.xag_not_count
        per_instance_xag_level_sum += result.xag_level_estimate
        all_unique_equivalent = all_unique_equivalent and result.source_equivalent and result.equivalent

    stem = safe_stem(dataset.name, seed, b)
    raw_network_blif = Path(args.out_dir) / "blif" / "networks" / f"{stem}.raw.blif"
    optimized_network_blif = Path(args.out_dir) / "blif" / "networks" / f"{stem}.abc_optimized.blif"
    network_log = Path(args.out_dir) / "abc_logs" / "networks" / f"{stem}.abc.log"
    write_network_blif(model, dataset.input_dim, raw_network_blif, stem)
    network_abc = run_abc_blif(Path(args.abc_path), raw_network_blif, optimized_network_blif, network_log)
    source_status, source_acc, source_loss, source_stats, source_note = evaluate_blif_safe(raw_network_blif, dataset, args.group_tau)
    if network_abc.abc_status == "ok":
        opt_status, opt_acc, opt_loss, opt_stats, opt_note = evaluate_blif_safe(optimized_network_blif, dataset, args.group_tau)
    else:
        opt_status, opt_acc, opt_loss, opt_stats, opt_note = network_abc.abc_status, math.nan, math.nan, {}, network_abc.abc_error

    raw_lut_gate_estimate = sum(int(row["raw_mux2_count"]) for row in instance_rows)
    raw_lut_bits_total = sum(int(row["raw_lut_bits"]) for row in instance_rows)
    raw_sop_literals_total = sum(int(row["raw_sop_literals"]) for row in instance_rows)
    raw_sop_not_total = sum(int(row["raw_sop_not_estimate"]) for row in instance_rows)
    source_delta = source_acc - discrete_acc if math.isfinite(source_acc) else math.nan
    opt_delta_disc = opt_acc - discrete_acc if math.isfinite(opt_acc) else math.nan
    opt_delta_source = opt_acc - source_acc if math.isfinite(opt_acc) and math.isfinite(source_acc) else math.nan
    accuracy_preserved = (
        source_status == "ok"
        and opt_status == "ok"
        and math.isfinite(source_delta)
        and math.isfinite(opt_delta_source)
        and abs(source_delta) <= args.accuracy_tolerance
        and abs(opt_delta_source) <= args.accuracy_tolerance
    )
    network_row = {
        "dataset": dataset.name,
        "seed": seed,
        "b": b,
        "width": width,
        "layers": args.layers,
        "teacher_acc": teacher_acc,
        "teacher_loss": teacher_loss,
        "teacher_train_time": teacher_train_time,
        "continuous_acc": continuous_acc,
        "continuous_loss": continuous_loss,
        "discrete_acc": discrete_acc,
        "discrete_loss": discrete_loss,
        "discrete_acc_gap": abs(continuous_acc - discrete_acc),
        "discrete_loss_gap": abs(continuous_loss - discrete_loss),
        "source_blif_status": source_status,
        "source_blif_acc": source_acc,
        "source_blif_loss": source_loss,
        "source_acc_delta_vs_discrete": source_delta,
        "minimized_blif_status": opt_status,
        "minimized_blif_acc": opt_acc,
        "minimized_blif_loss": opt_loss,
        "accuracy_delta": opt_delta_disc,
        "minimized_acc_delta_vs_source_blif": opt_delta_source,
        "accuracy_preserved": bool(accuracy_preserved),
        "lut_instance_count": len(instance_rows),
        "unique_lut_count_seen_so_far": len(minimizer.cache),
        "raw_lut_gate_estimate": raw_lut_gate_estimate,
        "raw_lut_bits_total": raw_lut_bits_total,
        "raw_sop_literals_total": raw_sop_literals_total,
        "raw_sop_not_estimate_total": raw_sop_not_total,
        "raw_lut_depth_estimate": len(model.layers) * b,
        "per_lut_abc_and_count_estimate": per_instance_abc_and,
        "per_lut_abc_level_sum_estimate": per_instance_abc_level_sum,
        "per_lut_xag_and_count_estimate": per_instance_xag_and,
        "per_lut_xag_xor_count_estimate": per_instance_xag_xor,
        "per_lut_xag_not_count_estimate": per_instance_xag_not,
        "per_lut_xag_level_sum_estimate": per_instance_xag_level_sum,
        "per_lut_abc_seconds_uncached_estimate": per_instance_abc_seconds_estimate,
        "unique_lut_abc_seconds_total": unique_abc_seconds,
        "all_unique_luts_equivalent": bool(all_unique_equivalent),
        "abc_status": network_abc.abc_status,
        "abc_returncode": network_abc.abc_returncode,
        "abc_and_count": network_abc.abc_and_count,
        "abc_level": network_abc.abc_level,
        "abc_node_count": network_abc.abc_node_count,
        "abc_edge_count": network_abc.abc_edge_count,
        "abc_cube_count": network_abc.abc_cube_count,
        "abc_stat_lines": network_abc.abc_stat_lines,
        "total_abc_seconds": network_abc.abc_seconds,
        "source_blif_fanout_max": source_stats.get("fanout_max", ""),
        "optimized_blif_fanout_max": opt_stats.get("fanout_max", ""),
        "source_blif_unused_node_ratio": source_stats.get("unused_node_ratio", ""),
        "optimized_blif_unused_node_ratio": opt_stats.get("unused_node_ratio", ""),
        "raw_network_blif": str(raw_network_blif),
        "optimized_network_blif": str(optimized_network_blif),
        "network_abc_log": str(network_log),
        "train_time": train_time,
        "parameter_count": parameter_count_model(model),
        "depth": len(model.layers),
        "fanout_max": fanout_max_model(list(model.layers)),
        "unused_gate_ratio": unused_gate_ratio_model(model, dataset.x_train, args.eval_batch_size, device),
        "train_forward_mode": args.train_forward_mode,
        "distill_weight": args.distill_weight,
        "distill_tau": args.distill_tau,
        "warmup_epochs": args.warmup_epochs,
        "warmup_mode": args.warmup_mode,
        "estimator": args.estimator,
        "init": args.init,
        "note": "; ".join(item for item in [source_note, opt_note] if item),
    }
    return network_row, instance_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["abc"], default="abc")
    parser.add_argument("--abc-path", default="/home/spco/boolean_sat/abc/abc")
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10", "digits"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--b-values", nargs="+", type=int, default=DEFAULT_B_VALUES)
    parser.add_argument("--accuracy-tolerance", type=float, default=1e-9)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
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
    parser.add_argument("--gumbel-temp-start", type=float, default=1.5)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.3)
    parser.add_argument("--anneal-train", action="store_true")
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--train-forward-mode", choices=["continuous", "st", "gumbel_st"], default="st")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-mode", choices=["continuous", "st", "gumbel_st"], default="continuous")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-tau", type=float, default=1.0)
    parser.add_argument("--teacher-width", type=int, default=0)
    parser.add_argument("--teacher-layers", type=int, default=0)
    parser.add_argument("--teacher-epochs", type=int, default=0)
    parser.add_argument("--teacher-init", choices=["residual", "and_or", "xor", "random", "uniform"], default="residual")
    parser.add_argument("--teacher-init-strength", type=float, default=0.98)
    parser.add_argument("--teacher-temp-eval", type=float, default=1.0)
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/lightlogic_b_lut_min_goal7_v1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--self-test-first", action="store_true", default=True)
    parser.add_argument("--no-self-test-first", action="store_false", dest="self_test_first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets = ["parity6", "majority7", "random_sparse8"]
        args.seeds = [0]
        if args.b_values == DEFAULT_B_VALUES:
            args.b_values = [3]
        args.width = min(args.width, 48)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 12)
    if any(b <= 2 for b in args.b_values):
        raise ValueError("--b-values must all be > 2 for this runner")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.self_test_first:
        self_test_rows = run_self_test(Path(args.abc_path), out_dir / "self_test")
        if not all(bool(row["self_test_pass"]) for row in self_test_rows):
            raise RuntimeError("LUT minimization self-test failed")

    minimizer = LutMinimizer(Path(args.abc_path), out_dir)
    network_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            for b in args.b_values:
                print(f"dataset={dataset_name} seed={seed} b={b}", flush=True)
                network_row, rows = run_dataset_b(dataset_name, seed, b, args, device, minimizer)
                network_rows.append(network_row)
                instance_rows.extend(rows)
                write_csv(out_dir / "network_minimization_results.partial.csv", network_rows)
                write_csv(out_dir / "lut_instances.partial.csv", instance_rows)
                write_csv(out_dir / "unique_lut_minimization.partial.csv", [vars(item) for item in minimizer.cache.values()])
                print(
                    "  width={} cont_acc={:.4f} disc_acc={:.4f} abc_acc={:.4f} abc_and={} abc_level={} abc_status={}".format(
                        network_row["width"],
                        network_row["continuous_acc"],
                        network_row["discrete_acc"],
                        network_row["minimized_blif_acc"],
                        network_row["abc_and_count"],
                        network_row["abc_level"],
                        network_row["abc_status"],
                    ),
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    for row in network_rows:
        row["elapsed_total_seconds"] = elapsed
    unique_rows = [vars(item) for item in minimizer.cache.values()]
    summary_rows = summarize_rows(network_rows)
    write_csv(out_dir / "network_minimization_results.csv", network_rows)
    write_csv(out_dir / "lut_instances.csv", instance_rows)
    write_csv(out_dir / "unique_lut_minimization.csv", unique_rows)
    write_csv(out_dir / "summary_by_dataset_b.csv", summary_rows)
    write_markdown_report(out_dir / "b_lut_minimization_summary.md", network_rows, summary_rows, unique_rows, args)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "network_minimization_results.csv")
    print(out_dir / "summary_by_dataset_b.csv")
    print(out_dir / "b_lut_minimization_summary.md")


if __name__ == "__main__":
    main()
