#!/usr/bin/env python3
"""Distill a K-expanded LightLogic student from a continuous teacher.

This is the Goal 6/6+ prototype.  The baseline keeps each expanded gate's
K-popcount truth table fixed and learns per-neuron thresholds with a soft
surrogate:

    soft_bit = sigmoid((popcount_over_K - threshold) / student_temp)

Goal 6+ can additionally learn a quantized truth-table adapter.  In
`truth_table_st` mode, the forward pass uses the rounded deployable r/K truth
table while the backward pass uses a straight-through gradient through the
continuous truth-table probabilities.

Hard evaluation still uses the deployable thresholded student.  The script
also reports a Goal 5 diagnostic: under unconstrained K hard truth-table
expansion, data-weighted local rounding has the same optimum as uniform
rounding whenever each truth-table entry can choose its own r_ab.
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

from hard_lgn_benchmark import DatasetBundle, GroupSum, fanout_max, load_dataset, make_loader, set_seed
from lightlogic_experiments import LightLogicLayer, LightNet, evaluate, temperature_at
from lightlogic_k_expansion import build_light_architecture, make_light_net, original_gate_count, train_teacher


@dataclass
class DistillRow:
    dataset: str
    seed: int
    k: int
    init_mode: str
    adapter_mode: str
    alpha: float
    distill_tau: float
    student_temp: float
    teacher_acc: float
    teacher_loss: float
    initial_hard_acc: float
    initial_hard_loss: float
    distilled_soft_acc: float
    distilled_soft_loss: float
    distilled_hard_acc: float
    distilled_hard_loss: float
    hard_acc_delta_vs_initial: float
    hard_gap_vs_teacher: float
    initial_gap_vs_teacher: float
    local_mae: float
    weighted_local_mse: float
    uniform_local_mse: float
    weighted_rounding_changed_ratio: float
    final_local_mae: float
    final_weighted_local_mse: float
    final_uniform_local_mse: float
    truth_table_changed_ratio: float
    truth_table_l1_from_initial: float
    gate_count: int
    expanded_gate_count: int
    depth: int
    fanout_max: int
    train_time: float
    distill_time: float


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
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


def logit_clamped(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-4, 1 - 1e-4))


class DistillKLayer(nn.Module):
    def __init__(
        self,
        source: LightLogicLayer,
        k: int,
        threshold: float,
        teacher_temperature: float,
        adapter_mode: str,
        adapter_init: str,
    ) -> None:
        super().__init__()
        self.in_dim = source.in_dim
        self.out_dim = source.out_dim
        self.k = int(k)
        self.teacher_temperature = float(teacher_temperature)
        self.adapter_mode = adapter_mode
        self.adapter_init = adapter_init
        self.register_buffer("indices_0", source.indices_0.detach().cpu().clone().long())
        self.register_buffer("indices_1", source.indices_1.detach().cpu().clone().long())
        q = source.probs(self.teacher_temperature).detach().cpu().clamp(0.0, 1.0)
        r = torch.floor(q * self.k + 0.5).clamp(0, self.k)
        qk = r / float(self.k)
        self.register_buffer("q", q)
        self.register_buffer("qk", qk)
        self.threshold_raw = nn.Parameter(logit_clamped(torch.full((source.out_dim,), float(threshold))))
        if adapter_mode in {"truth_table", "truth_table_st"}:
            if adapter_init == "rounded":
                prob_init = qk
            elif adapter_init == "teacher":
                prob_init = q
            elif adapter_init == "bin_center":
                # Keep the initial rounded r/K table unchanged, but avoid
                # saturated 0/1 logits so the adapter can cross a quantization
                # boundary when the task loss supports it.
                lower = (r - 0.5).clamp_min(0.0) / float(self.k)
                upper = (r + 0.5).clamp_max(float(self.k)) / float(self.k)
                prob_init = (lower + upper) * 0.5
            else:
                raise ValueError(adapter_init)
            self.prob_raw = nn.Parameter(logit_clamped(prob_init))
        elif adapter_mode == "none":
            self.prob_raw = None
        else:
            raise ValueError(adapter_mode)

    def thresholds(self) -> torch.Tensor:
        return torch.sigmoid(self.threshold_raw)

    def set_thresholds(self, thresholds: torch.Tensor | float) -> None:
        if isinstance(thresholds, float):
            value = torch.full((self.out_dim,), thresholds, dtype=torch.float32)
        else:
            value = thresholds.detach().cpu().float().reshape(-1)
            if value.numel() == 1:
                value = value.repeat(self.out_dim)
            if value.numel() != self.out_dim:
                raise ValueError((value.shape, self.out_dim))
        with torch.no_grad():
            self.threshold_raw.copy_(logit_clamped(value))

    def continuous_probs(self) -> torch.Tensor:
        if self.prob_raw is None:
            return self.qk
        return torch.sigmoid(self.prob_raw)

    def hard_probs(self) -> torch.Tensor:
        if self.prob_raw is None:
            return self.qk
        return torch.floor(self.continuous_probs() * self.k + 0.5).clamp(0, self.k) / float(self.k)

    def active_probs(self, mode: str) -> torch.Tensor:
        if self.prob_raw is None:
            return self.qk
        continuous = self.continuous_probs()
        hard = torch.floor(continuous * self.k + 0.5).clamp(0, self.k) / float(self.k)
        if mode == "hard":
            return hard
        if self.adapter_mode == "truth_table":
            return continuous
        if self.adapter_mode == "truth_table_st":
            return hard.detach() - continuous.detach() + continuous
        raise ValueError(self.adapter_mode)

    def scores(self, x: torch.Tensor, mode: str = "hard") -> torch.Tensor:
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]
        p = self.active_probs(mode).to(device=x.device, dtype=x.dtype)
        return (
            p[:, 0].view(1, -1) * (1 - a) * (1 - b)
            + p[:, 1].view(1, -1) * (1 - a) * b
            + p[:, 2].view(1, -1) * a * (1 - b)
            + p[:, 3].view(1, -1) * a * b
        )

    def forward(self, x: torch.Tensor, mode: str = "soft", student_temp: float = 0.1) -> torch.Tensor:
        threshold = self.thresholds().to(device=x.device, dtype=x.dtype).view(1, -1)
        score = self.scores(x, mode="hard" if mode == "hard" else "soft")
        if mode == "hard":
            return (score >= threshold).to(dtype=x.dtype)
        if mode == "soft":
            return torch.sigmoid((score - threshold) / max(float(student_temp), 1e-6))
        raise ValueError(mode)

    def local_stats(self, rho: torch.Tensor | None = None) -> dict[str, float]:
        qk_active = self.hard_probs().detach().to(device=self.q.device, dtype=self.q.dtype)
        diff = self.q - qk_active
        uniform_mse = float(diff.square().mean().item())
        if rho is None:
            weighted_mse = uniform_mse
        else:
            weights = rho.to(device=diff.device, dtype=diff.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            weighted_mse = float(((diff.square() * weights).sum(dim=1, keepdim=True) / denom).mean().item())
        changed = data_weighted_rounding_changed_ratio(self.q, self.k, rho)
        return {
            "mae": float(diff.abs().mean().item()),
            "uniform_mse": uniform_mse,
            "weighted_mse": weighted_mse,
            "weighted_changed_ratio": changed,
        }

    def adapter_anchor_loss(self) -> torch.Tensor:
        if self.prob_raw is None:
            return self.q.new_tensor(0.0)
        return (self.continuous_probs() - self.q.to(device=self.prob_raw.device, dtype=self.prob_raw.dtype)).square().mean()

    def adapter_stats(self) -> dict[str, float]:
        hard = self.hard_probs().detach().to(device=self.qk.device, dtype=self.qk.dtype)
        diff = hard - self.qk
        return {
            "changed_ratio": float((diff.abs() > 1e-8).float().mean().item()),
            "l1_from_initial": float(diff.abs().mean().item()),
        }


class DistillKNet(nn.Module):
    def __init__(self, layers: list[DistillKLayer], num_classes: int, group_tau: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(num_classes, tau=group_tau)

    def forward(self, x: torch.Tensor, mode: str = "soft", student_temp: float = 0.1) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mode=mode, student_temp=student_temp)
        return self.group_sum(x)

    def adapter_anchor_loss(self) -> torch.Tensor:
        losses = [layer.adapter_anchor_loss() for layer in self.layers]
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def adapter_stats(self) -> dict[str, float]:
        stats = [layer.adapter_stats() for layer in self.layers]
        if not stats:
            return {"changed_ratio": 0.0, "l1_from_initial": 0.0}
        return {
            "changed_ratio": sum(item["changed_ratio"] for item in stats) / len(stats),
            "l1_from_initial": sum(item["l1_from_initial"] for item in stats) / len(stats),
        }


def make_student(
    teacher: LightNet,
    num_classes: int,
    group_tau: float,
    k: int,
    threshold: float,
    temp_eval: float,
    adapter_mode: str,
    adapter_init: str,
) -> DistillKNet:
    layers = []
    for layer in teacher.layers:
        if not isinstance(layer, LightLogicLayer):
            raise TypeError(type(layer))
        layers.append(DistillKLayer(layer, k, threshold, temp_eval, adapter_mode, adapter_init))
    return DistillKNet(layers, num_classes=num_classes, group_tau=group_tau)


@torch.no_grad()
def eval_student(
    model: DistillKNet,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    student_temp: float,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits = model(xb, mode=mode, student_temp=student_temp)
        loss_sum += float(F.cross_entropy(logits, yb, reduction="sum").detach().cpu().item())
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    if was_training:
        model.train()
    return correct / max(total, 1), loss_sum / max(total, 1)


@torch.no_grad()
def calibrate_thresholds(
    teacher: LightNet,
    student: DistillKNet,
    x_cal: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    grid_size: int,
    temp_eval: float,
) -> None:
    if mode == "fixed":
        return
    thresholds = torch.linspace(0.0, 1.0, grid_size, device=device)
    current_chunks = [x_cal[start : start + batch_size].to(device) for start in range(0, x_cal.shape[0], batch_size)]
    teacher_layers = teacher.light_layers()
    for teacher_layer, student_layer in zip(teacher_layers, student.layers, strict=True):
        scores = []
        targets = []
        for xb in current_chunks:
            score = student_layer.scores(xb)
            target = (teacher_layer(xb, mode="continuous", temperature=temp_eval) >= 0.5).to(score.dtype)
            scores.append(score)
            targets.append(target)
        score_all = torch.cat(scores, dim=0)
        target_all = torch.cat(targets, dim=0)
        if mode == "per_layer":
            losses = [((score_all >= t).to(target_all.dtype) != target_all).float().mean() for t in thresholds]
            student_layer.set_thresholds(float(thresholds[int(torch.stack(losses).argmin().item())].item()))
        elif mode == "per_neuron":
            chosen = []
            for neuron in range(score_all.shape[1]):
                losses = [
                    ((score_all[:, neuron] >= t).to(target_all.dtype) != target_all[:, neuron]).float().mean()
                    for t in thresholds
                ]
                chosen.append(thresholds[int(torch.stack(losses).argmin().item())])
            student_layer.set_thresholds(torch.stack(chosen).detach().cpu())
        else:
            raise ValueError(mode)
        current_chunks = [student_layer(xb, mode="hard") for xb in current_chunks]


@torch.no_grad()
def pattern_rho_for_layer(layer: DistillKLayer, x: torch.Tensor) -> torch.Tensor:
    a = x[:, layer.indices_0]
    b = x[:, layer.indices_1]
    idx = (a >= 0.5).long() * 2 + (b >= 0.5).long()
    rho = torch.zeros((layer.out_dim, 4), dtype=torch.float32, device=x.device)
    for pattern in range(4):
        rho[:, pattern] = (idx == pattern).float().mean(dim=0)
    return rho.detach().cpu()


@torch.no_grad()
def student_local_stats(student: DistillKNet, x_cal: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float]:
    current = x_cal.to(device)
    stats = []
    for layer in student.layers:
        rho = pattern_rho_for_layer(layer, current)
        stats.append(layer.local_stats(rho))
        outputs = []
        for start in range(0, current.shape[0], batch_size):
            outputs.append(layer(current[start : start + batch_size], mode="hard"))
        current = torch.cat(outputs, dim=0)
    return {
        "mae": sum(item["mae"] for item in stats) / len(stats),
        "uniform_mse": sum(item["uniform_mse"] for item in stats) / len(stats),
        "weighted_mse": sum(item["weighted_mse"] for item in stats) / len(stats),
        "weighted_changed_ratio": sum(item["weighted_changed_ratio"] for item in stats) / len(stats),
    }


def data_weighted_rounding_changed_ratio(q: torch.Tensor, k: int, rho: torch.Tensor | None) -> float:
    if rho is None:
        return 0.0
    rho = rho.to(device=q.device, dtype=q.dtype)
    uniform_r = torch.floor(q * k + 0.5).clamp(0, k)
    # For independent r_ab entries, positive rho only rescales the same
    # squared error objective.  Zero-rho entries are ties; keep uniform_r for
    # deterministic comparison.
    best_r = uniform_r.clone()
    for value in range(k + 1):
        candidate = torch.full_like(q, float(value))
        candidate_error = rho * (q - candidate / float(k)).square()
        best_error = rho * (q - best_r / float(k)).square()
        improve = candidate_error < best_error - 1e-12
        best_r = torch.where(improve, candidate, best_r)
    return float((best_r != uniform_r).float().mean().item())


def distill_student(
    teacher: LightNet,
    student: DistillKNet,
    dataset: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    alpha: float,
    distill_tau: float,
) -> float:
    optimizer = torch.optim.Adam(student.parameters(), lr=args.student_lr, weight_decay=args.student_weight_decay)
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed + 30001)
    started = time.perf_counter()
    teacher.eval()
    for _epoch in range(args.distill_epochs):
        student.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                teacher_logits = teacher(xb, mode="continuous", temperature=args.temp_eval)
            student_logits = student(xb, mode="soft", student_temp=args.student_temp)
            ce = F.cross_entropy(student_logits, yb)
            kl = F.kl_div(
                F.log_softmax(student_logits / distill_tau, dim=1),
                F.softmax(teacher_logits / distill_tau, dim=1),
                reduction="batchmean",
            ) * (distill_tau**2)
            loss = ce + alpha * kl
            if args.adapter_anchor_weight > 0:
                loss = loss + args.adapter_anchor_weight * student.adapter_anchor_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def run_dataset(dataset_name: str, seed: int, args: argparse.Namespace, device: torch.device) -> list[DistillRow]:
    dataset = load_dataset(dataset_name, seed, args)
    teacher, teacher_train_time = train_teacher(dataset, args, device, seed)
    teacher_acc, teacher_loss = evaluate(
        teacher,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "continuous",
        args.temp_eval,
    )
    rows: list[DistillRow] = []
    base_gates = original_gate_count(teacher)
    for k in args.k_values:
        for init_mode in args.init_modes:
            for adapter_mode in args.adapter_modes:
                for alpha in args.alphas:
                    for distill_tau in args.distill_taus:
                        student = make_student(
                            teacher,
                            dataset.num_classes,
                            args.group_tau,
                            k,
                            args.threshold,
                            args.temp_eval,
                            adapter_mode,
                            args.adapter_init,
                        ).to(device)
                        calibrate_thresholds(
                            teacher,
                            student,
                            dataset.x_train,
                            args.eval_batch_size,
                            device,
                            init_mode,
                            args.calibration_grid_size,
                            args.temp_eval,
                        )
                        initial_hard_acc, initial_hard_loss = eval_student(
                            student, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", args.student_temp
                        )
                        local = student_local_stats(student, dataset.x_train, args.eval_batch_size, device)
                        distill_time = distill_student(teacher, student, dataset, args, device, seed, alpha, distill_tau)
                        soft_acc, soft_loss = eval_student(
                            student, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "soft", args.student_temp
                        )
                        hard_acc, hard_loss = eval_student(
                            student, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", args.student_temp
                        )
                        final_local = student_local_stats(student, dataset.x_train, args.eval_batch_size, device)
                        adapter_stats = student.adapter_stats()
                        rows.append(
                            DistillRow(
                                dataset=dataset.name,
                                seed=seed,
                                k=int(k),
                                init_mode=init_mode,
                                adapter_mode=adapter_mode,
                                alpha=float(alpha),
                                distill_tau=float(distill_tau),
                                student_temp=float(args.student_temp),
                                teacher_acc=teacher_acc,
                                teacher_loss=teacher_loss,
                                initial_hard_acc=initial_hard_acc,
                                initial_hard_loss=initial_hard_loss,
                                distilled_soft_acc=soft_acc,
                                distilled_soft_loss=soft_loss,
                                distilled_hard_acc=hard_acc,
                                distilled_hard_loss=hard_loss,
                                hard_acc_delta_vs_initial=hard_acc - initial_hard_acc,
                                hard_gap_vs_teacher=abs(teacher_acc - hard_acc),
                                initial_gap_vs_teacher=abs(teacher_acc - initial_hard_acc),
                                local_mae=local["mae"],
                                weighted_local_mse=local["weighted_mse"],
                                uniform_local_mse=local["uniform_mse"],
                                weighted_rounding_changed_ratio=local["weighted_changed_ratio"],
                                final_local_mae=final_local["mae"],
                                final_weighted_local_mse=final_local["weighted_mse"],
                                final_uniform_local_mse=final_local["uniform_mse"],
                                truth_table_changed_ratio=adapter_stats["changed_ratio"],
                                truth_table_l1_from_initial=adapter_stats["l1_from_initial"],
                                gate_count=base_gates,
                                expanded_gate_count=base_gates * int(k),
                                depth=len(teacher.layers),
                                fanout_max=fanout_max(student.layers),
                                train_time=teacher_train_time,
                                distill_time=distill_time,
                            )
                        )
    return rows


def markdown_report(rows: list[DistillRow], args: argparse.Namespace) -> str:
    best = sorted(rows, key=lambda row: (-row.distilled_hard_acc, row.dataset, row.k))
    lines = [
        "# LightLogic Distilled K-Expansion Report",
        "",
        "Goal 6: learn student thresholds by CE + teacher-logit KL distillation.",
        "",
        "| dataset | seed | K | init | adapter | alpha | tau | teacher_acc | initial_hard_acc | distilled_hard_acc | delta | hard_gap | tt_changed | weighted_changed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in best:
        lines.append(
            "| {dataset} | {seed} | {k} | {init_mode} | {adapter_mode} | {alpha:.6g} | {distill_tau:.6g} | {teacher_acc:.6g} | {initial_hard_acc:.6g} | {distilled_hard_acc:.6g} | {hard_acc_delta_vs_initial:.6g} | {hard_gap_vs_teacher:.6g} | {truth_table_changed_ratio:.6g} | {weighted_rounding_changed_ratio:.6g} |".format(
                **asdict(row)
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `weighted_rounding_changed_ratio` is expected to be zero for unconstrained per-entry K truth-table expansion; this is the Goal 5 degeneracy check.",
            "- Hard accuracy is still measured with thresholded deployable student outputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["majority9", "random_sparse10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--init-modes", nargs="+", choices=["fixed", "per_layer", "per_neuron"], default=["fixed", "per_neuron"])
    parser.add_argument("--adapter-modes", nargs="+", choices=["none", "truth_table", "truth_table_st"], default=["none"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 2.0])
    parser.add_argument("--distill-taus", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--distill-epochs", type=int, default=80)
    parser.add_argument("--student-lr", type=float, default=0.05)
    parser.add_argument("--student-weight-decay", type=float, default=0.0)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--adapter-anchor-weight", type=float, default=0.0)
    parser.add_argument("--adapter-init", choices=["bin_center", "rounded", "teacher"], default="bin_center")
    parser.add_argument("--threshold", type=float, default=0.5)
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
    parser.add_argument("--out-dir", default="runs/lightlogic_distill_k_v1")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets = ["majority7", "random_sparse8"]
        args.k_values = [2]
        args.init_modes = ["fixed", "per_neuron"]
        args.adapter_modes = ["none", "truth_table_st"]
        args.alphas = [0.0, 1.0]
        args.distill_taus = [1.0]
        args.width = min(args.width, 32)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 12)
        args.distill_epochs = min(args.distill_epochs, 10)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[DistillRow] = []
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            dataset_rows = run_dataset(dataset_name, seed, args, device)
            rows.extend(dataset_rows)
            for row in dataset_rows:
                print(
                    "dataset={} K={} init={} adapter={} alpha={} tau={} initial={:.4f} distilled={:.4f} delta={:.4f} tt_changed={:.4f}".format(
                        row.dataset,
                        row.k,
                        row.init_mode,
                        row.adapter_mode,
                        row.alpha,
                        row.distill_tau,
                        row.initial_hard_acc,
                        row.distilled_hard_acc,
                        row.hard_acc_delta_vs_initial,
                        row.truth_table_changed_ratio,
                    ),
                    flush=True,
                )
            write_csv(out_dir / "distill_results.partial.csv", [asdict(row) for row in rows])
    write_csv(out_dir / "distill_results.csv", [asdict(row) for row in rows])
    (out_dir / "distill_summary.md").write_text(markdown_report(rows, args), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'distill_results.csv'}")


if __name__ == "__main__":
    main()
