#!/usr/bin/env python3
"""Redundant-gate LGN training plus task-aware hardening.

This implements the first Part C slice from ``next_stage_protocol.md``. The
primary objective is deployable hard-path accuracy. Candidate hard networks are
selected on validation hard accuracy first within each method's eligible
candidate subset, then hard loss / refit / cost.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from hard_lgn_benchmark import (
    DatasetBundle,
    FrozenHardLogicLayer,
    GroupSum,
    SoftLogicLayer,
    apply_layers,
    compute_unused_gate_ratio,
    fanout_max,
    gate_count,
    load_dataset,
    make_loader,
    random_connections,
    refit_truth_table_layer,
    set_seed,
    weighted_gate,
)


RESULT_FIELDS = [
    "method",
    "dataset",
    "seed",
    "redundancy_factor",
    "train_estimator",
    "hardening",
    "checkpoint_source",
    "hard_acc",
    "soft_acc",
    "full_gap",
    "native_gap",
    "hard_loss",
    "soft_loss",
    "train_time",
    "hardening_time",
    "train_valid",
    "invalid_reason",
    "best_soft_val_loss",
    "best_soft_ckpt_hard_val_acc",
    "best_hard_ckpt_hard_val_acc",
    "hard_ckpt_beats_soft_ckpt",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "final_votes_per_class",
    "source_note",
]

CANDIDATE_FIELDS = [
    "method",
    "dataset",
    "seed",
    "redundancy_factor",
    "train_estimator",
    "eligible_for_method",
    "selection_policy",
    "candidate",
    "checkpoint_source",
    "val_hard_acc",
    "val_hard_loss",
    "val_soft_acc",
    "val_soft_loss",
    "val_full_gap",
    "refit_error",
    "refit_delta",
    "op_change_ratio",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "score",
    "selected",
]

CHECKPOINT_FIELDS = [
    "method",
    "dataset",
    "seed",
    "redundancy_factor",
    "train_estimator",
    "epoch",
    "val_soft_acc",
    "val_soft_loss",
    "val_native_acc",
    "val_native_loss",
    "val_hard_acc",
    "val_hard_loss",
    "hard_gap",
    "native_gap",
    "train_valid",
    "invalid_reason",
]


@dataclass
class TrainOutcome:
    model: "VoteLogicNet"
    best_soft_state: dict[str, torch.Tensor]
    best_hard_state: dict[str, torch.Tensor]
    best_soft_val_loss: float
    best_soft_ckpt_hard_val_acc: float
    best_hard_ckpt_hard_val_acc: float
    train_valid: bool
    invalid_reason: str
    train_time: float
    checkpoint_rows: list[dict[str, object]]


class VoteLogicNet(nn.Module):
    def __init__(
        self,
        layers: list[nn.Module],
        num_classes: int,
        votes_per_class: int,
        group_tau: float = 1.0,
        redundancy_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_classes = num_classes
        self.votes_per_class = votes_per_class
        self.output_dim = num_classes * votes_per_class
        self.group_sum = GroupSum(num_classes, tau=group_tau)
        self.redundancy_dropout = redundancy_dropout

    def forward(self, x: torch.Tensor, mode: str = "soft", tau: float = 1.0, return_bits: bool = False):
        h = x
        for layer in self.layers:
            if isinstance(layer, SoftLogicLayer):
                if mode == "argmax_st":
                    probs = F.softmax(layer.logits / tau, dim=-1)
                    hard = F.one_hot(probs.argmax(dim=-1), probs.shape[-1]).to(dtype=probs.dtype)
                    weights = hard - probs.detach() + probs
                    a = h[..., layer.indices_0]
                    b = h[..., layer.indices_1]
                    h = weighted_gate(a, b, weights.to(dtype=h.dtype))
                elif mode == "gumbel_st":
                    h = layer(h, mode="gumbel", tau=tau, gumbel_hard=True)
                else:
                    h = layer(h, mode=mode, tau=tau)
            elif isinstance(layer, FrozenHardLogicLayer):
                h = layer(h)
            else:
                raise TypeError(type(layer))
        bits = h[:, : self.output_dim]
        vote_bits = bits
        if self.training and self.redundancy_dropout > 0.0:
            keep = 1.0 - self.redundancy_dropout
            mask = torch.bernoulli(torch.full_like(vote_bits, keep))
            vote_bits = vote_bits * mask / max(keep, 1e-6)
        logits = self.group_sum(vote_bits)
        if return_bits:
            return logits, bits
        return logits


def build_layers(
    input_dim: int,
    num_classes: int,
    base_width: int,
    base_votes: int,
    depth: int,
    redundancy_factor: int,
    seed: int,
) -> tuple[list[SoftLogicLayer], int]:
    if redundancy_factor < 1:
        raise ValueError("redundancy_factor must be >= 1")
    if depth < 1:
        raise ValueError("--layers must be >= 1")
    generator = torch.Generator().manual_seed(seed)
    hidden_width = base_width * redundancy_factor
    votes = base_votes * redundancy_factor
    final_out = num_classes * votes
    layers: list[SoftLogicLayer] = []
    in_dim = input_dim
    for layer_id in range(depth):
        out_dim = final_out if layer_id == depth - 1 else hidden_width
        if out_dim * 2 < in_dim:
            raise ValueError(f"layer {layer_id}: out_dim={out_dim} cannot cover in_dim={in_dim}")
        idx0, idx1 = random_connections(in_dim, out_dim, generator)
        logits = torch.randn(out_dim, 16, generator=generator)
        layers.append(SoftLogicLayer(in_dim, out_dim, idx0, idx1, logits))
        in_dim = out_dim
    return layers, votes


def split_train_val(dataset: DatasetBundle, seed: int, val_frac: float) -> tuple[DatasetBundle, DatasetBundle]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset.x_train.shape[0], generator=generator)
    val_n = int(round(dataset.x_train.shape[0] * val_frac))
    val_n = max(1, min(dataset.x_train.shape[0] - 1, val_n))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    val_dataset = DatasetBundle(
        dataset.name,
        dataset.x_train[train_idx],
        dataset.y_train[train_idx],
        dataset.x_train[val_idx],
        dataset.y_train[val_idx],
        dataset.input_dim,
        dataset.num_classes,
        dataset.target_acc,
    )
    test_dataset = DatasetBundle(
        dataset.name,
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
        dataset.input_dim,
        dataset.num_classes,
        dataset.target_acc,
    )
    return val_dataset, test_dataset


def state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()})


def model_tensors_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(param).all().item() for param in model.parameters())


def model_gradients_finite(model: nn.Module) -> bool:
    return all(param.grad is None or torch.isfinite(param.grad).all().item() for param in model.parameters())


def margin_loss(logits: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(y, logits.shape[1]).bool(), -1e9).max(dim=1).values
    return F.relu(margin - (true - other)).mean()


def activity_balance_loss(bits: torch.Tensor) -> torch.Tensor:
    return ((bits.mean(dim=0) - 0.5) ** 2).mean()


def diversity_loss(bits: torch.Tensor, num_classes: int, votes_per_class: int) -> torch.Tensor:
    if votes_per_class < 2:
        return bits.new_tensor(0.0)
    losses = []
    grouped = bits.reshape(bits.shape[0], num_classes, votes_per_class)
    for class_id in range(num_classes):
        z = grouped[:, class_id, :] - grouped[:, class_id, :].mean(dim=0, keepdim=True)
        denom = (z.square().sum(dim=0) + 1e-6).sqrt()
        corr = (z.T @ z) / (denom[:, None] * denom[None, :])
        offdiag = corr[~torch.eye(votes_per_class, dtype=torch.bool, device=bits.device)]
        losses.append(offdiag.square().mean())
    return torch.stack(losses).mean() if losses else bits.new_tensor(0.0)


@torch.no_grad()
def evaluate_model(
    model: VoteLogicNet,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    tau: float,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total = 0
    correct = 0
    losses = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits = model(xb, mode=mode, tau=tau)
        losses.append(F.cross_entropy(logits, yb, reduction="sum").detach().cpu())
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.numel()
    if was_training:
        model.train()
    return correct / total, torch.stack(losses).sum().item() / total


def train_redundant_model(
    method: str,
    dataset: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    redundancy_factor: int,
    regularized: bool,
) -> TrainOutcome:
    layers, votes = build_layers(
        dataset.input_dim,
        dataset.num_classes,
        args.base_width,
        args.base_votes,
        args.layers,
        redundancy_factor,
        seed,
    )
    model = VoteLogicNet(
        layers,
        dataset.num_classes,
        votes,
        args.group_tau,
        redundancy_dropout=args.redundancy_dropout if regularized else 0.0,
    ).to(device)
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_soft_state = state_copy(model)
    best_hard_state = state_copy(model)
    best_soft_val_loss = math.inf
    best_soft_ckpt_hard_val_acc = 0.0
    best_hard_ckpt_hard_val_acc = -math.inf
    best_hard_val_loss = math.inf
    checkpoint_rows: list[dict[str, object]] = []
    train_valid = True
    invalid_reason = ""
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tau = temp_schedule(epoch, args.epochs, args.temp_start, args.temp_end)
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, bits = model(xb, mode=args.train_estimator, tau=tau, return_bits=True)
            loss = F.cross_entropy(logits, yb)
            if regularized:
                loss = loss + args.margin_coef * margin_loss(logits, yb, args.margin)
                loss = loss + args.activity_coef * activity_balance_loss(bits)
                loss = loss + args.diversity_coef * diversity_loss(bits, dataset.num_classes, model.votes_per_class)
            if not torch.isfinite(loss).item():
                train_valid = False
                invalid_reason = f"nonfinite_train_loss_epoch_{epoch}"
                break
            loss.backward()
            if not model_gradients_finite(model):
                train_valid = False
                invalid_reason = f"nonfinite_gradient_epoch_{epoch}"
                break
            if args.train_estimator != "soft" and args.st_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.st_grad_clip)
            opt.step()
            if not model_tensors_finite(model):
                train_valid = False
                invalid_reason = f"nonfinite_parameter_epoch_{epoch}"
                break

        if not train_valid:
            break

        val_soft_acc, val_soft_loss = evaluate_model(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "soft", tau)
        val_native_acc, val_native_loss = evaluate_model(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, args.train_estimator, tau)
        val_hard_acc, val_hard_loss = evaluate_model(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
        if not all(math.isfinite(value) for value in (val_soft_loss, val_native_loss, val_hard_loss)):
            train_valid = False
            invalid_reason = f"nonfinite_eval_loss_epoch_{epoch}"
        if val_soft_loss < best_soft_val_loss:
            best_soft_val_loss = val_soft_loss
            best_soft_state = state_copy(model)
            best_soft_ckpt_hard_val_acc = val_hard_acc
        if val_hard_acc > best_hard_ckpt_hard_val_acc or (
            val_hard_acc == best_hard_ckpt_hard_val_acc and val_hard_loss < best_hard_val_loss
        ):
            best_hard_ckpt_hard_val_acc = val_hard_acc
            best_hard_val_loss = val_hard_loss
            best_hard_state = state_copy(model)
        checkpoint_rows.append(
            {
                "method": method,
                "dataset": dataset.name,
                "seed": seed,
                "redundancy_factor": redundancy_factor,
                "train_estimator": args.train_estimator,
                "epoch": epoch,
                "val_soft_acc": val_soft_acc,
                "val_soft_loss": val_soft_loss,
                "val_native_acc": val_native_acc,
                "val_native_loss": val_native_loss,
                "val_hard_acc": val_hard_acc,
                "val_hard_loss": val_hard_loss,
                "hard_gap": abs(val_soft_acc - val_hard_acc),
                "native_gap": abs(val_native_acc - val_hard_acc),
                "train_valid": int(train_valid),
                "invalid_reason": invalid_reason,
            }
        )
        if not train_valid:
            break

    return TrainOutcome(
        model=model,
        best_soft_state=best_soft_state,
        best_hard_state=best_hard_state,
        best_soft_val_loss=best_soft_val_loss,
        best_soft_ckpt_hard_val_acc=best_soft_ckpt_hard_val_acc,
        best_hard_ckpt_hard_val_acc=best_hard_ckpt_hard_val_acc,
        train_valid=train_valid,
        invalid_reason=invalid_reason,
        train_time=time.perf_counter() - start_time,
        checkpoint_rows=checkpoint_rows,
    )


def temp_schedule(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    ratio = (epoch - 1) / (epochs - 1)
    return start * ((end / start) ** ratio)


def argmax_layers(model: VoteLogicNet) -> list[FrozenHardLogicLayer]:
    hard_layers = []
    for layer in model.layers:
        if not isinstance(layer, SoftLogicLayer):
            raise TypeError(type(layer))
        hard_layers.append(
            FrozenHardLogicLayer(
                layer.in_dim,
                layer.out_dim,
                layer.indices_0.detach().cpu(),
                layer.indices_1.detach().cpu(),
                layer.hard_ops_argmax(),
            )
        )
    return hard_layers


def sampled_layers(model: VoteLogicNet, generator: torch.Generator, temperature: float) -> list[FrozenHardLogicLayer]:
    hard_layers = []
    for layer in model.layers:
        if not isinstance(layer, SoftLogicLayer):
            raise TypeError(type(layer))
        probs = F.softmax(layer.logits.detach().cpu() / temperature, dim=-1)
        if not torch.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("cannot sample hard gates from non-finite gate probabilities")
        ops = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        hard_layers.append(
            FrozenHardLogicLayer(
                layer.in_dim,
                layer.out_dim,
                layer.indices_0.detach().cpu(),
                layer.indices_1.detach().cpu(),
                ops,
            )
        )
    return hard_layers


def refit_layers(model: VoteLogicNet, train_x: torch.Tensor, args: argparse.Namespace, device: torch.device) -> tuple[list[FrozenHardLogicLayer], dict[str, float]]:
    hard_layers: list[FrozenHardLogicLayer] = []
    x_ref = train_x
    stats_rows = []
    for layer in model.layers:
        if not isinstance(layer, SoftLogicLayer):
            raise TypeError(type(layer))
        hard_layer, stats = refit_truth_table_layer(layer, x_ref, args, device)
        hard_layers.append(hard_layer)
        stats_rows.append(stats)
        x_ref = apply_layers([hard_layer], x_ref, args.eval_batch_size, device, mode="hard")
    return hard_layers, {
        "refit_error": mean_stat(stats_rows, "truth_refit_mse"),
        "refit_delta": mean_stat(stats_rows, "refit_mse_delta"),
        "op_change_ratio": mean_stat(stats_rows, "op_change_ratio"),
    }


def mean_stat(rows: list[dict[str, object]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row.get(key, math.nan))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / len(vals) if vals else math.nan


def hard_model_from_layers(reference: VoteLogicNet, layers: list[FrozenHardLogicLayer]) -> VoteLogicNet:
    return VoteLogicNet(
        list(layers),
        reference.num_classes,
        reference.votes_per_class,
        reference.group_sum.tau,
        redundancy_dropout=0.0,
    )


def build_candidates(
    method: str,
    outcome: TrainOutcome,
    val_dataset: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    redundancy_factor: int,
) -> list[dict[str, object]]:
    candidates = []
    hardening_start = time.perf_counter()
    checkpoint_specs = [
        ("argmax_best_soft", "best_soft", outcome.best_soft_state),
        ("argmax_best_hard", "best_hard", outcome.best_hard_state),
    ]
    for candidate_name, checkpoint_source, state in checkpoint_specs:
        load_state(outcome.model, state, device)
        candidates.append(
            candidate_metrics(
                method,
                val_dataset,
                outcome.model,
                argmax_layers(outcome.model),
                args,
                device,
                seed,
                redundancy_factor,
                candidate_name,
                checkpoint_source,
                refit_error=math.nan,
                refit_delta=math.nan,
                op_change_ratio=0.0,
                hardening_start=hardening_start,
            )
        )

    if args.best_of_n > 0:
        load_state(outcome.model, outcome.best_hard_state, device)
        generator = torch.Generator().manual_seed(args.hardening_seed + seed)
        for sample_id in range(args.best_of_n):
            candidates.append(
                candidate_metrics(
                    method,
                    val_dataset,
                    outcome.model,
                    sampled_layers(outcome.model, generator, args.sample_temp),
                    args,
                    device,
                    seed,
                    redundancy_factor,
                    f"gumbel_sample_{sample_id}",
                    "best_hard",
                    refit_error=math.nan,
                    refit_delta=math.nan,
                    op_change_ratio=math.nan,
                    hardening_start=hardening_start,
                )
            )

    load_state(outcome.model, outcome.best_hard_state, device)
    refit_hard_layers, refit_stats = refit_layers(outcome.model, val_dataset.x_train, args, device)
    candidates.append(
        candidate_metrics(
            method,
            val_dataset,
            outcome.model,
            refit_hard_layers,
            args,
            device,
            seed,
            redundancy_factor,
            "truth_table_refit",
            "best_hard",
            refit_error=refit_stats["refit_error"],
            refit_delta=refit_stats["refit_delta"],
            op_change_ratio=refit_stats["op_change_ratio"],
            hardening_start=hardening_start,
        )
    )
    return candidates


def candidate_metrics(
    method: str,
    val_dataset: DatasetBundle,
    soft_model: VoteLogicNet,
    hard_layers: list[FrozenHardLogicLayer],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    redundancy_factor: int,
    candidate_name: str,
    checkpoint_source: str,
    *,
    refit_error: float,
    refit_delta: float,
    op_change_ratio: float,
    hardening_start: float,
) -> dict[str, object]:
    hard_model = hard_model_from_layers(soft_model, hard_layers).to(device)
    val_hard_acc, val_hard_loss = evaluate_model(hard_model, val_dataset.x_test, val_dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
    val_soft_acc, val_soft_loss = evaluate_model(soft_model, val_dataset.x_test, val_dataset.y_test, args.eval_batch_size, device, "soft", args.temp_end)
    unused = compute_unused_gate_ratio(list(hard_model.layers), val_dataset.x_train, args.eval_batch_size, device)
    gates = gate_count(list(hard_model.layers))
    depth = len(hard_model.layers)
    fanout = fanout_max(list(hard_model.layers))
    refit_penalty = 0.0 if not math.isfinite(refit_error) else args.lambda_refit * refit_error
    score = val_hard_loss + refit_penalty + args.lambda_cost * gates + args.lambda_unused * unused
    return {
        "method": method,
        "dataset": val_dataset.name,
        "seed": seed,
        "redundancy_factor": redundancy_factor,
        "train_estimator": args.train_estimator,
        "candidate": candidate_name,
        "checkpoint_source": checkpoint_source,
        "val_hard_acc": val_hard_acc,
        "val_hard_loss": val_hard_loss,
        "val_soft_acc": val_soft_acc,
        "val_soft_loss": val_soft_loss,
        "val_full_gap": abs(val_soft_acc - val_hard_acc),
        "refit_error": refit_error,
        "refit_delta": refit_delta,
        "op_change_ratio": op_change_ratio,
        "unused_gate_ratio": unused,
        "gate_count": gates,
        "depth": depth,
        "fanout_max": fanout,
        "score": score,
        "selected": 0,
        "_hard_layers": hard_layers,
        "_hardening_time": time.perf_counter() - hardening_start,
    }


def select_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["val_hard_acc"]),
            float(row["score"]),
            float(row["unused_gate_ratio"]),
            float(row["fanout_max"]),
            str(row["candidate"]),
        ),
    )[0]
    selected["selected"] = 1
    return selected


def final_row(
    method: str,
    test_dataset: DatasetBundle,
    outcome: TrainOutcome,
    selected: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    redundancy_factor: int,
) -> dict[str, object]:
    hard_layers = selected["_hard_layers"]  # type: ignore[assignment]
    assert isinstance(hard_layers, list)
    hard_model = hard_model_from_layers(outcome.model, hard_layers).to(device)
    load_state(outcome.model, outcome.best_hard_state if selected["checkpoint_source"] == "best_hard" else outcome.best_soft_state, device)
    hard_acc, hard_loss = evaluate_model(hard_model, test_dataset.x_test, test_dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
    soft_acc, soft_loss = evaluate_model(outcome.model, test_dataset.x_test, test_dataset.y_test, args.eval_batch_size, device, "soft", args.temp_end)
    native_acc, _native_loss = evaluate_model(outcome.model, test_dataset.x_test, test_dataset.y_test, args.eval_batch_size, device, args.train_estimator, args.temp_end)
    unused = compute_unused_gate_ratio(list(hard_model.layers), test_dataset.x_train, args.eval_batch_size, device)
    gates = gate_count(list(hard_model.layers))
    return {
        "method": method,
        "dataset": test_dataset.name,
        "seed": seed,
        "redundancy_factor": redundancy_factor,
        "train_estimator": args.train_estimator,
        "hardening": selected["candidate"],
        "checkpoint_source": selected["checkpoint_source"],
        "hard_acc": hard_acc,
        "soft_acc": soft_acc,
        "full_gap": abs(soft_acc - hard_acc),
        "native_gap": abs(native_acc - hard_acc),
        "hard_loss": hard_loss,
        "soft_loss": soft_loss,
        "train_time": outcome.train_time,
        "hardening_time": selected["_hardening_time"],
        "train_valid": int(outcome.train_valid),
        "invalid_reason": outcome.invalid_reason,
        "best_soft_val_loss": outcome.best_soft_val_loss,
        "best_soft_ckpt_hard_val_acc": outcome.best_soft_ckpt_hard_val_acc,
        "best_hard_ckpt_hard_val_acc": outcome.best_hard_ckpt_hard_val_acc,
        "hard_ckpt_beats_soft_ckpt": int(outcome.best_hard_ckpt_hard_val_acc > outcome.best_soft_ckpt_hard_val_acc),
        "unused_gate_ratio": unused,
        "gate_count": gates,
        "depth": len(hard_model.layers),
        "fanout_max": fanout_max(list(hard_model.layers)),
        "final_votes_per_class": outcome.model.votes_per_class,
        "source_note": f"train_estimator={args.train_estimator}; native_gap compares the training estimator path with final hard inference",
    }


def invalid_final_row(
    method: str,
    test_dataset: DatasetBundle,
    outcome: TrainOutcome,
    args: argparse.Namespace,
    seed: int,
    redundancy_factor: int,
) -> dict[str, object]:
    return {
        "method": method,
        "dataset": test_dataset.name,
        "seed": seed,
        "redundancy_factor": redundancy_factor,
        "train_estimator": args.train_estimator,
        "hardening": "training_invalid",
        "checkpoint_source": "none",
        "hard_acc": math.nan,
        "soft_acc": math.nan,
        "full_gap": math.nan,
        "native_gap": math.nan,
        "hard_loss": math.nan,
        "soft_loss": math.nan,
        "train_time": outcome.train_time,
        "hardening_time": 0.0,
        "train_valid": 0,
        "invalid_reason": outcome.invalid_reason,
        "best_soft_val_loss": outcome.best_soft_val_loss,
        "best_soft_ckpt_hard_val_acc": outcome.best_soft_ckpt_hard_val_acc,
        "best_hard_ckpt_hard_val_acc": outcome.best_hard_ckpt_hard_val_acc,
        "hard_ckpt_beats_soft_ckpt": 0,
        "unused_gate_ratio": math.nan,
        "gate_count": math.nan,
        "depth": args.layers,
        "fanout_max": math.nan,
        "final_votes_per_class": math.nan,
        "source_note": f"train_estimator={args.train_estimator}; invalid training run excluded from hardening evidence",
    }


def clean_candidate(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def applicable(method: str, factor: int) -> bool:
    if method == "no_redundancy":
        return factor == 1
    if method in {"more_gates_only", "redundant_regularized", "redundant_task_hardened", "redundant_truth_refit_only"}:
        return factor > 1
    return False


def regularized_method(method: str) -> bool:
    return method in {"redundant_regularized", "redundant_task_hardened", "redundant_truth_refit_only"}


def candidate_subset(method: str, candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in candidates if is_candidate_eligible(method, row)]


def is_candidate_eligible(method: str, row: dict[str, object]) -> bool:
    if method in {"no_redundancy", "more_gates_only"}:
        return row["candidate"] == "argmax_best_soft"
    if method == "redundant_regularized":
        return row["candidate"] == "argmax_best_hard"
    if method == "redundant_truth_refit_only":
        return row["candidate"] == "truth_table_refit"
    return True


def selection_policy(method: str) -> str:
    if method == "no_redundancy":
        return "argmax_best_soft_only"
    if method == "more_gates_only":
        return "argmax_best_soft_only"
    if method == "redundant_regularized":
        return "argmax_best_hard_only"
    if method == "redundant_truth_refit_only":
        return "truth_table_refit_only"
    if method == "redundant_task_hardened":
        return "all_candidates_validation_hard_acc"
    return "all_candidates_validation_hard_acc"


def annotate_candidate_eligibility(method: str, candidates: list[dict[str, object]]) -> None:
    policy = selection_policy(method)
    for row in candidates:
        row["eligible_for_method"] = int(is_candidate_eligible(method, row))
        row["selection_policy"] = policy


def markdown_report(rows: list[dict[str, object]], candidates: list[dict[str, object]]) -> str:
    lines = [
        "# Redundant Hardening Report",
        "",
        "Candidate hardening is selected on validation hard accuracy first, then score, within each method's eligible candidate subset.",
        "Only `redundant_task_hardened` makes all listed hardening candidates eligible.",
        "",
        "## Final Results",
        "",
        markdown_table(rows, RESULT_FIELDS),
        "",
        "## Hardening Candidates",
        "",
        markdown_table(candidates, CANDIDATE_FIELDS),
        "",
        "Notes:",
        "- `no_redundancy` uses 1x votes and argmax from the best-soft-loss checkpoint.",
        "- `more_gates_only` increases gate/vote budget without activity/diversity/dropout regularization.",
        "- `redundant_regularized` uses margin/activity/diversity/dropout and argmax from the best-hard checkpoint.",
        "- `redundant_truth_refit_only` uses the same regularized training but forces truth-table refit as the final hardening candidate.",
        "- `redundant_task_hardened` uses the same regularized training but selects among argmax, hard samples, and truth-table refit by validation hard accuracy.",
        "- `eligible_for_method=1` marks candidates that were allowed by that row's `selection_policy`; `selected=1` is chosen only among eligible candidates.",
        "- `full_gap` compares soft evaluation with final hard inference; `native_gap` compares the configured training estimator path with final hard inference.",
        "- `train_valid=0` rows stopped on non-finite training/evaluation values and are excluded from hardening evidence.",
    ]
    return "\n".join(lines) + "\n"


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_table(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument("--methods", nargs="+", default=["no_redundancy", "more_gates_only", "redundant_regularized", "redundant_task_hardened"])
    parser.add_argument("--redundancy-factors", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--base-votes", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--temp-start", type=float, default=1.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument(
        "--train-estimator",
        choices=["soft", "gumbel_st", "argmax_st"],
        default="soft",
        help="Forward estimator used during redundant training; gumbel_st/argmax_st make training closer to hard inference.",
    )
    parser.add_argument(
        "--st-grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm used only for gumbel_st/argmax_st training; <=0 disables clipping.",
    )
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--margin-coef", type=float, default=0.1)
    parser.add_argument("--activity-coef", type=float, default=0.01)
    parser.add_argument("--diversity-coef", type=float, default=0.01)
    parser.add_argument("--redundancy-dropout", type=float, default=0.1)
    parser.add_argument("--best-of-n", type=int, default=8)
    parser.add_argument("--sample-temp", type=float, default=0.5)
    parser.add_argument("--lambda-refit", type=float, default=0.01)
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--lambda-unused", type=float, default=0.01)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--exact-truth-max", type=int, default=12)
    parser.add_argument("--refit-samples", type=int, default=4096)
    parser.add_argument("--refit-seed", type=int, default=12345)
    parser.add_argument("--hardening-seed", type=int, default=54321)
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/redundant_hardening_v1")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets = ["parity6", "majority7"]
        args.redundancy_factors = [1, 2]
        args.seeds = [0]
        args.base_width = min(args.base_width, 24)
        args.base_votes = min(args.base_votes, 8)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 12)
        args.best_of_n = min(args.best_of_n, 4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            dataset = load_dataset(dataset_name, seed, args)
            val_dataset, test_dataset = split_train_val(dataset, seed + 70001, args.val_frac)
            print(f"dataset={dataset.name} seed={seed} train={len(val_dataset.y_train)} val={len(val_dataset.y_test)} test={len(test_dataset.y_test)}")
            for method in args.methods:
                for factor in args.redundancy_factors:
                    if not applicable(method, factor):
                        continue
                    print(f"  method={method} factor={factor}", flush=True)
                    outcome = train_redundant_model(method, val_dataset, args, device, seed + factor * 100, factor, regularized_method(method))
                    checkpoint_rows.extend(outcome.checkpoint_rows)
                    if not outcome.train_valid:
                        row = invalid_final_row(method, test_dataset, outcome, args, seed, factor)
                        final_rows.append(row)
                        print(f"    INVALID {outcome.invalid_reason}", flush=True)
                        write_table(out_dir / "redundant_hardening_results.partial.csv", final_rows, RESULT_FIELDS)
                        write_table(out_dir / "hardening_candidates.partial.csv", candidate_rows, CANDIDATE_FIELDS)
                        write_table(out_dir / "checkpoint_trace.partial.csv", checkpoint_rows, CHECKPOINT_FIELDS)
                        continue
                    all_candidates = build_candidates(method, outcome, val_dataset, args, device, seed, factor)
                    annotate_candidate_eligibility(method, all_candidates)
                    candidates = candidate_subset(method, all_candidates)
                    selected = select_candidate(candidates)
                    candidate_rows.extend(clean_candidate(row) for row in all_candidates)
                    row = final_row(method, test_dataset, outcome, selected, args, device, seed, factor)
                    final_rows.append(row)
                    print(
                        "    hard_acc={:.4f} soft_acc={:.4f} full_gap={:.4f} hardening={}".format(
                            float(row["hard_acc"]),
                            float(row["soft_acc"]),
                            float(row["full_gap"]),
                            row["hardening"],
                        ),
                        flush=True,
                    )
                    write_table(out_dir / "redundant_hardening_results.partial.csv", final_rows, RESULT_FIELDS)
                    write_table(out_dir / "hardening_candidates.partial.csv", candidate_rows, CANDIDATE_FIELDS)
                    write_table(out_dir / "checkpoint_trace.partial.csv", checkpoint_rows, CHECKPOINT_FIELDS)
    write_table(out_dir / "redundant_hardening_results.csv", final_rows, RESULT_FIELDS)
    write_table(out_dir / "hardening_candidates.csv", candidate_rows, CANDIDATE_FIELDS)
    write_table(out_dir / "checkpoint_trace.csv", checkpoint_rows, CHECKPOINT_FIELDS)
    (out_dir / "redundant_hardening_report.md").write_text(markdown_report(final_rows, candidate_rows))
    print(f"wrote {len(final_rows)} final rows, {len(candidate_rows)} candidates to {out_dir}")


if __name__ == "__main__":
    main()
