#!/usr/bin/env python3
"""Random-connectivity search then fixed-wiring LGN training.

This script implements Part A of the next-stage protocol. It reuses the compact
LGN implementation from ``hard_lgn_benchmark.py`` and keeps the validity checks
explicit:

- wiring seeds are separated from gate-logit initialization seeds;
- candidate wirings are selected on a validation split by hard accuracy;
- final metrics are reported on a held-out test split;
- search time and final training time are reported separately.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from hard_lgn_benchmark import (
    DatasetBundle,
    GroupSum,
    LogicNet,
    MetricsRow,
    argmax_hard_layer,
    build_architecture,
    compute_unused_gate_ratio,
    evaluate,
    fanout_max,
    gate_count,
    gate_outputs,
    load_dataset,
    make_loader,
    make_soft_net,
    random_connections,
    set_seed,
    temp_schedule,
    train_blockwise,
    train_end_to_end,
    write_csv,
)


FINAL_FIELDS = [
    "method",
    "base_method",
    "dataset",
    "seed",
    "wiring_seed",
    "init_seed",
    "selection_hard_val_acc",
    "selection_native_val_gap",
    "selection_unused_gate_ratio",
    "search_candidates",
    "search_time",
    "final_train_time",
    "train_time",
    "hard_acc",
    "soft_acc",
    "full_gap",
    "native_gap",
    "soft_loss",
    "hard_loss",
    "loss_gap",
    "native_loss_gap",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "epochs_to_target",
    "source_note",
]

CANDIDATE_FIELDS = [
    "method",
    "dataset",
    "seed",
    "candidate_rank",
    "wiring_seed",
    "init_seed",
    "base_method",
    "candidate_set_size",
    "hard_val_acc",
    "soft_val_acc",
    "full_val_gap",
    "native_val_gap",
    "soft_val_loss",
    "hard_val_loss",
    "native_val_loss_gap",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "candidate_train_time",
]


class CandidateSetLogicLayer(nn.Module):
    """Logic layer that softly selects one pair from a fixed candidate set."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        indices_0: torch.Tensor,
        indices_1: torch.Tensor,
        gate_logits: torch.Tensor,
        connection_logits: torch.Tensor,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer("indices_0", indices_0.clone().long())
        self.register_buffer("indices_1", indices_1.clone().long())
        self.gate_logits = nn.Parameter(gate_logits.clone().float())
        self.connection_logits = nn.Parameter(connection_logits.clone().float())

    def forward(self, x: torch.Tensor, mode: str = "soft", tau: float = 1.0) -> torch.Tensor:
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]
        if mode == "hard":
            conn_probs = F.one_hot(self.connection_logits.argmax(-1), self.connection_logits.shape[-1]).to(dtype=x.dtype)
            gate_probs = F.one_hot(self.gate_logits.argmax(-1), 16).to(dtype=x.dtype)
        elif mode == "soft":
            conn_probs = F.softmax(self.connection_logits / tau, dim=-1).to(dtype=x.dtype)
            gate_probs = F.softmax(self.gate_logits / tau, dim=-1).to(dtype=x.dtype)
        else:
            raise ValueError(mode)
        candidate_values = (gate_outputs(a, b) * gate_probs).sum(dim=-1)
        return (candidate_values * conn_probs).sum(dim=-1)

    def entropy(self, tau: float = 1.0) -> torch.Tensor:
        conn = F.softmax(self.connection_logits / tau, dim=-1)
        gate = F.softmax(self.gate_logits / tau, dim=-1)
        conn_entropy = -(conn * conn.clamp_min(1e-8).log()).sum(-1).mean()
        gate_entropy = -(gate * gate.clamp_min(1e-8).log()).sum(-1).mean()
        return conn_entropy + gate_entropy


class CandidateSetLogicNet(nn.Module):
    def __init__(self, layers: list[CandidateSetLogicLayer], num_classes: int, group_tau: float = 1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(num_classes, tau=group_tau)

    def forward(self, x: torch.Tensor, mode: str = "soft", tau: float = 1.0, gumbel_hard: bool = False) -> torch.Tensor:
        del gumbel_hard
        for layer in self.layers:
            x = layer(x, mode=mode, tau=tau)
        return self.group_sum(x)

    def entropy(self, tau: float = 1.0) -> torch.Tensor:
        return sum(layer.entropy(tau) for layer in self.layers)


def build_architecture_separate(
    input_dim: int,
    width: int,
    depth: int,
    *,
    wiring_seed: int,
    init_seed: int,
) -> list[dict[str, torch.Tensor | int]]:
    """Build an LGN architecture with independent wiring and logit seeds."""

    wiring_gen = torch.Generator().manual_seed(wiring_seed)
    init_gen = torch.Generator().manual_seed(init_seed)
    arch: list[dict[str, torch.Tensor | int]] = []
    in_dim = input_dim
    for _ in range(depth):
        idx0, idx1 = random_connections(in_dim, width, wiring_gen)
        logits = torch.randn(width, 16, generator=init_gen)
        arch.append({"in_dim": in_dim, "out_dim": width, "idx0": idx0, "idx1": idx1, "logits": logits})
        in_dim = width
    return arch


def build_candidate_set_layers(
    input_dim: int,
    width: int,
    depth: int,
    *,
    candidate_set_size: int,
    wiring_seed: int,
    init_seed: int,
) -> list[CandidateSetLogicLayer]:
    """Generate K candidate input pairs per gate and trainable selectors."""

    wiring_gen = torch.Generator().manual_seed(wiring_seed)
    init_gen = torch.Generator().manual_seed(init_seed)
    layers: list[CandidateSetLogicLayer] = []
    in_dim = input_dim
    for _ in range(depth):
        idx0_candidates = []
        idx1_candidates = []
        for _candidate in range(candidate_set_size):
            idx0, idx1 = random_connections(in_dim, width, wiring_gen)
            idx0_candidates.append(idx0)
            idx1_candidates.append(idx1)
        idx0_all = torch.stack(idx0_candidates, dim=1)
        idx1_all = torch.stack(idx1_candidates, dim=1)
        gate_logits = torch.randn(width, candidate_set_size, 16, generator=init_gen)
        connection_logits = 0.01 * torch.randn(width, candidate_set_size, generator=init_gen)
        layers.append(CandidateSetLogicLayer(in_dim, width, idx0_all, idx1_all, gate_logits, connection_logits))
        in_dim = width
    return layers


def extract_selected_architecture(
    layers: list[CandidateSetLogicLayer],
    *,
    init_seed: int,
    use_trained_gate_logits: bool,
) -> list[dict[str, torch.Tensor | int]]:
    """Freeze candidate selectors into a normal fixed-wiring architecture."""

    init_gen = torch.Generator().manual_seed(init_seed)
    arch: list[dict[str, torch.Tensor | int]] = []
    for layer in layers:
        selected = layer.connection_logits.detach().cpu().argmax(dim=-1)
        gate_ids = torch.arange(layer.out_dim)
        idx0 = layer.indices_0.detach().cpu()[gate_ids, selected]
        idx1 = layer.indices_1.detach().cpu()[gate_ids, selected]
        if use_trained_gate_logits:
            logits = layer.gate_logits.detach().cpu()[gate_ids, selected, :]
        else:
            logits = torch.randn(layer.out_dim, 16, generator=init_gen)
        arch.append({"in_dim": layer.in_dim, "out_dim": layer.out_dim, "idx0": idx0, "idx1": idx1, "logits": logits})
    return arch


def metrics_from_architecture(
    *,
    method: str,
    dataset: DatasetBundle,
    arch: list[dict[str, torch.Tensor | int]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    train_time: float,
) -> MetricsRow:
    model = make_soft_net(arch, dataset.num_classes, args.group_tau).to(device)
    soft_acc, soft_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "soft", 1.0)
    disc_acc, disc_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
    layers = list(model.layers)
    return MetricsRow(
        method=method,
        dataset=dataset.name,
        seed=seed,
        soft_acc=soft_acc,
        discrete_acc=disc_acc,
        acc_gap=abs(soft_acc - disc_acc),
        soft_loss=soft_loss,
        discrete_loss=disc_loss,
        loss_gap=abs(soft_loss - disc_loss),
        path_soft_acc=soft_acc,
        path_discrete_acc=disc_acc,
        path_acc_gap=abs(soft_acc - disc_acc),
        path_soft_loss=soft_loss,
        path_discrete_loss=disc_loss,
        path_loss_gap=abs(soft_loss - disc_loss),
        train_time=train_time,
        epochs_to_target=args.search_epochs if disc_acc >= dataset.target_acc else -1,
        time_to_target=train_time if disc_acc >= dataset.target_acc else -1.0,
        unused_gate_ratio=compute_unused_gate_ratio(layers, dataset.x_train, args.eval_batch_size, device),
        gate_count=gate_count(layers),
        depth=len(layers),
        fanout_max=fanout_max(layers),
        soft_inference_samples_per_sec=math.nan,
        discrete_inference_samples_per_sec=math.nan,
        inference_bench_repeats=0,
    )


def train_candidate_set_connectivity(
    dataset: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    *,
    wiring_seed: int,
    init_seed: int,
) -> tuple[MetricsRow, list[dict[str, torch.Tensor | int]], float]:
    """Learn a candidate-set selector, then return a fixed selected wiring."""

    layers = build_candidate_set_layers(
        dataset.input_dim,
        args.width,
        args.layers,
        candidate_set_size=args.candidate_set_size,
        wiring_seed=wiring_seed,
        init_seed=init_seed,
    )
    model = CandidateSetLogicNet(layers, dataset.num_classes, args.group_tau).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed + 424242)
    start_time = time.perf_counter()
    for epoch in range(1, args.search_epochs + 1):
        tau = temp_schedule(epoch, args.search_epochs, args.candidate_temp_start, args.candidate_temp_end)
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mode="soft", tau=tau)
            loss = F.cross_entropy(logits, yb)
            if args.candidate_entropy_coef:
                loss = loss + args.candidate_entropy_coef * model.entropy(tau)
            loss.backward()
            opt.step()
    search_time = time.perf_counter() - start_time

    trained_arch = extract_selected_architecture(
        list(model.layers),
        init_seed=init_seed,
        use_trained_gate_logits=True,
    )
    final_arch = extract_selected_architecture(
        list(model.layers),
        init_seed=init_seed,
        use_trained_gate_logits=args.candidate_use_trained_logits,
    )
    selection_row = metrics_from_architecture(
        method="candidate_set_learnable_freeze",
        dataset=dataset,
        arch=trained_arch,
        args=args,
        device=device,
        seed=seed,
        train_time=search_time,
    )
    return selection_row, final_arch, search_time


def split_train_val(dataset: DatasetBundle, seed: int, val_frac: float) -> tuple[DatasetBundle, DatasetBundle]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--val-frac must be in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset.x_train.shape[0], generator=generator)
    val_n = int(round(dataset.x_train.shape[0] * val_frac))
    val_n = max(1, min(dataset.x_train.shape[0] - 1, val_n))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    selection_dataset = DatasetBundle(
        name=dataset.name,
        x_train=dataset.x_train[train_idx],
        y_train=dataset.y_train[train_idx],
        x_test=dataset.x_train[val_idx],
        y_test=dataset.y_train[val_idx],
        input_dim=dataset.input_dim,
        num_classes=dataset.num_classes,
        target_acc=dataset.target_acc,
    )
    final_dataset = DatasetBundle(
        name=dataset.name,
        x_train=dataset.x_train,
        y_train=dataset.y_train,
        x_test=dataset.x_test,
        y_test=dataset.y_test,
        input_dim=dataset.input_dim,
        num_classes=dataset.num_classes,
        target_acc=dataset.target_acc,
    )
    return selection_dataset, final_dataset


def clone_args(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    cloned = copy.copy(args)
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def run_base_method(
    base_method: str,
    dataset: DatasetBundle,
    arch: list[dict[str, torch.Tensor | int]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> MetricsRow:
    if base_method in {"dlgn", "dlgn_anneal", "gumbel_st", "gumbel_soft"}:
        row, _, _, _ = train_end_to_end(base_method, dataset, arch, args, device, seed)
        return row
    if base_method in {"block_relaxed", "block_hard_refit"}:
        row, _, _, _, _ = train_blockwise(base_method, dataset, arch, args, device, seed)
        return row
    raise ValueError(f"unsupported --base-method {base_method!r}")


def final_result_row(
    *,
    method: str,
    base_method: str,
    dataset: str,
    seed: int,
    wiring_seed: int,
    init_seed: int,
    selection_row: MetricsRow | None,
    search_candidates: int,
    search_time: float,
    final_row: MetricsRow,
    source_note: str,
) -> dict[str, object]:
    return {
        "method": method,
        "base_method": base_method,
        "dataset": dataset,
        "seed": seed,
        "wiring_seed": wiring_seed,
        "init_seed": init_seed,
        "selection_hard_val_acc": selection_row.discrete_acc if selection_row else math.nan,
        "selection_native_val_gap": selection_row.path_acc_gap if selection_row else math.nan,
        "selection_unused_gate_ratio": selection_row.unused_gate_ratio if selection_row else math.nan,
        "search_candidates": search_candidates,
        "search_time": search_time,
        "final_train_time": final_row.train_time,
        "train_time": search_time + final_row.train_time,
        "hard_acc": final_row.discrete_acc,
        "soft_acc": final_row.soft_acc,
        "full_gap": final_row.acc_gap,
        "native_gap": final_row.path_acc_gap,
        "soft_loss": final_row.soft_loss,
        "hard_loss": final_row.discrete_loss,
        "loss_gap": final_row.loss_gap,
        "native_loss_gap": final_row.path_loss_gap,
        "unused_gate_ratio": final_row.unused_gate_ratio,
        "gate_count": final_row.gate_count,
        "depth": final_row.depth,
        "fanout_max": final_row.fanout_max,
        "epochs_to_target": final_row.epochs_to_target,
        "source_note": source_note,
    }


def candidate_row(
    *,
    method: str,
    dataset: str,
    seed: int,
    candidate_rank: int,
    wiring_seed: int,
    init_seed: int,
    base_method: str,
    candidate_set_size: int,
    row: MetricsRow,
) -> dict[str, object]:
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "candidate_rank": candidate_rank,
        "wiring_seed": wiring_seed,
        "init_seed": init_seed,
        "base_method": base_method,
        "candidate_set_size": candidate_set_size,
        "hard_val_acc": row.discrete_acc,
        "soft_val_acc": row.soft_acc,
        "full_val_gap": row.acc_gap,
        "native_val_gap": row.path_acc_gap,
        "soft_val_loss": row.soft_loss,
        "hard_val_loss": row.discrete_loss,
        "native_val_loss_gap": row.path_loss_gap,
        "unused_gate_ratio": row.unused_gate_ratio,
        "gate_count": row.gate_count,
        "depth": row.depth,
        "fanout_max": row.fanout_max,
        "candidate_train_time": row.train_time,
    }


def select_candidate(rows: list[tuple[int, MetricsRow]]) -> tuple[int, MetricsRow]:
    """Select by hard val acc, then native gap, unused ratio, fanout, train time."""

    return sorted(
        rows,
        key=lambda item: (
            -item[1].discrete_acc,
            item[1].path_acc_gap,
            item[1].unused_gate_ratio,
            item[1].fanout_max,
            item[1].train_time,
            item[0],
        ),
    )[0]


def markdown_report(final_rows: list[dict[str, object]], candidate_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Wiring Search Report",
        "",
        "Random wiring search uses a validation split to select fixed connectivity, then retrains gate choices on the full training split and reports held-out test metrics.",
        "",
        "Selection key: hard validation accuracy, then native gap, unused ratio, fanout, train time.",
        "",
        "## Final Comparison",
        "",
    ]
    display_fields = [
        "method",
        "dataset",
        "seed",
        "wiring_seed",
        "hard_acc",
        "soft_acc",
        "full_gap",
        "native_gap",
        "unused_gate_ratio",
        "train_time",
        "gate_count",
        "depth",
        "fanout_max",
    ]
    lines.extend(markdown_table(final_rows, display_fields))
    lines.extend(["", "## Selected Candidates", ""])
    selected = [row for row in final_rows if row["method"] in {"random_wiring_search_freeze", "candidate_set_learnable_freeze"}]
    selected_fields = [
        "method",
        "dataset",
        "seed",
        "wiring_seed",
        "selection_hard_val_acc",
        "selection_native_val_gap",
        "selection_unused_gate_ratio",
        "search_candidates",
        "search_time",
    ]
    lines.extend(markdown_table(selected, selected_fields))
    lines.extend(["", "## Candidate Search Rows", ""])
    cand_fields = [
        "method",
        "dataset",
        "seed",
        "candidate_rank",
        "wiring_seed",
        "candidate_set_size",
        "hard_val_acc",
        "native_val_gap",
        "unused_gate_ratio",
        "fanout_max",
        "candidate_train_time",
    ]
    lines.extend(markdown_table(candidate_rows, cand_fields))
    lines.extend(
        [
            "",
            "Notes:",
            "- `fixed_random` isolates the original single fixed-random wiring baseline.",
            "- `fixed_random_matched_budget` controls for extra search epochs when enabled.",
            "- `candidate_set_learnable_freeze` learns within a fixed per-gate candidate connection set, freezes the selected wiring, then retrains ordinary gate logits on the full training split.",
            "- `train_time` for search includes candidate search plus final retraining; `final_train_time` is available in CSV.",
            "- Do not use validation metrics as final claims; final hard accuracy is measured on held-out test data.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("nan")
                else:
                    values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--base-method", default="dlgn", choices=["dlgn", "dlgn_anneal", "gumbel_st", "block_relaxed", "block_hard_refit"])
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--candidate-set-learnable", action="store_true")
    parser.add_argument("--candidate-set-size", type=int, default=4)
    parser.add_argument("--candidate-set-seed-offset", type=int, default=100000)
    parser.add_argument("--candidate-temp-start", type=float, default=2.0)
    parser.add_argument("--candidate-temp-end", type=float, default=0.3)
    parser.add_argument("--candidate-entropy-coef", type=float, default=1e-3)
    parser.add_argument(
        "--candidate-use-trained-logits",
        action="store_true",
        help="Initialize final fixed-wiring gate logits from the candidate-set search instead of the common init seed.",
    )
    parser.add_argument("--wiring-seed-base", type=int, default=10000)
    parser.add_argument("--init-seed-base", type=int, default=20000)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--search-block-epochs", type=int, default=8)
    parser.add_argument("--final-block-epochs", type=int, default=24)
    parser.add_argument("--matched-budget-baseline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.2)
    parser.add_argument("--gumbel-temp-start", type=float, default=1.5)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.3)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--exact-truth-max", type=int, default=12)
    parser.add_argument("--refit-samples", type=int, default=4096)
    parser.add_argument("--refit-seed", type=int, default=12345)
    parser.add_argument("--target-acc-override", type=float)
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/wiring_search_v1")
    parser.add_argument("--quick", action="store_true")

    # Compatibility attributes used by hard_lgn_benchmark training helpers.
    parser.set_defaults(
        abc_stats=False,
        abc_path="/home/spco/boolean_sat/abc/abc",
        abc_max_gates=10000,
        inference_bench=False,
        inference_bench_device="cpu",
        inference_bench_repeats=0,
        inference_bench_warmup=0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be positive")
    if args.candidate_set_size < 1:
        raise ValueError("--candidate-set-size must be positive")
    if args.quick:
        args.datasets = ["parity6", "majority7"]
        args.seeds = [0]
        args.width = min(args.width, 32)
        args.layers = min(args.layers, 3)
        args.num_candidates = min(args.num_candidates, 4)
        args.candidate_set_size = min(args.candidate_set_size, 3)
        args.search_epochs = min(args.search_epochs, 4)
        args.final_epochs = min(args.final_epochs, 8)
        args.search_block_epochs = min(args.search_block_epochs, 3)
        args.final_block_epochs = min(args.final_block_epochs, 6)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    start_all = time.perf_counter()
    print(f"device={device} out_dir={out_dir} base_method={args.base_method}")

    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            dataset = load_dataset(dataset_name, seed, args)
            if args.target_acc_override is not None:
                dataset.target_acc = args.target_acc_override
            if args.width % dataset.num_classes != 0:
                raise ValueError(f"width={args.width} must be divisible by num_classes={dataset.num_classes}")
            if args.width * 2 < dataset.input_dim:
                raise ValueError(f"width={args.width} too small for input_dim={dataset.input_dim}")

            selection_dataset, final_dataset = split_train_val(dataset, seed + 131071, args.val_frac)
            init_seed = args.init_seed_base + seed
            baseline_wiring_seed = seed
            print(
                f"dataset={dataset.name} seed={seed} selection_train={len(selection_dataset.y_train)} val={len(selection_dataset.y_test)} test={len(final_dataset.y_test)}",
                flush=True,
            )

            final_args = clone_args(args, epochs=args.final_epochs, block_epochs=args.final_block_epochs)
            baseline_arch = build_architecture_separate(
                dataset.input_dim,
                args.width,
                args.layers,
                wiring_seed=baseline_wiring_seed,
                init_seed=init_seed,
            )
            baseline_final = run_base_method(args.base_method, final_dataset, baseline_arch, final_args, device, seed)
            final_rows.append(
                final_result_row(
                    method="fixed_random",
                    base_method=args.base_method,
                    dataset=dataset.name,
                    seed=seed,
                    wiring_seed=baseline_wiring_seed,
                    init_seed=init_seed,
                    selection_row=None,
                    search_candidates=1,
                    search_time=0.0,
                    final_row=baseline_final,
                    source_note="single fixed-random wiring; held-out test",
                )
            )

            if args.matched_budget_baseline:
                matched_epochs = args.final_epochs + args.num_candidates * args.search_epochs
                matched_block_epochs = args.final_block_epochs + args.num_candidates * args.search_block_epochs
                matched_args = clone_args(args, epochs=matched_epochs, block_epochs=matched_block_epochs)
                matched_final = run_base_method(args.base_method, final_dataset, baseline_arch, matched_args, device, seed)
                final_rows.append(
                    final_result_row(
                        method="fixed_random_matched_budget",
                        base_method=args.base_method,
                        dataset=dataset.name,
                        seed=seed,
                        wiring_seed=baseline_wiring_seed,
                        init_seed=init_seed,
                        selection_row=None,
                        search_candidates=1,
                        search_time=0.0,
                        final_row=matched_final,
                        source_note="single fixed-random wiring; epoch budget includes search-equivalent epochs",
                    )
                )

            search_args = clone_args(args, epochs=args.search_epochs, block_epochs=args.search_block_epochs)
            candidate_metrics: list[tuple[int, MetricsRow]] = []
            search_start = time.perf_counter()
            for candidate_id in range(args.num_candidates):
                wiring_seed = args.wiring_seed_base + seed * 1000 + candidate_id
                arch = build_architecture_separate(
                    dataset.input_dim,
                    args.width,
                    args.layers,
                    wiring_seed=wiring_seed,
                    init_seed=init_seed,
                )
                row = run_base_method(args.base_method, selection_dataset, arch, search_args, device, seed)
                candidate_metrics.append((wiring_seed, row))
                print(
                    f"  candidate={candidate_id} wiring_seed={wiring_seed} hard_val_acc={row.discrete_acc:.4f} native_gap={row.path_acc_gap:.4f}",
                    flush=True,
                )
            search_time = time.perf_counter() - search_start
            sorted_candidates = sorted(
                candidate_metrics,
                key=lambda item: (-item[1].discrete_acc, item[1].path_acc_gap, item[1].unused_gate_ratio, item[1].fanout_max, item[1].train_time),
            )
            for rank, (wiring_seed, row) in enumerate(sorted_candidates, start=1):
                candidate_rows.append(
                    candidate_row(
                        method="random_wiring_search_freeze",
                        dataset=dataset.name,
                        seed=seed,
                        candidate_rank=rank,
                        wiring_seed=wiring_seed,
                        init_seed=init_seed,
                        base_method=args.base_method,
                        candidate_set_size=1,
                        row=row,
                    )
                )

            selected_wiring_seed, selected_val = select_candidate(candidate_metrics)
            selected_arch = build_architecture_separate(
                dataset.input_dim,
                args.width,
                args.layers,
                wiring_seed=selected_wiring_seed,
                init_seed=init_seed,
            )
            selected_final = run_base_method(args.base_method, final_dataset, selected_arch, final_args, device, seed)
            final_rows.append(
                final_result_row(
                    method="random_wiring_search_freeze",
                    base_method=args.base_method,
                    dataset=dataset.name,
                    seed=seed,
                    wiring_seed=selected_wiring_seed,
                    init_seed=init_seed,
                    selection_row=selected_val,
                    search_candidates=args.num_candidates,
                    search_time=search_time,
                    final_row=selected_final,
                    source_note="selected on validation hard accuracy; retrained on full train; held-out test",
                )
            )

            if args.candidate_set_learnable:
                candidate_set_seed = args.wiring_seed_base + seed * 1000 + args.candidate_set_seed_offset
                print(
                    f"  candidate_set_learnable wiring_seed={candidate_set_seed} set_size={args.candidate_set_size}",
                    flush=True,
                )
                candidate_selection, candidate_arch, candidate_search_time = train_candidate_set_connectivity(
                    selection_dataset,
                    args,
                    device,
                    seed,
                    wiring_seed=candidate_set_seed,
                    init_seed=init_seed,
                )
                candidate_rows.append(
                    candidate_row(
                        method="candidate_set_learnable_freeze",
                        dataset=dataset.name,
                        seed=seed,
                        candidate_rank=1,
                        wiring_seed=candidate_set_seed,
                        init_seed=init_seed,
                        base_method=args.base_method,
                        candidate_set_size=args.candidate_set_size,
                        row=candidate_selection,
                    )
                )
                candidate_final = run_base_method(args.base_method, final_dataset, candidate_arch, final_args, device, seed)
                final_rows.append(
                    final_result_row(
                        method="candidate_set_learnable_freeze",
                        base_method=args.base_method,
                        dataset=dataset.name,
                        seed=seed,
                        wiring_seed=candidate_set_seed,
                        init_seed=init_seed,
                        selection_row=candidate_selection,
                        search_candidates=args.candidate_set_size,
                        search_time=candidate_search_time,
                        final_row=candidate_final,
                        source_note=(
                            "candidate-set connection logits selected fixed wiring; "
                            "final ordinary gate logits initialized from trained selection logits"
                            if args.candidate_use_trained_logits
                            else "candidate-set connection logits selected fixed wiring; final ordinary gate logits reinitialized from common init seed"
                        ),
                    )
                )

            write_csv(out_dir / "wiring_search_results.partial.csv", final_rows)
            write_csv(out_dir / "wiring_search_candidates.partial.csv", candidate_rows)

    elapsed = time.perf_counter() - start_all
    write_csv(out_dir / "wiring_search_results.csv", final_rows)
    write_csv(out_dir / "wiring_search_candidates.csv", candidate_rows)
    (out_dir / "wiring_search_report.md").write_text(markdown_report(final_rows, candidate_rows))
    (out_dir / "run_meta.txt").write_text(f"elapsed_seconds={elapsed:.6f}\n")
    print(f"wrote {len(final_rows)} final rows and {len(candidate_rows)} candidate rows to {out_dir}")


if __name__ == "__main__":
    main()
