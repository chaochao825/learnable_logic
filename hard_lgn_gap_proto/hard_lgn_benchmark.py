#!/usr/bin/env python3
"""Compare DLGN, Gumbel-ST LGN, and block-wise Hard-LGN prototypes.

This is a compact research prototype, not a full reproduction of Mind the Gap.
It keeps the comparison axes explicit: same architecture, same wiring, same
initial logits, same datasets, same seeds, and direct soft-vs-discrete metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - only used when sklearn is unavailable.
    load_digits = None
    train_test_split = None

try:
    from torchvision import datasets as tv_datasets
    from torchvision import transforms as tv_transforms
except Exception:  # pragma: no cover - torchvision is optional for Boolean runs.
    tv_datasets = None
    tv_transforms = None


GATE_NAMES = [
    "zero",
    "and",
    "not_implies",
    "a",
    "not_implied_by",
    "b",
    "xor",
    "or",
    "not_or",
    "not_xor",
    "not_b",
    "implied_by",
    "not_a",
    "implies",
    "not_and",
    "one",
]

# Gate table follows difflogic.functional order: columns are AB=00,01,10,11.
GATE_TRUTH = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    dtype=torch.float32,
)


@dataclass
class DatasetBundle:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    input_dim: int
    num_classes: int
    target_acc: float


@dataclass
class MetricsRow:
    method: str
    dataset: str
    seed: int
    soft_acc: float
    discrete_acc: float
    acc_gap: float
    soft_loss: float
    discrete_loss: float
    loss_gap: float
    path_soft_acc: float
    path_discrete_acc: float
    path_acc_gap: float
    path_soft_loss: float
    path_discrete_loss: float
    path_loss_gap: float
    train_time: float
    epochs_to_target: int
    time_to_target: float
    unused_gate_ratio: float
    gate_count: int
    depth: int
    fanout_max: int
    soft_inference_samples_per_sec: float
    discrete_inference_samples_per_sec: float
    inference_bench_repeats: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gate_outputs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return all 16 differentiable two-input gate outputs."""

    return torch.stack(
        [
            torch.zeros_like(a),
            a * b,
            a - a * b,
            a,
            b - a * b,
            b,
            a + b - 2 * a * b,
            a + b - a * b,
            1 - (a + b - a * b),
            1 - (a + b - 2 * a * b),
            1 - b,
            1 - b + a * b,
            1 - a,
            1 - a + a * b,
            1 - a * b,
            torch.ones_like(a),
        ],
        dim=-1,
    )


def weighted_gate(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (gate_outputs(a, b) * weights).sum(dim=-1)


def hard_gate(a: torch.Tensor, b: torch.Tensor, op_ids: torch.Tensor) -> torch.Tensor:
    truth = GATE_TRUTH.to(device=a.device, dtype=a.dtype)
    idx = (a.round().to(torch.long) * 2 + b.round().to(torch.long)).clamp(0, 3)
    return truth[op_ids.to(a.device).view(1, -1), idx]


class GroupSum(nn.Module):
    def __init__(self, num_classes: int, tau: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.num_classes != 0:
            raise ValueError((x.shape, self.num_classes))
        per_class = x.shape[-1] // self.num_classes
        return x.reshape(*x.shape[:-1], self.num_classes, per_class).sum(-1) / self.tau


class SoftLogicLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        indices_0: torch.Tensor,
        indices_1: torch.Tensor,
        init_logits: torch.Tensor,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer("indices_0", indices_0.clone().long())
        self.register_buffer("indices_1", indices_1.clone().long())
        self.logits = nn.Parameter(init_logits.clone().float())

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "soft",
        tau: float = 1.0,
        gumbel_hard: bool = False,
    ) -> torch.Tensor:
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]

        if mode == "hard":
            probs = F.one_hot(self.logits.argmax(-1), 16).to(dtype=x.dtype)
        elif mode == "gumbel":
            probs = F.gumbel_softmax(self.logits, tau=tau, hard=gumbel_hard, dim=-1).to(dtype=x.dtype)
        elif mode == "soft":
            probs = F.softmax(self.logits / tau, dim=-1).to(dtype=x.dtype)
        else:
            raise ValueError(mode)
        return weighted_gate(a, b, probs)

    def entropy(self, tau: float = 1.0) -> torch.Tensor:
        probs = F.softmax(self.logits / tau, dim=-1)
        return -(probs * (probs.clamp_min(1e-8)).log()).sum(-1).mean()

    def hard_ops_argmax(self) -> torch.Tensor:
        return self.logits.detach().argmax(-1).cpu()


class FrozenHardLogicLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        indices_0: torch.Tensor,
        indices_1: torch.Tensor,
        op_ids: torch.Tensor,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer("indices_0", indices_0.clone().long())
        self.register_buffer("indices_1", indices_1.clone().long())
        self.register_buffer("op_ids", op_ids.clone().long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]
        return hard_gate(a, b, self.op_ids)


class LogicNet(nn.Module):
    def __init__(self, layers: Iterable[nn.Module], num_classes: int, group_tau: float = 1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        self.group_sum = GroupSum(num_classes, tau=group_tau)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "soft",
        tau: float = 1.0,
        gumbel_hard: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, SoftLogicLayer):
                layer_mode = "gumbel" if mode == "gumbel" else mode
                x = layer(x, mode=layer_mode, tau=tau, gumbel_hard=gumbel_hard)
            else:
                x = layer(x)
        return self.group_sum(x)

    def soft_layers(self) -> list[SoftLogicLayer]:
        return [layer for layer in self.layers if isinstance(layer, SoftLogicLayer)]


def random_connections(in_dim: int, out_dim: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if out_dim * 2 < in_dim:
        raise ValueError(f"out_dim={out_dim} cannot cover in_dim={in_dim}")
    c = torch.randperm(2 * out_dim, generator=generator) % in_dim
    c = torch.randperm(in_dim, generator=generator)[c].reshape(2, out_dim)
    return c[0].long(), c[1].long()


def check_difflogic_compatibility(out_dir: Path) -> dict[str, object]:
    """Verify that this prototype matches the installed difflogic primitives.

    The benchmark keeps its layers compact so block-wise refitting can inspect
    every intermediate tensor. This check pins that implementation against the
    reusable `/home/spco/convlogic` DLGN primitives used as the reference.
    """

    try:
        from difflogic import GroupSum as DiffLogicGroupSum
        from difflogic import LogicLayer as DiffLogicLayer
        from difflogic.functional import bin_op, bin_op_s
    except Exception as exc:  # pragma: no cover - environment-dependent check.
        report: dict[str, object] = {
            "status": "import_failed",
            "error": repr(exc),
            "hint": "Set PYTHONPATH=/home/spco/convlogic/src:$PYTHONPATH on the 210 server.",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "difflogic_compat.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    our_gate_values = gate_outputs(a, b)
    ref_gate_values = torch.stack([bin_op(a, b, gate_id) for gate_id in range(16)], dim=-1)
    truth_table_values = GATE_TRUTH.T

    logits = torch.randn(4, 16, generator=torch.Generator().manual_seed(123))
    probs = torch.softmax(logits, dim=-1)
    a_batch = torch.tensor([[0.0, 0.2, 0.7, 1.0], [1.0, 0.8, 0.3, 0.0]])
    b_batch = torch.tensor([[0.0, 0.6, 0.1, 1.0], [0.0, 0.4, 0.9, 1.0]])
    weighted_diff = (weighted_gate(a_batch, b_batch, probs) - bin_op_s(a_batch, b_batch, probs)).abs().max()

    groupsum_input = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    groupsum_tau = 1.7
    groupsum_diff = (
        GroupSum(3, tau=groupsum_tau)(groupsum_input)
        - DiffLogicGroupSum(3, tau=groupsum_tau)(groupsum_input)
    ).abs().max()

    in_dim, out_dim = 6, 8
    generator = torch.Generator().manual_seed(456)
    idx0, idx1 = random_connections(in_dim, out_dim, generator)
    layer_logits = torch.randn(out_dim, 16, generator=torch.Generator().manual_seed(789))
    x = torch.rand(5, in_dim, generator=torch.Generator().manual_seed(321))

    our_layer = SoftLogicLayer(in_dim, out_dim, idx0, idx1, layer_logits)
    ref_layer = DiffLogicLayer(in_dim=in_dim, out_dim=out_dim, implementation="python")
    with torch.no_grad():
        ref_layer.indices_0.copy_(idx0)
        ref_layer.indices_1.copy_(idx1)
        ref_layer.weights.copy_(layer_logits)

    our_soft = our_layer(x, mode="soft", tau=1.0)
    ref_layer.train()
    ref_soft = ref_layer(x)
    our_hard = our_layer(x, mode="hard", tau=1.0)
    ref_layer.eval()
    ref_hard = ref_layer(x)

    connection_ref = DiffLogicLayer(in_dim=in_dim, out_dim=out_dim, implementation="python")
    torch.manual_seed(654)
    ref_idx0, ref_idx1 = connection_ref.get_connections("random")
    our_idx0, our_idx1 = random_connections(in_dim, out_dim, torch.Generator().manual_seed(654))

    report = {
        "status": "ok",
        "checks": {
            "gate_outputs_max_abs_diff": float((our_gate_values - ref_gate_values).abs().max().item()),
            "gate_truth_table_max_abs_diff": float((our_gate_values - truth_table_values).abs().max().item()),
            "weighted_gate_max_abs_diff": float(weighted_diff.item()),
            "groupsum_max_abs_diff": float(groupsum_diff.item()),
            "logiclayer_soft_forward_max_abs_diff": float((our_soft - ref_soft).abs().max().item()),
            "logiclayer_hard_forward_max_abs_diff": float((our_hard - ref_hard).abs().max().item()),
            "random_connection_indices_match": bool(torch.equal(ref_idx0, our_idx0) and torch.equal(ref_idx1, our_idx1)),
        },
        "reference": {
            "logic_layer": "/home/spco/convlogic/src/difflogic/difflogic.py",
            "functional": "/home/spco/convlogic/src/difflogic/functional.py",
        },
    }
    checks = report["checks"]
    assert isinstance(checks, dict)
    numeric_ok = all(
        abs(float(value)) <= 1e-6
        for key, value in checks.items()
        if key.endswith("_max_abs_diff")
    )
    report["all_pass"] = bool(numeric_ok and checks["random_connection_indices_match"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "difflogic_compat.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_architecture(input_dim: int, width: int, depth: int, seed: int) -> list[dict[str, torch.Tensor | int]]:
    generator = torch.Generator().manual_seed(seed)
    arch = []
    in_dim = input_dim
    for _ in range(depth):
        idx0, idx1 = random_connections(in_dim, width, generator)
        logits = torch.randn(width, 16, generator=generator)
        arch.append({"in_dim": in_dim, "out_dim": width, "idx0": idx0, "idx1": idx1, "logits": logits})
        in_dim = width
    return arch


def make_soft_layer(spec: dict[str, torch.Tensor | int]) -> SoftLogicLayer:
    return SoftLogicLayer(
        int(spec["in_dim"]),
        int(spec["out_dim"]),
        spec["idx0"],  # type: ignore[arg-type]
        spec["idx1"],  # type: ignore[arg-type]
        spec["logits"],  # type: ignore[arg-type]
    )


def make_soft_net(arch: list[dict[str, torch.Tensor | int]], num_classes: int, group_tau: float) -> LogicNet:
    return LogicNet([make_soft_layer(spec) for spec in arch], num_classes=num_classes, group_tau=group_tau)


def clone_soft_layer(layer: SoftLogicLayer) -> SoftLogicLayer:
    return SoftLogicLayer(
        layer.in_dim,
        layer.out_dim,
        layer.indices_0.detach().cpu(),
        layer.indices_1.detach().cpu(),
        layer.logits.detach().cpu(),
    )


def exact_binary_inputs(n_bits: int) -> torch.Tensor:
    values = torch.arange(2**n_bits, dtype=torch.long)
    shifts = torch.arange(n_bits - 1, -1, -1, dtype=torch.long)
    return ((values[:, None] >> shifts[None, :]) & 1).float()


def split_all(x: torch.Tensor, y: torch.Tensor, seed: int, train_frac: float = 0.75) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)
    cut = int(round(train_frac * x.shape[0]))
    if x.shape[0] > 1:
        cut = max(1, min(x.shape[0] - 1, cut))
    tr, te = perm[:cut], perm[cut:]
    return x[tr], y[tr], x[te], y[te]


def make_boolean_dataset(name: str, seed: int) -> DatasetBundle:
    if name.startswith("parity"):
        n_bits = int(name.removeprefix("parity"))
        x = exact_binary_inputs(n_bits)
        y = (x.sum(dim=1).long() % 2)
        target = 0.99
    elif name.startswith("majority"):
        n_bits = int(name.removeprefix("majority"))
        x = exact_binary_inputs(n_bits)
        y = (x.sum(dim=1) >= ((n_bits + 1) // 2)).long()
        target = 0.99
    elif name.startswith("random_sparse"):
        n_bits = int(name.removeprefix("random_sparse"))
        x = exact_binary_inputs(n_bits)
        rng = np.random.default_rng(seed + 1009)
        n_terms = max(4, n_bits // 2)
        term_width = min(3, n_bits)
        y_np = np.zeros(x.shape[0], dtype=np.bool_)
        x_np = x.numpy().astype(np.bool_)
        for _ in range(n_terms):
            dims = rng.choice(n_bits, size=term_width, replace=False)
            signs = rng.integers(0, 2, size=term_width).astype(np.bool_)
            term = np.ones(x.shape[0], dtype=np.bool_)
            for dim, sign in zip(dims, signs, strict=False):
                term &= x_np[:, dim] if sign else ~x_np[:, dim]
            y_np |= term
        y = torch.from_numpy(y_np.astype(np.int64))
        target = 0.95
    else:
        raise ValueError(f"Unknown Boolean dataset: {name}")

    x_train, y_train, x_test, y_test = split_all(x, y, seed=seed)
    return DatasetBundle(name, x_train, y_train, x_test, y_test, x.shape[1], 2, target)


def make_digits_dataset(seed: int) -> DatasetBundle:
    if load_digits is None or train_test_split is None:
        raise RuntimeError("sklearn is required for the digits dataset")
    data = load_digits()
    x = (torch.tensor(data.data, dtype=torch.float32) > 8.0).float()
    y = torch.tensor(data.target, dtype=torch.long)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=0.25,
        random_state=seed,
        stratify=y.numpy(),
    )
    return DatasetBundle(
        "digits",
        x[idx_train],
        y[idx_train],
        x[idx_test],
        y[idx_test],
        x.shape[1],
        10,
        0.80,
    )


def threshold_flatten_image(x: torch.Tensor, threshold_levels: int) -> torch.Tensor:
    thresholds = torch.linspace(1, threshold_levels, threshold_levels, dtype=torch.float32) / (threshold_levels + 1)
    bits = [(x > threshold).float().flatten() for threshold in thresholds]
    return torch.cat(bits, dim=0)


def stratified_subset_indices(labels: torch.Tensor, max_count: int | None, seed: int) -> torch.Tensor:
    if max_count is None or max_count <= 0 or max_count >= labels.numel():
        return torch.arange(labels.numel())
    generator = torch.Generator().manual_seed(seed)
    classes = labels.unique(sorted=True)
    per_class = max(1, max_count // len(classes))
    selected = []
    for cls in classes.tolist():
        cls_idx = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        cls_idx = cls_idx[torch.randperm(cls_idx.numel(), generator=generator)]
        selected.append(cls_idx[:per_class])
    idx = torch.cat(selected)
    if idx.numel() < max_count:
        remaining_mask = torch.ones(labels.numel(), dtype=torch.bool)
        remaining_mask[idx] = False
        remaining = torch.nonzero(remaining_mask, as_tuple=False).flatten()
        remaining = remaining[torch.randperm(remaining.numel(), generator=generator)]
        idx = torch.cat([idx, remaining[: max_count - idx.numel()]])
    idx = idx[torch.randperm(idx.numel(), generator=generator)]
    return idx[:max_count]


def tensorize_torchvision_dataset(
    dataset: torch.utils.data.Dataset,
    threshold_levels: int,
    max_count: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.tensor([int(dataset[i][1]) for i in range(len(dataset))], dtype=torch.long)
    idx = stratified_subset_indices(labels, max_count, seed)
    xs = []
    ys = []
    for item_idx in idx.tolist():
        x, y = dataset[item_idx]
        xs.append(threshold_flatten_image(x, threshold_levels))
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_torchvision_dataset(name: str, seed: int, args: argparse.Namespace) -> DatasetBundle:
    if tv_datasets is None or tv_transforms is None:
        raise RuntimeError("torchvision is required for MNIST/CIFAR datasets")
    dataset_key = name.lower()
    if dataset_key in {"mnist", "binarized_mnist", "mnist1"}:
        dataset_class = tv_datasets.MNIST
        target = 0.80
        canonical_name = "binarized_mnist"
    elif dataset_key.startswith("cifar10"):
        dataset_class = tv_datasets.CIFAR10
        target = 0.30
        canonical_name = "cifar10_small"
    else:
        raise ValueError(f"Unknown torchvision dataset: {name}")

    transform = tv_transforms.ToTensor()
    try:
        train_set = dataset_class(root=args.data_dir, train=True, download=args.download_data, transform=transform)
        test_set = dataset_class(root=args.data_dir, train=False, download=args.download_data, transform=transform)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{canonical_name} not found under {args.data_dir!r}. "
            "Pass --download-data to allow torchvision download, or set --data-dir to an existing dataset root."
        ) from exc

    x_train, y_train = tensorize_torchvision_dataset(train_set, args.threshold_levels, args.image_max_train, seed)
    x_test, y_test = tensorize_torchvision_dataset(test_set, args.threshold_levels, args.image_max_test, seed + 17)
    return DatasetBundle(
        canonical_name,
        x_train,
        y_train,
        x_test,
        y_test,
        x_train.shape[1],
        10,
        target,
    )


def load_dataset(name: str, seed: int, args: argparse.Namespace) -> DatasetBundle:
    if name == "digits":
        return make_digits_dataset(seed)
    if name in {"mnist", "binarized_mnist", "mnist1"} or name.startswith("cifar10"):
        return make_torchvision_dataset(name, seed, args)
    return make_boolean_dataset(name, seed)


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=generator)


@torch.no_grad()
def evaluate(
    model: LogicNet,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    tau: float,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    losses = []
    correct = 0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits = model(xb, mode=mode, tau=tau)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        losses.append(loss.detach().cpu())
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.numel()
    if was_training:
        model.train()
    return correct / total, torch.stack(losses).sum().item() / total


@torch.no_grad()
def benchmark_inference_samples_per_sec(
    model: LogicNet,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str,
    tau: float,
    repeats: int,
    warmup: int,
) -> float:
    if repeats <= 0:
        return math.nan
    was_training = model.training
    model.to(device)
    model.eval()
    x_device = x.to(device)

    def run_once() -> None:
        for start in range(0, x_device.shape[0], batch_size):
            xb = x_device[start : start + batch_size]
            _ = model(xb, mode=mode, tau=tau)

    for _ in range(max(warmup, 0)):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    for _ in range(repeats):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    if was_training:
        model.train()
    return (x_device.shape[0] * repeats) / max(elapsed, 1e-12)


def resolve_inference_bench_device(args: argparse.Namespace) -> torch.device:
    if args.inference_bench_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_benchmark_inference_pair(
    args: argparse.Namespace,
    soft_model: LogicNet,
    hard_model: LogicNet,
    x: torch.Tensor,
    soft_tau: float,
) -> tuple[float, float, int]:
    if not args.inference_bench:
        return math.nan, math.nan, 0
    bench_device = resolve_inference_bench_device(args)
    soft_throughput = benchmark_inference_samples_per_sec(
        soft_model,
        x,
        args.eval_batch_size,
        bench_device,
        "soft",
        soft_tau,
        args.inference_bench_repeats,
        args.inference_bench_warmup,
    )
    hard_throughput = benchmark_inference_samples_per_sec(
        hard_model,
        x,
        args.eval_batch_size,
        bench_device,
        "hard",
        1.0,
        args.inference_bench_repeats,
        args.inference_bench_warmup,
    )
    return soft_throughput, hard_throughput, args.inference_bench_repeats


@torch.no_grad()
def collect_layer_outputs(
    layers: Iterable[nn.Module],
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str = "hard",
    tau: float = 1.0,
) -> list[torch.Tensor]:
    layer_list = [layer.to(device) for layer in layers]
    outputs = [torch.empty((x.shape[0], layer.out_dim), dtype=torch.float32) for layer in layer_list]  # type: ignore[attr-defined]
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        for i, layer in enumerate(layer_list):
            if isinstance(layer, SoftLogicLayer):
                xb = layer(xb, mode=mode, tau=tau)
            else:
                xb = layer(xb)
            outputs[i][start : start + xb.shape[0]] = xb.detach().cpu()
    return outputs


@torch.no_grad()
def apply_layers(
    layers: Iterable[nn.Module],
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    mode: str = "soft",
    tau: float = 1.0,
) -> torch.Tensor:
    layer_list = [layer.to(device) for layer in layers]
    if not layer_list:
        return x.clone()
    out = None
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        for layer in layer_list:
            if isinstance(layer, SoftLogicLayer):
                xb = layer(xb, mode=mode, tau=tau)
            else:
                xb = layer(xb)
        xb_cpu = xb.detach().cpu()
        if out is None:
            out = torch.empty((x.shape[0], xb_cpu.shape[1]), dtype=torch.float32)
        out[start : start + xb_cpu.shape[0]] = xb_cpu
    assert out is not None
    return out


def compute_unused_gate_ratio(
    layers: Iterable[nn.Module],
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    outputs = collect_layer_outputs(layers, x, batch_size, device, mode="hard")
    inactive = 0
    total = 0
    for y in outputs:
        total += y.shape[1]
        inactive += ((y.max(dim=0).values - y.min(dim=0).values) < 1e-6).sum().item()
    return inactive / max(total, 1)


@torch.no_grad()
def layer_gap_diagnostics(
    method: str,
    dataset_name: str,
    seed: int,
    comparison: str,
    soft_layers: Iterable[nn.Module],
    hard_layers: Iterable[nn.Module],
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    soft_tau: float = 1.0,
) -> list[dict[str, float | int | str]]:
    soft_outputs = collect_layer_outputs(soft_layers, x, batch_size, device, mode="soft", tau=soft_tau)
    hard_outputs = collect_layer_outputs(hard_layers, x, batch_size, device, mode="hard", tau=1.0)
    if len(soft_outputs) != len(hard_outputs):
        raise ValueError((len(soft_outputs), len(hard_outputs)))

    rows: list[dict[str, float | int | str]] = []
    for layer_id, (soft_y, hard_y) in enumerate(zip(soft_outputs, hard_outputs, strict=True)):
        diff = (soft_y - hard_y).abs()
        soft_inactive = ((soft_y.max(dim=0).values - soft_y.min(dim=0).values) < 1e-6).float().mean().item()
        hard_inactive = ((hard_y.max(dim=0).values - hard_y.min(dim=0).values) < 1e-6).float().mean().item()
        rows.append(
            {
                "method": method,
                "dataset": dataset_name,
                "seed": seed,
                "comparison": comparison,
                "layer": layer_id,
                "prefix_depth": layer_id + 1,
                "mean_abs_diff": float(diff.mean().item()),
                "max_abs_diff": float(diff.max().item()),
                "mse": float((diff.square()).mean().item()),
                "binary_flip_ratio": float(((soft_y >= 0.5) != (hard_y >= 0.5)).float().mean().item()),
                "soft_inactive_ratio": float(soft_inactive),
                "hard_inactive_ratio": float(hard_inactive),
            }
        )
    return rows


def fanout_max(layers: Iterable[nn.Module]) -> int:
    max_fanout = 0
    for layer in layers:
        indices = torch.cat([layer.indices_0.detach().cpu(), layer.indices_1.detach().cpu()])  # type: ignore[attr-defined]
        counts = torch.bincount(indices, minlength=layer.in_dim)  # type: ignore[attr-defined]
        max_fanout = max(max_fanout, int(counts.max().item()))
    return max_fanout


def gate_count(layers: Iterable[nn.Module]) -> int:
    return sum(int(layer.out_dim) for layer in layers)  # type: ignore[attr-defined]


def blif_cover_lines(op_id: int) -> list[str]:
    truth = GATE_TRUTH[op_id].to(torch.long).tolist()
    lines = []
    for idx, value in enumerate(truth):
        if value:
            a = (idx >> 1) & 1
            b = idx & 1
            lines.append(f"{a}{b} 1")
    return lines


def export_hard_layers_to_blif(
    layers: Iterable[nn.Module],
    input_dim: int,
    path: Path,
    model_name: str,
) -> None:
    hard_layers = list(layers)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_names = [f"i{i}" for i in range(input_dim)]
    outputs = [f"l{len(hard_layers) - 1}_g{j}" for j in range(hard_layers[-1].out_dim)] if hard_layers else prev_names
    lines = [
        f".model {model_name}",
        ".inputs " + " ".join(prev_names),
        ".outputs " + " ".join(outputs),
    ]

    for layer_id, layer in enumerate(hard_layers):
        if not isinstance(layer, FrozenHardLogicLayer):
            raise TypeError(f"ABC export requires frozen hard layers, got {type(layer)}")
        next_names = []
        idx0 = layer.indices_0.detach().cpu().tolist()
        idx1 = layer.indices_1.detach().cpu().tolist()
        ops = layer.op_ids.detach().cpu().tolist()
        for gate_id, (a_idx, b_idx, op_id) in enumerate(zip(idx0, idx1, ops, strict=False)):
            out_name = f"l{layer_id}_g{gate_id}"
            next_names.append(out_name)
            if int(op_id) == 0:
                lines.append(f".names {out_name}")
            elif int(op_id) == 15:
                lines.append(f".names {out_name}")
                lines.append("1")
            else:
                lines.append(f".names {prev_names[a_idx]} {prev_names[b_idx]} {out_name}")
                lines.extend(blif_cover_lines(int(op_id)))
        prev_names = next_names

    lines.append(".end")
    path.write_text("\n".join(lines) + "\n")


def parse_abc_print_stats(output: str) -> dict[str, float | int | str]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    stat_lines = [line for line in clean.splitlines() if "i/o =" in line]
    stats: dict[str, float | int | str] = {}
    for prefix, line in zip(["abc_pre", "abc_post"], stat_lines[-2:], strict=False):
        io_match = re.search(r"i/o\s*=\s*([0-9]+)\s*/\s*([0-9]+)", line)
        if io_match:
            stats[f"{prefix}_inputs"] = int(io_match.group(1))
            stats[f"{prefix}_outputs"] = int(io_match.group(2))
        for key, value in re.findall(r"(lat|nd|edge|cube|lev|and)\s*=\s*([0-9]+)", line):
            stats[f"{prefix}_{key}"] = int(value)
    stats["abc_stat_lines"] = len(stat_lines)
    return stats


def maybe_run_abc_stats(
    args: argparse.Namespace,
    method: str,
    dataset: str,
    seed: int,
    hard_layers: list[nn.Module],
    input_dim: int,
) -> dict[str, float | int | str] | None:
    if not args.abc_stats:
        return None
    base = Path(args.out_dir) / "synthesis"
    safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    stem = f"{safe_dataset}__{safe_method}__seed{seed}"
    blif_name = f"{stem}.blif"
    log_name = f"{stem}.abc.log"
    blif_path = base / blif_name
    log_path = base / log_name
    row: dict[str, float | int | str] = {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "abc_status": "not_run",
        "abc_path": args.abc_path,
        "blif_path": str(Path("synthesis") / blif_name),
        "abc_log_path": str(Path("synthesis") / log_name),
        "pre_gate_count": gate_count(hard_layers),
        "pre_depth": len(hard_layers),
        "pre_fanout_max": fanout_max(hard_layers),
    }
    if gate_count(hard_layers) > args.abc_max_gates:
        row["abc_status"] = "skipped_gate_limit"
        return row
    abc_path = Path(args.abc_path)
    if not abc_path.exists():
        row["abc_status"] = "missing_abc"
        return row

    try:
        export_hard_layers_to_blif(hard_layers, input_dim, blif_path, stem)
        cmd = [
            str(abc_path),
            "-c",
            f"read_blif {row['blif_path']}; print_stats; strash; dc2; print_stats",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, cwd=Path(args.out_dir))
        log_path.write_text(proc.stdout + proc.stderr)
        row["abc_returncode"] = proc.returncode
        parsed = parse_abc_print_stats(proc.stdout + proc.stderr)
        row.update(parsed)
        row["abc_status"] = "ok" if proc.returncode == 0 and parsed.get("abc_stat_lines", 0) >= 2 else "abc_failed"
    except Exception as exc:  # pragma: no cover - operational path.
        row["abc_status"] = "exception"
        row["abc_error"] = repr(exc)
    return row


def temp_schedule(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    ratio = (epoch - 1) / (epochs - 1)
    return start * ((end / start) ** ratio)


def train_end_to_end(
    method: str,
    dataset: DatasetBundle,
    arch: list[dict[str, torch.Tensor | int]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[
    MetricsRow,
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    dict[str, float | int | str] | None,
]:
    model = make_soft_net(arch, dataset.num_classes, args.group_tau).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(dataset.x_train, dataset.y_train, args.batch_size, seed)
    epoch_rows = []
    epochs_to_target = -1
    time_to_target = -1.0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if method == "dlgn_anneal":
            tau = temp_schedule(epoch, args.epochs, args.temp_start, args.temp_end)
            entropy_coef = args.entropy_coef * epoch / args.epochs
        elif method in {"gumbel_st", "gumbel_soft"}:
            tau = temp_schedule(epoch, args.epochs, args.gumbel_temp_start, args.gumbel_temp_end)
            entropy_coef = 0.0
        else:
            tau = 1.0
            entropy_coef = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if method == "gumbel_st":
                logits = model(xb, mode="gumbel", tau=tau, gumbel_hard=True)
            elif method == "gumbel_soft":
                logits = model(xb, mode="gumbel", tau=tau, gumbel_hard=False)
            else:
                logits = model(xb, mode="soft", tau=tau)
            loss = F.cross_entropy(logits, yb)
            if entropy_coef:
                loss = loss + entropy_coef * sum(layer.entropy(tau) for layer in model.soft_layers())
            loss.backward()
            opt.step()

        soft_acc, soft_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "soft", tau)
        disc_acc, disc_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", tau)
        elapsed_time = time.perf_counter() - start_time
        if epochs_to_target < 0 and disc_acc >= dataset.target_acc:
            epochs_to_target = epoch
            time_to_target = elapsed_time
        epoch_rows.append(
            {
                "method": method,
                "dataset": dataset.name,
                "seed": seed,
                "epoch": epoch,
                "global_epoch": epoch,
                "elapsed_time": elapsed_time,
                "soft_acc": soft_acc,
                "discrete_acc": disc_acc,
                "soft_loss": soft_loss,
                "discrete_loss": disc_loss,
                "tau": tau,
            }
        )

    train_time = time.perf_counter() - start_time
    final_tau = 1.0
    if method == "dlgn_anneal":
        final_tau = args.temp_end
    elif method in {"gumbel_st", "gumbel_soft"}:
        final_tau = args.gumbel_temp_end
    soft_acc, soft_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "soft", final_tau)
    disc_acc, disc_loss = evaluate(model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", final_tau)
    layers = list(model.layers)
    hard_layers = [argmax_hard_layer(layer) for layer in model.soft_layers()]
    hard_model = LogicNet(hard_layers, dataset.num_classes, args.group_tau)
    soft_throughput, discrete_throughput, inference_repeats = maybe_benchmark_inference_pair(
        args,
        model,
        hard_model,
        dataset.x_test,
        final_tau,
    )
    row = MetricsRow(
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
        epochs_to_target=epochs_to_target,
        time_to_target=time_to_target,
        unused_gate_ratio=compute_unused_gate_ratio(layers, dataset.x_train, args.eval_batch_size, device),
        gate_count=gate_count(layers),
        depth=len(layers),
        fanout_max=fanout_max(layers),
        soft_inference_samples_per_sec=soft_throughput,
        discrete_inference_samples_per_sec=discrete_throughput,
        inference_bench_repeats=inference_repeats,
    )
    layer_diag_rows = layer_gap_diagnostics(
        method,
        dataset.name,
        seed,
        "full_soft_vs_discrete",
        layers,
        hard_layers,
        dataset.x_test,
        args.eval_batch_size,
        device,
        final_tau,
    )
    synth_row = maybe_run_abc_stats(args, method, dataset.name, seed, hard_layers, dataset.input_dim)
    return row, epoch_rows, layer_diag_rows, synth_row


def train_one_block(
    layer: SoftLogicLayer,
    prefix_layers: list[nn.Module],
    dataset: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    method: str,
    block_id: int,
    run_start_time: float,
) -> tuple[SoftLogicLayer, list[dict[str, float | int | str]], float]:
    prefix_mode = "soft" if method == "block_relaxed" else "hard"
    x_train_block = apply_layers(prefix_layers, dataset.x_train, args.eval_batch_size, device, mode=prefix_mode)
    x_test_block = apply_layers(prefix_layers, dataset.x_test, args.eval_batch_size, device, mode=prefix_mode)
    block_model = LogicNet([layer], dataset.num_classes, group_tau=args.group_tau).to(device)
    loader = make_loader(x_train_block, dataset.y_train, args.batch_size, seed + 7919 * (block_id + 1))
    opt = torch.optim.AdamW(block_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_rows = []
    start = time.perf_counter()

    for epoch in range(1, args.block_epochs + 1):
        block_model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = block_model(xb, mode="soft", tau=1.0)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

        soft_acc, soft_loss = evaluate(block_model, x_test_block, dataset.y_test, args.eval_batch_size, device, "soft", 1.0)
        disc_acc, disc_loss = evaluate(block_model, x_test_block, dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
        elapsed_time = time.perf_counter() - run_start_time
        epoch_rows.append(
            {
                "method": method,
                "dataset": dataset.name,
                "seed": seed,
                "block": block_id,
                "epoch": epoch,
                "global_epoch": block_id * args.block_epochs + epoch,
                "elapsed_time": elapsed_time,
                "block_elapsed_time": time.perf_counter() - start,
                "soft_acc": soft_acc,
                "discrete_acc": disc_acc,
                "soft_loss": soft_loss,
                "discrete_loss": disc_loss,
                "tau": 1.0,
            }
        )

    return block_model.layers[0].cpu(), epoch_rows, time.perf_counter() - start  # type: ignore[return-value]


@torch.no_grad()
def refit_truth_table_layer(
    layer: SoftLogicLayer,
    x_ref: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FrozenHardLogicLayer, dict[str, float | int | str]]:
    layer = clone_soft_layer(layer).to(device)
    layer.eval()
    if layer.in_dim <= args.exact_truth_max:
        x_fit = exact_binary_inputs(layer.in_dim)
        fit_source = "exact"
    else:
        if x_ref.shape[0] > args.refit_samples:
            generator = torch.Generator().manual_seed(args.refit_seed)
            idx = torch.randperm(x_ref.shape[0], generator=generator)[: args.refit_samples]
            x_fit = x_ref[idx]
        else:
            x_fit = x_ref
        fit_source = "sampled"

    y_target_chunks = []
    x_fit = x_fit.float()
    for start in range(0, x_fit.shape[0], args.eval_batch_size):
        xb = x_fit[start : start + args.eval_batch_size].to(device)
        y_target_chunks.append(layer(xb, mode="soft", tau=1.0).detach().cpu())
    y_target = torch.cat(y_target_chunks, dim=0)

    x_fit_hard = x_fit.round().clamp(0, 1)
    truth = GATE_TRUTH
    argmax_ops = layer.hard_ops_argmax()
    op_ids = []
    argmax_mse_values = []
    refit_mse_values = []
    for gate_id in range(layer.out_dim):
        a = x_fit_hard[:, layer.indices_0[gate_id].cpu()]
        b = x_fit_hard[:, layer.indices_1[gate_id].cpu()]
        idx = (a.long() * 2 + b.long()).clamp(0, 3)
        candidates = truth[:, idx].T
        mse = ((candidates - y_target[:, gate_id : gate_id + 1]) ** 2).mean(dim=0)
        refit_op = int(mse.argmin().item())
        op_ids.append(refit_op)
        argmax_mse_values.append(float(mse[int(argmax_ops[gate_id])].item()))
        refit_mse_values.append(float(mse[refit_op].item()))

    op_tensor = torch.tensor(op_ids, dtype=torch.long)
    stats = {
        "fit_source": fit_source,
        "fit_rows": int(x_fit.shape[0]),
        "argmax_refit_mse": float(np.mean(argmax_mse_values)),
        "truth_refit_mse": float(np.mean(refit_mse_values)),
        "refit_mse_delta": float(np.mean(argmax_mse_values) - np.mean(refit_mse_values)),
        "op_change_ratio": float((op_tensor != argmax_ops).float().mean().item()),
    }

    return FrozenHardLogicLayer(
        layer.in_dim,
        layer.out_dim,
        layer.indices_0.detach().cpu(),
        layer.indices_1.detach().cpu(),
        op_tensor,
    ), stats


def argmax_hard_layer(layer: SoftLogicLayer) -> FrozenHardLogicLayer:
    return FrozenHardLogicLayer(
        layer.in_dim,
        layer.out_dim,
        layer.indices_0.detach().cpu(),
        layer.indices_1.detach().cpu(),
        layer.hard_ops_argmax(),
    )


def train_blockwise(
    method: str,
    dataset: DatasetBundle,
    arch: list[dict[str, torch.Tensor | int]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[
    MetricsRow,
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    dict[str, float | int | str] | None,
]:
    assert method in {"block_relaxed", "block_hard_refit"}
    epoch_rows: list[dict[str, float | int | str]] = []
    block_diag_rows: list[dict[str, float | int | str]] = []
    frozen_or_soft_prefix: list[nn.Module] = []
    relaxed_layers: list[SoftLogicLayer] = []
    start_total = time.perf_counter()

    for block_id, spec in enumerate(arch):
        layer = make_soft_layer(spec)
        trained_layer, block_rows, block_time = train_one_block(
            layer,
            frozen_or_soft_prefix,
            dataset,
            args,
            device,
            seed,
            method,
            block_id,
            start_total,
        )
        _ = block_time
        epoch_rows.extend(block_rows)
        relaxed_layers.append(clone_soft_layer(trained_layer))

        if method == "block_relaxed":
            frozen_or_soft_prefix.append(clone_soft_layer(trained_layer))
            hard_prefix_layers = [argmax_hard_layer(layer) for layer in relaxed_layers]
            hard_prefix_model = LogicNet(hard_prefix_layers, dataset.num_classes, args.group_tau).to(device)
            path_soft_model = LogicNet([clone_soft_layer(layer) for layer in relaxed_layers], dataset.num_classes, args.group_tau).to(device)
            refit_stats: dict[str, float | int | str] = {
                "fit_source": "argmax_only",
                "fit_rows": 0,
                "argmax_refit_mse": float("nan"),
                "truth_refit_mse": float("nan"),
                "refit_mse_delta": float("nan"),
                "op_change_ratio": 0.0,
            }
        else:
            prefix_input = apply_layers(frozen_or_soft_prefix, dataset.x_train, args.eval_batch_size, device, mode="hard")
            hard_layer, refit_stats = refit_truth_table_layer(trained_layer, prefix_input, args, device)
            path_layers = list(frozen_or_soft_prefix) + [clone_soft_layer(trained_layer)]
            frozen_or_soft_prefix.append(hard_layer)
            hard_prefix_model = LogicNet(frozen_or_soft_prefix, dataset.num_classes, args.group_tau).to(device)
            path_soft_model = LogicNet(path_layers, dataset.num_classes, args.group_tau).to(device)

        path_block_acc, path_block_loss = evaluate(
            path_soft_model,
            dataset.x_test,
            dataset.y_test,
            args.eval_batch_size,
            device,
            "soft",
            1.0,
        )
        hard_block_acc, hard_block_loss = evaluate(
            hard_prefix_model,
            dataset.x_test,
            dataset.y_test,
            args.eval_batch_size,
            device,
            "hard",
            1.0,
        )
        block_diag_rows.append(
            {
                "method": method,
                "dataset": dataset.name,
                "seed": seed,
                "block": block_id,
                "global_epoch_end": (block_id + 1) * args.block_epochs,
                "elapsed_time": time.perf_counter() - start_total,
                "prefix_reaches_target": int(hard_block_acc >= dataset.target_acc),
                "block_train_time": block_time,
                "path_soft_acc": path_block_acc,
                "path_soft_loss": path_block_loss,
                "hard_prefix_acc": hard_block_acc,
                "hard_prefix_loss": hard_block_loss,
                "path_acc_gap": abs(path_block_acc - hard_block_acc),
                "path_loss_gap": abs(path_block_loss - hard_block_loss),
                **refit_stats,
            }
        )

    soft_model = LogicNet([clone_soft_layer(layer) for layer in relaxed_layers], dataset.num_classes, args.group_tau).to(device)
    if method == "block_relaxed":
        hard_layers = [argmax_hard_layer(layer) for layer in relaxed_layers]
        hard_model = LogicNet(hard_layers, dataset.num_classes, args.group_tau).to(device)
        path_soft_model = soft_model
    else:
        hard_model = LogicNet(frozen_or_soft_prefix, dataset.num_classes, args.group_tau).to(device)
        path_layers = list(frozen_or_soft_prefix[:-1]) + [clone_soft_layer(relaxed_layers[-1])]
        path_soft_model = LogicNet(path_layers, dataset.num_classes, args.group_tau).to(device)

    total_train_time = time.perf_counter() - start_total
    soft_acc, soft_loss = evaluate(soft_model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "soft", 1.0)
    disc_acc, disc_loss = evaluate(hard_model, dataset.x_test, dataset.y_test, args.eval_batch_size, device, "hard", 1.0)
    path_soft_acc, path_soft_loss = evaluate(
        path_soft_model,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "soft",
        1.0,
    )
    epochs_to_target = args.block_epochs * len(arch) if disc_acc >= dataset.target_acc else -1
    time_to_target = total_train_time if disc_acc >= dataset.target_acc else -1.0

    hard_layers_for_stats = list(hard_model.layers)
    soft_throughput, discrete_throughput, inference_repeats = maybe_benchmark_inference_pair(
        args,
        soft_model,
        hard_model,
        dataset.x_test,
        1.0,
    )
    row = MetricsRow(
        method=method,
        dataset=dataset.name,
        seed=seed,
        soft_acc=soft_acc,
        discrete_acc=disc_acc,
        acc_gap=abs(soft_acc - disc_acc),
        soft_loss=soft_loss,
        discrete_loss=disc_loss,
        loss_gap=abs(soft_loss - disc_loss),
        path_soft_acc=path_soft_acc,
        path_discrete_acc=disc_acc,
        path_acc_gap=abs(path_soft_acc - disc_acc),
        path_soft_loss=path_soft_loss,
        path_discrete_loss=disc_loss,
        path_loss_gap=abs(path_soft_loss - disc_loss),
        train_time=total_train_time,
        epochs_to_target=epochs_to_target,
        time_to_target=time_to_target,
        unused_gate_ratio=compute_unused_gate_ratio(hard_layers_for_stats, dataset.x_train, args.eval_batch_size, device),
        gate_count=gate_count(hard_layers_for_stats),
        depth=len(hard_layers_for_stats),
        fanout_max=fanout_max(hard_layers_for_stats),
        soft_inference_samples_per_sec=soft_throughput,
        discrete_inference_samples_per_sec=discrete_throughput,
        inference_bench_repeats=inference_repeats,
    )
    layer_diag_rows = layer_gap_diagnostics(
        method,
        dataset.name,
        seed,
        "full_soft_vs_discrete",
        list(soft_model.layers),
        list(hard_model.layers),
        dataset.x_test,
        args.eval_batch_size,
        device,
        1.0,
    )
    if method == "block_hard_refit":
        layer_diag_rows.extend(
            layer_gap_diagnostics(
                method,
                dataset.name,
                seed,
                "path_soft_vs_discrete",
                list(path_soft_model.layers),
                list(hard_model.layers),
                dataset.x_test,
                args.eval_batch_size,
                device,
                1.0,
            )
        )
    synth_row = maybe_run_abc_stats(args, method, dataset.name, seed, hard_layers_for_stats, dataset.input_dim)
    return row, epoch_rows, block_diag_rows, layer_diag_rows, synth_row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def write_summary(path: Path, rows: list[MetricsRow], args: argparse.Namespace, elapsed: float) -> None:
    fields = [
        "method",
        "dataset",
        "seed",
        "soft_acc",
        "discrete_acc",
        "acc_gap",
        "soft_loss",
        "discrete_loss",
        "loss_gap",
        "path_soft_acc",
        "path_discrete_acc",
        "path_acc_gap",
        "path_soft_loss",
        "path_discrete_loss",
        "path_loss_gap",
        "train_time",
        "epochs_to_target",
        "time_to_target",
        "unused_gate_ratio",
        "gate_count",
        "depth",
        "fanout_max",
        "soft_inference_samples_per_sec",
        "discrete_inference_samples_per_sec",
        "inference_bench_repeats",
    ]
    lines = [
        "# Hard-LGN Gap Prototype Summary",
        "",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        "Metric notes:",
        "",
        "- `soft_acc`/`soft_loss` evaluate all trained relaxed blocks in soft mode.",
        "- `discrete_acc`/`discrete_loss` evaluate the corresponding hard network: argmax gates for DLGN-style methods and refit gates for `block_hard_refit`.",
        "- `path_*` metrics evaluate the method-native training path. For `block_hard_refit`, this means hard frozen prefixes plus the final relaxed block compared with the final refit hard network.",
        "- `layer_diagnostics.csv` tracks relaxed-path versus hard-path representation mismatch after each layer prefix for depth-wise gap accumulation checks.",
        "- `*_inference_samples_per_sec` are optional PyTorch forward-pass throughput measurements from `--inference-bench`; they are not bit-packed Boolean inference kernels.",
        "- For block-wise methods, `epochs_to_target` is conservative: total block epochs if the final hard model reaches the dataset target, otherwise `-1`.",
        "- `time_to_target` is wall-clock seconds to first discrete target hit for end-to-end methods; for block-wise methods it is conservative final train time if the final hard model reaches target, otherwise `-1`.",
        "",
        "Configuration:",
        "",
        "```json",
        json.dumps(vars(args), indent=2, sort_keys=True),
        "```",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        d = asdict(row)
        lines.append("| " + " | ".join(format_float(d[field]) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def accuracy_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid float") from exc
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("target accuracy override must be between 0.0 and 1.0")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dlgn", "dlgn_anneal", "gumbel_st", "block_relaxed", "block_hard_refit"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--block-epochs", type=int, default=40)
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
    parser.add_argument(
        "--target-acc-override",
        type=accuracy_threshold,
        help="Override dataset target accuracy for convergence-speed sweeps.",
    )
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--abc-stats", action="store_true", help="Export final hard networks to BLIF and run ABC stats.")
    parser.add_argument("--abc-path", default="/home/spco/boolean_sat/abc/abc")
    parser.add_argument("--abc-max-gates", type=int, default=10000)
    parser.add_argument("--inference-bench", action="store_true", help="Measure optional PyTorch soft/discrete inference throughput.")
    parser.add_argument("--inference-bench-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--inference-bench-repeats", type=int, default=20)
    parser.add_argument("--inference-bench-warmup", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/quick")
    parser.add_argument("--quick", action="store_true", help="Small run for code-path validation.")
    parser.add_argument(
        "--difflogic-compat-check",
        action="store_true",
        help="Write difflogic_compat.json comparing prototype primitives against /home/spco/convlogic.",
    )
    parser.add_argument(
        "--compat-only",
        action="store_true",
        help="Run only --difflogic-compat-check and skip benchmark datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inference_bench and args.inference_bench_repeats <= 0:
        raise ValueError("--inference-bench-repeats must be positive when --inference-bench is enabled")
    if args.inference_bench and args.inference_bench_warmup < 0:
        raise ValueError("--inference-bench-warmup must be non-negative when --inference-bench is enabled")
    if args.quick:
        args.datasets = ["parity6", "majority7", "random_sparse8"]
        args.seeds = [0]
        args.width = min(args.width, 32)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 20)
        args.block_epochs = min(args.block_epochs, 15)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.difflogic_compat_check or args.compat_only:
        compat_report = check_difflogic_compatibility(out_dir)
        print(f"difflogic_compat all_pass={compat_report.get('all_pass')} status={compat_report.get('status')}")
        if not compat_report.get("all_pass"):
            raise RuntimeError(f"difflogic compatibility check failed; see {out_dir / 'difflogic_compat.json'}")
        if args.compat_only:
            return

    all_rows: list[MetricsRow] = []
    all_epoch_rows: list[dict[str, float | int | str]] = []
    all_block_diag_rows: list[dict[str, float | int | str]] = []
    all_layer_diag_rows: list[dict[str, float | int | str]] = []
    all_synthesis_rows: list[dict[str, float | int | str]] = []
    start_all = time.perf_counter()

    print(f"device={device} out_dir={out_dir}")
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

            arch = build_architecture(dataset.input_dim, args.width, args.layers, seed)
            print(f"dataset={dataset.name} seed={seed} train={len(dataset.y_train)} test={len(dataset.y_test)}")
            for method in args.methods:
                print(f"  method={method}", flush=True)
                if method in {"dlgn", "dlgn_anneal", "gumbel_st", "gumbel_soft"}:
                    row, epoch_rows, layer_diag_rows, synth_row = train_end_to_end(
                        method, dataset, arch, args, device, seed
                    )
                elif method in {"block_relaxed", "block_hard_refit"}:
                    row, epoch_rows, block_diag_rows, layer_diag_rows, synth_row = train_blockwise(
                        method, dataset, arch, args, device, seed
                    )
                    all_block_diag_rows.extend(block_diag_rows)
                else:
                    raise ValueError(f"Unknown method: {method}")
                if synth_row is not None:
                    all_synthesis_rows.append(synth_row)
                print(
                    "    soft_acc={:.4f} discrete_acc={:.4f} gap={:.4f} unused={:.4f} time={:.2f}s".format(
                        row.soft_acc,
                        row.discrete_acc,
                        row.acc_gap,
                        row.unused_gate_ratio,
                        row.train_time,
                    ),
                    flush=True,
                )
                all_rows.append(row)
                all_epoch_rows.extend(epoch_rows)
                all_layer_diag_rows.extend(layer_diag_rows)
                write_csv(out_dir / "results.partial.csv", [asdict(r) for r in all_rows])
                write_csv(out_dir / "per_epoch.partial.csv", all_epoch_rows)
                write_csv(out_dir / "block_diagnostics.partial.csv", all_block_diag_rows)
                write_csv(out_dir / "layer_diagnostics.partial.csv", all_layer_diag_rows)
                write_csv(out_dir / "synthesis_stats.partial.csv", all_synthesis_rows)

    elapsed = time.perf_counter() - start_all
    write_csv(out_dir / "results.csv", [asdict(r) for r in all_rows])
    write_csv(out_dir / "per_epoch.csv", all_epoch_rows)
    write_csv(out_dir / "block_diagnostics.csv", all_block_diag_rows)
    write_csv(out_dir / "layer_diagnostics.csv", all_layer_diag_rows)
    write_csv(out_dir / "synthesis_stats.csv", all_synthesis_rows)
    write_summary(out_dir / "summary.md", all_rows, args, elapsed)
    print(f"wrote {out_dir / 'results.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
