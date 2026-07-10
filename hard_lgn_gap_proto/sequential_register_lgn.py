#!/usr/bin/env python3
"""Minimal synchronous register-LGN prototype for stateful sequence tasks.

This implements Part B of ``next_stage_protocol.md``:

    state_{t+1}, output_t = logic_block(concat(input_t, state_t))

The implementation is intentionally small and controlled. Feedforward LGN
baselines see only the current input bit at each time step, while register LGNs
carry explicit Boolean state bits across time.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hard_lgn_benchmark import (
    FrozenHardLogicLayer,
    GroupSum,
    SoftLogicLayer,
    fanout_max,
    gate_count,
    random_connections,
    weighted_gate,
    write_csv,
)


RESULT_FIELDS = [
    "method",
    "task",
    "seed",
    "state_bits",
    "time_steps",
    "hard_acc",
    "soft_acc",
    "sequence_acc",
    "native_gap",
    "full_gap",
    "hard_loss",
    "soft_loss",
    "train_time",
    "register_utilization",
    "gate_count",
    "depth",
    "fanout_max",
    "param_count",
    "source_note",
]


@dataclass
class SequenceTask:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    input_bits: int
    num_classes: int
    time_steps: int
    output_kind: str
    eval_start: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sequence_task(name: str, seed: int, n_train: int, n_test: int, time_steps: int, delay: int) -> SequenceTask:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n_train + n_test, time_steps, 1), generator=generator).float()
    name = name.lower()
    if name == "sequence_parity":
        y = (x.long().sum(dim=(1, 2)) % 2).long()
        output_kind = "final"
        eval_start = 0
    elif name == "temporal_majority":
        y = (x.long().sum(dim=(1, 2)) >= math.ceil(time_steps / 2)).long()
        output_kind = "final"
        eval_start = 0
    elif name == "delayed_copy":
        if delay < 1 or delay >= time_steps:
            raise ValueError("--delay must be in [1, time_steps - 1] for delayed_copy")
        y = torch.zeros((n_train + n_test, time_steps), dtype=torch.long)
        y[:, delay:] = x[:, :-delay, 0].long()
        output_kind = "sequence"
        eval_start = delay
    elif name == "fsm_endswith101":
        bits = x[:, :, 0].long()
        y = ((bits[:, -3] == 1) & (bits[:, -2] == 0) & (bits[:, -1] == 1)).long()
        output_kind = "final"
        eval_start = 0
    else:
        raise ValueError(f"unknown task: {name}")
    return SequenceTask(
        name=name,
        x_train=x[:n_train],
        y_train=y[:n_train],
        x_test=x[n_train:],
        y_test=y[n_train:],
        input_bits=1,
        num_classes=2,
        time_steps=time_steps,
        output_kind=output_kind,
        eval_start=eval_start,
    )


def st_weights(logits: torch.Tensor, tau: float) -> torch.Tensor:
    probs = F.softmax(logits / tau, dim=-1)
    hard = F.one_hot(probs.argmax(dim=-1), probs.shape[-1]).to(dtype=probs.dtype)
    return hard - probs.detach() + probs


def logic_layer_forward(layer: nn.Module, x: torch.Tensor, mode: str, tau: float) -> torch.Tensor:
    if isinstance(layer, FrozenHardLogicLayer):
        return layer(x)
    if not isinstance(layer, SoftLogicLayer):
        raise TypeError(type(layer))
    if mode == "st":
        a = x[..., layer.indices_0]
        b = x[..., layer.indices_1]
        return weighted_gate(a, b, st_weights(layer.logits, tau).to(dtype=x.dtype))
    if mode in {"soft", "hard"}:
        return layer(x, mode=mode, tau=tau)
    raise ValueError(mode)


class SequentialRegisterLGN(nn.Module):
    def __init__(
        self,
        *,
        input_bits: int,
        state_bits: int,
        num_classes: int,
        width: int,
        depth: int,
        votes: int,
        wiring_seed: int,
        init_seed: int,
        group_tau: float,
    ) -> None:
        super().__init__()
        if state_bits < 1:
            raise ValueError("register LGN requires state_bits >= 1")
        self.input_bits = input_bits
        self.state_bits = state_bits
        self.num_classes = num_classes
        self.votes = votes
        self.group_sum = GroupSum(num_classes, group_tau)
        self.layers = nn.ModuleList(
            build_logic_layers(input_bits + state_bits, state_bits + num_classes * votes, width, depth, wiring_seed, init_seed)
        )

    def forward(self, x: torch.Tensor, mode: str = "soft", tau: float = 1.0, return_states: bool = False):
        batch, steps, _ = x.shape
        state = torch.zeros((batch, self.state_bits), dtype=x.dtype, device=x.device)
        logits_by_step = []
        states = []
        for t in range(steps):
            h = torch.cat([x[:, t, :], state], dim=-1)
            for layer in self.layers:
                h = logic_layer_forward(layer, h, mode, tau)
            next_state = h[:, : self.state_bits]
            vote_bits = h[:, self.state_bits : self.state_bits + self.num_classes * self.votes]
            logits_by_step.append(self.group_sum(vote_bits))
            if mode == "hard":
                state = next_state.round().clamp(0, 1)
            else:
                state = next_state
            states.append(state)
        logits = torch.stack(logits_by_step, dim=1)
        if return_states:
            return logits, torch.stack(states, dim=1)
        return logits

    def harden(self) -> "SequentialRegisterLGN":
        clone = SequentialRegisterLGN(
            input_bits=self.input_bits,
            state_bits=self.state_bits,
            num_classes=self.num_classes,
            width=max(layer.out_dim for layer in self.layers),  # type: ignore[attr-defined]
            depth=len(self.layers),
            votes=self.votes,
            wiring_seed=0,
            init_seed=0,
            group_tau=self.group_sum.tau,
        )
        clone.layers = nn.ModuleList([argmax_layer(layer) for layer in self.layers])
        return clone


class FeedForwardStepLGN(nn.Module):
    def __init__(
        self,
        *,
        input_bits: int,
        num_classes: int,
        width: int,
        depth: int,
        votes: int,
        wiring_seed: int,
        init_seed: int,
        group_tau: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.votes = votes
        self.group_sum = GroupSum(num_classes, group_tau)
        self.layers = nn.ModuleList(
            build_logic_layers(input_bits, num_classes * votes, width, depth, wiring_seed, init_seed)
        )

    def forward(self, x: torch.Tensor, mode: str = "soft", tau: float = 1.0, return_states: bool = False):
        logits_by_step = []
        for t in range(x.shape[1]):
            h = x[:, t, :]
            for layer in self.layers:
                h = logic_layer_forward(layer, h, mode, tau)
            vote_bits = h[:, : self.num_classes * self.votes]
            logits_by_step.append(self.group_sum(vote_bits))
        logits = torch.stack(logits_by_step, dim=1)
        if return_states:
            return logits, torch.empty((x.shape[0], x.shape[1], 0), dtype=x.dtype, device=x.device)
        return logits

    def harden(self) -> "FeedForwardStepLGN":
        clone = FeedForwardStepLGN(
            input_bits=int(self.layers[0].in_dim),  # type: ignore[attr-defined]
            num_classes=self.num_classes,
            width=max(layer.out_dim for layer in self.layers),  # type: ignore[attr-defined]
            depth=len(self.layers),
            votes=self.votes,
            wiring_seed=0,
            init_seed=0,
            group_tau=self.group_sum.tau,
        )
        clone.layers = nn.ModuleList([argmax_layer(layer) for layer in self.layers])
        return clone


class TorchRecurrentBaseline(nn.Module):
    def __init__(self, kind: str, input_bits: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.kind = kind
        if kind == "rnn_small":
            self.recurrent = nn.RNN(input_bits, hidden_size, batch_first=True)
        elif kind == "gru_small":
            self.recurrent = nn.GRU(input_bits, hidden_size, batch_first=True)
        else:
            raise ValueError(kind)
        self.readout = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        h, _ = self.recurrent(x)
        logits = self.readout(h)
        if return_states:
            return logits, h
        return logits


def build_logic_layers(
    input_dim: int,
    output_needed: int,
    width: int,
    depth: int,
    wiring_seed: int,
    init_seed: int,
) -> list[SoftLogicLayer]:
    if depth < 1:
        raise ValueError("--depth must be >= 1")
    wiring_gen = torch.Generator().manual_seed(wiring_seed)
    init_gen = torch.Generator().manual_seed(init_seed)
    layers = []
    in_dim = input_dim
    for layer_id in range(depth):
        out_dim = width if layer_id < depth - 1 else max(output_needed, math.ceil(in_dim / 2))
        if out_dim * 2 < in_dim:
            raise ValueError(f"layer {layer_id}: out_dim={out_dim} cannot cover in_dim={in_dim}")
        idx0, idx1 = random_connections(in_dim, out_dim, wiring_gen)
        logits = torch.randn(out_dim, 16, generator=init_gen)
        layers.append(SoftLogicLayer(in_dim, out_dim, idx0, idx1, logits))
        in_dim = out_dim
    return layers


def argmax_layer(layer: nn.Module) -> FrozenHardLogicLayer:
    if isinstance(layer, FrozenHardLogicLayer):
        return layer
    if not isinstance(layer, SoftLogicLayer):
        raise TypeError(type(layer))
    return FrozenHardLogicLayer(
        layer.in_dim,
        layer.out_dim,
        layer.indices_0.detach().cpu(),
        layer.indices_1.detach().cpu(),
        layer.hard_ops_argmax(),
    )


def task_loss(logits: torch.Tensor, y: torch.Tensor, task: SequenceTask) -> torch.Tensor:
    if task.output_kind == "final":
        return F.cross_entropy(logits[:, -1, :], y)
    active_logits = logits[:, task.eval_start :, :].reshape(-1, task.num_classes)
    active_y = y[:, task.eval_start :].reshape(-1)
    return F.cross_entropy(active_logits, active_y)


@torch.no_grad()
def evaluate_logic_model(model: nn.Module, task: SequenceTask, device: torch.device, mode: str, tau: float) -> dict[str, float]:
    model.eval()
    x = task.x_test.to(device)
    y = task.y_test.to(device)
    logits, states = model(x, mode=mode, tau=tau, return_states=True)  # type: ignore[misc]
    loss = float(task_loss(logits, y, task).item())
    pred = logits.argmax(dim=-1)
    if task.output_kind == "final":
        correct = pred[:, -1] == y
        acc = float(correct.float().mean().item())
        sequence_acc = acc
    else:
        active_pred = pred[:, task.eval_start :]
        active_y = y[:, task.eval_start :]
        acc = float((active_pred == active_y).float().mean().item())
        sequence_acc = float((active_pred == active_y).all(dim=1).float().mean().item())
    return {
        "acc": acc,
        "sequence_acc": sequence_acc,
        "loss": loss,
        "register_utilization": register_utilization(states),
    }


@torch.no_grad()
def evaluate_torch_recurrent(model: TorchRecurrentBaseline, task: SequenceTask, device: torch.device) -> dict[str, float]:
    model.eval()
    x = task.x_test.to(device)
    y = task.y_test.to(device)
    logits, states = model(x, return_states=True)
    loss = float(task_loss(logits, y, task).item())
    pred = logits.argmax(dim=-1)
    if task.output_kind == "final":
        acc = float((pred[:, -1] == y).float().mean().item())
        sequence_acc = acc
    else:
        active_pred = pred[:, task.eval_start :]
        active_y = y[:, task.eval_start :]
        acc = float((active_pred == active_y).float().mean().item())
        sequence_acc = float((active_pred == active_y).all(dim=1).float().mean().item())
    return {
        "acc": acc,
        "sequence_acc": sequence_acc,
        "loss": loss,
        "register_utilization": register_utilization(states),
    }


def register_utilization(states: torch.Tensor) -> float:
    if states.numel() == 0 or states.shape[-1] == 0:
        return 0.0
    flat = states.detach().float().reshape(-1, states.shape[-1])
    active = (flat.max(dim=0).values - flat.min(dim=0).values) > 1e-4
    return float(active.float().mean().item())


def train_logic_method(method: str, task: SequenceTask, args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, object]:
    wiring_seed = args.wiring_seed_base + seed
    init_seed = args.init_seed_base + seed
    if method == "feedforward_lgn":
        model: nn.Module = FeedForwardStepLGN(
            input_bits=task.input_bits,
            num_classes=task.num_classes,
            width=args.width,
            depth=args.depth,
            votes=args.votes,
            wiring_seed=wiring_seed,
            init_seed=init_seed,
            group_tau=args.group_tau,
        )
        train_mode = "soft"
        source_note = "no registers; each step sees only current input bits"
    elif method in {"register_lgn_soft", "register_lgn_st"}:
        model = SequentialRegisterLGN(
            input_bits=task.input_bits,
            state_bits=args.state_bits,
            num_classes=task.num_classes,
            width=args.width,
            depth=args.depth,
            votes=args.votes,
            wiring_seed=wiring_seed,
            init_seed=init_seed,
            group_tau=args.group_tau,
        )
        train_mode = "st" if method == "register_lgn_st" else "soft"
        source_note = "explicit Boolean registers; ST mode uses deterministic hard-forward soft-backward gates" if train_mode == "st" else "explicit registers trained with relaxed soft gates/state"
    else:
        raise ValueError(method)

    set_seed(seed)
    model.to(device)
    loader = DataLoader(TensorDataset(task.x_train, task.y_train), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tau = temp_schedule(epoch, args.epochs, args.temp_start, args.temp_end)
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mode=train_mode, tau=tau)  # type: ignore[misc]
            loss = task_loss(logits, yb, task)
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - start

    soft = evaluate_logic_model(model, task, device, "soft", args.temp_end)
    native = evaluate_logic_model(model, task, device, train_mode, args.temp_end)
    hard = evaluate_logic_model(model, task, device, "hard", 1.0)
    layers = list(model.layers)  # type: ignore[attr-defined]
    return {
        "method": method,
        "task": task.name,
        "seed": seed,
        "state_bits": args.state_bits if method.startswith("register") else 0,
        "time_steps": task.time_steps,
        "hard_acc": hard["acc"],
        "soft_acc": soft["acc"],
        "sequence_acc": hard["sequence_acc"],
        "native_gap": abs(native["acc"] - hard["acc"]),
        "full_gap": abs(soft["acc"] - hard["acc"]),
        "hard_loss": hard["loss"],
        "soft_loss": soft["loss"],
        "train_time": train_time,
        "register_utilization": hard["register_utilization"],
        "gate_count": gate_count(layers),
        "depth": len(layers),
        "fanout_max": fanout_max(layers),
        "param_count": sum(p.numel() for p in model.parameters()),
        "source_note": source_note,
    }


def train_torch_method(method: str, task: SequenceTask, args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, object]:
    set_seed(seed)
    model = TorchRecurrentBaseline(method, task.input_bits, args.state_bits, task.num_classes).to(device)
    loader = DataLoader(TensorDataset(task.x_train, task.y_train), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = task_loss(model(xb), yb, task)
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - start
    metrics = evaluate_torch_recurrent(model, task, device)
    params = sum(p.numel() for p in model.parameters())
    return {
        "method": method,
        "task": task.name,
        "seed": seed,
        "state_bits": args.state_bits,
        "time_steps": task.time_steps,
        "hard_acc": metrics["acc"],
        "soft_acc": metrics["acc"],
        "sequence_acc": metrics["sequence_acc"],
        "native_gap": 0.0,
        "full_gap": 0.0,
        "hard_loss": metrics["loss"],
        "soft_loss": metrics["loss"],
        "train_time": train_time,
        "register_utilization": metrics["register_utilization"],
        "gate_count": math.nan,
        "depth": 1,
        "fanout_max": math.nan,
        "param_count": params,
        "source_note": "continuous-state neural recurrent reference; gate_count is not comparable",
    }


def temp_schedule(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    ratio = (epoch - 1) / (epochs - 1)
    return start * ((end / start) ** ratio)


def markdown_report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Sequential Register-LGN Report",
        "",
        "Feedforward LGN baselines see only the current input bit at each step. Register LGNs concatenate current input bits with Boolean register bits and update registers synchronously.",
        "",
        "| " + " | ".join(RESULT_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(RESULT_FIELDS)) + " |",
    ]
    for row in rows:
        values = []
        for field in RESULT_FIELDS:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- `hard_acc` is final deployable hard-path accuracy for LGN methods.",
            "- `sequence_acc` is exact sequence accuracy for sequence-output tasks; for final-label tasks it equals hard final-label accuracy.",
            "- `register_utilization` is the fraction of state bits whose value changes over the held-out test trajectories.",
            "- RNN/GRU rows are continuous-state references; their `gate_count` and hard/soft gaps are not logic-hardware metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["sequence_parity", "delayed_copy"])
    parser.add_argument("--methods", nargs="+", default=["feedforward_lgn", "register_lgn_soft", "register_lgn_st", "rnn_small", "gru_small"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--state-bits", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--votes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--temp-start", type=float, default=1.0)
    parser.add_argument("--temp-end", type=float, default=0.3)
    parser.add_argument("--wiring-seed-base", type=int, default=30000)
    parser.add_argument("--init-seed-base", type=int, default=40000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/sequential_register_smoke_v1")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.tasks = ["sequence_parity", "delayed_copy"]
        args.methods = ["feedforward_lgn", "register_lgn_st", "rnn_small"]
        args.seeds = [0]
        args.n_train = min(args.n_train, 256)
        args.n_test = min(args.n_test, 128)
        args.time_steps = min(args.time_steps, 6)
        args.state_bits = min(args.state_bits, 6)
        args.width = min(args.width, 24)
        args.depth = min(args.depth, 2)
        args.votes = min(args.votes, 12)
        args.epochs = min(args.epochs, 12)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    start_all = time.perf_counter()
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        for task_name in args.tasks:
            task = make_sequence_task(task_name, seed, args.n_train, args.n_test, args.time_steps, args.delay)
            print(f"task={task.name} seed={seed} train={args.n_train} test={args.n_test} steps={args.time_steps}", flush=True)
            for method in args.methods:
                print(f"  method={method}", flush=True)
                if method in {"feedforward_lgn", "register_lgn_soft", "register_lgn_st"}:
                    row = train_logic_method(method, task, args, device, seed)
                elif method in {"rnn_small", "gru_small"}:
                    row = train_torch_method(method, task, args, device, seed)
                else:
                    raise ValueError(f"unknown method: {method}")
                rows.append(row)
                print(
                    "    hard_acc={:.4f} soft_acc={:.4f} seq_acc={:.4f} native_gap={:.4f} reg_util={:.4f}".format(
                        float(row["hard_acc"]),
                        float(row["soft_acc"]),
                        float(row["sequence_acc"]),
                        float(row["native_gap"]),
                        float(row["register_utilization"]),
                    ),
                    flush=True,
                )
                write_csv(out_dir / "sequential_results.partial.csv", rows)
    elapsed = time.perf_counter() - start_all
    write_csv(out_dir / "sequential_results.csv", rows)
    (out_dir / "sequential_report.md").write_text(markdown_report(rows))
    (out_dir / "run_meta.txt").write_text(f"elapsed_seconds={elapsed:.6f}\n")
    print(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
