from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


GATE_NAMES = (
    "zero",
    "and",
    "not_implies",
    "a",
    "not_implied_by",
    "b",
    "xor",
    "or",
    "nor",
    "xnor",
    "not_b",
    "implied_by",
    "not_a",
    "implies",
    "nand",
    "one",
)

# Rows are gates, columns are input addresses 00, 01, 10, 11.
TRUTH_TABLE = torch.tensor(
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
    dtype=torch.bool,
)


def relaxed_gate_outputs(a: Tensor, b: Tensor) -> Tensor:
    return torch.stack(
        (
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
        ),
        dim=-1,
    )


def evaluate_gate_bits(a: Tensor, b: Tensor, op_ids: Tensor) -> Tensor:
    if a.dtype != torch.bool or b.dtype != torch.bool:
        raise TypeError("bit execution requires torch.bool inputs")
    address = (a.to(torch.long) << 1) | b.to(torch.long)
    truth = TRUTH_TABLE.to(device=a.device)
    shape = (1,) * (address.ndim - 1) + (op_ids.numel(),)
    return truth[op_ids.to(a.device).reshape(shape), address]


def random_connections(
    in_dim: int,
    out_dim: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if in_dim < 2 or out_dim < 1:
        raise ValueError((in_dim, out_dim))
    first = torch.randint(in_dim, (out_dim,), generator=generator)
    second = torch.randint(in_dim - 1, (out_dim,), generator=generator)
    second = second + (second >= first).long()
    return first, second


class HardSTGateLayer(nn.Module):
    """Fixed-wiring 2-input gate layer with a bit-exact deployment path."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        indices_0: Tensor,
        indices_1: Tensor,
        *,
        init_ops: Tensor | None = None,
        init_strength: float = 2.0,
        surrogate_inputs: bool = False,
    ) -> None:
        super().__init__()
        if indices_0.shape != (out_dim,) or indices_1.shape != (out_dim,):
            raise ValueError((indices_0.shape, indices_1.shape, out_dim))
        if int(indices_0.min()) < 0 or int(indices_1.min()) < 0:
            raise ValueError("connection indices must be non-negative")
        if int(indices_0.max()) >= in_dim or int(indices_1.max()) >= in_dim:
            raise ValueError("connection index exceeds input width")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.surrogate_inputs = bool(surrogate_inputs)
        self.register_buffer("indices_0", indices_0.detach().clone().long())
        self.register_buffer("indices_1", indices_1.detach().clone().long())
        logits = torch.zeros(out_dim, 16)
        if init_ops is None:
            logits.normal_(mean=0.0, std=0.25)
        else:
            if init_ops.shape != (out_dim,):
                raise ValueError(init_ops.shape)
            logits.fill_(-float(init_strength))
            logits.scatter_(1, init_ops.long().view(-1, 1), float(init_strength))
        self.logits = nn.Parameter(logits)

    def hard_ops(self) -> Tensor:
        return self.logits.detach().argmax(dim=-1)

    def confidence(self) -> Tensor:
        return F.softmax(self.logits, dim=-1).amax(dim=-1).mean()

    def forward(self, x: Tensor, *, mode: str = "hard_st", tau: float = 1.0) -> Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError((x.shape, self.in_dim))
        if tau <= 0.0:
            raise ValueError(tau)
        a = x[..., self.indices_0]
        b = x[..., self.indices_1]
        gate_values = relaxed_gate_outputs(a, b)
        soft = F.softmax(self.logits / tau, dim=-1).to(x.dtype)
        hard = F.one_hot(self.logits.argmax(dim=-1), 16).to(x.dtype)
        if mode == "soft":
            weights = soft
            return (gate_values * weights).sum(dim=-1)
        if mode == "hard":
            return (gate_values * hard).sum(dim=-1)
        if mode == "gumbel_st":
            sampled_soft = F.gumbel_softmax(self.logits, tau=tau, hard=False, dim=-1).to(x.dtype)
            sampled_hard = F.one_hot(sampled_soft.argmax(dim=-1), 16).to(x.dtype)
            if self.surrogate_inputs:
                hard_value = (gate_values * sampled_hard).sum(dim=-1)
                soft_value = (gate_values * sampled_soft).sum(dim=-1)
                return hard_value.detach() + soft_value - soft_value.detach()
            weights = sampled_hard + sampled_soft - sampled_soft.detach()
            return (gate_values * weights).sum(dim=-1)
        if mode != "hard_st":
            raise ValueError(mode)
        if self.surrogate_inputs:
            hard_value = (gate_values * hard).sum(dim=-1)
            soft_value = (gate_values * soft).sum(dim=-1)
            return hard_value.detach() + soft_value - soft_value.detach()
        weights = hard + soft - soft.detach()
        return (gate_values * weights).sum(dim=-1)

    def forward_bits(self, x: Tensor) -> Tensor:
        if x.dtype != torch.bool:
            raise TypeError("forward_bits requires torch.bool")
        if x.shape[-1] != self.in_dim:
            raise ValueError((x.shape, self.in_dim))
        return evaluate_gate_bits(
            x[..., self.indices_0],
            x[..., self.indices_1],
            self.hard_ops(),
        )

    def fanout_max(self) -> int:
        connections = torch.cat((self.indices_0.detach().cpu(), self.indices_1.detach().cpu()))
        return int(torch.bincount(connections, minlength=self.in_dim).max().item())

    def deployment_payload(self) -> dict[str, Tensor | int | str]:
        return {
            "kind": "fixed_wiring_2input_lut",
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "indices_0": self.indices_0.detach().cpu().to(torch.int32),
            "indices_1": self.indices_1.detach().cpu().to(torch.int32),
            "op_ids": self.hard_ops().cpu().to(torch.uint8),
        }


def total_gate_count(layers: Iterable[HardSTGateLayer]) -> int:
    return sum(layer.out_dim for layer in layers)
