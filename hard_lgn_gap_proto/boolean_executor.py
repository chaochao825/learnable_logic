"""Strict Boolean/integer executor for frozen two-input logic networks.

Training may use floating logits, softmax, Gumbel noise, and STE. Deployment
does not. This executor accepts only CPU Boolean features, applies hardened
wiring and one of the sixteen Boolean truth tables, and returns integer class
counts. GroupSum temperature scaling and cross entropy belong to measurement,
not to the deployable decision path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch.utils._python_dispatch import TorchDispatchMode


BOOLEAN_GATE_TRUTH = torch.tensor(
    [
        [False, False, False, False],
        [False, False, False, True],
        [False, False, True, False],
        [False, False, True, True],
        [False, True, False, False],
        [False, True, False, True],
        [False, True, True, False],
        [False, True, True, True],
        [True, False, False, False],
        [True, False, False, True],
        [True, False, True, False],
        [True, False, True, True],
        [True, True, False, False],
        [True, True, False, True],
        [True, True, True, False],
        [True, True, True, True],
    ],
    dtype=torch.bool,
)


def _tensor_leaves(value: object):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensor_leaves(child)


class BooleanRuntimeAudit(TorchDispatchMode):
    """Fail when a strict hard-network operator sees a real-valued tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.operations = 0

    @staticmethod
    def _check(value: object) -> None:
        for tensor in _tensor_leaves(value):
            if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
                raise TypeError(f"real-valued runtime tensor detected: {tensor.dtype}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        keyword = kwargs or {}
        self._check((args, keyword))
        output = func(*args, **keyword)
        self._check(output)
        self.operations += 1
        return output


def _require_cpu_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    return value


def _require_integer(name: str, value: torch.Tensor) -> torch.Tensor:
    value = _require_cpu_tensor(name, value)
    if value.dtype not in {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(f"{name} must have an integer dtype")
    return value


@dataclass(frozen=True)
class BooleanLayerPayload:
    in_dim: int
    out_dim: int
    left_source: torch.Tensor
    right_source: torch.Tensor
    truth_table_id: torch.Tensor

    def __post_init__(self) -> None:
        if self.in_dim < 1 or self.out_dim < 1:
            raise ValueError("layer dimensions must be positive")
        for name, value in (
            ("left_source", self.left_source),
            ("right_source", self.right_source),
            ("truth_table_id", self.truth_table_id),
        ):
            _require_integer(name, value)
            if value.dtype != torch.int64 or value.shape != (self.out_dim,):
                raise TypeError(f"{name} must be int64 with shape [out_dim]")
        if int(self.left_source.min().item()) < 0 or int(
            self.left_source.max().item()
        ) >= self.in_dim:
            raise ValueError("left wiring index is out of range")
        if int(self.right_source.min().item()) < 0 or int(
            self.right_source.max().item()
        ) >= self.in_dim:
            raise ValueError("right wiring index is out of range")
        if int(self.truth_table_id.min().item()) < 0 or int(
            self.truth_table_id.max().item()
        ) > 15:
            raise ValueError("truth-table id must be in [0,15]")

    @classmethod
    def from_frozen_layer(cls, layer: object) -> "BooleanLayerPayload":
        required = ("in_dim", "out_dim", "indices_0", "indices_1", "op_ids")
        if any(not hasattr(layer, name) for name in required):
            raise TypeError("frozen layer does not expose the required hard payload")
        return cls(
            int(layer.in_dim),
            int(layer.out_dim),
            layer.indices_0.detach().cpu().to(torch.int64),
            layer.indices_1.detach().cpu().to(torch.int64),
            layer.op_ids.detach().cpu().to(torch.int64),
        )


def boolean_gate_bank(
    left: torch.Tensor, right: torch.Tensor, truth_table_id: torch.Tensor
) -> torch.Tensor:
    left = _require_cpu_tensor("left", left)
    right = _require_cpu_tensor("right", right)
    truth_table_id = _require_integer("truth_table_id", truth_table_id)
    if left.dtype != torch.bool or right.dtype != torch.bool:
        raise TypeError("Boolean gate inputs must use torch.bool")
    if left.shape != right.shape or left.shape[-1] != truth_table_id.numel():
        raise ValueError("Boolean gate input/payload shapes do not match")
    address = torch.bitwise_or(
        torch.bitwise_left_shift(left.to(torch.uint8), 1), right.to(torch.uint8)
    ).to(torch.int64)
    return BOOLEAN_GATE_TRUTH[
        truth_table_id.reshape(1, -1), address
    ]


class StrictBooleanLogicExecutor:
    """Frozen DLGN/Hard-LGN forward with no real-valued runtime state."""

    def __init__(self, layers: Iterable[object], num_classes: int | None = None) -> None:
        self.layers = tuple(
            BooleanLayerPayload.from_frozen_layer(layer) for layer in layers
        )
        if not self.layers:
            raise ValueError("at least one hard layer is required")
        if num_classes is not None:
            if num_classes < 2:
                raise ValueError("num_classes must be at least two")
            if self.layers[-1].out_dim % num_classes:
                raise ValueError("final width must be divisible by num_classes")
        for previous, current in zip(self.layers, self.layers[1:]):
            if previous.out_dim != current.in_dim:
                raise ValueError("adjacent hard layer dimensions do not match")
        self.num_classes = None if num_classes is None else int(num_classes)

    def layer_outputs(self, features: torch.Tensor) -> list[torch.Tensor]:
        features = _require_cpu_tensor("features", features)
        if features.dtype != torch.bool:
            raise TypeError("strict hard-network input must use torch.bool")
        if features.ndim != 2 or features.shape[1] != self.layers[0].in_dim:
            raise ValueError("strict hard-network input shape mismatch")
        outputs = []
        current = features
        for layer in self.layers:
            left = current[:, layer.left_source]
            right = current[:, layer.right_source]
            current = boolean_gate_bank(left, right, layer.truth_table_id)
            outputs.append(current)
        return outputs

    def class_counts(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_classes is None:
            raise ValueError("num_classes is required for class counts")
        output = self.layer_outputs(features)[-1]
        per_class = output.shape[-1] // self.num_classes
        return output.reshape(
            output.shape[0], self.num_classes, per_class
        ).sum(dim=-1, dtype=torch.int64)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.class_counts(features).argmax(dim=-1).to(torch.int64)


__all__ = [
    "BOOLEAN_GATE_TRUTH",
    "BooleanLayerPayload",
    "BooleanRuntimeAudit",
    "StrictBooleanLogicExecutor",
    "boolean_gate_bank",
]
