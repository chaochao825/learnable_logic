"""Trainable shadows with multiplier-free LUT deployment payloads.

The modules in this file deliberately do not depend on ``model.py``.  They can
be inserted into a full-discrete model without changing its integer carrier
contract:

* :class:`GroupwiseDiscreteActivationLUT` maps a signed A8 code through one
  256-entry signed-int8 table per feature group.  The output reuses the input
  power-of-two scale, so an identity table is exactly amplitude preserving.
* :class:`MonotonicHeadGapLUT` maps an integer attention-score gap through one
  monotonic table per head.  Its only deployable values are ``0, 1, 2, 4, 8``;
  multiplication is consequently a mask or a shift.

Training uses floating shadows only for gradients.  The numerical forward pass
is already the exact hard payload lookup through a straight-through estimator
(STE), and evaluation contains no shadow-path arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from .shiftadd import _power_of_two_scale


class _HardPayloadSoftGradient(torch.autograd.Function):
    """Return the hard payload exactly while routing gradients to the surrogate."""

    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, grad_output


class GroupwiseDiscreteActivationLUT(nn.Module):
    """A group-wise 8-bit-in/8-bit-out activation lookup table.

    The last feature dimension is partitioned into contiguous groups.  Each
    group is quantized to signed A8 with a runtime power-of-two scale and uses
    its own 256-entry table.  Table addresses are ``input_code + 128``.  Table
    entries are signed int8 output codes and the output reuses the input scale.

    Reusing the scale has two useful properties: identity initialization is
    exactly amplitude preserving, and deployed inference needs no table-output
    multiplier or additional scale metadata.  The table may still learn any
    group-specific mapping from an A8 code to another A8 code.

    During training, the returned value is the hard lookup.  Gradients follow a
    piecewise-linear interpolation through ``shadow_table`` with respect to
    both the input and the floating table shadow.  During evaluation only the
    rounded/clamped signed-int8 table is read.
    """

    table_entries = 256
    code_min = -128
    code_max = 127
    zero_point = 128

    def __init__(
        self,
        features: int,
        groups: int = 1,
        *,
        initialization: str = "identity",
        minimum_scale_exponent: int = -24,
    ) -> None:
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        if groups < 1 or features % groups:
            raise ValueError("groups must be positive and divide features")
        if initialization not in {"identity", "relu", "negative"}:
            raise ValueError("initialization must be identity, relu, or negative")

        self.features = int(features)
        self.groups = int(groups)
        self.group_size = self.features // self.groups
        self.minimum_scale_exponent = int(minimum_scale_exponent)

        signed_codes = torch.arange(self.code_min, self.code_max + 1, dtype=torch.float32)
        if initialization == "relu":
            signed_codes = signed_codes.clamp_min(0)
        elif initialization == "negative":
            signed_codes = -signed_codes
        self.shadow_table = nn.Parameter(signed_codes.repeat(self.groups, 1))

    def _group(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.features:
            raise ValueError(
                f"expected last dimension {self.features}, got {tuple(x.shape)}"
            )
        return x.reshape(*x.shape[:-1], self.groups, self.group_size)

    def integer_input_code_and_scale(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return signed-int8 input codes and per-item/per-group power-of-two scales."""

        grouped = self._group(x)
        maximum = grouped.detach().abs().amax(dim=-1, keepdim=True)
        scale = _power_of_two_scale(maximum / self.code_max, self.minimum_scale_exponent)
        code = torch.round(grouped / scale).clamp(self.code_min, self.code_max)
        return code.to(torch.int8), scale

    def integer_table(self) -> torch.Tensor:
        """Return the exact ``[groups, 256]`` signed-int8 deployment table."""

        return torch.round(self.shadow_table.detach()).clamp(
            self.code_min, self.code_max
        ).to(torch.int8)

    @staticmethod
    def _lookup(table: torch.Tensor, address: torch.Tensor) -> torch.Tensor:
        """Look up ``table[group, address]`` for arbitrary leading dimensions."""

        if address.ndim < 2:
            raise ValueError("address must include group and within-group dimensions")
        groups = table.shape[0]
        if address.shape[-2] != groups:
            raise ValueError("address group dimension does not match table")
        group_index = torch.arange(groups, device=address.device).view(
            *([1] * (address.ndim - 2)), groups, 1
        ).expand_as(address)
        return table[group_index, address]

    def integer_lookup(self, input_code: torch.Tensor) -> torch.Tensor:
        """Apply the deployment table to grouped signed-int8 codes.

        ``input_code`` must have shape ``[..., groups, group_size]``.  The
        returned tensor has the same shape and dtype ``torch.int8``.
        """

        if input_code.shape[-2:] != (self.groups, self.group_size):
            raise ValueError(
                "input_code must end in "
                f"({self.groups}, {self.group_size}), got {tuple(input_code.shape)}"
            )
        if input_code.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("input_code must be a signed integer tensor")
        if bool(((input_code < self.code_min) | (input_code > self.code_max)).any()):
            raise ValueError("input_code is outside signed-int8 range")
        address = input_code.to(torch.int64) + self.zero_point
        return self._lookup(self.integer_table(), address)

    def _soft_lookup(self, grouped: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        continuous_address = (grouped / scale + self.zero_point).clamp(
            0.0, float(self.table_entries - 1)
        )
        lower = continuous_address.floor().to(torch.int64)
        upper = (lower + 1).clamp_max(self.table_entries - 1)
        fraction = continuous_address - lower.to(continuous_address.dtype)
        lower_value = self._lookup(self.shadow_table, lower)
        upper_value = self._lookup(self.shadow_table, upper)
        return lower_value + fraction * (upper_value - lower_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grouped = self._group(x)
        input_code, scale = self.integer_input_code_and_scale(x)
        hard_code = self.integer_lookup(input_code).to(dtype=x.dtype)
        hard = hard_code * scale
        if self.training:
            soft_code = self._soft_lookup(grouped, scale)
            output = _HardPayloadSoftGradient.apply(hard, soft_code * scale)
        else:
            output = hard
        return output.reshape_as(x)

    def deployment_payload(self) -> dict[str, object]:
        """Return the complete non-floating payload needed by inference."""

        return {
            "table_int8": self.integer_table(),
            "input_zero_point_uint8": self.zero_point,
            "code_range": [self.code_min, self.code_max],
            "features": self.features,
            "groups": self.groups,
            "group_size": self.group_size,
            "minimum_scale_exponent": self.minimum_scale_exponent,
            "rounding": "IEEE ties-to-even for scale exponent and input/table codes",
        }

    def deployment_contract(self) -> dict[str, object]:
        return {
            "address": "uint8(input_signed_code + 128)",
            "table_shape": [self.groups, self.table_entries],
            "table_entry": "signed int8",
            "input_scale": "one runtime power-of-two per item and feature group",
            "output_scale": "reuse the input power-of-two scale",
            "rounding": "IEEE ties-to-even; clamp signed input/table codes to [-128,127]",
            "runtime": "group max/leading-one + shifts + 256:1 ROM lookup",
            "general_multipliers": 0,
            "training_only_state": "floating shadow_table",
        }


class MonotonicHeadGapLUT(nn.Module):
    """Per-head learned score-gap weights with an exact shift-only payload.

    The learned latent level for every head is constrained to be nonincreasing
    as integer ``gap`` grows.  Hard levels select exactly one member of
    ``{0, 1, 2, 4, 8}``; zero masks a candidate and the other values are left
    shifts by 0, 1, 2, or 3 bits.

    Monotonicity is structural rather than a penalty: the first level is at
    most four and every following level subtracts a nonnegative softplus drop.
    Rounding a nonincreasing level sequence and indexing an ordered value set
    preserves monotonicity for every possible parameter value.
    """

    allowed_weights = (0, 1, 2, 4, 8)

    def __init__(
        self,
        heads: int,
        max_gap: int,
        *,
        initial_weights: Sequence[int] | torch.Tensor | None = None,
        gap_shift: int = 1,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be positive")
        if max_gap < 0:
            raise ValueError("max_gap must be nonnegative")
        if gap_shift < 0:
            raise ValueError("gap_shift must be nonnegative")
        self.heads = int(heads)
        self.max_gap = int(max_gap)
        self.gap_count = self.max_gap + 1

        target = self._initial_level_table(initial_weights, gap_shift)
        # start_level = 4 - softplus(start_headroom).  A tiny positive
        # headroom avoids an infinite parameter while hardening exactly to 8.
        start_headroom = (4.0 - target[:, 0]).clamp_min(3.0e-4)
        self.raw_start_headroom = nn.Parameter(self._inverse_softplus(start_headroom))
        if self.max_gap:
            target_drop = (target[:, :-1] - target[:, 1:]).clamp_min(3.0e-4)
            self.raw_drops = nn.Parameter(self._inverse_softplus(target_drop))
        else:
            self.register_parameter("raw_drops", None)
        self.register_buffer(
            "_allowed_weight_tensor",
            torch.tensor(self.allowed_weights, dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        # log(expm1(x)) is accurate for the small values used for plateaus.
        return torch.log(torch.expm1(value))

    def _initial_level_table(
        self,
        initial_weights: Sequence[int] | torch.Tensor | None,
        gap_shift: int,
    ) -> torch.Tensor:
        if initial_weights is None:
            weights = [8 >> min(gap >> gap_shift, 3) for gap in range(self.gap_count)]
            table = torch.tensor(weights, dtype=torch.int64).unsqueeze(0).repeat(
                self.heads, 1
            )
        else:
            table = torch.as_tensor(initial_weights, dtype=torch.int64)
            if table.ndim == 1:
                if table.numel() != self.gap_count:
                    raise ValueError("initial_weights length must equal max_gap + 1")
                table = table.unsqueeze(0).repeat(self.heads, 1)
            elif table.shape != (self.heads, self.gap_count):
                raise ValueError(
                    "initial_weights must have shape [max_gap+1] or "
                    "[heads, max_gap+1]"
                )
        allowed = torch.tensor(self.allowed_weights, dtype=torch.int64)
        membership = (table.unsqueeze(-1) == allowed).any(dim=-1)
        if not bool(membership.all()):
            raise ValueError("initial_weights entries must be in {0,1,2,4,8}")
        if self.max_gap and not bool((table[:, 1:] <= table[:, :-1]).all()):
            raise ValueError("initial_weights must be nonincreasing with gap")
        level = torch.empty_like(table, dtype=torch.float32)
        for index, weight in enumerate(self.allowed_weights):
            level[table == weight] = float(index)
        return level

    def latent_level_table(self) -> torch.Tensor:
        start = 4.0 - torch.nn.functional.softplus(self.raw_start_headroom)
        start = start.clamp(0.0, 4.0)
        if not self.max_gap:
            return start.unsqueeze(-1)
        drops = torch.nn.functional.softplus(self.raw_drops)
        cumulative = torch.cat(
            (torch.zeros_like(start).unsqueeze(-1), drops.cumsum(dim=-1)), dim=-1
        )
        return (start.unsqueeze(-1) - cumulative).clamp(0.0, 4.0)

    def soft_weight_table(self) -> torch.Tensor:
        """Return the differentiable monotonic shadow table."""

        level = self.latent_level_table()
        lower = level.floor().to(torch.int64)
        upper = (lower + 1).clamp_max(len(self.allowed_weights) - 1)
        fraction = level - lower.to(level.dtype)
        values = self._allowed_weight_tensor.to(dtype=level.dtype, device=level.device)
        return values[lower] + fraction * (values[upper] - values[lower])

    def integer_table(self) -> torch.Tensor:
        """Return the exact monotonic ``[heads, max_gap+1]`` uint8 table."""

        level = torch.floor(self.latent_level_table().detach() + 0.5).to(torch.int64)
        level = level.clamp(0, len(self.allowed_weights) - 1)
        values = self._allowed_weight_tensor.to(device=level.device)
        return values[level].to(torch.uint8)

    def shift_code_table(self) -> torch.Tensor:
        """Return -1 for weight zero, otherwise log2(weight) in ``[0,3]``."""

        weight = self.integer_table().to(torch.int16)
        shift = torch.full_like(weight, -1, dtype=torch.int8)
        nonzero = weight > 0
        shift[nonzero] = torch.round(torch.log2(weight[nonzero].to(torch.float32))).to(
            torch.int8
        )
        return shift

    def _validate_gap(self, gap: torch.Tensor, head_axis: int) -> int:
        if gap.ndim < 1:
            raise ValueError("gap must have at least one dimension")
        axis = head_axis if head_axis >= 0 else gap.ndim + head_axis
        if axis < 0 or axis >= gap.ndim:
            raise ValueError("head_axis is outside the gap tensor")
        if gap.shape[axis] != self.heads:
            raise ValueError(
                f"gap head dimension must be {self.heads}, got {gap.shape[axis]}"
            )
        if gap.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("gap must be an integer tensor")
        return axis

    @staticmethod
    def _table_lookup(table: torch.Tensor, gap: torch.Tensor, head_axis: int) -> torch.Tensor:
        moved = gap.movedim(head_axis, -1)
        index = moved.to(torch.int64).clamp(0, table.shape[1] - 1)
        head_index = torch.arange(table.shape[0], device=gap.device).view(
            *([1] * (index.ndim - 1)), table.shape[0]
        )
        result = table[head_index, index]
        return result.movedim(-1, head_axis)

    def integer_lookup(self, gap: torch.Tensor, *, head_axis: int = 1) -> torch.Tensor:
        """Look up hard uint8 weights, clamping gaps above ``max_gap``."""

        axis = self._validate_gap(gap, head_axis)
        return self._table_lookup(self.integer_table(), gap, axis)

    def forward(self, gap: torch.Tensor, *, head_axis: int = 1) -> torch.Tensor:
        axis = self._validate_gap(gap, head_axis)
        hard = self._table_lookup(self.integer_table(), gap, axis).to(
            dtype=self.raw_start_headroom.dtype
        )
        if not self.training:
            return hard
        soft = self._table_lookup(self.soft_weight_table(), gap, axis)
        return _HardPayloadSoftGradient.apply(hard, soft)

    def deployment_payload(self) -> dict[str, object]:
        return {
            "weight_table_uint8": self.integer_table(),
            "shift_code_int8": self.shift_code_table(),
            "zero_shift_code": -1,
            "max_gap": self.max_gap,
            "heads": self.heads,
        }

    def deployment_contract(self) -> dict[str, object]:
        return {
            "address": f"clamp(unsigned integer gap, 0, {self.max_gap})",
            "table_shape": [self.heads, self.gap_count],
            "table_entry_values": list(self.allowed_weights),
            "table_entry_bits": 4,
            "monotonic": "nonincreasing along gap for every head",
            "runtime": "per-head ROM + zero mask or left shift by 0..3",
            "general_multipliers": 0,
            "training_only_state": "start headroom and nonnegative drop shadows",
        }


__all__ = ["GroupwiseDiscreteActivationLUT", "MonotonicHeadGapLUT"]
