"""Content-dependent global token fusion with explicit A8-by-A8 ROM trees.

The hard operator contains no learned matrix multiplication.  Patch codes are
reduced by a balanced binary tree whose two-input truth tables are shared over
all spatial nodes at the same stage and channel group.  A second table fuses
the root with CLS, and a third table broadcasts that context back to every
token.  All tables have a 16-bit address and a signed A8 payload.

Training uses a bilinear table surrogate only for derivatives.  The returned
forward value is always the exact rounded hard-table transaction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .enhancements_hadamard import rounded_power_of_two_divide
from .shiftadd import _power_of_two_scale, _ste


LUT_AXIS_SIZE = 256
LUT_ENTRIES = LUT_AXIS_SIZE * LUT_AXIS_SIZE


def signed_a8_lut_index(code: torch.Tensor) -> torch.Tensor:
    """Map signed A8 codes to monotone 0..255 ROM coordinates."""

    return (code.to(torch.int64) + 128).clamp(0, 255)


def _group_table_lookup(
    table: torch.Tensor,
    left_index: torch.Tensor,
    right_index: torch.Tensor,
) -> torch.Tensor:
    """Gather one payload per [batch,node,group,channel-in-group] address."""

    if table.ndim != 3 or table.shape[-2:] != (LUT_AXIS_SIZE, LUT_AXIS_SIZE):
        raise ValueError("table must have shape [groups,256,256]")
    if left_index.shape != right_index.shape or left_index.ndim != 4:
        raise ValueError("LUT indices must share shape [batch,node,group,channel]")
    if left_index.shape[2] != table.shape[0]:
        raise ValueError("LUT index group axis does not match table")
    address = torch.bitwise_or(
        torch.bitwise_left_shift(left_index.to(torch.int64), 8),
        right_index.to(torch.int64),
    )
    groups = table.shape[0]
    batch, nodes, _, channels = address.shape
    gather_index = address.permute(2, 0, 1, 3).reshape(groups, -1)
    gathered = torch.gather(table.reshape(groups, LUT_ENTRIES), 1, gather_index)
    return gathered.reshape(groups, batch, nodes, channels).permute(1, 2, 0, 3)


def hard_group_table_lookup(
    table: torch.Tensor, left_code: torch.Tensor, right_code: torch.Tensor
) -> torch.Tensor:
    """Exact integer A8-by-A8 lookup used by the deployable path."""

    if table.is_floating_point():
        raise TypeError("hard table payload must be integer")
    return _group_table_lookup(
        table,
        signed_a8_lut_index(left_code),
        signed_a8_lut_index(right_code),
    ).to(torch.int64)


def bilinear_group_table_lookup(
    table: torch.Tensor,
    left_code: torch.Tensor,
    right_code: torch.Tensor,
    qmin: int = -127,
    qmax: int = 127,
) -> torch.Tensor:
    """Training-only continuous derivative surrogate for the same table."""

    if not table.is_floating_point():
        raise TypeError("bilinear surrogate table must be floating point")
    left_position = (left_code + 128.0).clamp(0.0, 255.0)
    right_position = (right_code + 128.0).clamp(0.0, 255.0)
    left_floor = torch.floor(left_position).to(torch.int64)
    right_floor = torch.floor(right_position).to(torch.int64)
    left_ceil = (left_floor + 1).clamp_max(255)
    right_ceil = (right_floor + 1).clamp_max(255)
    left_fraction = left_position - left_floor.to(left_position.dtype)
    right_fraction = right_position - right_floor.to(right_position.dtype)
    bounded_table = table.clamp(float(qmin), float(qmax))

    value_00 = _group_table_lookup(bounded_table, left_floor, right_floor)
    value_01 = _group_table_lookup(bounded_table, left_floor, right_ceil)
    value_10 = _group_table_lookup(bounded_table, left_ceil, right_floor)
    value_11 = _group_table_lookup(bounded_table, left_ceil, right_ceil)
    left_fraction = left_fraction.to(table.dtype)
    right_fraction = right_fraction.to(table.dtype)
    top = value_00 + right_fraction * (value_01 - value_00)
    bottom = value_10 + right_fraction * (value_11 - value_10)
    return top + left_fraction * (bottom - top)


def _initial_average_table(qmin: int, qmax: int) -> torch.Tensor:
    codes = torch.arange(-128, 128, dtype=torch.int64)
    left = codes[:, None]
    right = codes[None, :]
    return rounded_power_of_two_divide(left + right, 1).clamp(qmin, qmax).to(torch.float32)


def _initial_projection_b_table(qmin: int, qmax: int) -> torch.Tensor:
    codes = torch.arange(-128, 128, dtype=torch.int64)
    return codes[None, :].expand(256, 256).clamp(qmin, qmax).to(torch.float32)


class A8GlobalLUTTreeMixer(nn.Module):
    """Balanced nonlinear global reduction/context/broadcast ROM network."""

    def __init__(
        self,
        dim: int,
        patch_tokens: int,
        block_index: int,
        activation_bits: int = 8,
        group_size: int = 32,
        branch_shift: int = 2,
    ) -> None:
        super().__init__()
        if activation_bits != 8:
            raise ValueError("A8GlobalLUTTreeMixer requires activation_bits=8")
        if patch_tokens < 2 or patch_tokens & (patch_tokens - 1):
            raise ValueError("patch_tokens must be a power of two greater than one")
        if group_size < 1 or dim % group_size:
            raise ValueError("group_size must be positive and divide dim")
        if branch_shift < 0 or branch_shift > 7:
            raise ValueError("branch_shift must be in [0,7]")
        self.dim = int(dim)
        self.patch_tokens = int(patch_tokens)
        self.block_index = int(block_index)
        self.activation_bits = int(activation_bits)
        self.group_size = int(group_size)
        self.groups = self.dim // self.group_size
        self.branch_shift = int(branch_shift)
        self.stages = int(math.log2(self.patch_tokens))
        self.qmin = -127
        self.qmax = 127
        self.shadow_dither = 0.49

        average = _initial_average_table(self.qmin, self.qmax)
        projection_b = _initial_projection_b_table(self.qmin, self.qmax)
        row = torch.arange(256, dtype=torch.int64)[:, None]
        column = torch.arange(256, dtype=torch.int64)[None, :]
        checker = torch.where(
            torch.bitwise_and(row + column, 1).bool(), 1.0, -1.0
        ).to(torch.float32) * self.shadow_dither
        reduce_polarity = torch.where(
            torch.bitwise_and(
                torch.arange(self.stages)[:, None]
                + torch.arange(self.groups)[None, :]
                + self.block_index,
                1,
            ).bool(),
            1.0,
            -1.0,
        ).to(torch.float32)
        group_polarity = torch.where(
            torch.bitwise_and(
                torch.arange(self.groups) + self.block_index, 1
            ).bool(),
            1.0,
            -1.0,
        ).to(torch.float32)
        self.register_buffer("_dither_checker", checker, persistent=False)
        self.register_buffer("_reduce_polarity", reduce_polarity, persistent=False)
        self.register_buffer("_group_polarity", group_polarity, persistent=False)
        self.reduce_table_shadow = nn.Parameter(
            average[None, None].repeat(self.stages, self.groups, 1, 1)
            + reduce_polarity[:, :, None, None] * checker[None, None]
        )
        self.context_table_shadow = nn.Parameter(
            average[None].repeat(self.groups, 1, 1)
            + group_polarity[:, None, None] * checker[None]
        )
        self.broadcast_table_shadow = nn.Parameter(
            projection_b[None].repeat(self.groups, 1, 1)
            - group_polarity[:, None, None] * checker[None]
        )

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch,1+patch_tokens,dim]")
        if tokens.shape[1:] != (self.patch_tokens + 1, self.dim):
            raise ValueError(
                f"tokens must have shape [batch,{self.patch_tokens + 1},{self.dim}]"
            )

    @staticmethod
    def _hard_table(shadow: torch.Tensor) -> torch.Tensor:
        return torch.round(shadow.detach()).clamp(-127, 127).to(torch.int8)

    def hard_table_payloads(self) -> dict[str, torch.Tensor]:
        return {
            "reduce_table_int8": self._hard_table(self.reduce_table_shadow),
            "context_table_int8": self._hard_table(self.context_table_shadow),
            "broadcast_table_int8": self._hard_table(self.broadcast_table_shadow),
        }

    def _soft_reduce_table(self, stage: int) -> torch.Tensor:
        dither = (
            self._reduce_polarity[stage, :, None, None]
            * self._dither_checker[None]
        )
        return self.reduce_table_shadow[stage] - dither

    def _soft_context_table(self) -> torch.Tensor:
        dither = self._group_polarity[:, None, None] * self._dither_checker[None]
        return self.context_table_shadow - dither

    def _soft_broadcast_table(self) -> torch.Tensor:
        dither = -self._group_polarity[:, None, None] * self._dither_checker[None]
        return self.broadcast_table_shadow - dither

    @torch.no_grad()
    def hard_table_change_statistics(self) -> dict[str, object]:
        current = self.hard_table_payloads()
        average = _initial_average_table(self.qmin, self.qmax).to(torch.int8)
        projection_b = _initial_projection_b_table(self.qmin, self.qmax).to(torch.int8)
        initial = {
            "reduce": average[None, None].expand(
                self.stages, self.groups, 256, 256
            ),
            "context": average[None].expand(self.groups, 256, 256),
            "broadcast": projection_b[None].expand(self.groups, 256, 256),
        }
        observed = {
            "reduce": current["reduce_table_int8"],
            "context": current["context_table_int8"],
            "broadcast": current["broadcast_table_int8"],
        }
        result: dict[str, object] = {}
        total_entries = total_changed = total_absolute_delta = 0
        for name in ("reduce", "context", "broadcast"):
            delta = observed[name].to(torch.int16) - initial[name].to(
                observed[name].device, torch.int16
            )
            changed = int((delta != 0).sum())
            entries = delta.numel()
            absolute_delta = int(delta.abs().sum())
            result[f"{name}_hard_changed_entries"] = changed
            result[f"{name}_hard_total_entries"] = entries
            result[f"{name}_hard_absolute_code_delta"] = absolute_delta
            result[f"{name}_hard_max_code_delta"] = int(delta.abs().max())
            total_entries += entries
            total_changed += changed
            total_absolute_delta += absolute_delta
        result["total_hard_changed_entries"] = total_changed
        result["total_hard_entries"] = total_entries
        result["total_hard_absolute_code_delta"] = total_absolute_delta
        return result

    def statistics(self) -> dict[str, object]:
        return self.hard_table_change_statistics()

    def integer_code_and_scale(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_tokens(tokens)
        grouped = tokens.detach().reshape(
            tokens.shape[0], tokens.shape[1], self.groups, self.group_size
        )
        maximum = grouped.abs().amax(dim=(1, 3), keepdim=True)
        scale = _power_of_two_scale(maximum / self.qmax)
        code = torch.round(grouped / scale).clamp(self.qmin, self.qmax)
        return code.to(torch.int64), scale

    def hard_reference_from_grouped_code(
        self, grouped_code: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if grouped_code.ndim != 4 or grouped_code.shape[1:] != (
            self.patch_tokens + 1, self.groups, self.group_size
        ):
            raise ValueError("grouped_code shape does not match mixer topology")
        if grouped_code.is_floating_point():
            raise TypeError("grouped_code must be integer")
        tables = self.hard_table_payloads()
        cls_code = grouped_code[:, :1].to(torch.int64)
        level = grouped_code[:, 1:].to(torch.int64)
        levels = [level]
        for stage in range(self.stages):
            left = level[:, 0::2]
            right = level[:, 1::2]
            level = hard_group_table_lookup(
                tables["reduce_table_int8"][stage], left, right
            )
            levels.append(level)
        root_code = level
        context_code = hard_group_table_lookup(
            tables["context_table_int8"], root_code, cls_code
        )
        patch_raw = hard_group_table_lookup(
            tables["broadcast_table_int8"],
            grouped_code[:, 1:].to(torch.int64),
            context_code.expand(-1, self.patch_tokens, -1, -1),
        )
        cls_raw = hard_group_table_lookup(
            tables["broadcast_table_int8"], cls_code, context_code
        )
        patch_update = rounded_power_of_two_divide(patch_raw, self.branch_shift)
        cls_update = rounded_power_of_two_divide(cls_raw, self.branch_shift)
        update_code = torch.cat((cls_update, patch_update), dim=1)
        return {
            "output_code": update_code,
            "root_code": root_code,
            "context_code": context_code,
            "reduction_levels": levels,
            **tables,
        }

    def integer_reference(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        grouped_code, scale = self.integer_code_and_scale(tokens)
        reference = self.hard_reference_from_grouped_code(grouped_code)
        output = (reference["output_code"].to(tokens.dtype) * scale).reshape_as(tokens)
        return {
            **reference,
            "output": output,
            "input_code": grouped_code,
            "input_scale": scale,
        }

    def soft_reference(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(tokens)
        grouped = tokens.reshape(
            tokens.shape[0], tokens.shape[1], self.groups, self.group_size
        )
        maximum = grouped.detach().abs().amax(dim=(1, 3), keepdim=True)
        scale = _power_of_two_scale(maximum / self.qmax)
        soft_code = (grouped / scale).clamp(float(self.qmin), float(self.qmax))
        cls_code = soft_code[:, :1]
        level = soft_code[:, 1:]
        for stage in range(self.stages):
            level = bilinear_group_table_lookup(
                self._soft_reduce_table(stage),
                level[:, 0::2],
                level[:, 1::2],
                self.qmin,
                self.qmax,
            )
        context = bilinear_group_table_lookup(
            self._soft_context_table(), level, cls_code, self.qmin, self.qmax
        )
        patch_raw = bilinear_group_table_lookup(
            self._soft_broadcast_table(),
            soft_code[:, 1:],
            context.expand(-1, self.patch_tokens, -1, -1),
            self.qmin,
            self.qmax,
        )
        cls_raw = bilinear_group_table_lookup(
            self._soft_broadcast_table(), cls_code, context, self.qmin, self.qmax
        )
        divisor = float(1 << self.branch_shift)
        update = torch.cat((cls_raw / divisor, patch_raw / divisor), dim=1)
        return (update * scale).reshape_as(tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hard = self.integer_reference(tokens)["output"]
        if not self.training:
            return hard
        return _ste(hard, self.soft_reference(tokens))

    def deployment_contract(self) -> dict[str, object]:
        tables_per_block = (self.stages + 2) * self.groups
        return {
            "operator": "group_shared_a8_pair_lut_global_tree",
            "patch_tokens": self.patch_tokens,
            "block_index": self.block_index,
            "activation_bits": self.activation_bits,
            "group_size": self.group_size,
            "groups": self.groups,
            "reduction_stages": self.stages,
            "tables_per_block": tables_per_block,
            "entries_per_table": LUT_ENTRIES,
            "payload_bits_per_entry": 8,
            "hard_payload_bits": tables_per_block * LUT_ENTRIES * 8,
            "address": "((signed_a8_left + 128) << 8) | (signed_a8_right + 128)",
            "spatial_sharing": "one table per stage/group across every tree node",
            "channel_sharing": "one table across channels inside each group",
            "root_context": "six-stage patch tree then root/CLS A8 table",
            "broadcast": "A8 pair table from token/context followed by rounded shift",
            "branch_right_shift": self.branch_shift,
            "output_code_signed_bits": self.activation_bits,
            "learned_connections": False,
            "general_multipliers_hard_forward": 0,
            "training_only_surrogate": "bilinear four-entry interpolation",
            "training_shadow_dither": (
                "plus_or_minus_0.49_removed_from_soft_surrogate; hard payload unchanged"
            ),
        }
