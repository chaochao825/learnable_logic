"""A pure-PyTorch, hardware-oriented spatial pyramid for logic-gate ViTs.

The module consumes ``[B, 1 + H*W, C]`` tokens (CLS first) and produces a
same-shaped residual delta by default.  For an 8x8 patch grid, three shared
four-input LUT stages form an 8 -> 4 -> 2 -> 1 hierarchy.  The root can update
only CLS, or the 4x4, 2x2, and 1x1 states can be broadcast back to patches.

Runtime hard semantics use only fixed wiring, threshold comparisons, integer
bit packing, LUT reads, low-bit additions, and a power-of-two shift.  Tanh and
round are used only to compile trainable parameters into LUT contents; exported
integer tables contain the values needed by a deployment implementation.

This is ConvLogic-inspired rather than an implementation of ConvLogic itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _validate_bits(bits: int) -> int:
    bits = int(bits)
    if bits < 1 or bits > 4:
        raise ValueError(
            "bits must be in [1, 4] for this PyTorch prototype; four 4-bit "
            "inputs already require 65,536 entries per channel and stage"
        )
    return bits


def _comparison_quantize(
    x: torch.Tensor,
    *,
    bits: int,
    clip: float,
    ste: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform quantization implemented as threshold comparisons.

    Returns a decoded floating tensor for the PyTorch interface and unsigned
    integer codes for LUT addressing.  The forward values are identical with
    or without STE; only the backward path differs.
    """

    levels = (1 << bits) - 1
    clip = float(clip)
    if clip <= 0.0:
        raise ValueError(f"clip must be positive, got {clip}")
    step = (2.0 * clip) / float(levels)
    thresholds = torch.linspace(
        -clip + 0.5 * step,
        clip - 0.5 * step,
        levels,
        dtype=x.dtype,
        device=x.device,
    )
    codes = (x.unsqueeze(-1) >= thresholds).sum(dim=-1).to(torch.long)
    decoded = codes.to(x.dtype) * step - clip
    if not ste:
        return decoded, codes
    clipped = x.clamp(-clip, clip)
    return clipped + (decoded - clipped).detach(), codes


def _compile_lut(
    logits: torch.Tensor,
    *,
    bits: int,
    clip: float,
    ste: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compile trainable logits into signed low-bit LUT values and codes."""

    levels = (1 << bits) - 1
    bounded = torch.tanh(logits) * float(clip)
    normalized = (bounded / float(clip) + 1.0) * 0.5
    codes = torch.round(normalized * levels).clamp(0, levels).to(torch.long)
    decoded = codes.to(logits.dtype) / float(levels) * (2.0 * clip) - clip
    if ste:
        decoded = bounded + (decoded - bounded).detach()
    return decoded, codes


def _mean_lut_logits(bits: int, clip: float, fan_in: int = 4) -> torch.Tensor:
    """Initialize an addressed LUT to the quantized mean of its inputs."""

    levels = (1 << bits) - 1
    table_size = 1 << (fan_in * bits)
    addresses = torch.arange(table_size, dtype=torch.long)
    decoded_children = []
    mask = levels
    for child in range(fan_in):
        code = (addresses >> (child * bits)) & mask
        decoded_children.append(code.to(torch.float32) / levels * (2.0 * clip) - clip)
    target = torch.stack(decoded_children, dim=0).mean(dim=0)
    normalized = (target / float(clip)).clamp(-0.999, 0.999)
    return torch.atanh(normalized)


def _identity_lut_logits(bits: int, clip: float) -> torch.Tensor:
    levels = (1 << bits) - 1
    values = torch.arange(levels + 1, dtype=torch.float32) / levels * (2.0 * clip) - clip
    return torch.atanh((values / float(clip)).clamp(-0.999, 0.999))


class _LogicTreeReduce2x2(nn.Module):
    """Shared 2x2 spatial reduction through a four-input low-bit LUT.

    Each output channel has a separate LUT.  Branch-specific channel rolls are
    fixed wires, not learned multiplexers.  LUT lookup provides gradients to
    visited table entries; a mean surrogate supplies input gradients while
    leaving the hard forward value unchanged.
    """

    def __init__(
        self,
        channels: int,
        bits: int,
        clip: float,
        channel_wiring: str,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.bits = _validate_bits(bits)
        self.clip = float(clip)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if channel_wiring not in {"identity", "fixed_roll"}:
            raise ValueError(f"unknown channel_wiring={channel_wiring!r}")
        self.channel_wiring = channel_wiring

        table_init = _mean_lut_logits(self.bits, self.clip)
        self.lut_logits = nn.Parameter(table_init.unsqueeze(0).repeat(self.channels, 1))

        base = torch.arange(self.channels, dtype=torch.long)
        if channel_wiring == "identity":
            wiring = base.unsqueeze(0).repeat(4, 1)
        else:
            offsets = (0, 1, max(self.channels // 2, 1), -1)
            wiring = torch.stack([(base + offset) % self.channels for offset in offsets], dim=0)
        self.register_buffer("channel_indices", wiring, persistent=True)

    @property
    def table_size(self) -> int:
        return 1 << (4 * self.bits)

    def _selected_children(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.channels:
            raise ValueError(f"expected [B,H,W,{self.channels}], got {tuple(x.shape)}")
        if x.shape[1] % 2 or x.shape[2] % 2:
            raise ValueError(f"2x2 reduction needs even spatial dimensions, got {tuple(x.shape[1:3])}")
        children = (
            x[:, 0::2, 0::2, :],
            x[:, 0::2, 1::2, :],
            x[:, 1::2, 0::2, :],
            x[:, 1::2, 1::2, :],
        )
        selected = [child.index_select(-1, self.channel_indices[i]) for i, child in enumerate(children)]
        return torch.stack(selected, dim=-2)  # [B,H/2,W/2,4,C]

    def forward(self, x: torch.Tensor, *, ste: bool) -> torch.Tensor:
        children = self._selected_children(x)
        quantized, codes = _comparison_quantize(children, bits=self.bits, clip=self.clip, ste=ste)
        address = torch.zeros_like(codes[..., 0, :])
        for branch in range(4):
            address = address | (codes[..., branch, :] << (branch * self.bits))

        table_values, _ = _compile_lut(self.lut_logits, bits=self.bits, clip=self.clip, ste=ste)
        table_values = table_values.to(dtype=x.dtype)
        flat_address = address.reshape(-1, self.channels).transpose(0, 1)
        hard = table_values.gather(1, flat_address).transpose(0, 1).reshape_as(address)
        if not ste:
            return hard
        proxy = quantized.mean(dim=-2)
        return hard + (proxy - proxy.detach())

    @torch.no_grad()
    def export_codes(self) -> torch.Tensor:
        return _compile_lut(self.lut_logits, bits=self.bits, clip=self.clip, ste=False)[1].clone()


class _ScalarChannelLUT(nn.Module):
    """Independent per-channel scalar LUT used at residual injection points."""

    def __init__(self, channels: int, bits: int, clip: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.bits = _validate_bits(bits)
        self.clip = float(clip)
        init = _identity_lut_logits(self.bits, self.clip)
        self.lut_logits = nn.Parameter(init.unsqueeze(0).repeat(self.channels, 1))

    def forward(self, x: torch.Tensor, *, ste: bool) -> torch.Tensor:
        if x.shape[-1] != self.channels:
            raise ValueError(f"expected last dimension {self.channels}, got {tuple(x.shape)}")
        quantized, codes = _comparison_quantize(x, bits=self.bits, clip=self.clip, ste=ste)
        table_values, _ = _compile_lut(self.lut_logits, bits=self.bits, clip=self.clip, ste=ste)
        table_values = table_values.to(dtype=x.dtype)
        flat_codes = codes.reshape(-1, self.channels).transpose(0, 1)
        hard = table_values.gather(1, flat_codes).transpose(0, 1).reshape_as(x)
        if not ste:
            return hard
        return hard + (quantized - quantized.detach())

    @torch.no_grad()
    def export_codes(self) -> torch.Tensor:
        return _compile_lut(self.lut_logits, bits=self.bits, clip=self.clip, ste=False)[1].clone()


class ConvLogicInspiredPyramidMixer(nn.Module):
    """Low-bit LUT spatial hierarchy with a ViT-compatible residual interface.

    Args:
        channels: Token embedding width.
        grid_size: Patch-grid side.  ``8`` gives 8 -> 4 -> 2 -> 1.
        bits: Unsigned code width for signed uniformly decoded states.
        clip: Symmetric activation/LUT range.
        injection_mode: ``"cls_only"`` or ``"topdown_broadcast"``.
        output_mode: ``"delta"`` for ``x + mixer(x)`` integration, or
            ``"residual"`` to return the already-added ``x + delta``.
        channel_wiring: ``"fixed_roll"`` adds zero-cost static cross-channel
            wiring; ``"identity"`` keeps channels independent.
    """

    def __init__(
        self,
        channels: int,
        grid_size: int = 8,
        bits: int = 3,
        clip: float = 1.0,
        injection_mode: str = "cls_only",
        output_mode: str = "delta",
        channel_wiring: str = "fixed_roll",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        self.bits = _validate_bits(bits)
        self.clip = float(clip)
        if self.grid_size < 2 or self.grid_size & (self.grid_size - 1):
            raise ValueError(f"grid_size must be a power of two >= 2, got {grid_size}")
        if injection_mode not in {"cls_only", "topdown_broadcast"}:
            raise ValueError(f"unknown injection_mode={injection_mode!r}")
        if output_mode not in {"delta", "residual"}:
            raise ValueError(f"unknown output_mode={output_mode!r}")
        self.injection_mode = injection_mode
        self.output_mode = output_mode
        self.channel_wiring = channel_wiring
        self.depth = int(math.log2(self.grid_size))

        self.reducers = nn.ModuleList(
            [
                _LogicTreeReduce2x2(
                    channels=self.channels,
                    bits=self.bits,
                    clip=self.clip,
                    channel_wiring=channel_wiring,
                )
                for _ in range(self.depth)
            ]
        )
        self.cls_injection = _ScalarChannelLUT(self.channels, self.bits, self.clip)
        self.patch_injection = (
            _ScalarChannelLUT(self.channels, self.bits, self.clip)
            if self.injection_mode == "topdown_broadcast"
            else None
        )

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    def _validate_input(self, x: torch.Tensor) -> None:
        expected = (self.num_patches + 1, self.channels)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"expected floating tokens, got dtype={x.dtype}")

    def _build_pyramid(self, patches: torch.Tensor, *, ste: bool) -> List[torch.Tensor]:
        current = patches.reshape(-1, self.grid_size, self.grid_size, self.channels)
        levels: List[torch.Tensor] = []
        for reducer in self.reducers:
            current = reducer(current, ste=ste)
            levels.append(current)
        return levels

    def _delta(self, x: torch.Tensor, *, ste: bool) -> torch.Tensor:
        self._validate_input(x)
        batch = x.shape[0]
        levels = self._build_pyramid(x[:, 1:], ste=ste)
        root = levels[-1].reshape(batch, self.channels)
        cls_delta = self.cls_injection(root, ste=ste)

        if self.injection_mode == "cls_only":
            patch_delta = x.new_zeros(batch, self.num_patches, self.channels)
        else:
            broadcasts = []
            for level in levels:
                side = level.shape[1]
                repeat = self.grid_size // side
                broadcasts.append(level.repeat_interleave(repeat, dim=1).repeat_interleave(repeat, dim=2))
            # Three terms for an 8x8 grid are divided by four: an arithmetic
            # right shift with one guard slot, avoiding a general divider.
            shift = int(math.ceil(math.log2(len(broadcasts))))
            context = torch.stack(broadcasts, dim=0).sum(dim=0) / float(1 << shift)
            patch_delta = self.patch_injection(context, ste=ste).reshape(
                batch, self.num_patches, self.channels
            )

        return torch.cat((cls_delta.unsqueeze(1), patch_delta), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training and inference have identical forward values.  Training adds
        # only surrogate gradients for hard addresses and quantizers.
        delta = self._delta(x, ste=self.training)
        return delta if self.output_mode == "delta" else x + delta

    @torch.no_grad()
    def forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        """Deployment-value reference without any surrogate-gradient path."""

        delta = self._delta(x, ste=False)
        return delta if self.output_mode == "delta" else x + delta

    @torch.no_grad()
    def export_integer_tables(self) -> Dict[str, torch.Tensor]:
        """Return the exact low-bit LUT payloads required at runtime."""

        tables: Dict[str, torch.Tensor] = {
            f"reduce_{index}": reducer.export_codes()
            for index, reducer in enumerate(self.reducers)
        }
        tables["cls_injection"] = self.cls_injection.export_codes()
        if self.patch_injection is not None:
            tables["patch_injection"] = self.patch_injection.export_codes()
        return tables

    def structural_dependency_masks(self) -> List[torch.Tensor]:
        """Return boolean patch-dependency masks for every pyramid level.

        The final tensor has shape ``[1, 1, H*W]`` and is all true, proving
        structural global coverage independently of current LUT contents.
        """

        dependencies = torch.eye(self.num_patches, dtype=torch.bool).reshape(
            self.grid_size, self.grid_size, self.num_patches
        )
        levels = []
        for _ in range(self.depth):
            dependencies = (
                dependencies[0::2, 0::2]
                | dependencies[0::2, 1::2]
                | dependencies[1::2, 0::2]
                | dependencies[1::2, 1::2]
            )
            levels.append(dependencies.clone())
        return levels

    def hardware_cost(self) -> Dict[str, Any]:
        """Return implementation-independent storage and operation counts.

        Counts cover this mixer only.  They exclude patch embedding, upstream
        token storage, physical SRAM/register overhead, routing/fanout buffers,
        clocking, and the rest of the ViT block.  LUTs are shared spatially per
        level and channel.
        """

        sides = [self.grid_size >> (level + 1) for level in range(self.depth)]
        cells = sum(side * side for side in sides)
        bottom_up_evaluations = cells * self.channels
        table_entries = 1 << (4 * self.bits)
        scalar_tables = 1 + int(self.patch_injection is not None)
        topdown_additions = 0
        topdown_shifts = 0
        if self.injection_mode == "topdown_broadcast":
            topdown_additions = (self.depth - 1) * self.num_patches * self.channels
            topdown_shifts = self.num_patches * self.channels
        scalar_injection_values = self.channels * (
            1 + int(self.patch_injection is not None) * self.num_patches
        )
        return {
            "grid_size": self.grid_size,
            "pyramid_sides": sides,
            "logic_tree_depth": self.depth,
            "channels": self.channels,
            "state_bits": self.bits,
            "four_input_lut_address_bits": 4 * self.bits,
            "entries_per_bottom_up_lut": table_entries,
            "shared_bottom_up_lut_count": self.depth * self.channels,
            "bottom_up_lut_table_bits": self.depth
            * self.channels
            * table_entries
            * self.bits,
            "bottom_up_lut_evaluations_per_sample": bottom_up_evaluations,
            "bottom_up_input_comparisons_per_sample": bottom_up_evaluations
            * 4
            * ((1 << self.bits) - 1),
            "materialized_pyramid_state_bits": cells * self.channels * self.bits,
            "scalar_injection_lut_table_bits": scalar_tables
            * self.channels
            * (1 << self.bits)
            * self.bits,
            "scalar_injection_lut_reads_per_sample": scalar_injection_values,
            "scalar_injection_input_comparisons_per_sample": scalar_injection_values
            * ((1 << self.bits) - 1),
            "topdown_low_bit_additions_per_sample": topdown_additions,
            "topdown_power_of_two_shifts_per_sample": topdown_shifts,
            "runtime_primitives": [
                "threshold_compare",
                "integer_shift_or",
                "lut_read",
                "fixed_channel_wiring",
                "low_bit_add",
                "power_of_two_shift",
            ],
            "scope_exclusions": [
                "patch_embedding and the rest of the ViT block",
                "SRAM/register periphery and physical routing",
                "training-only STE and LUT-parameter compilation",
            ],
        }
