"""Export a training checkpoint as an integer-only logic inference payload.

The modules in :mod:`vit_lgn.full_discrete` deliberately keep floating shadow
parameters so that they can be optimized by PyTorch.  Those shadows, optimizer
slots and RNG state are *not* an inference format.  This module is the boundary
between the two worlds: it hardens all learned choices and emits only integer
codes plus structural metadata.

By default a layer keeps its trained magnitude precision.  Thus Wmag7 remains one
sign plus seven magnitude bits and is emitted as low-4/high-3 unsigned chunks
with shifts ``[0,4]``.  A caller may explicitly request
``requantize_magnitude_bits=4``; only then are weights requantized directly
from their learned shadow to Wmag4, with the Wmag7 source precision recorded.  This
distinction prevents a deployment export from silently changing accuracy.
Per-output power-of-two scales are stored as signed integer exponents, never
as floating tensors.

This is an inference payload, not a bit-packed file format.  Tensor packing and
an RTL/C++ executor can be layered on top without reopening a checkpoint.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .enhanced_model import EnhancedFullDiscreteViT, StateSelectedFFNAdapter
from .enhancements_expert import (
    BitSliceBooleanLogicExpert,
    FourModeStateSelectedFFN,
    ParallelBitSliceLogicFFN,
)
from .enhancements_hadamard import FixedHadamardGlobalMixer
from .enhancements_lut import GroupwiseDiscreteActivationLUT
from .enhancements_logic_tree import SITE_OFFSETS, SharedLogicTreeConv3x3
from .enhancements_spatial import DiscreteDepthwiseLocalBranch
from .model import (
    DiscreteRMSNorm,
    FullDiscreteViT,
    HardXNORScoreGapAttention,
    IdentityDiscreteNorm,
    QuantizeOnlyNorm,
    ShiftRMSNorm,
)
from .shiftadd import ShiftAddLinear, _power_of_two_scale


SCHEMA_NAME = "learnable-logic-full-discrete-inference"
SCHEMA_VERSION = 2
SCHEMA_TOP_LEVEL_KEYS = (
    "schema",
    "source",
    "topology",
    "parameters",
    "shift_add_layers",
    "attention",
    "global_mixers",
    "rms_norms",
    "rms_luts",
    "activation_luts",
    "local_branches",
    "logic_experts",
    "logic_ffns",
    "state_ffns",
    "arithmetic_contract",
)

_FORBIDDEN_KEY_PARTS = (
    "optimizer",
    "scheduler",
    "sampler",
    "python_rng",
    "torch_rng",
    "cuda_rng",
    "latent",
    "shadow",
)


def _integer_dtype_for_bits(magnitude_bits: int) -> torch.dtype:
    return torch.int8 if magnitude_bits <= 4 else torch.int16


def _scale_to_exponent(scale: torch.Tensor) -> torch.Tensor:
    """Encode an exactly power-of-two floating scale as a signed exponent."""

    scale = scale.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("scale must contain finite positive powers of two")
    mantissa, exponent = torch.frexp(scale)
    if not bool((mantissa == 0.5).all()):
        raise ValueError("scale is not an exact power of two")
    return (exponent - 1).to(torch.int16)


@torch.no_grad()
def _harden_shiftadd(
    name: str,
    layer: ShiftAddLinear,
    requantize_magnitude_bits: int | None,
) -> dict[str, object]:
    if layer.bias is not None:
        raise ValueError(
            f"{name}: bias export is intentionally unsupported; fold it into an "
            "integer offset before producing a logic payload"
        )
    target_magnitude_bits = (
        layer.magnitude_bits
        if requantize_magnitude_bits is None
        else requantize_magnitude_bits
    )
    qmax = (1 << target_magnitude_bits) - 1
    maximum = layer.weight.detach().abs().amax(dim=1, keepdim=True)
    scale = _power_of_two_scale(maximum / qmax)
    code = torch.round(layer.weight.detach() / scale).clamp(-qmax, qmax)
    code = code.to(device="cpu", dtype=_integer_dtype_for_bits(target_magnitude_bits))
    sign = code < 0
    magnitude = code.abs().to(torch.int16)
    planes = torch.stack(
        [
            torch.bitwise_and(torch.bitwise_right_shift(magnitude, bit), 1)
            for bit in range(target_magnitude_bits)
        ],
        dim=-1,
    ).to(torch.uint8)
    chunk_shifts = list(range(0, target_magnitude_bits, 4))
    chunks = torch.stack(
        [
            torch.bitwise_and(
                torch.bitwise_right_shift(magnitude, shift), 0xF
            )
            for shift in chunk_shifts
        ],
        dim=-1,
    ).to(torch.uint8)
    return {
        "name": name,
        "operator": "signed_shift_add_linear",
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "weight_code": code,
        "weight_sign": sign.to(torch.uint8),
        "weight_magnitude_planes_lsb_first": planes,
        "weight_magnitude_chunks_u4_lsb_first": chunks,
        "magnitude_chunk_shift": torch.tensor(chunk_shifts, dtype=torch.uint8),
        "magnitude_chunk_valid_bits": torch.tensor(
            [min(4, target_magnitude_bits - shift) for shift in chunk_shifts],
            dtype=torch.uint8,
        ),
        "weight_scale_exponent": _scale_to_exponent(scale),
        "target_magnitude_bits": target_magnitude_bits,
        "target_signed_code_min": -qmax,
        "target_signed_code_max": qmax,
        "source_magnitude_bits": layer.magnitude_bits,
        "requantized_from_bits": (
            layer.magnitude_bits
            if requantize_magnitude_bits is not None
            and requantize_magnitude_bits != layer.magnitude_bits
            else 0
        ),
        "input_activation_bits": layer.input_quantizer.bits,
        "output_activation_bits": layer.output_quantizer.bits,
        "rounding": "nearest ties-to-even",
        "scale_floor_exponent": -24,
        "bias": "none",
        "magnitude_product": (
            "signed_A_code_by_U4_magnitude_ROM_per_chunk_then_chunk_shift_add"
        ),
    }


@torch.no_grad()
def _harden_parameter(
    name: str, parameter: torch.Tensor, quantizer: nn.Module
) -> dict[str, object]:
    code, scale = quantizer.integer_code_and_scale(parameter.detach())
    return {
        "name": name,
        "code": code.to(device="cpu", dtype=torch.int16),
        "scale_exponent": _scale_to_exponent(scale),
        "activation_bits": int(quantizer.bits),
        "signed_code_min": int(quantizer.qmin),
        "signed_code_max": int(quantizer.qmax),
        "rounding": "nearest ties-to-even",
        "scale_floor_exponent": -24,
    }


def _binary_rational(value: float) -> tuple[int, int]:
    """Return a reduced ``numerator / 2**shift`` for a finite binary float."""

    if not math.isfinite(value):
        raise ValueError("threshold fraction must be finite")
    numerator, denominator = float(value).as_integer_ratio()
    if denominator < 1 or denominator & (denominator - 1):
        raise ValueError("binary floating threshold has a non-power-of-two denominator")
    shift = denominator.bit_length() - 1
    while shift and numerator % 2 == 0:
        numerator //= 2
        shift -= 1
    return numerator, shift


@torch.no_grad()
def _attention_payload(
    name: str, module: HardXNORScoreGapAttention
) -> dict[str, object]:
    rationals = [_binary_rational(float(value)) for value in module.threshold_fractions.cpu()]
    numerator = torch.tensor([item[0] for item in rationals], dtype=torch.int64)
    shift = torch.tensor([item[1] for item in rationals], dtype=torch.uint8)
    if module.gap_lut is None:
        gap: dict[str, object] = {
            "kind": "fixed_bucket_shift",
            "gap_right_shift": module.gap_shift,
            "max_gap_bucket": module.max_gap_bucket,
            "weight_rule": "1 << (max_gap_bucket - clipped_bucket)",
            "possible_weights": torch.tensor(
                [1 << bit for bit in range(module.max_gap_bucket + 1)],
                dtype=torch.uint8,
            ),
        }
    else:
        table = module.gap_lut.integer_table().detach().cpu().to(torch.uint8)
        shift_code = torch.full(table.shape, -1, dtype=torch.int8)
        nonzero = table > 0
        # The allowed learned values are powers of two.  This loop is export
        # time only; the deployed datapath reads the already hardened shifts.
        for amount in range(8):
            shift_code[table == (1 << amount)] = amount
        gap = {
            "kind": "per_head_monotone_lut",
            "weight_table": table,
            "shift_code": shift_code,
            "zero_shift_code": -1,
            "max_gap": int(module.gap_lut.max_gap),
        }
    return {
        "name": name,
        "operator": "xnor_popcount_hard_topk",
        "heads": module.heads,
        "head_dim": module.head_dim,
        "topk": module.topk,
        "topk_tie_rule": "score_descending_then_key_index_ascending",
        "qk_lanes": module.qk_lanes,
        "popcount_width": module.head_dim * module.qk_lanes,
        "threshold_fraction_numerator": numerator,
        "threshold_fraction_denominator_shift": shift,
        "threshold_compare": (
            "(x_code << denominator_shift) >= max_abs_code * numerator"
        ),
        "threshold_constant_product": "signed constant shift-add",
        "qkv_projection": f"{name}.qkv",
        "output_projection": f"{name}.proj",
        "value_aggregation": {
            "selected_rows": module.topk,
            "numerator": "integer weighted sum",
            "denominator": "positive integer sum of selected weights",
            "division_rounding": "nearest, magnitude floor after adding denominator//2",
        },
        "gap": gap,
    }


@torch.no_grad()
def _hadamard_payload(
    name: str, module: FixedHadamardGlobalMixer
) -> dict[str, object]:
    return {
        "name": name,
        "operator": "fixed_hadamard_sign_hadamard_global_mixer",
        "patch_tokens": module.patch_tokens,
        "block_index": module.block_index,
        "activation_bits": module.activation_bits,
        "group_size": module.group_size,
        "runtime_scale_groups": module.groups,
        "normalization_right_shift": module.normalization_shift,
        "branch_right_shift": module.branch_shift,
        "rounding": "nearest_half_way_magnitude_away_from_zero",
        "sign_mask_int8": module.sign_mask.detach().cpu().to(torch.int8),
        "butterfly_stages_per_transform": module.normalization_shift,
        "transform_count": 2,
        "patch_add_sub_per_channel": (
            2 * module.patch_tokens * module.normalization_shift
        ),
        "cls_path": "rounded_patch_mean_then_broadcast",
        "learned_parameters": 0,
        "general_multipliers": 0,
    }


def _rounded_q15_reciprocal_sqrt(address: int) -> int:
    """Exact ties-to-even round of ``2**15 / sqrt(max(address,1))``."""

    address = max(int(address), 1)
    scale = 1 << 15
    lower = math.isqrt((scale * scale) // address)
    left = 4 * scale * scale
    right = address * (2 * lower + 1) ** 2
    if left > right or (left == right and lower & 1):
        lower += 1
    return min(lower, (1 << 16) - 1)


def _rms_lut(bits: int) -> dict[str, object]:
    qmax = (1 << (bits - 1)) - 1
    maximum_address = qmax * qmax
    # A8 is 16,130 entries.  Very wide activation codes should use a segmented
    # LUT design rather than accidentally materializing a multi-gigabyte ROM.
    if maximum_address > 1_048_576:
        raise ValueError(
            f"A{bits} RMS LUT would require {maximum_address + 1} entries; "
            "export a segmented LUT before using activation_bits > 11"
        )
    table = torch.tensor(
        [_rounded_q15_reciprocal_sqrt(address) for address in range(maximum_address + 1)],
        dtype=torch.int32,
    )
    return {
        "name": f"rms_reciprocal_sqrt_q15_a{bits}",
        "address_min": 1,
        "address_max": maximum_address,
        "address_zero_maps_to": 1,
        "entry": "unsigned_q0_15",
        "entry_bits": 16,
        "table_uint16_carried_as_int32": table,
        "generation": "exact integer ties-to-even round(2**15/sqrt(address))",
    }


def _model_topology(model: FullDiscreteViT) -> dict[str, object]:
    first_mixer = model.blocks[0].attn if model.blocks else None
    first_attention = (
        first_mixer if isinstance(first_mixer, HardXNORScoreGapAttention) else None
    )
    patch_features = model.patch_embed.projection.in_features
    patch_area = model.patch_embed.patch_size * model.patch_embed.patch_size
    return {
        "model_type": type(model).__name__,
        "patch_size": model.patch_embed.patch_size,
        "input_channels": patch_features // patch_area,
        "patch_tokens": model.patch_embed.num_patches,
        "sequence_tokens": int(model.position.shape[1]),
        "embedding_dim": int(model.position.shape[2]),
        "depth": len(model.blocks),
        "heads": int(first_attention.heads) if first_attention is not None else 0,
        "topk": int(first_attention.topk) if first_attention is not None else 0,
        "qk_lanes": int(first_attention.qk_lanes) if first_attention is not None else 0,
        "classes": model.head.out_features,
        "activation_bits": model.activation_bits,
        "norm_kind": model.norm_kind,
        "final_norm_kind": model.final_norm_kind,
        "block_order": [f"blocks.{index}" for index in range(len(model.blocks))],
        "classifier": "head",
    }


@torch.no_grad()
def _local_branch_payload(
    name: str,
    module: DiscreteDepthwiseLocalBranch,
    requantize_magnitude_bits: int | None,
) -> dict[str, object]:
    target_bits = (
        module.weight_bits
        if requantize_magnitude_bits is None
        else min(requantize_magnitude_bits, 4)
    )
    qmax = (1 << target_bits) - 1
    maximum = module.kernel.detach().abs().amax(dim=(1, 2), keepdim=True)
    scale = _power_of_two_scale(maximum / qmax)
    code = torch.round(module.kernel.detach() / scale).clamp(-qmax, qmax)
    effective_exponent = _scale_to_exponent(scale) - module.branch_shift
    code = code.cpu().to(_integer_dtype_for_bits(target_bits))
    magnitude = code.abs().to(torch.int16)
    return {
        "name": name,
        "operator": "zero_padded_depthwise_3x3_shift_add",
        "grid_height": module.grid_size[0],
        "grid_width": module.grid_size[1],
        "channels": module.dim,
        "kernel_code": code,
        "kernel_sign": (code < 0).to(torch.uint8),
        "kernel_magnitude_planes_lsb_first": torch.stack(
            [
                torch.bitwise_and(torch.bitwise_right_shift(magnitude, bit), 1)
                for bit in range(target_bits)
            ],
            dim=-1,
        ).to(torch.uint8),
        "kernel_effective_scale_exponent": effective_exponent.to(torch.int16),
        "target_magnitude_bits": target_bits,
        "source_magnitude_bits": module.weight_bits,
        "requantized_from_bits": (
            module.weight_bits
            if requantize_magnitude_bits is not None
            and target_bits != module.weight_bits
            else 0
        ),
        "branch_right_shift": module.branch_shift,
        "activation_bits": module.activation_bits,
        "cls_path": "bypass_then_requantize",
    }


@torch.no_grad()
def _logic_tree_local_payload(
    name: str, module: SharedLogicTreeConv3x3
) -> dict[str, object]:
    truth_entries = module.hard_truth_table_bits().cpu().to(torch.uint8)
    truth_nibbles = module.hard_truth_nibbles().cpu().to(torch.uint8)
    leaf_site = module.leaf_site.detach().cpu().to(torch.int8)
    site_offsets = module.site_offsets.detach().cpu().to(torch.int8)
    gate_children = torch.tensor(
        [
            [0, 1], [2, 3], [4, 5], [6, 7],
            [8, 9], [10, 11], [12, 13],
        ],
        dtype=torch.int8,
    )
    return {
        "name": name,
        "operator": "shared_logic_tree3x3_bitplane",
        "grid_height": module.grid_size,
        "grid_width": module.grid_size,
        "channels": module.dim,
        "activation_bits": module.activation_bits,
        "input_encoding": "signed_magnitude_sign_then_lsb_planes_per_channel",
        "tree_depth": module.TREE_DEPTH,
        "leaf_count": module.NUM_LEAVES,
        "gate_count": module.NUM_GATES,
        "truth_address": "(A<<1)|B_little_address_endian",
        "truth_table_00_01_10_11": truth_entries,
        "truth_nibble_lsb_address": truth_nibbles,
        "site_offsets_dy_dx": site_offsets,
        "leaf_site": leaf_site,
        "leaf_offsets_dy_dx": site_offsets[leaf_site.to(torch.long)],
        "gate_children_node_index": gate_children,
        "root_gate_node_index": 14,
        "padding": "constant_logic_0",
        "spatial_sharing": True,
        "learned_connections": False,
        "cls_path": "exact_bypass",
        "root_state": "boolean_bitplanes",
        "output_projection": "none",
        "output_scale": "reuse_per_token_input_power_of_two_scale",
    }


@torch.no_grad()
def _logic_expert_payload(name: str, module: BitSliceBooleanLogicExpert) -> dict[str, object]:
    return {
        "name": name,
        "operator": "two_input_boolean_lut_bank",
        "input_dim": module.dim,
        "activation_bits": module.activation_bits,
        "input_encoding": "signed_magnitude_sign_then_lsb_planes_per_channel",
        "source_count": module.source_count,
        "gate_count": module.expert_width,
        "left_source": module.left_connection_logits.argmax(dim=-1).cpu().to(torch.int32),
        "right_source": module.right_connection_logits.argmax(dim=-1).cpu().to(torch.int32),
        "truth_table_00_01_10_11": module.truth_table_logits.ge(0).cpu().to(torch.uint8),
        "output_projection": f"{name}.output_projection",
    }


def _logic_ffn_payload(name: str, module: ParallelBitSliceLogicFFN) -> dict[str, object]:
    return {
        "name": name,
        "operator": "parallel_shift_add_and_boolean_ffn",
        "base_gate": f"{name}.base.gate",
        "base_up": f"{name}.base.up",
        "base_down": f"{name}.base.down",
        "logic_experts": [f"{name}.logic_experts.{index}" for index in range(len(module.logic_experts))],
        "logic_output_right_shift": module.logic_output_shift,
        "merge": "align_exponents_integer_add_then_activation_requantize",
    }


def _state_ffn_payload(name: str, adapter: StateSelectedFFNAdapter) -> dict[str, object]:
    module: FourModeStateSelectedFFN = adapter.module
    modes = [
        [f"{name}.module.mode_experts.{mode}.{index}" for index in range(len(bank))]
        for mode, bank in enumerate(module.mode_experts)
    ]
    return {
        "name": name,
        "operator": "two_bit_state_selected_boolean_ffn",
        "selected_control": adapter.control,
        "block_index": adapter.block_index,
        "state_bits": 2,
        "state_count": 4,
        "state_controller": f"{name}.module.state_controller",
        "static_state_code": module.static_state_code.cpu().to(torch.uint8),
        "script_codes": module.script_codes.cpu().to(torch.uint8),
        "mode_logic_experts": modes,
        "base_gate": f"{name}.module.base.gate",
        "base_up": f"{name}.module.base.up",
        "base_down": f"{name}.module.base.down",
        "logic_output_right_shift": module.logic_output_shift,
        "mode_mux": "hard_2bit_4to1",
    }


def _activation_lut_payload(
    name: str, module: GroupwiseDiscreteActivationLUT
) -> dict[str, object]:
    return {
        "name": name,
        "operator": "groupwise_256_entry_activation_rom",
        "table_int8": module.integer_table().detach().cpu(),
        "input_zero_point_uint8": module.zero_point,
        "features": module.features,
        "groups": module.groups,
        "group_size": module.group_size,
        "minimum_scale_exponent": module.minimum_scale_exponent,
        "output_scale": "reuse_input_power_of_two_exponent",
    }


def _a8_u4_product_rom() -> dict[str, object]:
    """Canonical small ROM replacing one signed-A8 by U4 multiplication."""

    raw_address = torch.arange(256, dtype=torch.int16)
    signed_code = torch.where(
        raw_address < 128, raw_address, raw_address - 256
    ).unsqueeze(1)
    nibble = torch.arange(16, dtype=torch.int16).unsqueeze(0)
    return {
        "address": "concat(uint8_twos_complement_A8_code, U4_magnitude_nibble)",
        "address_bits": 12,
        "activation_code_min": -128,
        "activation_code_max": 127,
        "address_row_zero": "A8_code_zero",
        "address_row_127": "A8_code_127",
        "address_row_128": "A8_code_minus_128",
        "address_row_255": "A8_code_minus_1",
        "magnitude_nibble_min": 0,
        "magnitude_nibble_max": 15,
        "entry": "signed_int16_exact_product",
        "product_rom_int16": signed_code * nibble,
        "sign_application": "conditional_twos_complement_after_unsigned_magnitude_lookup",
    }


def _source_metadata(
    model: FullDiscreteViT, checkpoint_metadata: Mapping[str, Any] | None
) -> dict[str, object]:
    checkpoint_metadata = checkpoint_metadata or {}
    step = checkpoint_metadata.get("step")
    protocol = checkpoint_metadata.get("protocol_sha256")
    return {
        "model_type": type(model).__name__,
        "checkpoint_step": int(step) if step is not None else -1,
        "protocol_sha256": str(protocol) if protocol is not None else "unknown",
        "training_state_included": False,
    }


@torch.no_grad()
def export_logic_payload(
    model: FullDiscreteViT,
    *,
    requantize_magnitude_bits: int | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Harden ``model`` into a stable, integer-only inference dictionary."""

    if not isinstance(model, FullDiscreteViT):
        raise TypeError("model must be FullDiscreteViT or EnhancedFullDiscreteViT")
    if model.activation_bits != 8:
        raise ValueError(
            "schema v2 defines an A8-by-U4 product ROM; activation_bits must be 8"
        )
    if requantize_magnitude_bits is not None and not (
        1 <= requantize_magnitude_bits <= 8
    ):
        raise ValueError("requantize_magnitude_bits must be None or in [1,8]")

    named_modules = list(model.named_modules())
    shift_layers = [
        _harden_shiftadd(name, module, requantize_magnitude_bits)
        for name, module in named_modules
        if isinstance(module, ShiftAddLinear)
    ]
    attentions = [
        _attention_payload(name, module)
        for name, module in named_modules
        if isinstance(module, HardXNORScoreGapAttention)
    ]
    global_mixers = [
        _hadamard_payload(name, module)
        for name, module in named_modules
        if isinstance(module, FixedHadamardGlobalMixer)
    ]
    norms: list[dict[str, object]] = []
    for name, module in named_modules:
        if isinstance(module, DiscreteRMSNorm):
            norms.append({
                "name": name,
                "operator": "integer_rmsnorm_q15_lut",
                "dimension": module.dim,
                "input_activation_bits": module.input_quantizer.bits,
                "output_activation_bits": module.output_quantizer.bits,
                "mean_square": "(sum_square + dimension//2) // dimension, clamp_min_1",
                "reciprocal_lut": (
                    f"rms_reciprocal_sqrt_q15_a{module.input_quantizer.bits}"
                ),
                "product_integer_bits": 23,
                "product_exponent_delta": -15,
                "pre_requant_right_shift": 0,
                "requantization": (
                    "consume the full signed code_times_q15 product and its "
                    "-15 exponent; do not truncate fractional levels first"
                ),
            })
        elif isinstance(module, ShiftRMSNorm):
            norms.append({
                "name": name,
                "operator": "integer_shift_rms",
                "dimension": module.dim,
                "input_activation_bits": module.input_quantizer.bits,
                "output_activation_bits": module.output_quantizer.bits,
                "statistic": "exact integer sum_square",
                "rms_shift": (
                    "count candidates where sum_square >= "
                    "dimension * 2**(2*candidate-1)"
                ),
                "tie_rule": "half-way logarithmic cases round upward",
                "zero_and_subunit_policy": "rms_shift_saturates_at_zero",
                "runtime": "constant_shift_compare_then_output_exponent_replace",
                "output_exponent": (
                    "-rms_shift; input exponent cancels under RMS normalization"
                ),
            })
        elif isinstance(module, QuantizeOnlyNorm):
            norms.append({
                "name": name,
                "operator": "activation_requant_only",
                "dimension": module.dim,
                "output_activation_bits": module.output_quantizer.bits,
            })
        elif isinstance(module, IdentityDiscreteNorm):
            norms.append({
                "name": name,
                "operator": "identity_wire",
                "dimension": module.dim,
                "activation_bits": module.bits,
            })
    rms_bits = sorted({
        item["input_activation_bits"]
        for item in norms
        if item["operator"] == "integer_rmsnorm_q15_lut"
    })
    rms_luts = [_rms_lut(int(bits)) for bits in rms_bits]
    activation_luts = [
        _activation_lut_payload(name, module)
        for name, module in named_modules
        if isinstance(module, GroupwiseDiscreteActivationLUT)
    ]
    local_branches = []
    for name, module in named_modules:
        if isinstance(module, DiscreteDepthwiseLocalBranch):
            local_branches.append(
                _local_branch_payload(name, module, requantize_magnitude_bits)
            )
        elif isinstance(module, SharedLogicTreeConv3x3):
            local_branches.append(_logic_tree_local_payload(name, module))
    logic_experts = [
        _logic_expert_payload(name, module)
        for name, module in named_modules
        if isinstance(module, BitSliceBooleanLogicExpert)
    ]
    logic_ffns = [
        _logic_ffn_payload(name, module)
        for name, module in named_modules
        if isinstance(module, ParallelBitSliceLogicFFN)
    ]
    state_ffns = [
        _state_ffn_payload(name, module)
        for name, module in named_modules
        if isinstance(module, StateSelectedFFNAdapter)
    ]

    payload: dict[str, object] = {
        "schema": {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "encoding": "torch_save_integer_tensors_and_structural_metadata",
        },
        "source": _source_metadata(model, checkpoint_metadata),
        "topology": _model_topology(model),
        "parameters": {
            "cls_token": _harden_parameter(
                "cls_token", model.cls_token, model.parameter_quantizer
            ),
            "position": _harden_parameter(
                "position", model.position, model.parameter_quantizer
            ),
        },
        "shift_add_layers": shift_layers,
        "attention": attentions,
        "global_mixers": global_mixers,
        "rms_norms": norms,
        "rms_luts": rms_luts,
        "activation_luts": activation_luts,
        "local_branches": local_branches,
        "logic_experts": logic_experts,
        "logic_ffns": logic_ffns,
        "state_ffns": state_ffns,
        "arithmetic_contract": {
            "learned_matrix_runtime": "sign_xor_bitplane_mux_shift_integer_accumulate",
            "a8_by_u4_product_rom": _a8_u4_product_rom(),
            "power_of_two_scale_runtime": "signed_exponent_alignment_only",
            "activation_requantization": "leading_one_exponent_and_nearest_ties_to_even",
            "normalization_operators": sorted({item["operator"] for item in norms}),
            "global_mixer_operators": sorted({
                item["operator"] for item in global_mixers
            }),
            "rms_reciprocal_sqrt_runtime": "integer_rom_lookup_if_present",
            "general_learned_multipliers": 0,
            "floating_inference_state": False,
        },
    }
    if tuple(payload) != SCHEMA_TOP_LEVEL_KEYS:
        raise AssertionError("internal schema key order changed without a version bump")
    validate_logic_payload(payload)
    return payload


def validate_logic_payload(payload: Mapping[str, object]) -> None:
    """Reject training-only keys, floating tensors and floating scalar state."""

    if tuple(payload) != SCHEMA_TOP_LEVEL_KEYS:
        raise ValueError("payload top-level schema does not match version 2")

    def require_keys(value: object, required: tuple[str, ...], path: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        return value

    schema = require_keys(payload["schema"], ("name", "version", "encoding"), "schema")
    if schema["name"] != SCHEMA_NAME or schema["version"] != SCHEMA_VERSION:
        raise ValueError("payload schema identity does not match version 2")
    if schema["encoding"] != "torch_save_integer_tensors_and_structural_metadata":
        raise ValueError("payload schema encoding does not match version 1")
    parameters = require_keys(payload["parameters"], ("cls_token", "position"), "parameters")
    for parameter_name in ("cls_token", "position"):
        require_keys(
            parameters[parameter_name],
            ("code", "scale_exponent", "activation_bits", "signed_code_min", "signed_code_max"),
            f"parameters.{parameter_name}",
        )
    if not isinstance(payload["shift_add_layers"], list):
        raise ValueError("shift_add_layers must be a list")
    for index, layer in enumerate(payload["shift_add_layers"]):
        item = require_keys(
            layer,
            (
                "name", "in_features", "out_features", "weight_code", "weight_sign",
                "weight_magnitude_planes_lsb_first", "weight_magnitude_chunks_u4_lsb_first",
                "magnitude_chunk_shift", "weight_scale_exponent", "target_magnitude_bits",
                "source_magnitude_bits", "input_activation_bits", "output_activation_bits",
            ),
            f"shift_add_layers[{index}]",
        )
        bits = int(item["target_magnitude_bits"])
        code = item["weight_code"]
        sign = item["weight_sign"]
        planes = item["weight_magnitude_planes_lsb_first"]
        chunks = item["weight_magnitude_chunks_u4_lsb_first"]
        exponent = item["weight_scale_exponent"]
        expected_matrix = (int(item["out_features"]), int(item["in_features"]))
        if not isinstance(code, torch.Tensor) or tuple(code.shape) != expected_matrix:
            raise ValueError(f"shift_add_layers[{index}].weight_code shape mismatch")
        if not isinstance(sign, torch.Tensor) or tuple(sign.shape) != expected_matrix:
            raise ValueError(f"shift_add_layers[{index}].weight_sign shape mismatch")
        if not isinstance(planes, torch.Tensor) or tuple(planes.shape) != (*expected_matrix, bits):
            raise ValueError(f"shift_add_layers[{index}].planes shape mismatch")
        if not isinstance(chunks, torch.Tensor) or tuple(chunks.shape) != (
            *expected_matrix, (bits + 3) // 4
        ):
            raise ValueError(f"shift_add_layers[{index}].chunks shape mismatch")
        if not isinstance(exponent, torch.Tensor) or tuple(exponent.shape) != (
            expected_matrix[0], 1
        ):
            raise ValueError(f"shift_add_layers[{index}].scale exponent shape mismatch")
    if not isinstance(payload["attention"], list):
        raise ValueError("attention must be a list")
    for index, attention in enumerate(payload["attention"]):
        item = require_keys(
            attention,
            ("name", "heads", "head_dim", "topk", "qk_lanes",
             "topk_tie_rule", "threshold_fraction_numerator",
             "threshold_fraction_denominator_shift", "gap"),
            f"attention[{index}]",
        )
        if item["topk_tie_rule"] != "score_descending_then_key_index_ascending":
            raise ValueError(f"attention[{index}] Top-K tie ABI mismatch")
        numerator = item["threshold_fraction_numerator"]
        shift = item["threshold_fraction_denominator_shift"]
        expected = (int(item["qk_lanes"]),)
        if not isinstance(numerator, torch.Tensor) or tuple(numerator.shape) != expected:
            raise ValueError(f"attention[{index}] threshold numerator shape mismatch")
        if not isinstance(shift, torch.Tensor) or tuple(shift.shape) != expected:
            raise ValueError(f"attention[{index}] threshold shift shape mismatch")
    if not isinstance(payload["global_mixers"], list):
        raise ValueError("global_mixers must be a list")
    for index, mixer in enumerate(payload["global_mixers"]):
        item = require_keys(
            mixer,
            (
                "name", "operator", "patch_tokens", "block_index",
                "activation_bits", "group_size", "runtime_scale_groups",
                "normalization_right_shift", "branch_right_shift", "rounding",
                "sign_mask_int8", "butterfly_stages_per_transform",
                "transform_count", "patch_add_sub_per_channel", "cls_path",
                "learned_parameters", "general_multipliers",
            ),
            f"global_mixers[{index}]",
        )
        patch_tokens = int(item["patch_tokens"])
        if (item["operator"] != "fixed_hadamard_sign_hadamard_global_mixer"
                or patch_tokens < 2 or patch_tokens & (patch_tokens - 1)):
            raise ValueError(f"global_mixers[{index}] operator/topology mismatch")
        stages = patch_tokens.bit_length() - 1
        if (
            int(item["normalization_right_shift"]) != stages
            or int(item["butterfly_stages_per_transform"]) != stages
            or int(item["transform_count"]) != 2
            or int(item["patch_add_sub_per_channel"]) != 2 * patch_tokens * stages
            or int(item["learned_parameters"]) != 0
            or int(item["general_multipliers"]) != 0
        ):
            raise ValueError(f"global_mixers[{index}] arithmetic ABI mismatch")
        sign_mask = item["sign_mask_int8"]
        if not isinstance(sign_mask, torch.Tensor) or tuple(sign_mask.shape) != (
            patch_tokens,
        ):
            raise ValueError(f"global_mixers[{index}] sign mask shape mismatch")
        if bool(((sign_mask != -1) & (sign_mask != 1)).any()) or int(sign_mask[0]) != 1:
            raise ValueError(f"global_mixers[{index}] sign mask value mismatch")
    if not isinstance(payload["rms_norms"], list):
        raise ValueError("rms_norms must be a list")
    for index, rms_norm in enumerate(payload["rms_norms"]):
        item = require_keys(rms_norm, ("name", "operator", "dimension"),
                            f"rms_norms[{index}]")
        if item["operator"] == "integer_rmsnorm_q15_lut":
            require_keys(
                item,
                ("product_integer_bits", "product_exponent_delta",
                 "pre_requant_right_shift", "requantization"),
                f"rms_norms[{index}]",
            )
            if (item["product_integer_bits"], item["product_exponent_delta"],
                    item["pre_requant_right_shift"]) != (23, -15, 0):
                raise ValueError(f"rms_norms[{index}] product ABI mismatch")
    if not isinstance(payload["rms_luts"], list):
        raise ValueError("rms_luts must be a list")
    for index, rms_lut in enumerate(payload["rms_luts"]):
        item = require_keys(
            rms_lut,
            ("name", "address_min", "address_max", "entry_bits", "table_uint16_carried_as_int32"),
            f"rms_luts[{index}]",
        )
        table = item["table_uint16_carried_as_int32"]
        if not isinstance(table, torch.Tensor) or tuple(table.shape) != (
            int(item["address_max"]) + 1,
        ):
            raise ValueError(f"rms_luts[{index}] table shape mismatch")
    if not isinstance(payload["local_branches"], list):
        raise ValueError("local_branches must be a list")
    for index, branch in enumerate(payload["local_branches"]):
        item = require_keys(
            branch,
            ("name", "operator", "grid_height", "grid_width", "channels", "activation_bits"),
            f"local_branches[{index}]",
        )
        if item["operator"] == "zero_padded_depthwise_3x3_shift_add":
            require_keys(
                item,
                ("kernel_code", "kernel_sign", "kernel_magnitude_planes_lsb_first",
                 "kernel_effective_scale_exponent", "target_magnitude_bits"),
                f"local_branches[{index}]",
            )
            continue
        if item["operator"] != "shared_logic_tree3x3_bitplane":
            raise ValueError(f"local_branches[{index}] unsupported operator")
        tree = require_keys(
            item,
            (
                "input_encoding", "tree_depth", "leaf_count", "gate_count",
                "truth_address", "truth_table_00_01_10_11",
                "truth_nibble_lsb_address", "site_offsets_dy_dx", "leaf_site",
                "leaf_offsets_dy_dx", "gate_children_node_index",
                "root_gate_node_index", "padding", "spatial_sharing",
                "learned_connections", "cls_path", "root_state",
                "output_projection", "output_scale",
            ),
            f"local_branches[{index}]",
        )
        channels = int(tree["channels"])
        activation_bits = int(tree["activation_bits"])
        if (activation_bits, int(tree["tree_depth"]), int(tree["leaf_count"]),
                int(tree["gate_count"])) != (8, 3, 8, 7):
            raise ValueError(f"local_branches[{index}] logic-tree topology mismatch")
        expected_truth_shape = (channels, activation_bits, 7, 4)
        truth = tree["truth_table_00_01_10_11"]
        if not isinstance(truth, torch.Tensor) or tuple(truth.shape) != expected_truth_shape:
            raise ValueError(f"local_branches[{index}] truth-table shape mismatch")
        if bool(((truth != 0) & (truth != 1)).any()):
            raise ValueError(f"local_branches[{index}] truth-table is not binary")
        nibbles = tree["truth_nibble_lsb_address"]
        if not isinstance(nibbles, torch.Tensor) or tuple(nibbles.shape) != (
            channels, activation_bits, 7
        ):
            raise ValueError(f"local_branches[{index}] truth-nibble shape mismatch")
        shifts = torch.arange(4, dtype=torch.uint8)
        expected_nibbles = torch.sum(
            torch.bitwise_left_shift(truth.to(torch.uint8), shifts), dim=-1
        ).to(torch.uint8)
        if not torch.equal(nibbles.to(torch.uint8), expected_nibbles):
            raise ValueError(f"local_branches[{index}] truth packing mismatch")
        leaf_site = tree["leaf_site"]
        expected_leaf_shape = (channels, activation_bits, 8)
        if not isinstance(leaf_site, torch.Tensor) or tuple(leaf_site.shape) != expected_leaf_shape:
            raise ValueError(f"local_branches[{index}] leaf-site shape mismatch")
        if not bool((leaf_site[..., 0] == 0).all()):
            raise ValueError(f"local_branches[{index}] leaf zero is not the centre")
        if bool(((leaf_site < 0) | (leaf_site > 8)).any()):
            raise ValueError(f"local_branches[{index}] leaf-site index out of range")
        expected_leaf_site = torch.empty(
            channels, activation_bits, 8, dtype=torch.int8
        )
        for channel in range(channels):
            for bitplane in range(activation_bits):
                omitted = 1 + ((channel + bitplane) % 8)
                expected_leaf_site[channel, bitplane] = torch.tensor(
                    [0, *[site for site in range(1, 9) if site != omitted]],
                    dtype=torch.int8,
                )
        if not torch.equal(leaf_site.to(torch.int8), expected_leaf_site):
            raise ValueError(f"local_branches[{index}] fixed leaf-map ABI mismatch")
        offsets = tree["site_offsets_dy_dx"]
        expected_offsets = torch.tensor(SITE_OFFSETS, dtype=torch.int8)
        if not isinstance(offsets, torch.Tensor) or not torch.equal(
            offsets.to(torch.int8), expected_offsets
        ):
            raise ValueError(f"local_branches[{index}] site-offset ABI mismatch")
        leaf_offsets = tree["leaf_offsets_dy_dx"]
        if not isinstance(leaf_offsets, torch.Tensor) or not torch.equal(
            leaf_offsets.to(torch.int8), expected_offsets[leaf_site.to(torch.long)]
        ):
            raise ValueError(f"local_branches[{index}] leaf-offset payload mismatch")
        expected_children = torch.tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
            dtype=torch.int8,
        )
        children = tree["gate_children_node_index"]
        if not isinstance(children, torch.Tensor) or not torch.equal(
            children.to(torch.int8), expected_children
        ):
            raise ValueError(f"local_branches[{index}] gate topology mismatch")
        if (
            tree["root_gate_node_index"] != 14
            or tree["output_projection"] != "none"
            or tree["cls_path"] != "exact_bypass"
            or tree["learned_connections"] is not False
            or tree["spatial_sharing"] is not True
            or tree["truth_address"] != "(A<<1)|B_little_address_endian"
            or tree["padding"] != "constant_logic_0"
            or tree["root_state"] != "boolean_bitplanes"
            or tree["output_scale"] != "reuse_per_token_input_power_of_two_scale"
            or tree["input_encoding"]
            != "signed_magnitude_sign_then_lsb_planes_per_channel"
        ):
            raise ValueError(f"local_branches[{index}] logic-tree ABI mismatch")
    arithmetic = require_keys(
        payload["arithmetic_contract"], ("a8_by_u4_product_rom",),
        "arithmetic_contract",
    )
    product_rom = require_keys(
        arithmetic["a8_by_u4_product_rom"], ("address", "product_rom_int16"),
        "arithmetic_contract.a8_by_u4_product_rom",
    )
    expected_rom = _a8_u4_product_rom()
    if product_rom["address"] != expected_rom["address"]:
        raise ValueError("A8-by-U4 product ROM address ABI mismatch")
    table = product_rom["product_rom_int16"]
    if (not isinstance(table, torch.Tensor) or tuple(table.shape) != (256, 16)
            or not torch.equal(table, expected_rom["product_rom_int16"])):
        raise ValueError("A8-by-U4 product ROM payload mismatch")

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
                    raise ValueError(f"training-only key at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, torch.Tensor):
            if value.is_floating_point() or value.is_complex():
                raise ValueError(f"floating tensor at {path}")
            if value.device.type != "cpu":
                raise ValueError(f"non-CPU deployment tensor at {path}")
        elif isinstance(value, float):
            raise ValueError(f"floating scalar at {path}")
        elif value is not None and not isinstance(value, (str, int, bool)):
            raise TypeError(f"unsupported payload value {type(value).__name__} at {path}")

    visit(payload, "payload")


def _model_kwargs_from_checkpoint_args(args: Mapping[str, Any]) -> dict[str, object]:
    return {
        "image_size": int(args.get("image_size", 32)),
        "patch_size": int(args.get("patch_size", 4)),
        "channels": int(args.get("channels", 3)),
        "classes": int(args.get("classes", 10)),
        "dim": int(args.get("dim", 192)),
        "depth": int(args.get("depth", 6)),
        "heads": int(args.get("heads", 6)),
        "topk": int(args.get("topk", 8)),
        "mlp_ratio": float(args.get("mlp_ratio", 4.0)),
        "weight_bits": int(args.get("weight_magnitude_bits", args.get("weight_bits", 4))),
        "activation_bits": int(args.get("activation_bits", 8)),
        "qk_lanes": int(args.get("qk_lanes", 7)),
        "norm_kind": str(args.get("norm_kind", "rms_lut")),
        "final_norm_kind": str(args.get("final_norm_kind", "same")),
        "learned_gap": bool(args.get("learned_gap", False)),
        "group_lut_groups": int(args.get("group_lut_groups", 0)),
        "local_layers": int(args.get("local_layers", 0)),
        "local_operator": str(args.get("local_operator", "depthwise_shiftadd")),
        "global_mixer": str(args.get("global_mixer", "attention")),
        "hadamard_group_size": int(args.get("hadamard_group_size", 32)),
        "hadamard_branch_shift": int(args.get("hadamard_branch_shift", 2)),
        "logic_expert_width": int(args.get("logic_expert_width", 0)),
        "logic_expert_count": int(args.get("logic_expert_count", 1)),
        "state_control": str(args.get("state_control", "none")),
        "state_expert_width": int(args.get("state_expert_width", 0)),
    }


def load_checkpoint_model(
    checkpoint_path: str | Path,
) -> tuple[EnhancedFullDiscreteViT, Mapping[str, Any]]:
    """Reconstruct an enhanced/base-compatible model from a training checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("checkpoint must contain model and args entries")
    args = checkpoint.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("checkpoint args are required to reconstruct model topology")
    model = EnhancedFullDiscreteViT(**_model_kwargs_from_checkpoint_args(args))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def export_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    requantize_magnitude_bits: int | None = None,
) -> dict[str, object]:
    """Load a training checkpoint and atomically save its hardened payload."""

    model, checkpoint = load_checkpoint_model(checkpoint_path)
    payload = export_logic_payload(
        model,
        requantize_magnitude_bits=requantize_magnitude_bits,
        checkpoint_metadata=checkpoint,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a full-discrete checkpoint as integer-only logic payload"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--requantize-magnitude-bits",
        type=int,
        default=None,
        help="explicitly requantize all learned matrices (for example 4); default preserves each layer",
    )
    args = parser.parse_args()
    payload = export_checkpoint(
        args.checkpoint,
        args.output,
        requantize_magnitude_bits=args.requantize_magnitude_bits,
    )
    print(
        f"exported schema v{payload['schema']['version']} with "
        f"{len(payload['shift_add_layers'])} shift-add layers to {args.output}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SCHEMA_TOP_LEVEL_KEYS",
    "export_logic_payload",
    "validate_logic_payload",
    "load_checkpoint_model",
    "export_checkpoint",
]
