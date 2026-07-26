"""Strict integer executor for hardened FullDiscreteViT payloads.

The training modules use floating shadow parameters and STE gradients.  This
module is the deployment boundary: its public input is a CPU ``uint8`` image
tensor and every tensor created by the executor is Boolean or integer.  A
runtime value is represented by an integer code and a signed integer exponent;
no power-of-two scale is materialized as a floating tensor.

The implementation is a transaction-level reference.  ``int_matmul`` is an
integer-only acceleration of the fixed-weight shift/add network.  ``lut`` uses
the slower A8-by-U4 product-ROM oracle and is intended for primitive-level
checks.  Both backends have the same integer ABI and output codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .export_logic_payload import validate_logic_payload
from .logic_backend import lut_linear_accumulator


_INTEGER_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_INT64_MAX = (1 << 63) - 1
_INT32_MAX = (1 << 31) - 1
_MINIMUM_SCALE_EXPONENT = -24
_BYTE_POPCOUNT = torch.tensor(
    [bin(value).count("1") for value in range(256)], dtype=torch.uint8
)


def _require_integer_tensor(
    name: str, value: torch.Tensor, *, boolean_allowed: bool = False
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    allowed = value.dtype in _INTEGER_DTYPES
    if boolean_allowed:
        allowed = allowed or value.dtype == torch.bool
    if not allowed:
        raise TypeError(f"{name} must be Boolean/integer, got {value.dtype}")
    return value


def _tensor_leaves(value: object):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensor_leaves(child)


class IntegerRuntimeAudit(TorchDispatchMode):
    """Reject any floating/complex tensor crossing an executed Torch operator."""

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


def _check_int64_bound(maximum_magnitude: int, context: str) -> None:
    if maximum_magnitude < 0 or maximum_magnitude > _INT64_MAX:
        raise OverflowError(f"signed int64 overflow while {context}")


@dataclass(frozen=True)
class IntegerTensor:
    """Dyadic tensor represented solely as ``code * 2**exponent`` metadata."""

    code: torch.Tensor
    exponent: torch.Tensor

    def __post_init__(self) -> None:
        _require_integer_tensor("code", self.code)
        _require_integer_tensor("exponent", self.exponent)
        if self.code.dtype != torch.int64:
            raise TypeError("IntegerTensor.code must use torch.int64")
        if self.exponent.dtype != torch.int32:
            raise TypeError("IntegerTensor.exponent must use torch.int32")
        try:
            torch.broadcast_shapes(self.code.shape, self.exponent.shape)
        except RuntimeError as error:
            raise ValueError("code and exponent shapes are not broadcastable") from error

    def expanded_exponent(self) -> torch.Tensor:
        return torch.broadcast_to(self.exponent, self.code.shape)


def _round_divide_nearest_even(
    numerator: torch.Tensor, denominator: int | torch.Tensor
) -> torch.Tensor:
    """Signed integer division rounded to nearest with half-to-even ties."""

    numerator = _require_integer_tensor("numerator", numerator).to(torch.int64)
    if isinstance(denominator, int):
        if denominator < 1 or denominator > _INT64_MAX:
            raise ValueError("denominator must be a positive signed-int64 integer")
        divisor = torch.full_like(numerator, denominator)
    else:
        divisor = _require_integer_tensor("denominator", denominator).to(torch.int64)
        divisor = torch.broadcast_to(divisor, numerator.shape)
        if divisor.numel() and int(divisor.min().item()) < 1:
            raise ValueError("denominator must be positive")
    magnitude = numerator.abs()
    quotient = torch.div(magnitude, divisor, rounding_mode="floor")
    remainder = magnitude - quotient * divisor
    mirrored = divisor - remainder
    increment = (remainder > mirrored) | (
        (remainder == mirrored) & ((quotient & 1) != 0)
    )
    rounded = quotient + increment.to(torch.int64)
    return torch.sign(numerator) * rounded


def _round_shift_right_nearest_even(
    value: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    value = _require_integer_tensor("value", value).to(torch.int64)
    shift = _require_integer_tensor("shift", shift).to(torch.int64)
    shift = torch.broadcast_to(shift, value.shape)
    if shift.numel() and int(shift.min().item()) < 0:
        raise ValueError("right-shift count must be non-negative")
    output = torch.zeros_like(value)
    for amount_tensor in torch.unique(shift):
        amount = int(amount_tensor.item())
        mask = shift == amount
        selected = value[mask]
        if amount == 0:
            output[mask] = selected
        elif amount < 63:
            output[mask] = _round_divide_nearest_even(selected, 1 << amount)
        elif amount == 63:
            magnitude = selected.abs()
            half = 1 << 62
            rounded = (magnitude > half).to(torch.int64)
            output[mask] = torch.sign(selected) * rounded
        else:
            output[mask] = 0
    return output


def _shift_left_exact(value: torch.Tensor, shift: torch.Tensor, context: str) -> torch.Tensor:
    value = _require_integer_tensor("value", value).to(torch.int64)
    shift = _require_integer_tensor("shift", shift).to(torch.int64)
    shift = torch.broadcast_to(shift, value.shape)
    if shift.numel() and int(shift.min().item()) < 0:
        raise ValueError("left-shift count must be non-negative")
    maximum = int(value.abs().max().item()) if value.numel() else 0
    maximum_shift = int(shift.max().item()) if shift.numel() else 0
    _check_int64_bound(maximum << maximum_shift, context)
    return torch.bitwise_left_shift(value, shift)


def _shift_signed_nearest_even(
    value: torch.Tensor, signed_shift: torch.Tensor, context: str
) -> torch.Tensor:
    """Positive counts shift left; negative counts right-round to nearest-even."""

    value = _require_integer_tensor("value", value).to(torch.int64)
    signed_shift = _require_integer_tensor("signed_shift", signed_shift).to(torch.int64)
    signed_shift = torch.broadcast_to(signed_shift, value.shape)
    output = torch.empty_like(value)
    left = signed_shift >= 0
    if bool(left.any()):
        output[left] = _shift_left_exact(
            value[left], signed_shift[left], context
        )
    if bool((~left).any()):
        output[~left] = _round_shift_right_nearest_even(
            value[~left], -signed_shift[~left]
        )
    return output


def _floor_log2_ratio(numerator: int, denominator: int) -> int:
    if numerator < 1 or denominator < 1:
        raise ValueError("log2 ratio operands must be positive")
    candidate = numerator.bit_length() - denominator.bit_length()
    if candidate >= 0:
        if numerator < (denominator << candidate):
            candidate -= 1
    elif (numerator << (-candidate)) < denominator:
        candidate -= 1
    return candidate


def _nearest_log2_ratio_exponent(
    numerator: int,
    denominator: int,
    base_exponent: int,
    minimum_exponent: int,
) -> int:
    """Exact ``round(log2(numerator/denominator))+base`` with even ties."""

    if numerator == 0:
        return minimum_exponent
    lower = _floor_log2_ratio(numerator, denominator)
    boundary_power = 2 * lower + 1
    left = numerator * numerator
    right = denominator * denominator
    if boundary_power >= 0:
        right <<= boundary_power
    else:
        left <<= -boundary_power
    lower_absolute = base_exponent + lower
    if left > right:
        rounded = lower_absolute + 1
    elif left < right or (lower_absolute & 1) == 0:
        rounded = lower_absolute
    else:
        rounded = lower_absolute + 1
    return max(minimum_exponent, rounded)


def _select_scale_exponent(
    maximum_magnitude: torch.Tensor,
    base_exponent: torch.Tensor,
    denominator: int,
    minimum_exponent: int = _MINIMUM_SCALE_EXPONENT,
) -> torch.Tensor:
    maximum_magnitude = _require_integer_tensor(
        "maximum_magnitude", maximum_magnitude
    ).to(torch.int64)
    base_exponent = _require_integer_tensor("base_exponent", base_exponent).to(
        torch.int32
    )
    base_exponent = torch.broadcast_to(base_exponent, maximum_magnitude.shape)
    values = []
    for maximum, base in zip(
        maximum_magnitude.reshape(-1).tolist(),
        base_exponent.reshape(-1).tolist(),
    ):
        values.append(
            _nearest_log2_ratio_exponent(
                int(maximum), denominator, int(base), minimum_exponent
            )
        )
    return torch.tensor(values, dtype=torch.int32).reshape(maximum_magnitude.shape)


def _align_to_axis_min(state: IntegerTensor, axis: int) -> IntegerTensor:
    exponent = state.expanded_exponent()
    common = exponent.amin(dim=axis, keepdim=True)
    shift = exponent.to(torch.int64) - common.to(torch.int64)
    aligned = _shift_left_exact(state.code, shift, "aligning dyadic exponents")
    return IntegerTensor(aligned, common.to(torch.int32))


def requantize(
    state: IntegerTensor,
    bits: int,
    *,
    minimum_exponent: int = _MINIMUM_SCALE_EXPONENT,
) -> IntegerTensor:
    """Requantize the last axis using exact integer leading-one comparisons."""

    if bits < 2 or bits > 16:
        raise ValueError("activation bits must be in [2,16]")
    qmax = (1 << (bits - 1)) - 1
    aligned = _align_to_axis_min(state, -1)
    maximum = aligned.code.abs().amax(dim=-1, keepdim=True)
    output_exponent = _select_scale_exponent(
        maximum, aligned.exponent, qmax, minimum_exponent
    )
    shift = aligned.exponent.to(torch.int64) - output_exponent.to(torch.int64)
    code = _shift_signed_nearest_even(
        aligned.code, shift, "requantizing activation"
    ).clamp(-qmax, qmax)
    return IntegerTensor(code.to(torch.int64), output_exponent.to(torch.int32))


def _quantize_unsigned_rational(
    value: torch.Tensor,
    *,
    denominator: int,
    bits: int,
    minimum_exponent: int = _MINIMUM_SCALE_EXPONENT,
) -> IntegerTensor:
    """Quantize non-negative integer numerators interpreted as value/denominator."""

    value = _require_integer_tensor("value", value).to(torch.int64)
    if value.numel() and int(value.min().item()) < 0:
        raise ValueError("unsigned rational input cannot contain negative values")
    if denominator < 1:
        raise ValueError("denominator must be positive")
    qmax = (1 << (bits - 1)) - 1
    maximum = value.amax(dim=-1, keepdim=True)
    zero_base = torch.zeros_like(maximum, dtype=torch.int32)
    output_exponent = _select_scale_exponent(
        maximum, zero_base, denominator * qmax, minimum_exponent
    )
    expanded_exponent = torch.broadcast_to(output_exponent, value.shape)
    code = torch.empty_like(value)
    for exponent_tensor in torch.unique(expanded_exponent):
        exponent = int(exponent_tensor.item())
        mask = expanded_exponent == exponent
        selected = value[mask]
        if exponent < 0:
            numerator = _shift_left_exact(
                selected,
                torch.full_like(selected, -exponent),
                "quantizing rational input",
            )
            divisor = denominator
        else:
            numerator = selected
            divisor = denominator << exponent
        code[mask] = _round_divide_nearest_even(numerator, divisor)
    return IntegerTensor(code.clamp(0, qmax), output_exponent)


def _add_exact(states: tuple[IntegerTensor, ...]) -> IntegerTensor:
    if not states:
        raise ValueError("at least one state is required")
    shape = states[0].code.shape
    if any(state.code.shape != shape for state in states):
        raise ValueError("addends must have identical shapes")
    exponents = [state.expanded_exponent() for state in states]
    common = exponents[0]
    for exponent in exponents[1:]:
        common = torch.minimum(common, exponent)
    aligned_codes = []
    bound = 0
    for state, exponent in zip(states, exponents):
        shift = exponent.to(torch.int64) - common.to(torch.int64)
        aligned = _shift_left_exact(state.code, shift, "aligning residual addend")
        aligned_codes.append(aligned)
        bound += int(aligned.abs().max().item()) if aligned.numel() else 0
    _check_int64_bound(bound, "adding residual branches")
    total = torch.stack(aligned_codes, dim=0).sum(dim=0)
    return IntegerTensor(total, common.to(torch.int32))


def _add_and_requantize(states: tuple[IntegerTensor, ...], bits: int) -> IntegerTensor:
    return requantize(_add_exact(states), bits)


def _concatenate(states: tuple[IntegerTensor, ...], dim: int) -> IntegerTensor:
    return IntegerTensor(
        torch.cat([state.code for state in states], dim=dim),
        torch.cat([state.expanded_exponent() for state in states], dim=dim).to(
            torch.int32
        ),
    )


class StrictIntegerExecutor:
    """Execute the supported hardened topology without real-valued tensors."""

    def __init__(
        self,
        payload: Mapping[str, object],
        *,
        linear_backend: str = "int_matmul",
        score_backend: str = "vectorized",
    ) -> None:
        validate_logic_payload(payload)
        if linear_backend not in {"int_matmul", "lut"}:
            raise ValueError("linear_backend must be int_matmul or lut")
        if score_backend != "vectorized":
            raise ValueError("score_backend must be vectorized")
        self.payload = payload
        self.linear_backend = linear_backend
        self.score_backend = score_backend
        self.topology = payload["topology"]
        if self.topology["global_mixer_mode"] != "attention":
            raise NotImplementedError(
                "strict executor currently supports the attention global mixer"
            )
        unsupported = (
            "global_mixers",
            "activation_luts",
            "logic_experts",
            "logic_ffns",
            "state_ffns",
        )
        for key in unsupported:
            if payload[key]:
                raise NotImplementedError(f"strict executor does not support {key}")
        self.layers = {item["name"]: item for item in payload["shift_add_layers"]}
        self.attentions = {item["name"]: item for item in payload["attention"]}
        self.norms = {item["name"]: item for item in payload["rms_norms"]}
        self.local_branches = {
            item["name"]: item for item in payload["local_branches"]
        }
        unsupported_local = [
            name
            for name, item in self.local_branches.items()
            if item["operator"] != "zero_padded_depthwise_3x3_shift_add"
        ]
        if unsupported_local:
            raise NotImplementedError(
                "strict executor supports only depthwise shift/add local branches"
            )
        self.rms_luts = {item["name"]: item for item in payload["rms_luts"]}
        self.trace: list[dict[str, object]] = []

    def _record(self, name: str, state: IntegerTensor) -> IntegerTensor:
        self.trace.append(
            {
                "name": name,
                "code_dtype": str(state.code.dtype),
                "exponent_dtype": str(state.exponent.dtype),
                "code_shape": list(state.code.shape),
                "exponent_shape": list(state.exponent.shape),
            }
        )
        return state

    def _parameter(self, name: str) -> IntegerTensor:
        item = self.payload["parameters"][name]
        return IntegerTensor(
            item["code"].to(torch.int64),
            item["scale_exponent"].to(torch.int32),
        )

    def _linear(self, name: str, state: IntegerTensor) -> IntegerTensor:
        item = self.layers[name]
        quantized = requantize(state, int(item["input_activation_bits"]))
        weight = item["weight_code"].to(torch.int64)
        fanin_bound = (
            int(quantized.code.abs().max().item())
            * int(weight.abs().max().item())
            * int(weight.shape[1])
        )
        _check_int64_bound(fanin_bound, f"accumulating {name}")
        if self.linear_backend == "lut":
            accumulator = lut_linear_accumulator(
                quantized.code,
                weight,
                magnitude_bits=int(item["target_magnitude_bits"]),
            )
        else:
            if fanin_bound <= _INT32_MAX:
                accumulator = torch.matmul(
                    quantized.code.to(torch.int32),
                    weight.to(torch.int32).transpose(0, 1),
                ).to(torch.int64)
            else:
                accumulator = torch.matmul(
                    quantized.code, weight.transpose(0, 1)
                )
        weight_exponent = item["weight_scale_exponent"].to(torch.int32).squeeze(-1)
        accumulator_exponent = quantized.exponent + weight_exponent
        output = requantize(
            IntegerTensor(accumulator.to(torch.int64), accumulator_exponent),
            int(item["output_activation_bits"]),
        )
        return self._record(name, output)

    def _norm(self, name: str, state: IntegerTensor) -> IntegerTensor:
        item = self.norms[name]
        operator = item["operator"]
        if operator == "identity_wire":
            return self._record(name, state)
        bits = int(item.get("input_activation_bits", item.get("output_activation_bits", 8)))
        quantized = requantize(state, bits)
        if operator == "activation_requant_only":
            return self._record(name, quantized)
        integer = quantized.code
        sum_square = integer.square().sum(dim=-1, keepdim=True)
        if operator == "integer_shift_rms":
            rms_shift = torch.zeros_like(sum_square, dtype=torch.int32)
            dimension = int(item["dimension"])
            for candidate in range(1, bits):
                threshold = dimension << (2 * candidate - 1)
                rms_shift += (sum_square >= threshold).to(torch.int32)
            normalized = IntegerTensor(integer, -rms_shift)
        elif operator == "integer_rmsnorm_q15_lut":
            dimension = int(item["dimension"])
            mean_square = torch.div(
                sum_square + dimension // 2, dimension, rounding_mode="floor"
            ).clamp_min(1)
            lut_name = f"rms_reciprocal_sqrt_q15_a{bits}"
            table = self.rms_luts[lut_name]["table_uint16_carried_as_int32"]
            if int(mean_square.max().item()) >= table.numel():
                raise ValueError("RMS mean-square exceeds exported LUT")
            reciprocal = table[mean_square].to(torch.int64)
            normalized = IntegerTensor(
                integer * reciprocal,
                torch.full_like(mean_square, -15, dtype=torch.int32),
            )
        else:
            raise NotImplementedError(f"unsupported norm operator {operator}")
        output = requantize(normalized, int(item["output_activation_bits"]))
        return self._record(name, output)

    @staticmethod
    def _threshold_bits(code: torch.Tensor, item: Mapping[str, object]) -> torch.Tensor:
        maximum = code.abs().amax(dim=-1, keepdim=True).unsqueeze(-1)
        numerator = item["threshold_fraction_numerator"].to(torch.int64)
        denominator_shift = item[
            "threshold_fraction_denominator_shift"
        ].to(torch.int64)
        lanes = int(numerator.numel())
        expanded = code.unsqueeze(-1).expand(*code.shape, lanes)
        expanded_shift = denominator_shift.reshape(1, 1, 1, 1, lanes).expand_as(
            expanded
        )
        left = _shift_left_exact(
            expanded,
            expanded_shift,
            "applying attention thresholds",
        )
        right = maximum * numerator
        return left >= right

    @staticmethod
    def _score(q_bits: torch.Tensor, k_bits: torch.Tensor) -> torch.Tensor:
        q_flat = q_bits.flatten(-2)
        k_flat = k_bits.flatten(-2)
        valid_bits = int(q_flat.shape[-1])
        padding = (-valid_bits) % 8
        if padding:
            q_flat = torch.cat(
                [q_flat, torch.zeros(*q_flat.shape[:-1], padding, dtype=torch.bool)],
                dim=-1,
            )
            k_flat = torch.cat(
                [k_flat, torch.zeros(*k_flat.shape[:-1], padding, dtype=torch.bool)],
                dim=-1,
            )
        shifts = torch.arange(8, dtype=torch.uint8)

        def pack(bits: torch.Tensor) -> torch.Tensor:
            reshaped = bits.reshape(*bits.shape[:-1], -1, 8).to(torch.uint8)
            return torch.bitwise_left_shift(reshaped, shifts).sum(
                dim=-1, dtype=torch.int16
            ).to(torch.uint8)

        q_packed = pack(q_flat)
        k_packed = pack(k_flat)
        difference = torch.bitwise_xor(
            q_packed.unsqueeze(-2), k_packed.unsqueeze(-3)
        )
        mismatch = _BYTE_POPCOUNT[difference.to(torch.int64)].sum(
            dim=-1, dtype=torch.int64
        )
        return valid_bits - mismatch

    @staticmethod
    def _attention_weights(
        scores: torch.Tensor, item: Mapping[str, object]
    ) -> torch.Tensor:
        best = scores.max(dim=-1, keepdim=True).values
        raw_gap = best - scores
        gap = item["gap"]
        if gap["kind"] == "fixed_bucket_shift":
            bucket = torch.bitwise_right_shift(
                raw_gap, int(gap["gap_right_shift"])
            ).clamp(0, int(gap["max_gap_bucket"]))
            return torch.bitwise_left_shift(
                torch.ones_like(bucket), int(gap["max_gap_bucket"]) - bucket
            )
        if gap["kind"] == "per_head_monotone_lut":
            table = gap["weight_table"].to(torch.int64)
            address = raw_gap.clamp(0, int(gap["max_gap"]))
            heads = int(table.shape[0])
            head = torch.arange(heads, dtype=torch.int64).reshape(1, heads, 1, 1)
            return table[head, address]
        raise NotImplementedError(f"unsupported attention gap {gap['kind']}")

    def _attention(self, name: str, state: IntegerTensor) -> IntegerTensor:
        item = self.attentions[name]
        qkv = self._linear(item["qkv_projection"], state)
        batch, tokens, _ = qkv.code.shape
        heads = int(item["heads"])
        head_dim = int(item["head_dim"])
        reshaped = qkv.code.reshape(batch, tokens, 3, heads, head_dim)
        q, k, v_code = reshaped.permute(2, 0, 3, 1, 4)
        q_bits = self._threshold_bits(q, item)
        k_bits = self._threshold_bits(k, item)
        scores = self._score(q_bits, k_bits)
        count = min(int(item["topk"]), tokens)
        indices = torch.argsort(
            scores, dim=-1, descending=True, stable=True
        )[..., :count]
        weights = self._attention_weights(scores, item)

        qkv_exponent = qkv.exponent.reshape(batch, tokens, 1, 1)
        v_exponent = qkv_exponent.permute(0, 2, 1, 3)
        value = requantize(
            IntegerTensor(v_code.to(torch.int64), v_exponent.to(torch.int32)),
            int(self.layers[item["qkv_projection"]]["output_activation_bits"]),
        )
        aligned_value = _align_to_axis_min(value, -2)
        expanded_code = aligned_value.code.unsqueeze(-3).expand(
            batch, heads, tokens, tokens, head_dim
        )
        gather_index = indices.unsqueeze(-1).expand(
            batch, heads, tokens, count, head_dim
        )
        selected_code = torch.gather(expanded_code, -2, gather_index)
        selected_weight = torch.gather(weights, -1, indices).to(torch.int64)
        numerator_bound = (
            int(selected_code.abs().max().item())
            * int(selected_weight.max().item())
            * count
        )
        _check_int64_bound(numerator_bound, f"aggregating {name} values")
        numerator = (
            selected_code * selected_weight.unsqueeze(-1)
        ).sum(dim=-2)
        denominator = selected_weight.sum(dim=-1, keepdim=True).clamp_min(1)
        quotient = torch.sign(numerator) * torch.div(
            numerator.abs() + denominator // 2,
            denominator,
            rounding_mode="floor",
        )
        aggregate_exponent = aligned_value.exponent.expand(
            batch, heads, tokens, head_dim
        )
        aggregate = requantize(
            IntegerTensor(quotient, aggregate_exponent),
            int(self.topology["activation_bits"]),
        )
        merged = IntegerTensor(
            aggregate.code.transpose(1, 2).reshape(batch, tokens, heads * head_dim),
            aggregate.expanded_exponent().transpose(1, 2).reshape(
                batch, tokens, heads * head_dim
            ),
        )
        return self._linear(item["output_projection"], merged)

    def _ffn(self, block_name: str, state: IntegerTensor) -> IntegerTensor:
        gate = self._linear(f"{block_name}.ffn.gate", state)
        up = self._linear(f"{block_name}.ffn.up", state)
        gated = IntegerTensor(
            torch.where(gate.code >= 0, up.code, torch.zeros_like(up.code)),
            up.exponent,
        )
        hidden = requantize(gated, int(self.topology["activation_bits"]))
        return self._linear(f"{block_name}.ffn.down", hidden)

    def _local_branch(self, name: str, state: IntegerTensor) -> IntegerTensor:
        item = self.local_branches[name]
        kernel = item["kernel_code"].to(torch.int64)
        if bool((kernel == 0).all()):
            return self._record(name, state)
        batch, token_count, channels = state.code.shape
        height = int(item["grid_height"])
        width = int(item["grid_width"])
        if token_count != height * width + 1 or channels != int(item["channels"]):
            raise ValueError(f"{name} token topology mismatch")
        cls = IntegerTensor(state.code[:, :1], state.expanded_exponent()[:, :1])
        patches = IntegerTensor(state.code[:, 1:], state.expanded_exponent()[:, 1:])
        channel_state = IntegerTensor(
            patches.code.transpose(1, 2),
            patches.expanded_exponent().transpose(1, 2),
        )
        channel_state = requantize(channel_state, int(item["activation_bits"]))
        grid_code = channel_state.code.reshape(batch, channels, height, width)
        grid_exponent = channel_state.exponent.reshape(batch, channels, 1, 1)
        padded = torch.zeros(
            batch, channels, height + 2, width + 2, dtype=torch.int64
        )
        padded[:, :, 1:-1, 1:-1] = grid_code
        windows = padded.unfold(2, 3, 1).unfold(3, 3, 1)
        accumulator_bound = (
            int(grid_code.abs().max().item())
            * int(kernel.abs().max().item())
            * 9
        )
        _check_int64_bound(accumulator_bound, f"accumulating {name}")
        accumulator = (
            windows * kernel.reshape(1, channels, 1, 1, 3, 3)
        ).sum(dim=(-1, -2))
        branch_exponent = grid_exponent + item[
            "kernel_effective_scale_exponent"
        ].to(torch.int32).reshape(1, channels, 1, 1)
        patch_sum = _add_exact(
            (
                IntegerTensor(grid_code, grid_exponent),
                IntegerTensor(accumulator, branch_exponent),
            )
        )
        patch_output = requantize(
            IntegerTensor(
                patch_sum.code.flatten(2).transpose(1, 2),
                patch_sum.expanded_exponent().flatten(2).transpose(1, 2),
            ),
            int(item["activation_bits"]),
        )
        cls_output = requantize(
            requantize(cls, int(item["activation_bits"])),
            int(item["activation_bits"]),
        )
        return self._record(name, _concatenate((cls_output, patch_output), 1))

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        images = _require_integer_tensor("images", images)
        if images.dtype != torch.uint8:
            raise TypeError("strict image ABI requires torch.uint8")
        if images.ndim != 4:
            raise ValueError("images must have shape [batch,channels,height,width]")
        patch_size = int(self.topology["patch_size"])
        batch, channels, height, width = images.shape
        if channels != int(self.topology["input_channels"]):
            raise ValueError("input channel count does not match payload")
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch size")
        patches = images.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        grid_height = height // patch_size
        grid_width = width // patch_size
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch, grid_height * grid_width, channels * patch_size * patch_size
        )
        if patches.shape[1] != int(self.topology["patch_tokens"]):
            raise ValueError("image patch count does not match payload")
        return patches.to(torch.int64)

    def forward(self, images: torch.Tensor) -> IntegerTensor:
        """Return integer classifier codes and exponents for uint8 images."""

        self.trace = []
        patches = self._patchify(images)
        input_state = _quantize_unsigned_rational(
            patches,
            denominator=int(self.topology["input_denominator"]),
            bits=int(self.topology["activation_bits"]),
        )
        tokens = self._linear("patch_embed.projection", input_state)
        batch = int(images.shape[0])
        cls = self._parameter("cls_token")
        cls = IntegerTensor(
            cls.code.expand(batch, -1, -1),
            cls.expanded_exponent().expand(batch, -1, -1),
        )
        position = self._parameter("position")
        position = IntegerTensor(
            position.code.expand(batch, -1, -1),
            position.expanded_exponent().expand(batch, -1, -1),
        )
        tokens = _concatenate((cls, tokens), 1)
        tokens = _add_and_requantize(
            (tokens, position), int(self.topology["activation_bits"])
        )
        self._record("tokens.input", tokens)

        for block_name in self.topology["block_order"]:
            local_name = block_name.replace("blocks.", "local_branches.")
            if local_name in self.local_branches:
                tokens = self._local_branch(local_name, tokens)
            attention_input = self._norm(f"{block_name}.norm1", tokens)
            attention = self._attention(f"{block_name}.attn", attention_input)
            tokens = _add_and_requantize(
                (tokens, attention), int(self.topology["activation_bits"])
            )
            self._record(f"{block_name}.residual1", tokens)
            ffn_input = self._norm(f"{block_name}.norm2", tokens)
            ffn = self._ffn(block_name, ffn_input)
            tokens = _add_and_requantize(
                (tokens, ffn), int(self.topology["activation_bits"])
            )
            self._record(f"{block_name}.residual2", tokens)

        normalized = self._norm("norm", tokens)
        cls_state = IntegerTensor(
            normalized.code[:, 0], normalized.expanded_exponent()[:, 0]
        )
        logits = self._linear("head", cls_state)
        return self._record("logits", logits)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images)
        prediction = logits.code.argmax(dim=-1)
        return _require_integer_tensor("prediction", prediction).to(torch.int64)


__all__ = [
    "IntegerRuntimeAudit",
    "IntegerTensor",
    "StrictIntegerExecutor",
    "requantize",
]
