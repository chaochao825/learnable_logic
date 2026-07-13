from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    return soft + (hard - soft).detach()


def _power_of_two_scale(scale: torch.Tensor, minimum_exponent: int = -24) -> torch.Tensor:
    floor = torch.full_like(scale, float(2**minimum_exponent))
    safe = torch.maximum(scale, floor)
    return torch.pow(2.0, torch.round(torch.log2(safe)))


class PowerOfTwoActivationQuantizer(nn.Module):
    """Hard signed integer activation with a runtime power-of-two scale.

    The floating tensor is only a training/reference carrier.  Its hard forward
    value is exactly ``integer_code * 2**exponent``.  The scale can be obtained
    with a leading-one detector and shifts; no general multiplier is required.
    """

    def __init__(self, bits: int = 8, clip_percentile: float = 1.0) -> None:
        super().__init__()
        if bits < 2 or bits > 16:
            raise ValueError("activation bits must be in [2,16]")
        if not 0.0 < clip_percentile <= 1.0:
            raise ValueError("clip_percentile must be in (0,1]")
        self.bits = int(bits)
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -self.qmax
        self.clip_percentile = float(clip_percentile)

    def integer_code_and_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detached = x.detach().abs()
        if self.clip_percentile < 1.0 and detached.shape[-1] > 1:
            maximum = torch.quantile(
                detached, self.clip_percentile, dim=-1, keepdim=True
            )
        else:
            maximum = detached.amax(dim=-1, keepdim=True)
        scale = _power_of_two_scale(maximum / max(self.qmax, 1))
        code = torch.round(x / scale).clamp(self.qmin, self.qmax)
        return code, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code, scale = self.integer_code_and_scale(x)
        return _ste(code * scale, x)


class ShiftAddLinear(nn.Module):
    """Linear layer whose hard weights are signed {8,4,2,1} bit-plane sums.

    For ``magnitude_bits=4``, each integer coefficient is in [-15,15] and is
    implemented as a sign plus four independently enabled shifted copies of the
    input.  Per-output scales are powers of two and are therefore shifts too.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        magnitude_bits: int = 4,
        output_bits: int = 8,
        input_bits: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if magnitude_bits < 1 or magnitude_bits > 8:
            raise ValueError("magnitude_bits must be in [1,8]")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.magnitude_bits = int(magnitude_bits)
        self.qmax = (1 << magnitude_bits) - 1
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.input_quantizer = PowerOfTwoActivationQuantizer(
            output_bits if input_bits is None else input_bits
        )
        self.output_quantizer = PowerOfTwoActivationQuantizer(output_bits)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def integer_weight_and_scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = self.weight.detach().abs().amax(dim=1, keepdim=True)
        scale = _power_of_two_scale(maximum / max(self.qmax, 1))
        code = torch.round(self.weight / scale).clamp(-self.qmax, self.qmax)
        return code, scale

    def quantized_weight(self) -> torch.Tensor:
        code, scale = self.integer_weight_and_scale()
        return _ste(code * scale, self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_input = self.input_quantizer(x)
        output = F.linear(quantized_input, self.quantized_weight(), self.bias)
        return self.output_quantizer(output)

    def integer_accumulator(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bias is not None:
            raise ValueError("integer accumulator reference currently requires bias=False")
        input_code, input_scale = self.input_quantizer.integer_code_and_scale(x)
        weight_code, weight_scale = self.integer_weight_and_scale()
        accumulator = torch.matmul(
            input_code.to(torch.int64), weight_code.to(torch.int64).transpose(0, 1)
        )
        combined_scale = input_scale * weight_scale.squeeze(-1)
        return accumulator, combined_scale

    def bitplane_codes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code, scale = self.integer_weight_and_scale()
        integer = code.to(torch.int16)
        sign = integer < 0
        magnitude = integer.abs()
        planes = torch.stack(
            [torch.bitwise_and(torch.bitwise_right_shift(magnitude, bit), 1).bool()
             for bit in range(self.magnitude_bits)],
            dim=-1,
        )
        return sign, planes, scale

    def deployment_contract(self) -> dict[str, object]:
        return {
            "coefficient_integer_range": [-self.qmax, self.qmax],
            "magnitude_bit_values": [1 << bit for bit in range(self.magnitude_bits)],
            "per_output_scale": "signed power of two",
            "runtime": "bit-plane gate/XNOR or conditional shift plus integer accumulation",
            "general_multipliers": 0,
        }
