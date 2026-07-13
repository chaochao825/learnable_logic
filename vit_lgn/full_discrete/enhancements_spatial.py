from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shiftadd import (
    PowerOfTwoActivationQuantizer,
    _power_of_two_scale,
    _ste,
)


class _HardPayloadSoftGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, grad_output


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    else:
        if len(value) != 2:
            raise ValueError("grid_size must be an int or a pair")
        result = (int(value[0]), int(value[1]))
    if result[0] < 1 or result[1] < 1:
        raise ValueError("grid dimensions must be positive")
    return result


class DiscreteDepthwiseLocalBranch(nn.Module):
    """Shape-preserving, fully discrete 3x3 local branch for ViT tokens.

    The input layout is ``[batch, 1 + height * width, channels]``.  Token zero
    is the CLS token and never enters the spatial stencil.  Patch activations
    use one signed A8 code scale per sample and channel, while every depthwise
    channel has a signed integer 3x3 kernel and a power-of-two scale.
    Consequently the deployed stencil is composed of bit tests, shifts, and
    integer additions.

    The branch is residual and shape preserving, so instances can be inserted
    before several early transformer blocks or applied repeatedly.  Both the
    patch residual and the bypassed CLS token are requantized to A8 at the
    output boundary.  A non-negative ``branch_shift`` attenuates the local path
    with a right shift and is useful when many branches are stacked.

    Floating tensors are training/reference carriers only.  ``integer_reference``
    exposes the exact integer accumulator and output payload used by the hard
    forward path.
    """

    def __init__(
        self,
        dim: int,
        grid_size: int | Sequence[int] = 8,
        weight_bits: int = 4,
        activation_bits: int = 8,
        branch_shift: int = 2,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        # The hard carrier uses FP32 convolution only as an exact integer
        # accumulator surrogate.  A8/W4 bounds 3x3 sums below 2**24, where every
        # integer is exactly representable.  Wider endpoints require an int64
        # kernel and are deliberately rejected rather than silently inexact.
        if weight_bits < 1 or weight_bits > 4:
            raise ValueError("weight_bits must be in [1,4] for exact hard accumulation")
        if activation_bits < 2 or activation_bits > 8:
            raise ValueError("activation_bits must be in [2,8] for exact hard accumulation")
        if branch_shift < 0:
            raise ValueError("branch_shift must be non-negative")

        self.dim = int(dim)
        self.grid_size = _pair(grid_size)
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.branch_shift = int(branch_shift)
        self.weight_qmax = (1 << self.weight_bits) - 1

        # One independent integer 3x3 kernel per feature channel.  Zero
        # initialization makes repeated insertion an identity-at-start change;
        # the STE still supplies gradients to every kernel coefficient.
        self.kernel = nn.Parameter(torch.empty(self.dim, 3, 3))
        if zero_init:
            nn.init.zeros_(self.kernel)
        else:
            nn.init.kaiming_uniform_(self.kernel.unsqueeze(1), a=math.sqrt(5))

        self.input_quantizer = PowerOfTwoActivationQuantizer(self.activation_bits)
        self.residual_quantizer = PowerOfTwoActivationQuantizer(self.activation_bits)

    @property
    def num_patch_tokens(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, tokens, channels]")
        if tokens.shape[-1] != self.dim:
            raise ValueError(
                f"expected channel dimension {self.dim}, got {tokens.shape[-1]}"
            )
        expected = self.num_patch_tokens + 1
        if tokens.shape[1] != expected:
            raise ValueError(
                f"expected CLS + {self.grid_size[0]}x{self.grid_size[1]} patches "
                f"({expected} tokens), got {tokens.shape[1]}"
            )

    def patch_integer_code_and_scale(
        self, patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return A8 codes and one power-of-two scale per sample/channel.

        A common spatial scale within each depthwise channel makes every
        neighbor in its stencil directly addable without sacrificing the
        dynamic range of unrelated channels.  It avoids hidden general
        multipliers or per-neighbor requantizers inside the datapath.
        """

        if patches.ndim != 3 or patches.shape[-1] != self.dim:
            raise ValueError("patches must have shape [batch, patches, dim]")
        channel_major = patches.transpose(1, 2)
        channel_code, channel_scale = (
            self.input_quantizer.integer_code_and_scale(channel_major)
        )
        return channel_code.transpose(1, 2), channel_scale.transpose(1, 2)

    def integer_kernel_and_scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return signed kernel codes and effective per-channel shift scales."""

        maximum = self.kernel.detach().abs().amax(dim=(1, 2), keepdim=True)
        scale = _power_of_two_scale(maximum / max(self.weight_qmax, 1))
        code = torch.round(self.kernel / scale).clamp(
            -self.weight_qmax, self.weight_qmax
        )
        effective_scale = scale * float(2.0 ** (-self.branch_shift))
        return code, effective_scale

    def quantized_kernel(self) -> torch.Tensor:
        code, effective_scale = self.integer_kernel_and_scale()
        soft = self.kernel * float(2.0 ** (-self.branch_shift))
        return _ste(code * effective_scale, soft)

    def kernel_bitplanes(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sign bits, magnitude planes, and per-channel power-of-two scale."""

        code, effective_scale = self.integer_kernel_and_scale()
        integer = code.to(torch.int16)
        sign = integer < 0
        magnitude = integer.abs()
        planes = torch.stack(
            [
                torch.bitwise_and(
                    torch.bitwise_right_shift(magnitude, bit), 1
                ).bool()
                for bit in range(self.weight_bits)
            ],
            dim=-1,
        )
        return sign, planes, effective_scale

    def _integer_accumulator_from_codes(
        self, patch_code: torch.Tensor, kernel_code: torch.Tensor
    ) -> torch.Tensor:
        height, width = self.grid_size
        batch = patch_code.shape[0]
        grid = patch_code.to(torch.int64).transpose(1, 2).reshape(
            batch, self.dim, height, width
        )
        padded = F.pad(grid, (1, 1, 1, 1), mode="constant", value=0)
        windows = padded.unfold(2, 3, 1).unfold(3, 3, 1)
        return (
            windows
            * kernel_code.to(torch.int64)[None, :, None, None, :, :]
        ).sum(dim=(-1, -2))

    def integer_accumulator(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact depthwise accumulator and its power-of-two scale."""

        self._validate_tokens(tokens)
        patches = tokens[:, 1:]
        patch_code, input_scale = self.patch_integer_code_and_scale(patches)
        kernel_code, kernel_scale = self.integer_kernel_and_scale()
        accumulator = self._integer_accumulator_from_codes(patch_code, kernel_code)
        combined_scale = (
            input_scale.reshape(tokens.shape[0], self.dim, 1, 1)
            * kernel_scale.reshape(1, self.dim, 1, 1)
        )
        return accumulator, combined_scale

    def integer_reference(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build the hard output solely from integer payloads and shift scales."""

        self._validate_tokens(tokens)
        batch = tokens.shape[0]
        cls, patches = tokens[:, :1], tokens[:, 1:]

        patch_code, input_scale = self.patch_integer_code_and_scale(patches)
        kernel_code, kernel_scale = self.integer_kernel_and_scale()
        accumulator = self._integer_accumulator_from_codes(patch_code, kernel_code)
        branch_scale = (
            input_scale.reshape(batch, self.dim, 1, 1)
            * kernel_scale.reshape(1, self.dim, 1, 1)
        )
        height, width = self.grid_size
        hard_patch_input = patch_code.to(tokens.dtype).transpose(1, 2).reshape(
            batch, self.dim, height, width
        ) * input_scale.reshape(batch, self.dim, 1, 1)
        hard_branch = accumulator.to(tokens.dtype) * branch_scale
        patch_prequant = (hard_patch_input + hard_branch).flatten(2).transpose(1, 2)
        patch_output_code, patch_output_scale = (
            self.residual_quantizer.integer_code_and_scale(patch_prequant)
        )

        cls_input_code, cls_input_scale = self.input_quantizer.integer_code_and_scale(cls)
        hard_cls = cls_input_code * cls_input_scale
        cls_output_code, cls_output_scale = (
            self.residual_quantizer.integer_code_and_scale(hard_cls)
        )
        output_code = torch.cat((cls_output_code, patch_output_code), dim=1)
        output_scale = torch.cat((cls_output_scale, patch_output_scale), dim=1)
        output = output_code.to(tokens.dtype) * output_scale
        return {
            "output": output,
            "output_code": output_code.to(torch.int64),
            "output_scale": output_scale,
            "patch_input_code": patch_code.to(torch.int64),
            "patch_input_scale": input_scale,
            "kernel_code": kernel_code.to(torch.int64),
            "kernel_scale": kernel_scale,
            "branch_accumulator": accumulator,
            "branch_scale": branch_scale,
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate_tokens(tokens)
        batch = tokens.shape[0]
        height, width = self.grid_size
        cls, patches = tokens[:, :1], tokens[:, 1:]

        patch_code, input_scale = self.patch_integer_code_and_scale(patches)
        hard_patches = patch_code * input_scale
        quantized_patches = _ste(hard_patches, patches)
        patch_grid = quantized_patches.transpose(1, 2).reshape(
            batch, self.dim, height, width
        )
        local = F.conv2d(
            patch_grid,
            self.quantized_kernel().unsqueeze(1),
            padding=1,
            groups=self.dim,
        )
        soft_patch_output = self.residual_quantizer(
            (patch_grid + local).flatten(2).transpose(1, 2)
        )

        # CLS bypasses the local stencil.  It is still quantized at both the
        # input and output boundaries so a repeated stack has an A8 interface.
        cls_input = self.input_quantizer(cls)
        soft_cls_output = self.residual_quantizer(cls_input)
        soft_output = torch.cat((soft_cls_output, soft_patch_output), dim=1)

        reference = self.integer_reference(tokens)
        hard_output = reference["output"]
        if bool((reference["kernel_code"] == 0).all()):
            # A zero-initialized branch is an exact bypass at insertion time;
            # gradients still follow the quantized convolution surrogate.
            hard_output = tokens
        if self.training:
            return _HardPayloadSoftGradient.apply(hard_output, soft_output)
        return hard_output

    def deployment_contract(self) -> dict[str, object]:
        activation_qmax = (1 << (self.activation_bits - 1)) - 1
        maximum_accumulator = 9 * activation_qmax * self.weight_qmax
        accumulator_bits = math.ceil(math.log2(maximum_accumulator + 1)) + 1
        return {
            "token_layout": (
                f"CLS + row-major {self.grid_size[0]}x{self.grid_size[1]} patch grid"
            ),
            "cls_path": "bypass spatial stencil; signed-A requantization only",
            "spatial_operator": "zero-padded 3x3 depthwise correlation",
            "input_activation": (
                f"signed A{self.activation_bits}, one power-of-two scale per "
                "sample/channel"
            ),
            "output_activation": f"signed A{self.activation_bits}, requantized after residual add",
            "kernel_integer_range": [-self.weight_qmax, self.weight_qmax],
            "kernel_magnitude_bit_values": [
                1 << bit for bit in range(self.weight_bits)
            ],
            "kernel_scale": (
                f"per-channel signed power of two followed by >> {self.branch_shift}"
            ),
            "minimum_signed_accumulator_bits": accumulator_bits,
            "residual_add": (
                "align power-of-two exponents with shifts, add integers, then A requantize"
            ),
            "runtime": "bit-plane conditional shifts plus integer accumulation",
            "general_multipliers": 0,
        }


class RepeatedDiscreteDepthwiseLocalBranches(nn.Module):
    """Convenience stack for placing the local branch before early blocks."""

    def __init__(
        self,
        repeats: int,
        dim: int,
        grid_size: int | Sequence[int] = 8,
        weight_bits: int = 4,
        activation_bits: int = 8,
        branch_shift: int = 2,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if repeats < 1:
            raise ValueError("repeats must be positive")
        self.branches = nn.ModuleList(
            [
                DiscreteDepthwiseLocalBranch(
                    dim=dim,
                    grid_size=grid_size,
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    branch_shift=branch_shift,
                    zero_init=zero_init,
                )
                for _ in range(repeats)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for branch in self.branches:
            tokens = branch(tokens)
        return tokens

    def deployment_contract(self) -> dict[str, object]:
        branch_contract = self.branches[0].deployment_contract()
        return {
            "repeats": len(self.branches),
            "parameters_shared_across_repeats": False,
            "branch": branch_contract,
            "general_multipliers": 0,
        }
