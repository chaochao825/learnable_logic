"""Fixed integer Walsh--Hadamard global token mixing.

The deployed operator is ``H D H / N`` over patch tokens.  ``H`` is a
Sylvester Walsh--Hadamard transform and ``D`` is a block-specific frozen sign
mask.  Both transforms are butterfly networks containing only additions and
subtractions; ``D`` is conditional negation and division by the power-of-two
patch count is a rounded right shift.

The CLS token receives the rounded patch mean and is broadcast back to every
patch.  Consequently every block has bidirectional global communication while
preserving the original row-major patch coordinates for the local 3x3 branch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .shiftadd import _power_of_two_scale, _ste


def walsh_hadamard_transform(tokens: torch.Tensor) -> torch.Tensor:
    """Unnormalised FWHT along token axis 1 without matrix multiplication."""

    if tokens.ndim < 2:
        raise ValueError("FWHT input must have a token axis")
    length = tokens.shape[1]
    if length < 1 or length & (length - 1):
        raise ValueError("FWHT token length must be a positive power of two")
    output = tokens
    stride = 1
    while stride < length:
        shape = (
            output.shape[0],
            length // (2 * stride),
            2,
            stride,
            *output.shape[2:],
        )
        pair = output.reshape(shape)
        left, right = pair[:, :, 0], pair[:, :, 1]
        output = torch.stack((left + right, left - right), dim=2).reshape_as(output)
        stride <<= 1
    return output


def rounded_power_of_two_divide(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Signed nearest rounding with half-way magnitudes rounded away from zero."""

    if shift < 0:
        raise ValueError("right-shift amount must be non-negative")
    if shift == 0:
        return value
    offset = 1 << (shift - 1)
    magnitude = torch.div(value.abs() + offset, 1 << shift, rounding_mode="floor")
    return torch.where(value < 0, -magnitude, magnitude)


def fixed_hadamard_sign_mask(length: int, block_index: int) -> torch.Tensor:
    """Generate a reproducible balanced-looking Rademacher mask with xorshift32."""

    if length < 2 or length & (length - 1):
        raise ValueError("mask length must be a power of two greater than one")
    state = (0x9E3779B9 ^ ((int(block_index) + 1) * 0x85EBCA6B)) & 0xFFFFFFFF
    values: list[int] = []
    for _ in range(length):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        values.append(1 if state & 1 else -1)
    # Preserve a positive DC convention and rule out the degenerate all-equal
    # mask even if a future generator change is made.
    values[0] = 1
    if abs(sum(values)) == length:
        values[-1] = -values[-1]
    return torch.tensor(values, dtype=torch.int8)


class FixedHadamardGlobalMixer(nn.Module):
    """A8-input, parameter-free global mixer with a wide integer reference."""

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
        if activation_bits < 2 or activation_bits > 16:
            raise ValueError("activation_bits must be in [2,16]")
        if patch_tokens < 2 or patch_tokens & (patch_tokens - 1):
            raise ValueError("patch_tokens must be a power of two greater than one")
        if group_size < 1 or dim % group_size:
            raise ValueError("group_size must be positive and divide dim")
        if branch_shift < 0 or branch_shift > 8:
            raise ValueError("branch_shift must be in [0,8]")
        self.dim = int(dim)
        self.patch_tokens = int(patch_tokens)
        self.block_index = int(block_index)
        self.activation_bits = int(activation_bits)
        self.group_size = int(group_size)
        self.groups = self.dim // self.group_size
        self.branch_shift = int(branch_shift)
        self.normalization_shift = int(math.log2(self.patch_tokens))
        self.qmax = (1 << (self.activation_bits - 1)) - 1
        self.qmin = -self.qmax
        self.first_butterfly_signed_bits = (
            self.activation_bits + self.normalization_shift
        )
        self.second_butterfly_signed_bits = (
            self.activation_bits + 2 * self.normalization_shift
        )
        self.normalized_global_signed_bits = self.first_butterfly_signed_bits
        maximum_pre_branch = (self.patch_tokens + 1) * self.qmax
        self.patch_pre_branch_signed_bits = maximum_pre_branch.bit_length() + 1
        if self.branch_shift:
            maximum_branch_output = (
                maximum_pre_branch + (1 << (self.branch_shift - 1))
            ) >> self.branch_shift
        else:
            maximum_branch_output = maximum_pre_branch
        self.branch_output_accumulator_signed_bits = (
            maximum_branch_output.bit_length() + 1
        )
        self.register_buffer(
            "sign_mask",
            fixed_hadamard_sign_mask(self.patch_tokens, self.block_index),
            persistent=True,
        )

    def _validate(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch,1+patch_tokens,dim]")
        if tokens.shape[1:] != (self.patch_tokens + 1, self.dim):
            raise ValueError(
                "tokens must have shape "
                f"[batch,{self.patch_tokens + 1},{self.dim}]"
            )

    def integer_code_and_scale(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize with one runtime power-of-two scale per 32-channel group."""

        self._validate(tokens)
        grouped = tokens.detach().reshape(
            tokens.shape[0], tokens.shape[1], self.groups, self.group_size
        )
        maximum = grouped.abs().amax(dim=(1, 3), keepdim=True)
        scale = _power_of_two_scale(maximum / max(self.qmax, 1))
        code = torch.round(grouped / scale).clamp(self.qmin, self.qmax)
        return code.to(torch.int64).reshape_as(tokens), scale

    def integer_reference(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the exact add/sub/sign/shift branch update and its payloads."""

        code, scale = self.integer_code_and_scale(tokens)
        cls_code, patch_code = code[:, 0], code[:, 1:]
        first = walsh_hadamard_transform(patch_code)
        signed = first * self.sign_mask.to(first.dtype)[None, :, None]
        second = walsh_hadamard_transform(signed)
        global_patch_code = rounded_power_of_two_divide(
            second, self.normalization_shift
        )
        patch_mean_code = rounded_power_of_two_divide(
            patch_code.sum(dim=1), self.normalization_shift
        )
        patch_update_code = rounded_power_of_two_divide(
            global_patch_code + cls_code[:, None, :], self.branch_shift
        )
        cls_update_code = rounded_power_of_two_divide(
            patch_mean_code, self.branch_shift
        )[:, None, :]
        update_code = torch.cat((cls_update_code, patch_update_code), dim=1)
        grouped_update = update_code.reshape(
            tokens.shape[0], tokens.shape[1], self.groups, self.group_size
        )
        output = (grouped_update.to(tokens.dtype) * scale).reshape_as(tokens)
        return {
            "output": output,
            "output_code": update_code,
            "input_code": code,
            "input_scale": scale,
            "first_butterfly": first,
            "second_butterfly": second,
            "global_patch_code": global_patch_code,
            "patch_mean_code": patch_mean_code,
        }

    def soft_reference(self, tokens: torch.Tensor) -> torch.Tensor:
        """Differentiable arithmetic analogue used only for the backward path."""

        cls, patches = tokens[:, :1], tokens[:, 1:]
        first = walsh_hadamard_transform(patches)
        signed = first * self.sign_mask.to(tokens.dtype)[None, :, None]
        global_patches = walsh_hadamard_transform(signed) / self.patch_tokens
        patch_mean = patches.mean(dim=1, keepdim=True)
        scale = float(1 << self.branch_shift)
        return torch.cat(
            (patch_mean / scale, (global_patches + cls) / scale), dim=1
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hard = self.integer_reference(tokens)["output"]
        if not self.training:
            return hard
        return _ste(hard, self.soft_reference(tokens))

    def deployment_contract(self) -> dict[str, object]:
        return {
            "operator": "fixed_hadamard_sign_hadamard_global_mixer",
            "patch_tokens": self.patch_tokens,
            "block_index": self.block_index,
            "activation_bits": self.activation_bits,
            "group_size": self.group_size,
            "runtime_scale_groups": self.groups,
            "normalization_right_shift": self.normalization_shift,
            "branch_right_shift": self.branch_shift,
            "rounding": "nearest, half-way magnitude away from zero",
            "sign_mask": [int(value) for value in self.sign_mask.cpu().tolist()],
            "input_code_signed_bits": self.activation_bits,
            "first_butterfly_signed_bits": self.first_butterfly_signed_bits,
            "second_butterfly_signed_bits": self.second_butterfly_signed_bits,
            "normalized_global_signed_bits": self.normalized_global_signed_bits,
            "patch_pre_branch_signed_bits": self.patch_pre_branch_signed_bits,
            "branch_output_accumulator_signed_bits": (
                self.branch_output_accumulator_signed_bits
            ),
            "output_boundary": (
                "wide branch code; enclosing residual merge requantizes to A8"
            ),
            "patch_add_sub_per_channel": (
                2 * self.patch_tokens * self.normalization_shift
            ),
            "learned_parameters": 0,
            "general_multipliers": 0,
            "global_path": "two FWHTs + fixed sign XOR/negate + shifts",
            "cls_path": "rounded patch mean to CLS; CLS broadcast to patches",
        }
