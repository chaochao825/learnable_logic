from __future__ import annotations

import torch
import torch.nn as nn

from .logic_backend import ReciprocalSqrtQ15ROM, xnor_popcount_matrix
from .shiftadd import PowerOfTwoActivationQuantizer, ShiftAddLinear, _ste


_RMS_Q15_ROM = ReciprocalSqrtQ15ROM(
    maximum_index=((1 << 7) - 1) ** 2,
    zero_value=1 << 15,
)


def _nearest_rms_shift_from_sum_square(
    sum_square: torch.Tensor, dim: int, bits: int
) -> torch.Tensor:
    """Select nearest-log2 RMS exponent with exact integer comparisons."""

    rms_shift = torch.zeros_like(sum_square)
    for candidate in range(1, bits):
        threshold = int(dim) << (2 * candidate - 1)
        rms_shift += sum_square >= threshold
    return rms_shift


def _norm_statistics(
    shift_counts: torch.Tensor,
    saturated_count: torch.Tensor,
    code_count: torch.Tensor,
) -> dict[str, object]:
    return {
        "nearest_rms_shift_histogram": [
            int(value) for value in shift_counts.detach().cpu().tolist()
        ],
        "saturated_code_count": int(saturated_count.detach().cpu()),
        "code_count": int(code_count.detach().cpu()),
    }

class DiscreteRMSNorm(nn.Module):
    """Quantized RMS normalization reference with LUT-deployable reciprocal sqrt."""

    def __init__(self, dim: int, bits: int = 8, eps: float = 2**-12) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.input_quantizer = PowerOfTwoActivationQuantizer(bits)
        self.output_quantizer = PowerOfTwoActivationQuantizer(bits)
        self.register_buffer(
            "reciprocal_q15_lut",
            torch.tensor(_RMS_Q15_ROM.entries, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("shift_counts", torch.zeros(bits, dtype=torch.int64), persistent=False)
        self.register_buffer("saturated_count", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("code_count", torch.zeros((), dtype=torch.int64), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code, _ = self.input_quantizer.integer_code_and_scale(x)
        integer = code.to(torch.int64)
        sum_square = integer.square().sum(dim=-1, keepdim=True)
        mean_square = torch.div(
            sum_square + self.dim // 2, self.dim, rounding_mode="floor"
        ).clamp_min(1)
        if int(mean_square.max()) >= self.reciprocal_q15_lut.numel():
            raise ValueError("RMS mean-square exceeds the frozen Q0.15 ROM")
        reciprocal_q15 = self.reciprocal_q15_lut[mean_square]
        if not self.training:
            shift = _nearest_rms_shift_from_sum_square(
                sum_square, self.dim, self.input_quantizer.bits
            )
            self.shift_counts += torch.bincount(
                shift.reshape(-1), minlength=self.shift_counts.numel()
            ).to(self.shift_counts.device)
            self.saturated_count += (integer.abs() == self.input_quantizer.qmax).sum()
            self.code_count += integer.numel()
        product = integer * reciprocal_q15
        # Preserve the Q0.15 fractional payload until the output requantizer;
        # rounding it to a whole integer here would collapse RMSNorm to a few levels.
        hard = product.to(x.dtype) * float(2**-15)
        if self.training:
            soft = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
            hard = _ste(hard, soft)
        return self.output_quantizer(hard)

    def integer_reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code, _ = self.input_quantizer.integer_code_and_scale(x)
        integer = code.to(torch.int64)
        mean_square = torch.div(
            integer.square().sum(dim=-1, keepdim=True) + self.dim // 2,
            self.dim,
            rounding_mode="floor",
        ).clamp_min(1)
        if int(mean_square.max()) >= self.reciprocal_q15_lut.numel():
            raise ValueError("RMS mean-square exceeds the frozen Q0.15 ROM")
        reciprocal_q15 = self.reciprocal_q15_lut[mean_square]
        return integer, reciprocal_q15

    def deployment_contract(self) -> dict[str, object]:
        return {
            "kind": "rms_lut",
            "statistics": "rounded integer mean square",
            "normalizer": "unsigned Q0.15 reciprocal-square-root ROM",
            "runtime_code_product": True,
        }

    def reset_statistics(self) -> None:
        self.shift_counts.zero_()
        self.saturated_count.zero_()
        self.code_count.zero_()

    def statistics(self) -> dict[str, object]:
        return _norm_statistics(
            self.shift_counts, self.saturated_count, self.code_count
        )


class ShiftRMSNorm(nn.Module):
    """RMS-scale control using only a leading-one exponent adjustment.

    The hard value is ``A8_code * 2**(-rms_shift)``.  In an integer/exponent
    datapath this is an exponent relabel, not a multiply or a variable barrel
    shift of the payload.  The floating tensor produced here is only the QAT /
    transaction-reference carrier.  Training keeps the exact RMS surrogate
    gradient while evaluation uses the frozen integer leading-one rule.
    """

    def __init__(self, dim: int, bits: int = 8, eps: float = 2**-12) -> None:
        super().__init__()
        if bits != 8:
            raise ValueError("ShiftRMSNorm currently freezes the A8 control table")
        self.dim = int(dim)
        self.eps = float(eps)
        self.input_quantizer = PowerOfTwoActivationQuantizer(bits)
        self.output_quantizer = PowerOfTwoActivationQuantizer(bits)
        self.register_buffer("shift_counts", torch.zeros(bits, dtype=torch.int64), persistent=False)
        self.register_buffer("saturated_count", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("code_count", torch.zeros((), dtype=torch.int64), persistent=False)

    def integer_shift(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code, _ = self.input_quantizer.integer_code_and_scale(x)
        integer = code.to(torch.int64)
        sum_square = integer.square().sum(dim=-1, keepdim=True)
        # k = round(log2(RMS)), with exact half-way logarithmic cases upward.
        # Comparing S/D against 2**(2*k-1) directly avoids both division and a
        # premature rounded-mean threshold.  D is a compile-time constant.
        rms_shift = _nearest_rms_shift_from_sum_square(
            sum_square, self.dim, self.input_quantizer.bits
        )
        if not self.training:
            self.shift_counts += torch.bincount(
                rms_shift.reshape(-1), minlength=self.shift_counts.numel()
            ).to(self.shift_counts.device)
            self.saturated_count += (integer.abs() == self.input_quantizer.qmax).sum()
            self.code_count += integer.numel()
        return integer, rms_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        integer, rms_shift = self.integer_shift(x)
        # ldexp is the float-carrier spelling of exponent subtraction.  The
        # hardened payload stores rms_shift as integer control; no FP operator
        # or activation-code multiply exists in the deployed datapath.
        hard = torch.ldexp(integer.to(x.dtype), -rms_shift.to(torch.int32))
        if self.training:
            soft = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
            hard = _ste(hard, soft)
        return self.output_quantizer(hard)

    def deployment_contract(self) -> dict[str, object]:
        return {
            "kind": "shift_rms",
            "statistics": "exact integer sum_square with cross-comparison",
            "normalizer": (
                "rms_shift counts thresholds sum_square >= "
                "dimension * 2**(2*candidate-1)"
            ),
            "tie_rule": "half-way logarithmic cases round upward",
            "zero_and_subunit_policy": (
                "rms_shift saturates at zero, matching the rms_lut clamp_min(1) policy"
            ),
            "runtime": (
                "constant shift/compare control; output exponent is replaced by "
                "-rms_shift because the input exponent cancels"
            ),
            "runtime_code_product": False,
        }

    def reset_statistics(self) -> None:
        self.shift_counts.zero_()
        self.saturated_count.zero_()
        self.code_count.zero_()

    def statistics(self) -> dict[str, object]:
        return _norm_statistics(
            self.shift_counts, self.saturated_count, self.code_count
        )


class QuantizeOnlyNorm(nn.Module):
    """Matched ablation that retains only the A8 output requantizer."""

    def __init__(self, dim: int, bits: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.output_quantizer = PowerOfTwoActivationQuantizer(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_quantizer(x)

    def deployment_contract(self) -> dict[str, object]:
        return {"kind": "requant", "runtime": "A8 requantization only"}


class IdentityDiscreteNorm(nn.Module):
    """Strict no-normalization control with no arithmetic or state."""

    def __init__(self, dim: int, bits: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.bits = int(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def deployment_contract(self) -> dict[str, object]:
        return {"kind": "none", "runtime": "wire"}


def make_discrete_norm(kind: str, dim: int, bits: int) -> nn.Module:
    if kind == "rms_lut":
        return DiscreteRMSNorm(dim, bits)
    if kind == "shift_rms":
        return ShiftRMSNorm(dim, bits)
    if kind == "requant":
        return QuantizeOnlyNorm(dim, bits)
    if kind == "none":
        return IdentityDiscreteNorm(dim, bits)
    raise ValueError("norm_kind must be one of rms_lut, shift_rms, requant, none")


class DiscretePatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, channels: int, embed_dim: int,
                 weight_bits: int, activation_bits: int) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image size must be divisible by patch size")
        self.patch_size = int(patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.input_quantizer = PowerOfTwoActivationQuantizer(activation_bits)
        self.projection = ShiftAddLinear(
            channels * patch_size * patch_size,
            embed_dim,
            magnitude_bits=weight_bits,
            output_bits=activation_bits,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.input_quantizer(self.unfold(x).transpose(1, 2))
        return self.projection(patches)


class HardXNORScoreGapAttention(nn.Module):
    """Fully discrete projection, XNOR Top-K, and shift-weighted V aggregation."""

    def __init__(self, dim: int, heads: int, topk: int, weight_bits: int = 4,
                 activation_bits: int = 8, qk_lanes: int = 7,
                 gap_shift: int = 1, max_gap_bucket: int = 3,
                 gap_lut: nn.Module | None = None,
                 inference_backend: str = "fake_quant") -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dimension must be divisible by heads")
        if qk_lanes < 1:
            raise ValueError("qk_lanes must be positive")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.topk = int(topk)
        self.qk_lanes = int(qk_lanes)
        self.gap_shift = int(gap_shift)
        self.max_gap_bucket = int(max_gap_bucket)
        self.gap_lut = gap_lut
        if inference_backend not in {"fake_quant", "logic_lut"}:
            raise ValueError("inference_backend must be fake_quant or logic_lut")
        self.inference_backend = inference_backend
        self.qkv = ShiftAddLinear(
            dim, 3 * dim, weight_bits, activation_bits,
            inference_backend=inference_backend,
        )
        self.proj = ShiftAddLinear(
            dim, dim, weight_bits, activation_bits,
            inference_backend=inference_backend,
        )
        self.value_quantizer = PowerOfTwoActivationQuantizer(activation_bits)
        self.output_quantizer = PowerOfTwoActivationQuantizer(activation_bits)
        thresholds = torch.linspace(-0.75, 0.75, qk_lanes)
        self.register_buffer("threshold_fractions", thresholds, persistent=True)

    def _threshold_bits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(2**-12)
        thresholds = maximum.unsqueeze(-1) * self.threshold_fractions
        expanded = x.unsqueeze(-1)
        hard = expanded >= thresholds
        if not self.training:
            return hard, hard.to(x.dtype)
        span = (maximum / max(self.qk_lanes, 1)).unsqueeze(-1)
        soft = torch.sigmoid((expanded - thresholds) / span.clamp_min(2**-12))
        return hard, soft

    @staticmethod
    def xnor_popcount(q_bits: torch.Tensor, k_bits: torch.Tensor) -> torch.Tensor:
        q_flat = q_bits.flatten(-2).to(torch.float32)
        k_flat = k_bits.flatten(-2).to(torch.float32)
        return (
            torch.matmul(q_flat, k_flat.transpose(-1, -2))
            + torch.matmul(1.0 - q_flat, (1.0 - k_flat).transpose(-1, -2))
        ).round().to(torch.int32)

    @staticmethod
    def logic_xnor_popcount(q_bits: torch.Tensor, k_bits: torch.Tensor) -> torch.Tensor:
        q_flat = q_bits.flatten(-2)
        k_flat = k_bits.flatten(-2)
        return xnor_popcount_matrix(q_flat, k_flat).to(torch.int32)

    @staticmethod
    def stable_topk_indices(scores: torch.Tensor, count: int) -> torch.Tensor:
        """Canonical Top-K: score descending, then key index ascending."""

        return torch.argsort(
            scores, dim=-1, descending=True, stable=True
        )[..., :count]

    def set_inference_backend(self, backend: str) -> None:
        if backend not in {"fake_quant", "logic_lut"}:
            raise ValueError("backend must be fake_quant or logic_lut")
        self.inference_backend = backend
        self.qkv.set_inference_backend(backend)
        self.proj.set_inference_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        v_code, v_scale = self.value_quantizer.integer_code_and_scale(v)
        v = _ste(v_code * v_scale, v)
        q_hard, q_soft = self._threshold_bits(q)
        k_hard, k_soft = self._threshold_bits(k)
        if not self.training and self.inference_backend == "logic_lut":
            hard_scores = self.logic_xnor_popcount(q_hard, k_hard)
        else:
            hard_scores = self.xnor_popcount(q_hard, k_hard)
        if self.training:
            q_soft_flat, k_soft_flat = q_soft.flatten(-2), k_soft.flatten(-2)
            soft_scores = (
                torch.matmul(q_soft_flat, k_soft_flat.transpose(-1, -2))
                + torch.matmul(1.0 - q_soft_flat, (1.0 - k_soft_flat).transpose(-1, -2))
            )
            scores = _ste(hard_scores.to(soft_scores.dtype), soft_scores)
        else:
            scores = hard_scores.to(v.dtype)

        k_count = min(self.topk, tokens)
        # The hard selector used by QAT, ordinary eval, and logic_lut eval must
        # share the exact same tie ABI.  XNOR scores have many ties in practice.
        indices = self.stable_topk_indices(hard_scores, k_count)
        hard_selector = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
        if self.training:
            kth = scores.topk(k_count, dim=-1).values[..., -1:].detach()
            soft_selector = torch.sigmoid((scores - kth) * 0.5)
            soft_selector = soft_selector * (
                k_count / soft_selector.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            selector = _ste(hard_selector, soft_selector)
        else:
            selector = hard_selector

        minimum = torch.iinfo(torch.int64).min
        selected_scores = hard_scores.to(torch.int64).masked_fill(hard_selector == 0, minimum)
        best = selected_scores.max(dim=-1, keepdim=True).values
        raw_gap = best - hard_scores.to(torch.int64)
        if self.gap_lut is None:
            gap = torch.bitwise_right_shift(raw_gap, self.gap_shift)
            bucket = gap.clamp(0, self.max_gap_bucket)
            integer_weight = torch.bitwise_left_shift(
                torch.ones_like(bucket), self.max_gap_bucket - bucket
            ).to(v.dtype)
        else:
            integer_weight = self.gap_lut(raw_gap, head_axis=1).to(v.dtype)
        if self.training:
            routing_weight = selector * integer_weight
            soft_numerator = torch.matmul(routing_weight, v)
            soft_denominator = routing_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
            soft_aggregate = soft_numerator / soft_denominator

        # Align every V row to a common power-of-two exponent, gather only K
        # rows, and perform the hard path with integer shifts/adds/division.
        common_scale = v_scale.amin(dim=-2, keepdim=True)
        aligned_code = torch.round(v.detach() / common_scale).to(torch.int64)
        gather_index = indices.unsqueeze(-1).expand(*indices.shape, self.head_dim)
        expanded_code = aligned_code.unsqueeze(-3).expand(
            batch, self.heads, tokens, tokens, self.head_dim
        )
        selected_code = torch.gather(expanded_code, -2, gather_index)
        selected_integer_weight = torch.gather(integer_weight, -1, indices).to(torch.int64)
        integer_numerator = (
            selected_code * selected_integer_weight.unsqueeze(-1)
        ).sum(dim=-2)
        integer_denominator = selected_integer_weight.sum(dim=-1, keepdim=True).clamp_min(1)
        rounded_quotient = torch.sign(integer_numerator) * torch.div(
            integer_numerator.abs() + integer_denominator // 2,
            integer_denominator,
            rounding_mode="floor",
        )
        hard_aggregate = rounded_quotient.to(v.dtype) * common_scale
        if self.training:
            hard_aggregate = _ste(hard_aggregate, soft_aggregate)
        aggregated = self.output_quantizer(hard_aggregate)
        return self.proj(aggregated.transpose(1, 2).reshape(batch, tokens, self.dim))


class DiscreteGatedFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, weight_bits: int, activation_bits: int) -> None:
        super().__init__()
        self.gate = ShiftAddLinear(dim, hidden_dim, weight_bits, activation_bits)
        self.up = ShiftAddLinear(dim, hidden_dim, weight_bits, activation_bits)
        self.down = ShiftAddLinear(hidden_dim, dim, weight_bits, activation_bits)
        self.hidden_quantizer = PowerOfTwoActivationQuantizer(activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        hard_gate = (gate_logits >= 0).to(x.dtype)
        gate = _ste(hard_gate, torch.sigmoid(gate_logits)) if self.training else hard_gate
        hidden = self.hidden_quantizer(gate * self.up(x))
        return self.down(hidden)


class FullDiscreteBlock(nn.Module):
    def __init__(self, dim: int, heads: int, topk: int, mlp_ratio: float,
                 weight_bits: int, activation_bits: int, qk_lanes: int,
                 norm_kind: str = "rms_lut") -> None:
        super().__init__()
        self.norm1 = make_discrete_norm(norm_kind, dim, activation_bits)
        self.attn = HardXNORScoreGapAttention(
            dim, heads, topk, weight_bits, activation_bits, qk_lanes
        )
        self.norm2 = make_discrete_norm(norm_kind, dim, activation_bits)
        self.ffn = DiscreteGatedFFN(
            dim, int(dim * mlp_ratio), weight_bits, activation_bits
        )
        self.residual_quantizer = PowerOfTwoActivationQuantizer(activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_quantizer(x + self.attn(self.norm1(x)))
        return self.residual_quantizer(x + self.ffn(self.norm2(x)))


class FullDiscreteViT(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, channels: int = 3,
                 classes: int = 10, dim: int = 192, depth: int = 6, heads: int = 6,
                 topk: int = 8, mlp_ratio: float = 4.0, weight_bits: int = 4,
                 activation_bits: int = 8, qk_lanes: int = 7,
                 inference_backend: str = "fake_quant",
                 norm_kind: str = "rms_lut",
                 final_norm_kind: str | None = None) -> None:
        super().__init__()
        self.activation_bits = int(activation_bits)
        self.weight_magnitude_bits = int(weight_bits)
        self.norm_kind = str(norm_kind)
        self.final_norm_kind = (
            self.norm_kind if final_norm_kind in {None, "same"}
            else str(final_norm_kind)
        )
        self.patch_embed = DiscretePatchEmbedding(
            image_size, patch_size, channels, dim, weight_bits, activation_bits
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.parameter_quantizer = PowerOfTwoActivationQuantizer(activation_bits)
        self.token_quantizer = PowerOfTwoActivationQuantizer(activation_bits)
        self.blocks = nn.ModuleList([
            FullDiscreteBlock(
                dim, heads, topk, mlp_ratio, weight_bits, activation_bits,
                qk_lanes, norm_kind=self.norm_kind,
            )
            for _ in range(depth)
        ])
        self.norm = make_discrete_norm(self.final_norm_kind, dim, activation_bits)
        self.head = ShiftAddLinear(
            dim, classes, weight_bits, output_bits=16, input_bits=activation_bits
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        self.set_inference_backend(inference_backend)

    def set_inference_backend(self, backend: str) -> None:
        if backend not in {"fake_quant", "logic_lut"}:
            raise ValueError("backend must be fake_quant or logic_lut")
        self.inference_backend = backend
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, ShiftAddLinear):
                module.set_inference_backend(backend)
            elif isinstance(module, HardXNORScoreGapAttention):
                module.set_inference_backend(backend)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        cls = self.parameter_quantizer(self.cls_token).expand(images.shape[0], -1, -1)
        position = self.parameter_quantizer(self.position)
        tokens = self.token_quantizer(torch.cat((cls, tokens), dim=1) + position)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens)[:, 0])

    def reset_evaluation_statistics(self) -> None:
        for module in self.modules():
            if module is not self and hasattr(module, "reset_statistics"):
                module.reset_statistics()

    def evaluation_statistics(self) -> dict[str, object]:
        return {
            name: module.statistics()
            for name, module in self.named_modules()
            if module is not self and hasattr(module, "statistics")
        }

    def deployment_contract(self) -> dict[str, object]:
        linear_layers = [module for module in self.modules() if isinstance(module, ShiftAddLinear)]
        final_norm_contract = (
            self.norm.deployment_contract()
            if hasattr(self.norm, "deployment_contract")
            else {"kind": self.norm_kind}
        )
        block_norm_contract = (
            self.blocks[0].norm1.deployment_contract()
            if self.blocks and hasattr(self.blocks[0].norm1, "deployment_contract")
            else {"kind": self.norm_kind}
        )
        return {
            "all_learned_matrix_weights": (
                "signed bit-plane sums times power-of-two scales"
            ),
            "weight_magnitude_bit_values": [
                1 << bit for bit in range(self.weight_magnitude_bits)
            ],
            "linear_layer_count": len(linear_layers),
            "activation_bits": self.activation_bits,
            "classifier_output_bits": 16,
            "attention": "threshold bits + XNOR/popcount + hard Top-K + {8,4,2,1} score-gap",
            "ffn": "binary gate/MUX + shift-add bit-plane projections",
            "normalization": {
                "block": block_norm_contract,
                "final": final_norm_contract,
            },
            "general_matrix_multipliers": 0,
            "inference_backend": self.inference_backend,
            "logic_lut_scope": (
                "CPU bit-exact W-magnitude nibble LUT and packed-XNOR transaction "
                "reference; power-of-two requantization still uses a float carrier"
            ),
        }


def full_discrete_vit(**kwargs) -> FullDiscreteViT:
    return FullDiscreteViT(**kwargs)
