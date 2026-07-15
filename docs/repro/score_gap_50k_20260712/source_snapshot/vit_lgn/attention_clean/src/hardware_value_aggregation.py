from __future__ import annotations

import torch
import torch.nn as nn


class UnitIntervalFinalChannelQuantizer(nn.Module):
    """Uniform hard quantizer for decoded value channels with a clipped STE.

    The forward value is always one of ``2**bits`` uniformly spaced levels in
    ``[0, 1]``.  During training, gradients follow ``clamp(x, 0, 1)``; during
    evaluation the same hard value is returned without a surrogate.  Ties use
    round-half-up, so the 1-bit control is exactly ``decoded_value >= 0.5``.

    This module is intentionally applied *after* thermometer decoding.  It is
    therefore a final-channel bit-width control, unlike the legacy
    ``output_bits`` option in :class:`HardwareFriendlyTopKValueAggregator`,
    which quantizes every encoded thermometer lane before decoding.
    """

    def __init__(self, bits: int) -> None:
        super().__init__()
        if bits < 1:
            raise ValueError("bits must be positive")
        if bits > 16:
            raise ValueError("bits must not exceed 16 in the software reference")
        self.bits = int(bits)
        self.levels = (1 << self.bits) - 1

    def forward(self, decoded_values: torch.Tensor) -> torch.Tensor:
        if not decoded_values.is_floating_point():
            decoded_values = decoded_values.to(torch.float32)
        clipped = decoded_values.clamp(0.0, 1.0)
        hard = torch.floor(clipped * self.levels + 0.5) / self.levels
        if self.training and torch.is_grad_enabled() and decoded_values.requires_grad:
            return clipped + (hard - clipped).detach()
        return hard


class HardwareFriendlyTopKValueAggregator(nn.Module):
    """Low-bit Top-K value aggregation without softmax or multiplication.

    The score-gap mode maps integer XNOR/popcount score gaps to powers of two.
    A deployment implementation therefore needs only shifts, adders, and a small
    reciprocal LUT.  ``selector_mask`` may contain an STE surrogate during
    training; its hard forward value is still the exact Top-K mask.
    """

    MODES = ("uniform-mean", "score-gap-lut")

    def __init__(
        self,
        mode: str = "uniform-mean",
        gap_shift: int = 1,
        max_gap_bucket: int = 3,
        global_tail_weight: int = 0,
        global_tail_exclude_cls: bool = True,
        output_bits: int = 0,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if gap_shift < 0:
            raise ValueError("gap_shift must be non-negative")
        if gap_shift > 62:
            raise ValueError("gap_shift must not exceed 62 for int64 score arithmetic")
        if max_gap_bucket < 0:
            raise ValueError("max_gap_bucket must be non-negative")
        if max_gap_bucket > 30:
            raise ValueError("max_gap_bucket must not exceed 30 to avoid impractical/overflowing weights")
        if global_tail_weight < 0:
            raise ValueError("global_tail_weight must be non-negative")
        if global_tail_weight > (1 << 30):
            raise ValueError("global_tail_weight must not exceed 2^30")
        if output_bits < 0:
            raise ValueError("output_bits must be non-negative; use 0 to disable quantization")
        if output_bits > 16:
            raise ValueError("output_bits must not exceed 16 in the software reference")

        self.mode = mode
        self.gap_shift = int(gap_shift)
        self.max_gap_bucket = int(max_gap_bucket)
        self.global_tail_weight = int(global_tail_weight)
        self.global_tail_exclude_cls = bool(global_tail_exclude_cls)
        # Backward-compatible name: this is encoded-lane quantization, not
        # decoded final-channel quantization.  Keep it unchanged so old runs
        # using --value-output-bits remain reproducible.
        self.output_bits = int(output_bits)
        self.legacy_lane_output_bits = self.output_bits

    def _integer_candidate_weights(
        self,
        xnor_popcount: torch.Tensor,
        selector_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "uniform-mean":
            return torch.ones_like(xnor_popcount, dtype=torch.int64)

        integer_scores = xnor_popcount.to(torch.int64)
        # Calibrate gaps only against candidates that the hard forward reads.
        # For dynamic Top-K and dense-all this equals the global maximum.  For
        # static-K it removes a hidden dependency on all N scores and prevents
        # an unselected winner from saturating every selected weight.
        selected_hard = selector_mask.detach() > 0.5
        if not bool(selected_hard.any(dim=-1).all().item()):
            raise ValueError("score-gap weighting requires at least one selected key per query")
        minimum = torch.iinfo(torch.int64).min
        best_score = integer_scores.masked_fill(~selected_hard, minimum).max(
            dim=-1, keepdim=True
        ).values
        score_gap = best_score - integer_scores
        if self.gap_shift:
            score_gap = torch.bitwise_right_shift(score_gap, self.gap_shift)
        bucket = score_gap.clamp(min=0, max=self.max_gap_bucket)

        # Equivalent to 2**(-bucket) up to the common factor 2**A.  Keeping the
        # common factor makes both numerator and denominator integer-valued.
        shift = self.max_gap_bucket - bucket
        return torch.bitwise_left_shift(torch.ones_like(shift), shift)

    @staticmethod
    def _validate_shapes(
        selector_mask: torch.Tensor,
        xnor_popcount: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if selector_mask.ndim < 2 or values.ndim < 2:
            raise ValueError("selector_mask and values must each have at least two dimensions")
        if selector_mask.shape != xnor_popcount.shape:
            raise ValueError("selector_mask and xnor_popcount must have identical shapes")
        if selector_mask.shape[:-2] != values.shape[:-2]:
            raise ValueError("selector_mask and values must have identical prefix dimensions")
        if selector_mask.shape[-1] != values.shape[-2]:
            raise ValueError("selector_mask key dimension must match the values row dimension")

    def _quantize_hard_ste(self, output: torch.Tensor) -> torch.Tensor:
        """Legacy per-thermometer-lane quantizer (ties-to-even).

        Do not change its rounding rule: earlier experiments used
        ``torch.round``.  New matched final-channel controls use
        :class:`UnitIntervalFinalChannelQuantizer` after decoding instead.
        """
        if self.output_bits == 0:
            return output
        levels = (1 << self.output_bits) - 1
        clipped = output.clamp(0.0, 1.0)
        hard = torch.round(clipped * levels) / levels
        if self.training and torch.is_grad_enabled() and output.requires_grad:
            return clipped + (hard - clipped).detach()
        return hard

    def forward(
        self,
        selector_mask: torch.Tensor,
        xnor_popcount: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_shapes(selector_mask, xnor_popcount, values)

        value_dtype = values.dtype if values.is_floating_point() else torch.float32
        values_float = values.to(value_dtype)
        candidate_weights = self._integer_candidate_weights(
            xnor_popcount, selector_mask
        ).to(value_dtype)
        routing_weights = selector_mask.to(value_dtype) * candidate_weights

        numerator = torch.matmul(routing_weights, values_float)
        denominator = routing_weights.sum(dim=-1, keepdim=True)

        if self.global_tail_weight:
            tail_values = values_float
            if self.global_tail_exclude_cls and values_float.shape[-2] > 1:
                tail_values = values_float[..., 1:, :]
            global_tail = tail_values.mean(dim=-2, keepdim=True)
            numerator = numerator + float(self.global_tail_weight) * global_tail
            denominator = denominator + float(self.global_tail_weight)

        if torch.any(denominator.detach() <= 0):
            raise ValueError("every query row must select at least one value or use a positive global tail weight")

        output = numerator / denominator
        return self._quantize_hard_ste(output)

    @torch.no_grad()
    def integer_rational_reference(
        self,
        selector_mask: torch.Tensor,
        xnor_popcount: torch.Tensor,
        binary_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact int64 numerator/denominator before output quantization.

        This validates the shift/add semantics only.  It deliberately does not
        specify a fixed-point reciprocal, rounding rule, overflow policy, RTL,
        or physical implementation.
        """
        self._validate_shapes(selector_mask, xnor_popcount, binary_values)
        if not bool(torch.all((selector_mask == 0) | (selector_mask == 1)).item()):
            raise ValueError("integer reference requires a hard 0/1 selector_mask")
        if not bool(torch.all((binary_values == 0) | (binary_values == 1)).item()):
            raise ValueError("integer reference requires hard 0/1 values")

        candidate_weights = self._integer_candidate_weights(xnor_popcount, selector_mask)
        routing_weights = selector_mask.to(torch.int64) * candidate_weights
        selected_numerator = torch.matmul(routing_weights, binary_values.to(torch.int64))
        selected_denominator = routing_weights.sum(dim=-1, keepdim=True)

        if not self.global_tail_weight:
            return selected_numerator, selected_denominator

        tail_values = binary_values.to(torch.int64)
        if self.global_tail_exclude_cls and tail_values.shape[-2] > 1:
            tail_values = tail_values[..., 1:, :]
        tail_rows = int(tail_values.shape[-2])
        tail_sum = tail_values.sum(dim=-2, keepdim=True)
        numerator = selected_numerator * tail_rows + self.global_tail_weight * tail_sum
        denominator = (selected_denominator + self.global_tail_weight) * tail_rows
        return numerator, denominator

    def aggregation_cost_proxy(
        self,
        topk: int,
        value_width: int,
        num_value_rows: int = 0,
    ) -> dict[str, int | str]:
        """Return incremental aggregation counts, not full-attention cost or PPA."""
        if topk <= 0 or value_width <= 0:
            raise ValueError("topk and value_width must be positive")
        if self.global_tail_weight and num_value_rows <= int(self.global_tail_exclude_cls):
            raise ValueError("num_value_rows must cover the global-tail reduction when it is enabled")
        tail_rows = max(num_value_rows - int(self.global_tail_exclude_cls), 0)
        max_candidate_weight = (1 << self.max_gap_bucket) if self.mode == "score-gap-lut" else 1
        return {
            "scope": "value-aggregation-only; excludes QK, Top-K, gather, projection, and control",
            "value_width_semantics": "number of encoded thermometer lanes, not final decoded channels",
            "mode": self.mode,
            "selected_value_adds": max(topk - 1, 0) * value_width,
            "denominator_adds": max(topk - 1, 0),
            "shifted_value_lanes": (topk * value_width) if self.mode == "score-gap-lut" else 0,
            "gap_lut_entries": (self.max_gap_bucket + 1) if self.mode == "score-gap-lut" else 0,
            "reciprocal_lut_denominators_upper_bound": (
                topk * max_candidate_weight + self.global_tail_weight + 1
            ),
            "global_tail_enabled": int(self.global_tail_weight > 0),
            "global_tail_value_adds": (
                max(tail_rows - 1, 0) * value_width if self.global_tail_weight else 0
            ),
            "global_tail_fixed_reciprocal_lookups": int(self.global_tail_weight > 0),
            "global_tail_fusion_adds_per_query": value_width if self.global_tail_weight else 0,
            "final_reciprocal_lookups_per_query": 1,
            "output_bits": self.output_bits,
            "unmodeled": (
                "accumulator/address/data widths; tail broadcast; reciprocal multiply/scale/round; "
                "overflow policy; thermometer decode"
            ),
        }
