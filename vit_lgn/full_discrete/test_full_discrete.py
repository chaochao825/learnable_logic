from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn

from vit_lgn.full_discrete.model import (
    DiscreteRMSNorm,
    FullDiscreteViT,
    HardXNORScoreGapAttention,
    ShiftRMSNorm,
    _nearest_rms_shift_from_sum_square,
)
from vit_lgn.full_discrete.shiftadd import PowerOfTwoActivationQuantizer, ShiftAddLinear


class FullDiscreteTest(unittest.TestCase):
    def test_activation_codes_and_scales_are_discrete(self) -> None:
        quantizer = PowerOfTwoActivationQuantizer(bits=8)
        x = torch.tensor([[0.13, -0.71, 1.91, 0.0]])
        code, scale = quantizer.integer_code_and_scale(x)
        self.assertTrue(bool((code == code.round()).all()))
        self.assertTrue(bool((torch.log2(scale) == torch.log2(scale).round()).all()))
        torch.testing.assert_close(quantizer(x).detach(), code * scale)

    def test_shiftadd_bitplanes_reconstruct_integer_weight(self) -> None:
        layer = ShiftAddLinear(5, 3, magnitude_bits=4, output_bits=8)
        sign, planes, _ = layer.bitplane_codes()
        magnitude = sum(planes[..., bit].to(torch.int16) << bit for bit in range(4))
        reconstructed = torch.where(sign, -magnitude, magnitude)
        integer, _ = layer.integer_weight_and_scale()
        torch.testing.assert_close(reconstructed, integer.to(torch.int16))

    def test_shiftadd_float_carrier_matches_integer_accumulator(self) -> None:
        layer = ShiftAddLinear(5, 3, magnitude_bits=4, output_bits=8)
        x = torch.randn(2, 5)
        accumulator, combined_scale = layer.integer_accumulator(x)
        hard_prequant = accumulator.to(x.dtype) * combined_scale
        input_code, input_scale = layer.input_quantizer.integer_code_and_scale(x)
        weight_code, weight_scale = layer.integer_weight_and_scale()
        direct = torch.nn.functional.linear(
            input_code * input_scale, weight_code * weight_scale
        )
        torch.testing.assert_close(hard_prequant, direct)

    def test_logic_lut_shiftadd_matches_small_exact_float_carrier(self) -> None:
        for magnitude_bits in (4, 7):
            torch.manual_seed(100 + magnitude_bits)
            layer = ShiftAddLinear(
                9, 5, magnitude_bits=magnitude_bits, output_bits=8
            ).eval()
            x = torch.randn(2, 3, 9)
            expected = layer(x)
            layer.set_inference_backend("logic_lut")
            with mock.patch(
                "torch.nn.functional.linear",
                side_effect=AssertionError("logic_lut called F.linear"),
            ):
                actual = layer(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rms_q15_retains_fractional_levels(self) -> None:
        norm = DiscreteRMSNorm(8, bits=8)
        norm.eval()
        x = torch.tensor([[0.2, 0.4, 0.8, 1.1, -0.3, -0.7, 0.6, -1.2]])
        output = norm(x)
        self.assertGreater(torch.unique(output).numel(), 4)
        soft = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + norm.eps)
        self.assertLess(float((output - soft).abs().mean()), 0.08)

    def test_shift_rms_is_leading_one_exponent_control(self) -> None:
        norm = ShiftRMSNorm(8, bits=8).eval()
        x = torch.tensor([[0.2, 0.4, 0.8, 1.1, -0.3, -0.7, 0.6, -1.2]])
        integer, shift = norm.integer_shift(x)
        sum_square = integer.square().sum(-1, keepdim=True)
        expected = sum(
            sum_square >= (8 << (2 * candidate - 1))
            for candidate in range(1, 8)
        ).to(torch.int64)
        torch.testing.assert_close(shift, expected)
        with (
            mock.patch("torch.sqrt", side_effect=AssertionError("eval called sqrt")),
            mock.patch("torch.rsqrt", side_effect=AssertionError("eval called rsqrt")),
        ):
            output = norm(x)
        self.assertTrue(bool(torch.isfinite(output).all()))

        # Do not round S/D before selecting the exponent: 1.5 < S/D < 2
        # must remain below the sqrt(2) logarithmic boundary.
        crafted = torch.tensor([[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        crafted_integer = crafted.to(torch.int64)
        crafted_sum = crafted_integer.square().sum(-1, keepdim=True)
        self.assertGreater(float(crafted_sum / 8), 1.5)
        self.assertLess(float(crafted_sum / 8), 2.0)
        # Directly exercise the frozen control comparison on the exact codes.
        crafted_shift = sum(
            crafted_sum >= (8 << (2 * candidate - 1))
            for candidate in range(1, 8)
        ).to(torch.int64)
        self.assertEqual(int(crafted_shift), 0)

    def test_shift_rms_cross_comparison_boundaries_and_zero(self) -> None:
        dim = 10
        threshold = dim << (2 * 3 - 1)
        sums = torch.tensor([[0], [threshold - 1], [threshold]], dtype=torch.int64)
        shifts = _nearest_rms_shift_from_sum_square(sums, dim, bits=8)
        torch.testing.assert_close(
            shifts, torch.tensor([[0], [2], [3]], dtype=torch.int64)
        )

    def test_shift_rms_float_carrier_matches_code_exponent_pair(self) -> None:
        norm = ShiftRMSNorm(8, bits=8).eval()
        x = torch.tensor([
            [0.2, 0.4, 0.8, 1.1, -0.3, -0.7, 0.6, -1.2],
            [2.0, -0.5, 0.25, 1.0, -1.5, 0.75, -0.25, 0.5],
        ])
        code, shift = norm.integer_shift(x)
        carrier = torch.ldexp(code.to(x.dtype), -shift.to(torch.int32))
        output_code, output_scale = norm.output_quantizer.integer_code_and_scale(carrier)
        torch.testing.assert_close(output_code.to(torch.int64), code)
        torch.testing.assert_close(
            output_scale,
            torch.ldexp(torch.ones_like(output_scale), -shift.to(torch.int32)),
        )
        torch.testing.assert_close(norm(x), carrier)

    def test_norm_variants_keep_paired_matrix_initialization(self) -> None:
        matrices = []
        for norm_kind in ("rms_lut", "shift_rms", "requant", "none"):
            torch.manual_seed(1234)
            model = FullDiscreteViT(
                dim=24, depth=1, heads=3, mlp_ratio=2.0,
                weight_bits=4, norm_kind=norm_kind,
            )
            matrices.append([
                module.weight.detach().clone()
                for module in model.modules()
                if isinstance(module, ShiftAddLinear)
            ])
        for candidate in matrices[1:]:
            for expected, actual in zip(matrices[0], candidate):
                torch.testing.assert_close(actual, expected)

    def test_wmag4_and_wmag7_keep_paired_matrix_initialization(self) -> None:
        matrices = []
        for magnitude_bits in (4, 7):
            torch.manual_seed(5678)
            model = FullDiscreteViT(
                dim=24, depth=1, heads=3, mlp_ratio=2.0,
                weight_bits=magnitude_bits,
            )
            matrices.append([
                module.weight.detach().clone()
                for module in model.modules()
                if isinstance(module, ShiftAddLinear)
            ])
        for expected, actual in zip(*matrices):
            torch.testing.assert_close(actual, expected)

    def test_final_norm_can_be_removed_independently(self) -> None:
        model = FullDiscreteViT(
            dim=24,
            depth=1,
            heads=3,
            mlp_ratio=2.0,
            norm_kind="shift_rms",
            final_norm_kind="none",
        ).eval()
        self.assertEqual(model.norm_kind, "shift_rms")
        self.assertEqual(model.final_norm_kind, "none")
        self.assertEqual(
            model.deployment_contract()["normalization"]["block"]["kind"],
            "shift_rms",
        )
        self.assertEqual(
            model.deployment_contract()["normalization"]["final"]["kind"],
            "none",
        )
        logits = model(torch.rand(1, 3, 32, 32))
        self.assertEqual(logits.shape, (1, 10))

    def test_attention_is_differentiable_in_train_and_hard_in_eval(self) -> None:
        module = HardXNORScoreGapAttention(24, heads=3, topk=4)
        x = torch.randn(2, 9, 24, requires_grad=True)
        module.train()
        module(x).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.0)
        module.eval()
        with torch.no_grad():
            y = module(x.detach())
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(bool(torch.isfinite(y).all()))

    def test_xnor_gemm_matches_boolean_oracle(self) -> None:
        q = torch.tensor([[[[[True, False], [True, True]], [[False, False], [True, False]]]]])
        k = torch.tensor([[[[[True, False], [False, True]], [[False, True], [True, False]]]]])
        actual = HardXNORScoreGapAttention.xnor_popcount(q, k)
        oracle = (q.unsqueeze(-3) == k.unsqueeze(-4)).sum(dim=(-1, -2)).to(torch.int32)
        torch.testing.assert_close(actual, oracle)

    def test_hard_topk_ties_prefer_lower_key_index_in_every_backend(self) -> None:
        scores = torch.tensor([[[[7, 9, 9, 8, 9, 7]]]], dtype=torch.int32)
        expected = torch.tensor([[[[1, 2, 4, 3]]]], dtype=torch.int64)
        actual = HardXNORScoreGapAttention.stable_topk_indices(scores, 4)
        torch.testing.assert_close(actual, expected)

    def test_full_model_has_no_float_matrix_modules(self) -> None:
        model = FullDiscreteViT(dim=48, depth=2, heads=3, mlp_ratio=2.0)
        forbidden = (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.GELU, nn.Softmax)
        self.assertFalse(any(isinstance(module, forbidden) for module in model.modules()))
        logits = model(torch.rand(2, 3, 32, 32))
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(model.deployment_contract()["general_matrix_multipliers"], 0)
        for parameter in (model.cls_token, model.position):
            code, scale = model.parameter_quantizer.integer_code_and_scale(parameter)
            torch.testing.assert_close(
                model.parameter_quantizer(parameter).detach(), code * scale
            )
        self.assertEqual(model.head.input_quantizer.bits, model.activation_bits)
        self.assertEqual(model.head.output_quantizer.bits, 16)

    def test_logic_lut_full_model_avoids_float_matrix_reference_calls(self) -> None:
        model = FullDiscreteViT(
            dim=24,
            depth=1,
            heads=3,
            mlp_ratio=2.0,
            weight_bits=4,
        ).eval()
        image = torch.rand(1, 3, 32, 32)
        expected = model(image)
        model.set_inference_backend("logic_lut")
        with (
            mock.patch(
                "torch.nn.functional.linear",
                side_effect=AssertionError("logic backend called F.linear"),
            ),
            mock.patch(
                "torch.matmul",
                side_effect=AssertionError("logic backend called torch.matmul"),
            ),
            mock.patch(
                "torch.rsqrt",
                side_effect=AssertionError("logic backend called torch.rsqrt"),
            ),
        ):
            logits = model(image)
        self.assertEqual(logits.shape, (1, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        self.assertEqual(model.deployment_contract()["inference_backend"], "logic_lut")

    def test_full_model_backward_reaches_all_shiftadd_weights(self) -> None:
        model = FullDiscreteViT(dim=24, depth=1, heads=3, mlp_ratio=2.0)
        loss = model(torch.rand(2, 3, 32, 32)).square().mean()
        loss.backward()
        layers = [module for module in model.modules() if isinstance(module, ShiftAddLinear)]
        self.assertTrue(layers)
        for layer in layers:
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(bool(torch.isfinite(layer.weight.grad).all()))

    def test_shiftadd_weights_are_shared_initialization_compatible(self) -> None:
        torch.manual_seed(7)
        first = FullDiscreteViT(dim=48, depth=1, heads=3)
        torch.manual_seed(7)
        second = FullDiscreteViT(dim=48, depth=1, heads=3)
        for left, right in zip(first.state_dict().values(), second.state_dict().values()):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
