from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from vit_lgn.full_discrete.model import DiscreteRMSNorm, FullDiscreteViT, HardXNORScoreGapAttention
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

    def test_rms_q15_retains_fractional_levels(self) -> None:
        norm = DiscreteRMSNorm(8, bits=8)
        norm.eval()
        x = torch.tensor([[0.2, 0.4, 0.8, 1.1, -0.3, -0.7, 0.6, -1.2]])
        output = norm(x)
        self.assertGreater(torch.unique(output).numel(), 4)
        soft = x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + norm.eps)
        self.assertLess(float((output - soft).abs().mean()), 0.08)

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
