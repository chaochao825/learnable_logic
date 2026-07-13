from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from vit_lgn.full_discrete.enhancements_spatial import (
    DiscreteDepthwiseLocalBranch,
    RepeatedDiscreteDepthwiseLocalBranches,
)


class DiscreteDepthwiseLocalBranchTest(unittest.TestCase):
    def test_inexact_wide_accumulator_configs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DiscreteDepthwiseLocalBranch(dim=2, weight_bits=8, activation_bits=8)
        with self.assertRaises(ValueError):
            DiscreteDepthwiseLocalBranch(dim=2, weight_bits=4, activation_bits=16)
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_cls_plus_8x8_shape_and_validation(self) -> None:
        branch = DiscreteDepthwiseLocalBranch(dim=12)
        tokens = torch.randn(2, 65, 12)
        self.assertEqual(branch(tokens).shape, tokens.shape)
        with self.assertRaisesRegex(ValueError, "expected CLS"):
            branch(torch.randn(2, 64, 12))
        with self.assertRaisesRegex(ValueError, "channel dimension"):
            branch(torch.randn(2, 65, 13))

    def test_kernel_bitplanes_reconstruct_integer_kernel(self) -> None:
        branch = DiscreteDepthwiseLocalBranch(
            dim=5, weight_bits=4, branch_shift=1, zero_init=False
        )
        sign, planes, scale = branch.kernel_bitplanes()
        magnitude = sum(
            planes[..., bit].to(torch.int16) << bit for bit in range(4)
        )
        reconstructed = torch.where(sign, -magnitude, magnitude)
        code, expected_scale = branch.integer_kernel_and_scale()
        torch.testing.assert_close(reconstructed, code.to(torch.int16))
        torch.testing.assert_close(scale, expected_scale)
        self.assertTrue(bool((torch.log2(scale) == torch.log2(scale).round()).all()))

    def test_eval_hard_path_matches_integer_reference(self) -> None:
        branch = DiscreteDepthwiseLocalBranch(
            dim=7, weight_bits=4, activation_bits=8, branch_shift=2, zero_init=False
        ).eval()
        tokens = torch.randn(3, 65, 7)
        with torch.no_grad():
            actual = branch(tokens)
            reference = branch.integer_reference(tokens)
        torch.testing.assert_close(actual, reference["output"], rtol=0.0, atol=0.0)
        self.assertGreaterEqual(int(reference["output_code"].min()), -127)
        self.assertLessEqual(int(reference["output_code"].max()), 127)
        self.assertTrue(
            bool(
                (
                    torch.log2(reference["output_scale"])
                    == torch.log2(reference["output_scale"]).round()
                ).all()
            )
        )

    def test_local_stencil_never_reads_cls_and_has_3x3_support(self) -> None:
        branch = DiscreteDepthwiseLocalBranch(
            dim=2, branch_shift=0, zero_init=True
        ).eval()
        with torch.no_grad():
            branch.kernel.fill_(1.0)
        baseline = torch.zeros(1, 65, 2)
        baseline[:, 0] = torch.tensor([0.25, -0.5])
        perturbed = baseline.clone()
        source_row, source_col = 3, 3
        source_token = 1 + source_row * 8 + source_col
        perturbed[:, source_token, 0] = 1.0
        with torch.no_grad():
            base_output = branch(baseline)
            changed_output = branch(perturbed)
        torch.testing.assert_close(changed_output[:, 0], base_output[:, 0])

        difference = (changed_output - base_output).abs().sum(dim=-1)[0, 1:]
        changed = set(torch.nonzero(difference, as_tuple=False).flatten().tolist())
        expected = {
            row * 8 + col
            for row in range(source_row - 1, source_row + 2)
            for col in range(source_col - 1, source_col + 2)
        }
        self.assertEqual(changed, expected)

    def test_repeated_branches_are_shape_preserving_and_trainable(self) -> None:
        stack = RepeatedDiscreteDepthwiseLocalBranches(
            repeats=3, dim=6, branch_shift=2, zero_init=True
        )
        tokens = torch.randn(2, 65, 6, requires_grad=True)
        output = stack(tokens)
        self.assertEqual(output.shape, tokens.shape)
        output.square().mean().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)
        for branch in stack.branches:
            self.assertIsNotNone(branch.kernel.grad)
            self.assertTrue(bool(torch.isfinite(branch.kernel.grad).all()))

    def test_module_tree_contains_no_general_matrix_layer(self) -> None:
        branch = DiscreteDepthwiseLocalBranch(dim=8)
        forbidden = (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.GELU, nn.Softmax)
        self.assertFalse(any(isinstance(module, forbidden) for module in branch.modules()))
        contract = branch.deployment_contract()
        self.assertEqual(contract["general_multipliers"], 0)
        self.assertEqual(contract["kernel_magnitude_bit_values"], [1, 2, 4, 8])
        self.assertEqual(contract["minimum_signed_accumulator_bits"], 16)
        self.assertIn("requantized", contract["output_activation"])

    def test_fixed_seed_initialization_is_reproducible(self) -> None:
        torch.manual_seed(23)
        first = DiscreteDepthwiseLocalBranch(dim=4, zero_init=False)
        torch.manual_seed(23)
        second = DiscreteDepthwiseLocalBranch(dim=4, zero_init=False)
        torch.testing.assert_close(first.kernel, second.kernel)


if __name__ == "__main__":
    unittest.main()
