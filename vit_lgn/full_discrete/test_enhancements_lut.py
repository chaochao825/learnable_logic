from __future__ import annotations

import unittest

import torch

from vit_lgn.full_discrete.enhancements_lut import (
    GroupwiseDiscreteActivationLUT,
    MonotonicHeadGapLUT,
)


class GroupwiseDiscreteActivationLUTTest(unittest.TestCase):
    def test_identity_initialization_preserves_hard_a8_carrier(self) -> None:
        module = GroupwiseDiscreteActivationLUT(features=8, groups=2)
        module.eval()
        x = torch.tensor(
            [[0.13, -0.71, 1.91, 0.0, -2.1, 0.4, 1.2, -0.03]],
            dtype=torch.float32,
        )
        code, scale = module.integer_input_code_and_scale(x)
        expected = code.to(x.dtype) * scale
        torch.testing.assert_close(module(x), expected.reshape_as(x))
        torch.testing.assert_close(module.integer_lookup(code), code)

    def test_each_group_has_an_independent_signed_int8_table(self) -> None:
        module = GroupwiseDiscreteActivationLUT(features=4, groups=2)
        with torch.no_grad():
            module.shadow_table[0].zero_()
            module.shadow_table[1].copy_(
                -torch.arange(-128, 128, dtype=module.shadow_table.dtype)
            )
        module.eval()
        x = torch.tensor([[0.25, -0.5, 0.25, -0.5]])
        code, scale = module.integer_input_code_and_scale(x)
        code_i16 = code.to(torch.int16)
        expected_code = torch.stack(
            (torch.zeros_like(code_i16[..., 0, :]), -code_i16[..., 1, :]), -2
        )
        expected_code = expected_code.clamp(-128, 127)
        torch.testing.assert_close(module(x), (expected_code * scale).reshape_as(x))

        payload = module.deployment_payload()
        table = payload["table_int8"]
        self.assertEqual(table.shape, (2, 256))
        self.assertEqual(table.dtype, torch.int8)
        self.assertEqual(module.deployment_contract()["general_multipliers"], 0)

    def test_train_forward_is_hard_but_gradients_follow_shadow(self) -> None:
        torch.manual_seed(3)
        module = GroupwiseDiscreteActivationLUT(features=6, groups=3)
        module.train()
        x = (torch.randn(4, 6) * 0.8).requires_grad_()
        train_output = module(x)
        module.eval()
        eval_output = module(x.detach())
        torch.testing.assert_close(train_output.detach(), eval_output)

        module.train()
        loss = module(x).square().mean()
        loss.backward()
        self.assertIsNotNone(module.shadow_table.grad)
        self.assertGreater(float(module.shadow_table.grad.abs().sum()), 0.0)
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_extreme_shadow_cannot_change_hard_train_forward(self) -> None:
        module = GroupwiseDiscreteActivationLUT(features=4, groups=1)
        x = torch.tensor([[0.1, -0.2, 0.5, -1.0]])
        with torch.no_grad():
            module.shadow_table.fill_(1.0e20)
        module.train()
        train_output = module(x)
        module.eval()
        torch.testing.assert_close(train_output, module(x))

    def test_relu_initialization_is_an_integer_relu(self) -> None:
        module = GroupwiseDiscreteActivationLUT(
            features=4, groups=1, initialization="relu"
        ).eval()
        x = torch.tensor([[-1.0, -0.25, 0.25, 1.0]])
        code, scale = module.integer_input_code_and_scale(x)
        expected = code.clamp_min(0).to(x.dtype) * scale
        torch.testing.assert_close(module(x), expected.reshape_as(x))

    def test_invalid_grouping_and_integer_payload_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GroupwiseDiscreteActivationLUT(features=7, groups=2)
        module = GroupwiseDiscreteActivationLUT(features=4, groups=2)
        with self.assertRaises(TypeError):
            module.integer_lookup(torch.zeros(1, 2, 2))
        with self.assertRaises(ValueError):
            module.integer_lookup(torch.full((1, 2, 2), 200, dtype=torch.int16))


class MonotonicHeadGapLUTTest(unittest.TestCase):
    def test_default_hard_table_is_allowed_and_monotonic(self) -> None:
        module = MonotonicHeadGapLUT(heads=3, max_gap=12, gap_shift=1)
        table = module.integer_table().to(torch.int16)
        allowed = torch.tensor(module.allowed_weights, dtype=torch.int16)
        self.assertTrue(bool((table.unsqueeze(-1) == allowed).any(dim=-1).all()))
        self.assertTrue(bool((table[:, 1:] <= table[:, :-1]).all()))
        torch.testing.assert_close(
            table[0, :8], torch.tensor([8, 8, 4, 4, 2, 2, 1, 1], dtype=torch.int16)
        )

    def test_monotonicity_is_structural_under_arbitrary_parameters(self) -> None:
        torch.manual_seed(5)
        module = MonotonicHeadGapLUT(heads=4, max_gap=31)
        with torch.no_grad():
            module.raw_start_headroom.normal_(0.0, 4.0)
            module.raw_drops.normal_(0.0, 4.0)
        hard = module.integer_table().to(torch.int16)
        soft = module.soft_weight_table()
        self.assertTrue(bool((hard[:, 1:] <= hard[:, :-1]).all()))
        self.assertTrue(bool((soft[:, 1:] <= soft[:, :-1] + 1e-6).all()))

    def test_per_head_lookup_and_gap_clamping(self) -> None:
        initial = torch.tensor(
            [
                [8, 4, 2, 1, 0],
                [4, 4, 1, 0, 0],
            ]
        )
        module = MonotonicHeadGapLUT(
            heads=2, max_gap=4, initial_weights=initial
        ).eval()
        gap = torch.tensor(
            [
                [
                    [[0, 1, 2], [3, 4, 99]],
                    [[0, 1, 2], [3, 4, 99]],
                ]
            ],
            dtype=torch.int16,
        )
        actual = module.integer_lookup(gap, head_axis=1)
        expected = torch.tensor(
            [
                [
                    [[8, 4, 2], [1, 0, 0]],
                    [[4, 4, 1], [0, 0, 0]],
                ]
            ],
            dtype=torch.uint8,
        )
        torch.testing.assert_close(actual, expected)

    def test_train_forward_is_hard_and_shadow_is_trainable(self) -> None:
        module = MonotonicHeadGapLUT(heads=2, max_gap=7)
        gap = torch.arange(8, dtype=torch.int16).view(1, 1, 8).repeat(3, 2, 1)
        module.train()
        train_output = module(gap, head_axis=1)
        module.eval()
        eval_output = module(gap, head_axis=1)
        torch.testing.assert_close(train_output.detach(), eval_output)

        module.train()
        module(gap, head_axis=1).sum().backward()
        self.assertIsNotNone(module.raw_start_headroom.grad)
        self.assertGreater(float(module.raw_start_headroom.grad.abs().sum()), 0.0)
        self.assertIsNotNone(module.raw_drops.grad)
        self.assertGreater(float(module.raw_drops.grad.abs().sum()), 0.0)

    def test_shift_payload_exactly_reconstructs_weight_table(self) -> None:
        module = MonotonicHeadGapLUT(
            heads=2,
            max_gap=4,
            initial_weights=[8, 4, 2, 1, 0],
        )
        payload = module.deployment_payload()
        weight = payload["weight_table_uint8"]
        shift = payload["shift_code_int8"]
        reconstructed = torch.where(
            shift < 0,
            torch.zeros_like(weight, dtype=torch.int16),
            torch.bitwise_left_shift(
                torch.ones_like(weight, dtype=torch.int16), shift.to(torch.int16)
            ),
        )
        torch.testing.assert_close(reconstructed, weight.to(torch.int16))
        contract = module.deployment_contract()
        self.assertEqual(contract["table_entry_values"], [0, 1, 2, 4, 8])
        self.assertEqual(contract["general_multipliers"], 0)

    def test_bad_initial_table_and_noninteger_gap_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MonotonicHeadGapLUT(heads=2, max_gap=3, initial_weights=[8, 4, 8, 1])
        with self.assertRaises(ValueError):
            MonotonicHeadGapLUT(heads=2, max_gap=2, initial_weights=[8, 3, 1])
        module = MonotonicHeadGapLUT(heads=2, max_gap=3)
        with self.assertRaises(TypeError):
            module(torch.zeros(1, 2, 3), head_axis=1)


if __name__ == "__main__":
    unittest.main()
