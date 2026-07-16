from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from vit_lgn.full_discrete.enhancements_expert import (
    BitSliceBooleanLogicExpert,
    FourModeStateSelectedFFN,
    ParallelBitSliceLogicFFN,
)
from vit_lgn.full_discrete.shiftadd import ShiftAddLinear


class BitSliceBooleanLogicExpertTest(unittest.TestCase):
    def test_hard_payload_includes_output_projection_bitplanes(self) -> None:
        expert = BitSliceBooleanLogicExpert(dim=4, expert_width=8)
        payload = expert.hard_payload()
        self.assertIn("output_projection", payload)
        self.assertIn("weight_magnitude_planes", payload["output_projection"])
        self.assertEqual(payload["output_projection"]["input_activation_bits"], 8)
    def test_signed_magnitude_bit_slice_layout_is_exact(self) -> None:
        expert = BitSliceBooleanLogicExpert(
            dim=2, expert_width=1, activation_bits=4, weight_bits=2
        )
        code = torch.tensor([[-5, 2]], dtype=torch.int64)
        actual = expert.hard_bit_slices_from_code(code)
        # Per channel: [sign, magnitude bit0, bit1, bit2].
        expected = torch.tensor([[1, 1, 0, 1, 0, 0, 1, 0]], dtype=actual.dtype)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            expert.hard_bit_slices_from_code(torch.tensor([[8, 0]]))

    def test_hard_connections_and_four_bit_lut_match_xor(self) -> None:
        expert = BitSliceBooleanLogicExpert(
            dim=1, expert_width=1, activation_bits=2, weight_bits=2
        )
        with torch.no_grad():
            expert.left_connection_logits.copy_(torch.tensor([[8.0, -8.0]]))
            expert.right_connection_logits.copy_(torch.tensor([[-8.0, 8.0]]))
            # Truth-table order is 00, 01, 10, 11.
            expert.truth_table_logits.copy_(torch.tensor([[-8.0, 8.0, 8.0, -8.0]]))
        expert.eval()
        source_bits = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        )
        expected = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        torch.testing.assert_close(expert.boolean_gate_outputs(source_bits), expected)
        payload = expert.hard_payload()
        self.assertEqual(payload["left_source"].tolist(), [0])
        self.assertEqual(payload["right_source"].tolist(), [1])
        self.assertEqual(
            payload["truth_table_00_01_10_11"].tolist(), [[0, 1, 1, 0]]
        )

    def test_train_forward_is_hard_while_ste_reaches_all_shadow_parameters(self) -> None:
        torch.manual_seed(4)
        expert = BitSliceBooleanLogicExpert(
            dim=4, expert_width=7, activation_bits=4, weight_bits=3
        )
        x = torch.randn(3, 5, 4, requires_grad=True)
        expert.train()
        train_output = expert(x)
        train_output.square().mean().backward()
        for parameter in (
            expert.left_connection_logits,
            expert.right_connection_logits,
            expert.truth_table_logits,
            expert.output_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertIsNotNone(x.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

        expert.eval()
        with torch.no_grad():
            eval_output = expert(x.detach())
        torch.testing.assert_close(train_output.detach(), eval_output)

    def test_gate_width_and_parallel_count_scale_explicitly(self) -> None:
        narrow = ParallelBitSliceLogicFFN(
            4, 8, 4, num_logic_experts=1, activation_bits=4, weight_bits=2
        )
        wide = ParallelBitSliceLogicFFN(
            4, 8, 8, num_logic_experts=2, activation_bits=4, weight_bits=2
        )
        narrow_count = sum(parameter.numel() for parameter in narrow.parameters())
        wide_count = sum(parameter.numel() for parameter in wide.parameters())
        self.assertGreater(wide_count, narrow_count)
        self.assertEqual(narrow.deployment_contract()["boolean_gate_count"], 4)
        self.assertEqual(wide.deployment_contract()["boolean_gate_count"], 16)
        output = wide(torch.randn(2, 3, 4))
        self.assertEqual(output.shape, (2, 3, 4))
        self.assertFalse(any(isinstance(module, nn.Linear) for module in wide.modules()))
        self.assertEqual(wide.deployment_contract()["general_multipliers"], 0)


class FourModeStateSelectedFFNTest(unittest.TestCase):
    def test_full_payload_and_matching_scope_are_explicit(self) -> None:
        module = FourModeStateSelectedFFN(dim=4, hidden_dim=8, expert_width=8)
        payload = module.hard_payload()
        self.assertIn("base", payload)
        self.assertIn("state_controller", payload)
        self.assertEqual(len(payload["mode_experts"]), 4)
        self.assertIn("effective-update", payload["control_comparison_warning"])
        self.assertIn("effective bank updates differ", module.matched_control_contract()["matching_scope"])
    @staticmethod
    def _small_module() -> FourModeStateSelectedFFN:
        return FourModeStateSelectedFFN(
            dim=4,
            hidden_dim=8,
            expert_width=3,
            experts_per_mode=1,
            activation_bits=4,
            weight_bits=2,
            static_state=2,
            script=(3, 1, 0, 2),
        )

    def test_dynamic_static_random_and_script_controls_are_two_bit(self) -> None:
        module = self._small_module()
        module.eval()
        x = torch.randn(3, 5, 4)

        _, dynamic = module(x, control="dynamic", return_state=True)
        _, static = module(x, control="static", return_state=True)
        _, scripted = module(x, control="script", script_step=1, return_state=True)
        supplied = torch.tensor([0, 3, 1])
        _, random_control = module(
            x, control="random", control_state=supplied, return_state=True
        )

        self.assertTrue(bool(((dynamic >= 0) & (dynamic <= 3)).all()))
        self.assertEqual(static.tolist(), [2, 2, 2])
        self.assertEqual(scripted.tolist(), [1, 1, 1])
        torch.testing.assert_close(random_control, supplied)
        expected_bits = torch.tensor([[0, 0], [1, 1], [1, 0]], dtype=torch.uint8)
        torch.testing.assert_close(module.state_bits(supplied), expected_bits)

    def test_seeded_random_control_is_replayable(self) -> None:
        module = self._small_module()
        x = torch.randn(12, 4)
        first_generator = torch.Generator().manual_seed(9)
        second_generator = torch.Generator().manual_seed(9)
        first, _ = module.resolve_hard_state(
            x, control="random", generator=first_generator
        )
        second, _ = module.resolve_hard_state(
            x, control="random", generator=second_generator
        )
        torch.testing.assert_close(first, second)

    def test_state_selects_distinct_lut_computation_not_feature_rotation(self) -> None:
        module = self._small_module()
        with torch.no_grad():
            for layer in module.base.modules():
                if isinstance(layer, ShiftAddLinear):
                    layer.weight.zero_()
            # Constant-zero versus constant-one Boolean modes.  Their identical
            # output wiring therefore emits opposite bipolar values.
            for mode, experts in enumerate(module.mode_experts):
                for expert in experts:
                    expert.truth_table_logits.fill_(-8.0 if mode == 0 else 8.0)
                    expert.output_projection.weight.fill_(1.0)
        module.eval()
        x = torch.randn(2, 3, 4)
        zero_lut = module(x, control="static", control_state=0)
        one_lut = module(x, control="static", control_state=1)
        self.assertLess(float(zero_lut.mean()), 0.0)
        self.assertGreater(float(one_lut.mean()), 0.0)
        self.assertFalse(torch.equal(zero_lut, one_lut))

    def test_matched_ste_keeps_hard_state_and_trains_controller(self) -> None:
        module = self._small_module()
        module.train()
        x = torch.randn(2, 3, 4, requires_grad=True)
        output, state = module(
            x,
            control="static",
            control_state=torch.tensor([0, 3]),
            matched_ste=True,
            return_state=True,
        )
        self.assertEqual(state.tolist(), [0, 3])
        output.mean().backward()
        self.assertIsNotNone(module.state_controller.weight.grad)
        self.assertTrue(bool(torch.isfinite(module.state_controller.weight.grad).all()))
        self.assertIsNotNone(x.grad)
        contract = module.matched_control_contract()
        self.assertTrue(contract["shared_parameter_object"])
        self.assertTrue(contract["same_four_mode_payload"])

    def test_deployment_contract_exports_real_mode_mux(self) -> None:
        module = self._small_module()
        contract = module.deployment_contract()
        self.assertEqual(contract["state_bits"], 2)
        self.assertEqual(contract["state_count"], 4)
        self.assertEqual(contract["boolean_gate_count_total"], 12)
        self.assertEqual(contract["general_multipliers"], 0)
        self.assertIn("not a feature rotation", contract["mode_action"])
        self.assertIsNot(module.mode_experts[0][0], module.mode_experts[1][0])
        self.assertFalse(any(isinstance(child, nn.Linear) for child in module.modules()))

    def test_invalid_controls_and_states_fail_closed(self) -> None:
        module = self._small_module()
        x = torch.randn(2, 4)
        with self.assertRaises(ValueError):
            module(x, control="unknown")
        with self.assertRaises(ValueError):
            module(x, control="static", control_state=4)
        with self.assertRaises(ValueError):
            module(x, control="dynamic", control_state=0)
        with self.assertRaises(ValueError):
            module.state_bits(torch.tensor([-1]))


if __name__ == "__main__":
    unittest.main()
