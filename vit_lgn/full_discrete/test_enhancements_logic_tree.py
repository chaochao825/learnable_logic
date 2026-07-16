from __future__ import annotations

import ast
import inspect
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_lgn.full_discrete import enhancements_logic_tree as logic_tree_module
from vit_lgn.full_discrete.enhancements_logic_tree import (
    TRUTH_NIBBLE_A,
    SharedLogicTreeConv3x3,
    hard_truth_table_gate,
)
from vit_lgn.full_discrete.shiftadd import ShiftAddLinear


class SharedLogicTreeConv3x3Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260714)

    def test_all_sixteen_packed_truth_tables_are_addressed_as_a_shift_1_or_b(self) -> None:
        left = torch.tensor([0, 0, 1, 1], dtype=torch.uint8)
        right = torch.tensor([0, 1, 0, 1], dtype=torch.uint8)
        for nibble in range(16):
            actual = hard_truth_table_gate(left, right, torch.tensor(nibble))
            expected = torch.tensor(
                [(nibble >> address) & 1 for address in range(4)], dtype=torch.bool
            )
            torch.testing.assert_close(actual, expected)
        projection_a = hard_truth_table_gate(left, right, torch.tensor(TRUTH_NIBBLE_A))
        torch.testing.assert_close(projection_a, left.bool())
        self.assertEqual(TRUTH_NIBBLE_A, 0xC)

    def test_minimal_topology_and_identity_initialization(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=5, grid_size=4)
        self.assertEqual(tuple(branch.truth_table_logits.shape), (5, 8, 7, 4))
        self.assertEqual(tuple(branch.leaf_site.shape), (5, 8, 8))
        self.assertTrue(bool((branch.leaf_site[..., 0] == 0).all()))
        self.assertTrue(bool((branch.hard_truth_nibbles() == TRUTH_NIBBLE_A).all()))
        self.assertEqual(branch.num_trees, 40)
        self.assertEqual(branch.num_unique_shared_lut_gates, 280)
        self.assertEqual(branch.gate_evaluations_per_image, 4 * 4 * 280)

        bitplanes = torch.randint(0, 2, (2, 4, 4, 5, 8), dtype=torch.bool)
        roots = branch.forward_hard_bitplanes(bitplanes)
        self.assertEqual(roots.dtype, torch.bool)
        torch.testing.assert_close(roots, bitplanes)

    def test_a8_signed_magnitude_round_trip_covers_full_code_range(self) -> None:
        codes = torch.arange(-127, 128, dtype=torch.int16).view(1, 1, -1)
        bitplanes = SharedLogicTreeConv3x3.hard_bitplanes_from_code(codes)
        reconstructed = SharedLogicTreeConv3x3.decode_signed_magnitude(bitplanes)
        torch.testing.assert_close(reconstructed.to(torch.int16), codes)
        self.assertEqual(bitplanes.shape[-1], 8)

    def test_eval_is_bit_exact_to_reference_and_cls_is_exact_bypass(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=6, grid_size=4).eval()
        tokens = torch.randn(2, 17, 6)
        tokens[:, 0] = torch.tensor([0.125, -0.25, 0.5, -1.0, 2.0, -4.0])
        with torch.no_grad():
            actual = branch(tokens)
            reference, state = branch.hard_reference(tokens, return_state=True)
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[:, 0], tokens[:, 0], rtol=0.0, atol=0.0)
        self.assertEqual(state["root_bitplanes"].dtype, torch.bool)
        self.assertGreaterEqual(int(state["root_code"].min()), -127)
        self.assertLessEqual(int(state["root_code"].max()), 127)

    def test_identity_is_exact_for_an_already_a8_quantized_carrier(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=7, grid_size=4).eval()
        tokens = torch.randn(2, 17, 7)
        with torch.no_grad():
            quantized_patches = branch.input_quantizer(tokens[:, 1:])
            carrier = torch.cat((tokens[:, :1], quantized_patches), dim=1)
            actual = branch(carrier)
        torch.testing.assert_close(actual, carrier, rtol=0.0, atol=0.0)

    def test_fixed_leaf_map_is_spatially_shared_and_has_no_learned_connections(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=3, grid_size=5)
        parameter_names = {name for name, _ in branch.named_parameters()}
        self.assertEqual(parameter_names, {"truth_table_logits"})
        self.assertIn("leaf_site", dict(branch.named_buffers()))
        self.assertFalse(branch.leaf_site.requires_grad)
        for channel in range(3):
            for bitplane in range(8):
                sites = branch.leaf_site[channel, bitplane].tolist()
                self.assertEqual(sites[0], 0)
                self.assertEqual(len(set(sites)), 8)
                self.assertEqual(set(sites[1:]), set(range(1, 9)) - {1 + ((channel + bitplane) % 8)})

        bitplanes = torch.zeros(1, 5, 5, 3, 8, dtype=torch.bool)
        bitplanes[:, 2, 2] = True
        shifted = torch.zeros_like(bitplanes)
        shifted[:, 2, 3] = True
        with torch.no_grad():
            # Make every gate OR so the neighbourhood is visible.
            branch.truth_table_logits.copy_(
                torch.tensor([-2.0, 2.0, 2.0, 2.0]).expand_as(branch.truth_table_logits)
            )
        output = branch.forward_hard_bitplanes(bitplanes)
        shifted_output = branch.forward_hard_bitplanes(shifted)
        torch.testing.assert_close(shifted_output[:, :, 1:], output[:, :, :-1])

    def test_training_keeps_hard_forward_and_reaches_all_gate_levels(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=4, grid_size=4).train()
        tokens = torch.randn(3, 17, 4, requires_grad=True)
        output = branch(tokens)
        with torch.no_grad():
            hard = branch.hard_reference(tokens)
        torch.testing.assert_close(output, hard, rtol=0.0, atol=0.0)
        output.square().mean().backward()
        gradient = branch.truth_table_logits.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool((gradient.abs().sum(dim=(0, 1, 3)) > 0).all()))
        self.assertIsNotNone(tokens.grad)
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)

    def test_module_has_no_projection_or_general_matrix_layer(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=8, grid_size=4)
        forbidden_modules = (nn.Linear, nn.Conv2d, ShiftAddLinear)
        self.assertFalse(
            any(isinstance(module, forbidden_modules) for module in branch.modules())
        )
        contract = branch.deployment_contract()
        self.assertEqual(contract["gates_per_tree"], 7)
        self.assertEqual(contract["root_state"], "boolean_bitplane_inside_tree_core")
        self.assertFalse(contract["learned_routing"])

    def test_randomized_tree_matches_independent_scalar_oracle(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=2, grid_size=2)
        leaves = torch.randint(0, 2, (3, 2, 2, 2, 8, 8), dtype=torch.bool)
        nibbles = torch.randint(0, 16, (2, 8, 7), dtype=torch.uint8)
        actual = branch._hard_tree(leaves, nibbles)
        expected = torch.empty_like(actual)

        def scalar_gate(left: bool, right: bool, nibble: int) -> bool:
            address = (int(left) << 1) | int(right)
            return bool((nibble >> address) & 1)

        for batch in range(3):
            for row in range(2):
                for col in range(2):
                    for channel in range(2):
                        for bitplane in range(8):
                            nodes = [
                                bool(value)
                                for value in leaves[
                                    batch, row, col, channel, bitplane
                                ].tolist()
                            ]
                            tables = nibbles[channel, bitplane].tolist()
                            nodes.extend([
                                scalar_gate(nodes[0], nodes[1], tables[0]),
                                scalar_gate(nodes[2], nodes[3], tables[1]),
                                scalar_gate(nodes[4], nodes[5], tables[2]),
                                scalar_gate(nodes[6], nodes[7], tables[3]),
                            ])
                            nodes.extend([
                                scalar_gate(nodes[8], nodes[9], tables[4]),
                                scalar_gate(nodes[10], nodes[11], tables[5]),
                            ])
                            expected[batch, row, col, channel, bitplane] = scalar_gate(
                                nodes[12], nodes[13], tables[6]
                            )
        torch.testing.assert_close(actual, expected)

    def test_evaluation_statistics_measure_hard_lut_and_root_changes(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=2, grid_size=2).eval()
        branch.reset_statistics()
        tokens = torch.randn(3, 5, 2)
        with torch.no_grad():
            branch(tokens)
        identity = branch.statistics()
        self.assertEqual(identity["hard_lut_nibbles_not_0xC"], 0)
        self.assertEqual(identity["root_code_changed"], 0)
        self.assertEqual(identity["root_code_total"], 3 * 4 * 2)
        with torch.no_grad():
            branch.truth_table_logits[..., 0, :].copy_(
                torch.tensor([-2.0, 2.0, 2.0, 2.0])
            )
        branch.reset_statistics()
        with torch.no_grad():
            branch(tokens)
        changed = branch.statistics()
        self.assertGreater(changed["hard_lut_nibbles_not_0xC"], 0)
        self.assertGreater(changed["root_code_changed"], 0)

    def test_hard_branch_does_not_call_matrix_or_convolution_operators(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=4, grid_size=4).eval()
        tokens = torch.randn(1, 17, 4)

        def forbidden(*args, **kwargs):
            del args, kwargs
            raise AssertionError("forbidden dense/spatial arithmetic primitive")

        with (
            mock.patch.object(torch, "matmul", forbidden),
            mock.patch.object(torch, "mm", forbidden),
            mock.patch.object(torch, "bmm", forbidden),
            mock.patch.object(torch, "einsum", forbidden),
            mock.patch.object(F, "linear", forbidden),
            mock.patch.object(F, "conv2d", forbidden),
        ):
            output = branch(tokens)
        self.assertEqual(output.shape, tokens.shape)

    def test_source_ast_contains_no_dense_or_convolution_call(self) -> None:
        tree = ast.parse(inspect.getsource(logic_tree_module))
        forbidden = {"linear", "matmul", "mm", "bmm", "addmm", "einsum", "conv1d", "conv2d", "conv3d"}
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Attribute):
                    calls.append(function.attr)
                elif isinstance(function, ast.Name):
                    calls.append(function.id)
            self.assertNotIsInstance(node, ast.MatMult)
        self.assertFalse(forbidden.intersection(calls), sorted(forbidden.intersection(calls)))


if __name__ == "__main__":
    unittest.main()
