from __future__ import annotations

import ast
import inspect
import unittest

import torch

from vit_lgn.full_discrete.enhanced_model import (
    EnhancedFullDiscreteViT,
    ParallelContentGlobalLUTMixer,
)
from vit_lgn.full_discrete.enhancements_global_lut import (
    A8GlobalLUTTreeMixer,
    hard_group_table_lookup,
    signed_a8_lut_index,
)
from vit_lgn.full_discrete.model import HardXNORScoreGapAttention


def _round_shift(value: int, shift: int) -> int:
    if shift == 0:
        return value
    magnitude = (abs(value) + (1 << (shift - 1))) >> shift
    return -magnitude if value < 0 else magnitude


class A8GlobalLUTTreeMixerTest(unittest.TestCase):
    def test_signed_address_encoding_is_monotone_and_complete(self) -> None:
        code = torch.arange(-128, 128, dtype=torch.int64)
        torch.testing.assert_close(
            signed_a8_lut_index(code), torch.arange(256, dtype=torch.int64)
        )

    def test_hard_group_lookup_matches_independent_scalar_addresses(self) -> None:
        torch.manual_seed(2)
        table = torch.randint(-127, 128, (3, 256, 256), dtype=torch.int8)
        left = torch.randint(-127, 128, (2, 5, 3, 4), dtype=torch.int64)
        right = torch.randint(-127, 128, left.shape, dtype=torch.int64)
        actual = hard_group_table_lookup(table, left, right)
        expected = torch.empty_like(actual)
        for batch in range(left.shape[0]):
            for node in range(left.shape[1]):
                for group in range(left.shape[2]):
                    for channel in range(left.shape[3]):
                        a = int(left[batch, node, group, channel]) + 128
                        b = int(right[batch, node, group, channel]) + 128
                        expected[batch, node, group, channel] = int(table[group, a, b])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_average_reduce_projection_b_broadcast_initialization_is_exact(self) -> None:
        mixer = A8GlobalLUTTreeMixer(
            dim=4, patch_tokens=4, block_index=0, group_size=2, branch_shift=2
        ).eval()
        grouped_code = torch.empty(1, 5, 2, 2, dtype=torch.int64)
        grouped_code[:, :, 0] = 20
        grouped_code[:, :, 1] = -21
        reference = mixer.hard_reference_from_grouped_code(grouped_code)
        expected = torch.empty_like(grouped_code)
        expected[:, :, 0] = 5
        expected[:, :, 1] = -5
        torch.testing.assert_close(reference["output_code"], expected, rtol=0, atol=0)
        self.assertEqual(len(reference["reduction_levels"]), 3)

    def test_random_full_tree_matches_independent_scalar_oracle(self) -> None:
        torch.manual_seed(9)
        mixer = A8GlobalLUTTreeMixer(
            dim=4, patch_tokens=4, block_index=1, group_size=2, branch_shift=1
        ).eval()
        with torch.no_grad():
            mixer.reduce_table_shadow.copy_(
                torch.randint(-7, 8, mixer.reduce_table_shadow.shape).to(torch.float32)
            )
            mixer.context_table_shadow.copy_(
                torch.randint(-7, 8, mixer.context_table_shadow.shape).to(torch.float32)
            )
            mixer.broadcast_table_shadow.copy_(
                torch.randint(-7, 8, mixer.broadcast_table_shadow.shape).to(torch.float32)
            )
        grouped_code = torch.randint(-6, 7, (1, 5, 2, 2), dtype=torch.int64)
        payload = mixer.hard_table_payloads()
        actual = mixer.hard_reference_from_grouped_code(grouped_code)["output_code"]
        expected = torch.empty_like(actual)
        for group in range(2):
            for channel in range(2):
                nodes = [
                    int(grouped_code[0, token, group, channel])
                    for token in range(1, 5)
                ]
                for stage in range(2):
                    table = payload["reduce_table_int8"][stage, group]
                    nodes = [
                        int(table[nodes[index] + 128, nodes[index + 1] + 128])
                        for index in range(0, len(nodes), 2)
                    ]
                root = nodes[0]
                cls = int(grouped_code[0, 0, group, channel])
                context_table = payload["context_table_int8"][group]
                context = int(context_table[root + 128, cls + 128])
                broadcast = payload["broadcast_table_int8"][group]
                for token in range(5):
                    source = int(grouped_code[0, token, group, channel])
                    raw = int(broadcast[source + 128, context + 128])
                    expected[0, token, group, channel] = _round_shift(raw, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_value_is_hard_and_surrogate_reaches_every_table_family(self) -> None:
        torch.manual_seed(13)
        mixer = A8GlobalLUTTreeMixer(
            dim=8, patch_tokens=4, block_index=0, group_size=4, branch_shift=2
        ).train()
        tokens = torch.randn(2, 5, 8, requires_grad=True)
        expected = mixer.integer_reference(tokens)["output"]
        actual = mixer(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)
        for parameter in (
            mixer.reduce_table_shadow,
            mixer.context_table_shadow,
            mixer.broadcast_table_shadow,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_subthreshold_shadow_dither_preserves_init_and_exposes_hard_flips(self) -> None:
        mixer = A8GlobalLUTTreeMixer(
            dim=4, patch_tokens=4, block_index=0, group_size=2
        )
        initial = mixer.hard_table_change_statistics()
        self.assertEqual(initial["total_hard_changed_entries"], 0)
        before = mixer.hard_table_payloads()["broadcast_table_int8"]
        rounded = torch.round(mixer.broadcast_table_shadow.detach())
        margin = mixer.broadcast_table_shadow.detach() - rounded
        candidates = torch.nonzero((margin > 0.48) & (before < 127), as_tuple=False)
        self.assertGreater(candidates.shape[0], 0)
        index = tuple(int(value) for value in candidates[0])
        with torch.no_grad():
            mixer.broadcast_table_shadow[index] += 0.02
        after = mixer.hard_table_payloads()["broadcast_table_int8"]
        self.assertEqual(int(after[index]), int(before[index]) + 1)
        changed = mixer.hard_table_change_statistics()
        self.assertEqual(changed["broadcast_hard_changed_entries"], 1)
        self.assertEqual(changed["total_hard_changed_entries"], 1)

    def test_contract_counts_every_shared_rom_without_calling_it_gate_count(self) -> None:
        mixer = A8GlobalLUTTreeMixer(
            dim=384, patch_tokens=64, block_index=0, group_size=32
        )
        contract = mixer.deployment_contract()
        self.assertEqual(contract["reduction_stages"], 6)
        self.assertEqual(contract["groups"], 12)
        self.assertEqual(contract["tables_per_block"], 96)
        self.assertEqual(contract["entries_per_table"], 65_536)
        self.assertEqual(contract["hard_payload_bits"], 50_331_648)
        self.assertFalse(contract["learned_connections"])
        self.assertEqual(contract["general_multipliers_hard_forward"], 0)
        self.assertNotIn("gate_count", contract)

    def test_enhanced_model_keeps_content_router_and_adds_tree_per_block(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=8,
            patch_size=4,
            dim=8,
            depth=2,
            heads=2,
            topk=2,
            mlp_ratio=2.0,
            weight_bits=4,
            activation_bits=8,
            qk_lanes=2,
            global_mixer="parallel_lut_tree",
            global_lut_group_size=4,
            global_lut_branch_shift=2,
        )
        self.assertTrue(all(
            isinstance(block.attn, ParallelContentGlobalLUTMixer)
            for block in model.blocks
        ))
        self.assertEqual(sum(
            isinstance(module, HardXNORScoreGapAttention)
            for module in model.modules()
        ), 2)
        self.assertEqual(sum(
            isinstance(module, A8GlobalLUTTreeMixer)
            for module in model.modules()
        ), 2)
        output = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 10))
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(model.head.weight.grad).all()))
        contract = model.deployment_contract()
        self.assertIn("nonlinear A8 ROM tree", contract["attention"])
        self.assertTrue(all(
            item["global_mixer"]["operator"]
            == "parallel_content_and_a8_global_lut_tree"
            for item in contract["block_contracts"]
        ))

    def test_hard_reference_source_has_no_dense_global_primitive(self) -> None:
        source = "\n".join((
            inspect.getsource(hard_group_table_lookup),
            inspect.getsource(A8GlobalLUTTreeMixer.hard_reference_from_grouped_code),
        ))
        tree = ast.parse(source)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
        for forbidden in ("matmul", "einsum", "linear", "conv2d", "log2", "pow"):
            self.assertNotIn(forbidden, calls)

    def test_invalid_precision_and_patch_topology_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "activation_bits=8"):
            A8GlobalLUTTreeMixer(dim=8, patch_tokens=4, block_index=0,
                                activation_bits=4, group_size=4)
        with self.assertRaisesRegex(ValueError, "power of two"):
            A8GlobalLUTTreeMixer(dim=8, patch_tokens=6, block_index=0,
                                group_size=4)


if __name__ == "__main__":
    unittest.main()
