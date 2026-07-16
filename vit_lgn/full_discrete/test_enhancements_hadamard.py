from __future__ import annotations

import ast
import inspect
import unittest

import torch

from vit_lgn.full_discrete import enhancements_hadamard as hadamard_module
from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
from vit_lgn.full_discrete.enhancements_hadamard import (
    FixedHadamardGlobalMixer,
    fixed_hadamard_sign_mask,
    walsh_hadamard_transform,
)
from vit_lgn.full_discrete.model import HardXNORScoreGapAttention


class FixedHadamardGlobalMixerTest(unittest.TestCase):
    def test_fwht_is_exact_involution_for_integer_codes(self) -> None:
        torch.manual_seed(3)
        value = torch.randint(-127, 128, (2, 64, 12), dtype=torch.int64)
        transformed = walsh_hadamard_transform(value)
        recovered = walsh_hadamard_transform(transformed)
        torch.testing.assert_close(recovered, value * 64, rtol=0, atol=0)

    def test_frozen_masks_are_reproducible_and_block_specific(self) -> None:
        first = fixed_hadamard_sign_mask(64, 0)
        torch.testing.assert_close(first, fixed_hadamard_sign_mask(64, 0))
        self.assertFalse(torch.equal(first, fixed_hadamard_sign_mask(64, 1)))
        self.assertEqual(set(first.tolist()), {-1, 1})
        self.assertEqual(int(first[0]), 1)

    def test_training_value_equals_integer_reference_and_gradients_flow(self) -> None:
        torch.manual_seed(5)
        mixer = FixedHadamardGlobalMixer(
            dim=24,
            patch_tokens=16,
            block_index=2,
            activation_bits=8,
            group_size=8,
            branch_shift=2,
        ).train()
        tokens = torch.randn(2, 17, 24, requires_grad=True)
        expected = mixer.integer_reference(tokens)["output"]
        actual = mixer(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
        self.assertGreater(float(tokens.grad.abs().sum()), 0.0)

    def test_group_scales_and_deployment_cost_are_explicit(self) -> None:
        mixer = FixedHadamardGlobalMixer(
            dim=192, patch_tokens=64, block_index=0, group_size=32
        ).eval()
        tokens = torch.randn(1, 65, 192)
        reference = mixer.integer_reference(tokens)
        self.assertEqual(tuple(reference["input_scale"].shape), (1, 1, 6, 1))
        torch.testing.assert_close(mixer(tokens), reference["output"], rtol=0, atol=0)
        contract = mixer.deployment_contract()
        self.assertEqual(contract["patch_add_sub_per_channel"], 768)
        self.assertEqual(contract["runtime_scale_groups"], 6)
        self.assertEqual(contract["input_code_signed_bits"], 8)
        self.assertEqual(contract["first_butterfly_signed_bits"], 14)
        self.assertEqual(contract["second_butterfly_signed_bits"], 20)
        self.assertEqual(contract["normalized_global_signed_bits"], 14)
        self.assertEqual(contract["patch_pre_branch_signed_bits"], 15)
        self.assertEqual(contract["branch_output_accumulator_signed_bits"], 13)
        self.assertIn("requantizes to A8", contract["output_boundary"])
        self.assertEqual(contract["general_multipliers"], 0)
        self.assertEqual(contract["learned_parameters"], 0)

    def test_source_contains_no_dense_global_operator(self) -> None:
        tree = ast.parse(inspect.getsource(hadamard_module))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
        self.assertNotIn("matmul", calls)
        self.assertNotIn("einsum", calls)
        self.assertNotIn("linear", calls)
        self.assertNotIn("conv2d", calls)

    def test_enhanced_model_replaces_every_attention_block(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=16,
            patch_size=4,
            dim=24,
            depth=2,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=4,
            activation_bits=8,
            global_mixer="hadamard",
            hadamard_group_size=8,
            local_layers=1,
        )
        self.assertTrue(all(
            isinstance(block.attn, FixedHadamardGlobalMixer)
            for block in model.blocks
        ))
        self.assertFalse(any(
            isinstance(module, HardXNORScoreGapAttention)
            for module in model.modules()
        ))
        images = torch.randn(2, 3, 16, 16)
        output = model(images)
        self.assertEqual(tuple(output.shape), (2, 10))
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(model.head.weight.grad).all()))
        contract = model.deployment_contract()
        self.assertEqual(contract["attention"], "none")
        self.assertIn("H-D-H", contract["global_mixer"])

    def test_hadamard_rejects_attention_only_gap_lut(self) -> None:
        with self.assertRaisesRegex(ValueError, "learned_gap"):
            EnhancedFullDiscreteViT(
                image_size=16,
                patch_size=4,
                dim=24,
                depth=1,
                heads=3,
                topk=4,
                mlp_ratio=2.0,
                weight_bits=4,
                activation_bits=8,
                global_mixer="hadamard",
                hadamard_group_size=8,
                learned_gap=True,
            )

    def test_hybrid_keeps_periodic_content_routing_and_final_attention(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=16,
            patch_size=4,
            dim=24,
            depth=6,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=4,
            activation_bits=8,
            global_mixer="hybrid",
            hybrid_attention_period=3,
            hadamard_group_size=8,
            hadamard_branch_shift=3,
        )
        attention_indices = [
            index for index, block in enumerate(model.blocks)
            if isinstance(block.attn, HardXNORScoreGapAttention)
        ]
        hadamard_indices = [
            index for index, block in enumerate(model.blocks)
            if isinstance(block.attn, FixedHadamardGlobalMixer)
        ]
        self.assertEqual(attention_indices, [2, 5])
        self.assertEqual(hadamard_indices, [0, 1, 3, 4])
        self.assertIsInstance(model.blocks[-1].attn, HardXNORScoreGapAttention)
        contract = model.deployment_contract()
        self.assertIn("every 3 blocks", contract["attention"])
        self.assertEqual(
            sum("gap" in item for item in contract["block_contracts"]), 2
        )

    def test_parallel_keeps_all_content_routers_and_adds_fixed_mixers(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=16,
            patch_size=4,
            dim=24,
            depth=2,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=4,
            activation_bits=8,
            global_mixer="parallel",
            hadamard_group_size=8,
            hadamard_branch_shift=4,
        )
        self.assertEqual(sum(
            isinstance(module, HardXNORScoreGapAttention)
            for module in model.modules()
        ), 2)
        self.assertEqual(sum(
            isinstance(module, FixedHadamardGlobalMixer)
            for module in model.modules()
        ), 2)
        output = model(torch.randn(2, 3, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 10))
        contract = model.deployment_contract()
        self.assertIn("parallel", contract["attention"])
        self.assertTrue(all(
            "gap" in item for item in contract["block_contracts"]
        ))


if __name__ == "__main__":
    unittest.main()
