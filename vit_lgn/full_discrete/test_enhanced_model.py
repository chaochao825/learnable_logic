from __future__ import annotations

import unittest

import torch

from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
from vit_lgn.full_discrete.shiftadd import ShiftAddLinear


class EnhancedModelTest(unittest.TestCase):
    def _model(self, **kwargs) -> EnhancedFullDiscreteViT:
        return EnhancedFullDiscreteViT(
            dim=24, depth=1, heads=3, mlp_ratio=2.0,
            weight_bits=4, activation_bits=8, **kwargs
        )

    def test_every_enhancement_family_runs_and_backpropagates(self) -> None:
        variants = [
            {"learned_gap": True},
            {"group_lut_groups": 4},
            {"local_layers": 1},
            {"logic_expert_width": 32},
            {"state_control": "dynamic", "state_expert_width": 32},
            {"state_control": "script", "state_expert_width": 32},
        ]
        for config in variants:
            model = self._model(**config)
            output = model(torch.rand(2, 3, 32, 32))
            self.assertEqual(output.shape, (2, 10), config)
            output.square().mean().backward()
            self.assertTrue(all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ), config)

    def test_ffn_families_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            self._model(group_lut_groups=4, logic_expert_width=32)

    def test_replacement_ffns_inherit_logic_lut_backend(self) -> None:
        variants = [
            {"group_lut_groups": 4},
            {"logic_expert_width": 32},
            {"state_control": "dynamic", "state_expert_width": 32},
        ]
        for config in variants:
            model = self._model(inference_backend="logic_lut", **config)
            layers = [
                module for module in model.modules()
                if isinstance(module, ShiftAddLinear)
            ]
            self.assertTrue(layers, config)
            self.assertTrue(all(
                layer.inference_backend == "logic_lut" for layer in layers
            ), config)

    def test_common_ffn_branch_preserves_baseline_initialization(self) -> None:
        for config in (
            {"group_lut_groups": 4},
            {"logic_expert_width": 32},
            {"state_control": "dynamic", "state_expert_width": 32},
        ):
            torch.manual_seed(29)
            baseline = self._model()
            torch.manual_seed(29)
            enhanced = self._model(**config)
            target = enhanced.blocks[0].ffn
            if hasattr(target, "module"):
                target = target.module
            if hasattr(target, "base"):
                target = target.base
            for name in ("gate", "up", "down"):
                torch.testing.assert_close(
                    getattr(target, name).weight,
                    getattr(baseline.blocks[0].ffn, name).weight,
                )

    def test_local_branch_is_identity_at_initialization(self) -> None:
        torch.manual_seed(11)
        baseline = self._model()
        torch.manual_seed(11)
        local = self._model(local_layers=1)
        baseline.eval(), local.eval()
        images = torch.rand(2, 3, 32, 32)
        torch.testing.assert_close(local(images), baseline(images))

    def test_local_grid_is_derived_from_patch_embedding(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=16, patch_size=4, dim=24, depth=1, heads=3,
            mlp_ratio=2.0, local_layers=1,
        )
        self.assertEqual(model.local_branches["0"].grid_size, (4, 4))
        self.assertEqual(model(torch.rand(2, 3, 16, 16)).shape, (2, 10))
        model(torch.rand(2, 3, 16, 16)).square().mean().backward()
        self.assertGreater(float(model.local_branches["0"].kernel.grad.abs().sum()), 0.0)

    def test_contract_reports_actual_gap_and_ffn(self) -> None:
        learned = self._model(learned_gap=True)
        contract = learned.deployment_contract()
        self.assertIn("monotone learned gap", contract["attention"])
        self.assertIn("table_entry_values", contract["block_contracts"][0]["gap"])
        state = self._model(state_control="script", state_expert_width=32)
        state_contract = state.deployment_contract()
        self.assertEqual(state_contract["enhancements"]["effective_state_expert_width"], 32)
        state.reset_state_statistics()
        state.eval()
        state(torch.rand(3, 3, 32, 32))
        statistics = state.state_statistics()
        self.assertEqual(sum(statistics["0"]), 3)


if __name__ == "__main__":
    unittest.main()
