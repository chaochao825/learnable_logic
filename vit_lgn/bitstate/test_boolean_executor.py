from __future__ import annotations

import ast
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from hard_lgn_gap_proto.boolean_executor import BooleanRuntimeAudit
from vit_lgn.bitstate.boolean_executor import StrictBitStateExecutor
from vit_lgn.bitstate.export_boolean_payload import export_payload
from vit_lgn.bitstate.model import BitStateConfig, BitStateViT


class BitStateBooleanExecutorTest(unittest.TestCase):
    def _model(self, message_mode: str) -> BitStateViT:
        thresholds = (1, 2) if message_mode != "majority" else (1, 3, 5, 7)
        return BitStateViT(
            BitStateConfig(
                image_size=4,
                patch_size=2,
                in_channels=1,
                threshold_levels=2,
                state_width=16,
                encoder_kind="redundant_predicate",
                predicate_fanin=3,
                encoder_identity_width=8,
                global_token_mode="learned_count",
                local_depth=1,
                global_depth=1,
                heads=2,
                qk_bits=4,
                topk=2,
                num_classes=2,
                votes_per_class=4,
                message_mode=message_mode,
                message_count_thresholds=thresholds,
                message_count_fraction=0.5,
                seed=11,
            )
        ).eval()

    def test_strict_executor_matches_model_for_both_message_modes(self) -> None:
        images = torch.randint(0, 256, (5, 1, 4, 4), dtype=torch.uint8)
        for mode in ("majority", "count_threshold_hybrid"):
            model = self._model(mode)
            expected, expected_trace = model.forward_bits(images, return_trace=True)
            executor = StrictBitStateExecutor(model.deployment_payload())
            audit = BooleanRuntimeAudit()
            with audit:
                actual, actual_trace = executor.logits(images, return_trace=True)
            self.assertGreater(audit.operations, 0)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(len(actual_trace), len(expected_trace))
            for actual_state, expected_state in zip(actual_trace, expected_trace):
                self.assertEqual(actual_state.dtype, expected_state.dtype)
                torch.testing.assert_close(actual_state, expected_state)

    def test_executor_rejects_real_input_and_payload(self) -> None:
        model = self._model("majority")
        executor = StrictBitStateExecutor(model.deployment_payload())
        with self.assertRaises(TypeError):
            executor.logits(torch.rand(2, 1, 4, 4))
        payload = model.deployment_payload()
        payload["bad"] = torch.ones(1)
        with self.assertRaises(TypeError):
            StrictBitStateExecutor(payload)

    def test_executor_source_forbids_real_arithmetic(self) -> None:
        path = Path(__file__).with_name("boolean_executor.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        self.assertFalse(
            any(
                isinstance(node, ast.Constant) and isinstance(node.value, float)
                for node in ast.walk(tree)
            )
        )
        self.assertFalse(any(isinstance(node, ast.Div) for node in ast.walk(tree)))
        forbidden = {
            "float",
            "double",
            "half",
            "bfloat16",
            "true_divide",
            "softmax",
            "sigmoid",
            "exp",
            "log",
            "pow",
        }
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(called & forbidden)

    def test_exported_artifact_is_standalone_and_contains_no_real_tensors(self) -> None:
        model = self._model("count_threshold_hybrid")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            payload_path = root / "deployment_payload.pt"
            manifest_path = root / "run_manifest.json"
            torch.save(
                {"model": model.state_dict(), "config": asdict(model.config)},
                checkpoint,
            )
            manifest_path.write_text("{}\n", encoding="utf-8")
            result = export_payload(
                checkpoint,
                payload_path,
                manifest_path,
                verify_samples=2,
            )
            self.assertFalse(result["contains_real_values"])
            self.assertTrue(result["verification"]["exact_logits"])
            self.assertEqual(result["verification"]["floating_tensor_count"], 0)
            self.assertFalse(
                any("float" in dtype for dtype in result["tensor_element_inventory"])
            )
            reloaded = torch.load(payload_path, weights_only=False)
            StrictBitStateExecutor(reloaded)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["artifacts"]["deployment_payload"]["sha256"],
                result["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
