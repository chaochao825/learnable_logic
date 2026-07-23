from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import torch

from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
from vit_lgn.full_discrete.export_logic_payload import export_logic_payload
from vit_lgn.full_discrete.integer_executor import (
    IntegerRuntimeAudit,
    IntegerTensor,
    StrictIntegerExecutor,
    requantize,
)
from vit_lgn.full_discrete.inspect_logic_payload import inspect_payload
from vit_lgn.full_discrete.model import FullDiscreteViT


class IntegerPrimitiveTest(unittest.TestCase):
    def test_integer_tensor_rejects_real_carriers(self) -> None:
        with self.assertRaisesRegex(TypeError, "Boolean/integer"):
            IntegerTensor(
                torch.ones(2, dtype=torch.float32),
                torch.zeros(1, dtype=torch.int32),
            )
        with self.assertRaisesRegex(TypeError, "torch.int32"):
            IntegerTensor(
                torch.ones(2, dtype=torch.int64),
                torch.zeros(1, dtype=torch.int64),
            )

    def test_requantize_uses_exact_even_ties(self) -> None:
        state = IntegerTensor(
            torch.tensor([[5, 7, -5, -7]], dtype=torch.int64),
            torch.tensor([[-1]], dtype=torch.int32),
        )
        result = requantize(state, 3, minimum_exponent=0)
        torch.testing.assert_close(
            result.code,
            torch.tensor([[2, 3, -2, -3]], dtype=torch.int64),
        )
        torch.testing.assert_close(
            result.exponent, torch.tensor([[0]], dtype=torch.int32)
        )

    def test_executor_source_contains_no_real_arithmetic_spelling(self) -> None:
        path = Path(__file__).with_name("integer_executor.py")
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
            "log2",
            "pow",
            "sqrt",
            "rsqrt",
            "sigmoid",
            "softmax",
            "ldexp",
        }
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(called & forbidden)


class StrictIntegerExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.model = FullDiscreteViT(
            image_size=8,
            patch_size=4,
            dim=8,
            depth=1,
            heads=2,
            topk=2,
            mlp_ratio=1,
            weight_bits=4,
            activation_bits=8,
            qk_lanes=3,
        ).eval()
        self.payload = export_logic_payload(self.model)
        self.images = torch.randint(
            0, 256, (2, 3, 8, 8), dtype=torch.uint8
        )

    def test_payload_declares_strict_runtime_boundary(self) -> None:
        arithmetic = self.payload["arithmetic_contract"]
        self.assertTrue(arithmetic["training_reference_uses_float_carrier"])
        self.assertTrue(arithmetic["standalone_integer_executor_included"])
        self.assertFalse(arithmetic["runtime_contains_real_values"])
        self.assertEqual(self.payload["topology"]["input_denominator"], 255)

    def test_payload_capacity_inspection_distinguishes_serialized_and_logical_bits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.pt"
            torch.save(self.payload, path)
            report = inspect_payload(path)
        capacity = report["capacity"]
        self.assertFalse(report["artifact"]["contains_real_values"])
        self.assertEqual(len(report["artifact"]["sha256"]), 64)
        self.assertEqual(capacity["general_matrix_multipliers"], 0)
        self.assertGreater(capacity["dense_fanin_max"], 0)
        self.assertGreater(capacity["dense_nonzero_fanout_max"], 0)
        self.assertGreaterEqual(
            capacity["tensor_bits"], capacity["logical_bitpacked_weight_bits"]
        )

    def test_complete_forward_creates_no_floating_tensor(self) -> None:
        executor = StrictIntegerExecutor(self.payload)
        audit = IntegerRuntimeAudit()
        with audit:
            logits = executor.forward(self.images)
        self.assertGreater(audit.operations, 0)
        self.assertEqual(logits.code.dtype, torch.int64)
        self.assertEqual(logits.exponent.dtype, torch.int32)
        self.assertEqual(tuple(logits.code.shape), (2, 10))
        self.assertTrue(executor.trace)

    def test_integer_and_float_carrier_hard_results_match(self) -> None:
        executor = StrictIntegerExecutor(self.payload)
        integer_logits = executor.forward(self.images)
        reference = self.model(self.images.to(torch.float32) / 255)
        materialized = torch.ldexp(
            integer_logits.code.to(torch.float32), integer_logits.exponent
        )
        torch.testing.assert_close(materialized, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            executor.predict(self.images), reference.argmax(dim=-1)
        )

    def test_lut_and_integer_matrix_acceleration_match(self) -> None:
        fast = StrictIntegerExecutor(self.payload, linear_backend="int_matmul")
        primitive = StrictIntegerExecutor(self.payload, linear_backend="lut")
        fast_logits = fast.forward(self.images[:1])
        primitive_logits = primitive.forward(self.images[:1])
        torch.testing.assert_close(fast_logits.code, primitive_logits.code)
        torch.testing.assert_close(fast_logits.exponent, primitive_logits.exponent)

    def test_non_uint8_input_is_rejected(self) -> None:
        executor = StrictIntegerExecutor(self.payload)
        with self.assertRaisesRegex(TypeError, "uint8"):
            executor.forward(self.images.to(torch.int16))

    def test_depthwise_local_branch_remains_integer_and_matches_reference(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=8,
            patch_size=4,
            dim=8,
            depth=1,
            heads=2,
            topk=2,
            mlp_ratio=1,
            weight_bits=4,
            activation_bits=8,
            qk_lanes=3,
            local_layers=1,
        ).eval()
        with torch.no_grad():
            model.local_branches["0"].kernel.copy_(torch.randn(8, 3, 3))
        payload = export_logic_payload(model)
        executor = StrictIntegerExecutor(payload)
        audit = IntegerRuntimeAudit()
        with audit:
            integer_logits = executor.forward(self.images[:1])
        reference = model(self.images[:1].to(torch.float32) / 255)
        materialized = torch.ldexp(
            integer_logits.code.to(torch.float32), integer_logits.exponent
        )
        torch.testing.assert_close(materialized, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
