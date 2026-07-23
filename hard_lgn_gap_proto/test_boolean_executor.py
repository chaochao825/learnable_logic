from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from hard_lgn_gap_proto.boolean_executor import (
    BOOLEAN_GATE_TRUTH,
    BooleanRuntimeAudit,
    StrictBooleanLogicExecutor,
    boolean_gate_bank,
)
from hard_lgn_gap_proto.hard_lgn_benchmark import (
    FrozenHardLogicLayer,
    LogicNet,
    evaluate_strict_hard,
    exact_binary_inputs,
)


class BooleanExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = FrozenHardLogicLayer(
            2,
            4,
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 1, 0, 0]),
            torch.tensor([1, 6, 7, 14]),
        )
        self.second = FrozenHardLogicLayer(
            4,
            4,
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([3, 5, 6, 7]),
        )
        self.float_input = exact_binary_inputs(2)
        self.bool_input = self.float_input.to(torch.bool)

    def test_all_sixteen_boolean_gates_match_truth_payload(self) -> None:
        inputs = self.bool_input
        left = inputs[:, :1].expand(-1, 16)
        right = inputs[:, 1:].expand(-1, 16)
        operation = torch.arange(16, dtype=torch.int64)
        actual = boolean_gate_bank(left, right, operation)
        torch.testing.assert_close(actual, BOOLEAN_GATE_TRUTH.transpose(0, 1))

    def test_complete_hard_forward_is_bool_and_integer_only(self) -> None:
        executor = StrictBooleanLogicExecutor([self.first, self.second], 2)
        audit = BooleanRuntimeAudit()
        with audit:
            outputs = executor.layer_outputs(self.bool_input)
            counts = executor.class_counts(self.bool_input)
            prediction = executor.predict(self.bool_input)
        self.assertGreater(audit.operations, 0)
        self.assertTrue(all(output.dtype == torch.bool for output in outputs))
        self.assertEqual(counts.dtype, torch.int64)
        self.assertEqual(prediction.dtype, torch.int64)

    def test_integer_counts_match_legacy_float_carrier_exactly(self) -> None:
        tau = 0.5
        executor = StrictBooleanLogicExecutor([self.first, self.second], 2)
        counts = executor.class_counts(self.bool_input)
        legacy = LogicNet([self.first, self.second], 2, group_tau=tau)
        legacy_logits = legacy(self.float_input, mode="hard")
        torch.testing.assert_close(counts.to(torch.float32) / tau, legacy_logits)
        labels = legacy_logits.argmax(dim=-1)
        strict_acc, strict_loss, operations = evaluate_strict_hard(
            [self.first, self.second],
            self.float_input,
            labels,
            batch_size=2,
            num_classes=2,
            group_tau=tau,
        )
        self.assertEqual(strict_acc, 1)
        self.assertGreater(strict_loss, 0)
        self.assertGreater(operations, 0)

    def test_executor_rejects_float_input(self) -> None:
        executor = StrictBooleanLogicExecutor([self.first, self.second], 2)
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            executor.class_counts(self.float_input)

    def test_source_has_no_real_valued_arithmetic(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
