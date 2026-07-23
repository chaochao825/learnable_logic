from __future__ import annotations

import unittest

from research_registry.capacity import bitstate_capacity
from research_registry.promotion import compare_candidate, select_results
from research_registry.validate import validate


class _Config:
    in_channels = 3
    patch_size = 4
    threshold_levels = 4
    image_size = 32
    state_width = 4096
    qk_bits = 64
    topk = 8
    message_mode = "count_threshold"
    message_count_thresholds = (1, 3, 5, 7)
    message_count_fraction = 0.25
    global_depth = 2
    local_depth = 2
    heads = 8
    encoder_kind = "redundant_predicate"


class RegistryTest(unittest.TestCase):
    def test_registry_is_internally_consistent(self) -> None:
        summary = validate()
        self.assertGreaterEqual(summary["methods"], 15)
        self.assertGreaterEqual(summary["branches"], 12)
        self.assertGreaterEqual(summary["synthesis_rows"], 6)

    def test_bitstate_capacity_exposes_message_levels(self) -> None:
        metrics = bitstate_capacity(_Config())
        self.assertEqual(metrics["input_patch_bits"], 192)
        self.assertEqual(metrics["state_storage_bits_per_token"], 4096)
        self.assertEqual(metrics["message_levels"], 5)
        self.assertEqual(metrics["fixed_xnor_per_sample"], 4_326_400)
        self.assertGreater(metrics["value_count_inputs_per_sample"], 0)
        self.assertGreater(metrics["count_threshold_outputs_per_sample"], 0)

    def test_promotion_requires_strict_paired_improvement(self) -> None:
        policy = {
            "minimum_seeds": 3,
            "minimum_seed_wins": 2,
            "minimum_mean_hard_acc_delta": 0.002,
            "maximum_single_seed_hard_acc_regression": 0.02,
            "maximum_mean_acc_gap_increase": 0.01,
            "maximum_late_validation_drop": 0.02,
            "require_finite_curve": True,
            "required_runtime_compliance": "operator_audited_bool_int",
            "require_deployment_payload_hash": True,
        }
        baseline = []
        candidate = []
        for seed, delta in enumerate((0.01, 0.02, -0.005)):
            common = {
                "protocol_id": "matched",
                "dataset": "cifar10",
                "selection_split": "validation",
                "training_health": "pass",
                "runtime_compliance": "operator_audited_bool_int",
                "float_tensor_count": "0",
                "audit_ops": "100",
                "protocol_sha256": "a" * 64,
                "deployment_payload_sha256": f"{seed + 1:064x}",
                "seed": str(seed),
                "acc_gap": "0.01",
            }
            baseline.append(
                {
                    **common,
                    "hard_acc": "0.70",
                    "best_hard_acc": "0.70",
                    "final_hard_acc": "0.70",
                }
            )
            candidate.append(
                {
                    **common,
                    "hard_acc": str(0.70 + delta),
                    "best_hard_acc": str(0.70 + delta),
                    "final_hard_acc": str(0.70 + delta),
                }
            )
        decision = compare_candidate(candidate, baseline, policy)
        self.assertTrue(decision.passed, decision.reasons)
        self.assertEqual(decision.seed_wins, 2)

    def test_promotion_variant_filter_separates_capacity_rows(self) -> None:
        rows = [
            {
                "result_id": f"candidate_{width}_s{seed}",
                "method_id": "candidate",
                "protocol_id": "matched",
                "seed": str(seed),
            }
            for width in ("w672", "w832")
            for seed in range(3)
        ]
        selected = select_results(
            rows,
            method_id="candidate",
            protocol_id="matched",
            variant="w832",
        )
        self.assertEqual(len(selected), 3)
        self.assertTrue(all("w832" in row["result_id"] for row in selected))

    def test_promotion_ignores_a_test_only_gain(self) -> None:
        policy = {
            "minimum_seeds": 3,
            "minimum_seed_wins": 2,
            "minimum_mean_hard_acc_delta": 0.002,
            "maximum_single_seed_hard_acc_regression": 0.02,
            "maximum_mean_acc_gap_increase": 0.01,
            "maximum_late_validation_drop": 0.02,
            "require_finite_curve": True,
            "required_runtime_compliance": "operator_audited_bool_int",
            "require_deployment_payload_hash": True,
        }
        baseline = []
        candidate = []
        for seed, validation_delta in enumerate((-0.0035, 0.0005, 0.0015)):
            common = {
                "protocol_id": "matched",
                "protocol_sha256": "a" * 64,
                "dataset": "cifar10",
                "selection_split": "validation",
                "training_health": "pass",
                "runtime_compliance": "operator_audited_bool_int",
                "float_tensor_count": "0",
                "audit_ops": "100",
                "deployment_payload_sha256": f"{seed + 1:064x}",
                "seed": str(seed),
                "acc_gap": "0.01",
            }
            baseline.append(
                {
                    **common,
                    "hard_acc": "0.20",
                    "best_hard_acc": "0.20",
                    "final_hard_acc": "0.20",
                    "test_hard_acc": "0.20",
                }
            )
            candidate.append(
                {
                    **common,
                    "hard_acc": str(0.20 + validation_delta),
                    "best_hard_acc": str(0.20 + validation_delta),
                    "final_hard_acc": str(0.20 + validation_delta),
                    "test_hard_acc": "0.99",
                }
            )
        decision = compare_candidate(candidate, baseline, policy)
        self.assertFalse(decision.passed)
        self.assertAlmostEqual(decision.hard_acc_delta or 0.0, -0.0005)

    def test_promotion_rejects_late_validation_collapse(self) -> None:
        policy = {
            "minimum_seeds": 3,
            "minimum_seed_wins": 2,
            "minimum_mean_hard_acc_delta": 0.002,
            "maximum_single_seed_hard_acc_regression": 0.02,
            "maximum_mean_acc_gap_increase": 0.01,
            "maximum_late_validation_drop": 0.02,
            "require_finite_curve": True,
            "required_runtime_compliance": "operator_audited_bool_int",
            "require_deployment_payload_hash": True,
        }
        baseline = []
        candidate = []
        for seed in range(3):
            common = {
                "protocol_id": "matched",
                "protocol_sha256": "a" * 64,
                "dataset": "cifar10",
                "selection_split": "validation",
                "training_health": "pass",
                "runtime_compliance": "operator_audited_bool_int",
                "float_tensor_count": "0",
                "audit_ops": "100",
                "deployment_payload_sha256": f"{seed + 1:064x}",
                "seed": str(seed),
                "acc_gap": "0.01",
            }
            baseline.append(
                {
                    **common,
                    "hard_acc": "0.70",
                    "best_hard_acc": "0.70",
                    "final_hard_acc": "0.70",
                }
            )
            candidate.append(
                {
                    **common,
                    "hard_acc": "0.71",
                    "best_hard_acc": "0.74",
                    "final_hard_acc": "0.71",
                }
            )
        decision = compare_candidate(candidate, baseline, policy)
        self.assertFalse(decision.passed)
        self.assertTrue(
            any("late validation drop" in reason for reason in decision.reasons)
        )


if __name__ == "__main__":
    unittest.main()
