from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from hard_lgn_gap_proto.hard_lgn_benchmark import (
    CageTemperature,
    GATE_TRUTH,
    SoftLogicLayer,
    _block_metrics_from_ops,
    exact_binary_inputs,
    gate_outputs,
    refit_task_aware_layer,
)


class GateLibraryTest(unittest.TestCase):
    def test_all_sixteen_gates_match_their_boolean_truth_tables(self) -> None:
        inputs = exact_binary_inputs(2)
        actual = gate_outputs(inputs[:, 0], inputs[:, 1])
        self.assertTrue(torch.equal(actual, GATE_TRUTH.T))

    def test_hard_st_matches_hard_forward_and_keeps_logit_gradients(self) -> None:
        layer = SoftLogicLayer(
            2,
            3,
            torch.tensor([0, 0, 1]),
            torch.tensor([1, 1, 0]),
            torch.randn(3, 16, generator=torch.Generator().manual_seed(7)),
        )
        inputs = exact_binary_inputs(2)
        hard = layer(inputs, mode="hard")
        hard_st = layer(inputs, mode="hard_st", tau=1.7)
        self.assertTrue(torch.equal(hard, hard_st))
        weights = torch.arange(1, hard_st.numel() + 1, dtype=hard_st.dtype).reshape_as(hard_st)
        (hard_st * weights).sum().backward()
        self.assertIsNotNone(layer.logits.grad)
        self.assertGreater(float(layer.logits.grad.abs().sum()), 0.0)


class CageTemperatureTest(unittest.TestCase):
    def test_confidence_maps_random_logits_to_high_tau_and_committed_to_low_tau(self) -> None:
        layer = SoftLogicLayer(
            2,
            4,
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
            torch.zeros(4, 16),
        )
        cage = CageTemperature(tau_max=3.0, tau_min=0.5, beta=0.0)
        tau, confidence = cage.update([layer])
        self.assertAlmostEqual(confidence, 1.0 / 16.0, places=7)
        self.assertAlmostEqual(tau, 3.0, places=7)

        with torch.no_grad():
            layer.logits.fill_(-20.0)
            layer.logits[:, 6] = 20.0
        tau, confidence = cage.update([layer])
        self.assertGreater(confidence, 0.999)
        self.assertAlmostEqual(tau, 0.5, places=5)


class TaskAwareRefitTest(unittest.TestCase):
    def test_coordinate_refit_can_override_locally_best_but_task_wrong_gates(self) -> None:
        logits = torch.full((2, 16), -8.0)
        logits[:, 3] = 8.0  # Both relaxed gates locally prefer projection A.
        layer = SoftLogicLayer(
            2,
            2,
            torch.tensor([0, 0]),
            torch.tensor([1, 1]),
            logits,
        )
        inputs = exact_binary_inputs(2).repeat(64, 1)
        labels = (inputs[:, 0].long() ^ inputs[:, 1].long())
        args = SimpleNamespace(
            exact_truth_max=2,
            refit_samples=256,
            refit_seed=19,
            refit_validation_fraction=0.25,
            refit_candidate_topk=16,
            refit_coordinate_passes=4,
            refit_local_weight=0.0,
            refit_distill_weight=0.0,
            refit_inactive_weight=0.0,
            refit_selection_mode="task_first",
            eval_batch_size=128,
            group_tau=1.0,
        )

        hard_layer, stats = refit_task_aware_layer(
            layer,
            inputs,
            labels,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        )
        accuracy, _ = _block_metrics_from_ops(
            layer,
            inputs,
            labels,
            hard_layer.op_ids,
            num_classes=2,
            group_tau=1.0,
        )

        self.assertEqual(stats["selected_candidate"], "task_coordinate_refit")
        self.assertGreater(float(stats["coordinate_updates"]), 0)
        self.assertGreater(
            float(stats["task_coordinate_refit_validation_acc"]),
            float(stats["truth_table_refit_validation_acc"]),
        )
        self.assertAlmostEqual(accuracy, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
