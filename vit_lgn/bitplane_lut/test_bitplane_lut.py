from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from vit_lgn.bitplane_lut.executor import (
    BooleanRuntimeAudit,
    StrictBitPlaneLUTExecutor,
    validate_hard_payload,
)
from vit_lgn.bitplane_lut.layers import (
    LearnableLUTLayer,
    bitplanes_to_uint8,
    sequence_candidate_indices,
    spatial_candidate_indices,
    uint8_to_bitplanes,
)
from vit_lgn.bitplane_lut.model import (
    BitPlaneLUTClassifier,
    boolean_state_diagnostics,
)
from vit_lgn.bitplane_lut.train_shakespeare import load_hard_prefix
from vit_lgn.bitplane_lut.synthesize_payloads import (
    essential_input_count,
    realized_truth,
)


def _select_source(layer: LearnableLUTLayer, slot: int, source: int) -> None:
    with torch.no_grad():
        match = layer.candidate_indices[:, slot] == source
        if not bool(match.any(dim=1).all()):
            raise AssertionError("source is absent from a candidate pool")
        choice = match.to(torch.int64).argmax(dim=1)
        layer.wiring_logits[:, slot].fill_(-8.0)
        layer.wiring_logits[:, slot].scatter_(1, choice.unsqueeze(1), 8.0)


class BitPlaneCodecTest(unittest.TestCase):
    def test_all_a8_codes_round_trip(self) -> None:
        symbols = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        planes = uint8_to_bitplanes(symbols)
        self.assertEqual(planes.dtype, torch.bool)
        self.assertEqual(planes.shape, (16, 128))
        torch.testing.assert_close(bitplanes_to_uint8(planes), symbols)

    def test_state_diagnostics_separate_marginal_and_joint_use(self) -> None:
        symbols = torch.tensor([[0], [1], [2], [3]], dtype=torch.uint8)
        state = uint8_to_bitplanes(symbols)
        metrics = boolean_state_diagnostics(state)
        self.assertEqual(metrics["joint_unique_states"], 4)
        self.assertEqual(len(metrics["per_plane_entropy"]), 8)
        self.assertGreater(metrics["inactive_bit_ratio"], 0.0)


class LearnableLUTLayerTest(unittest.TestCase):
    def test_sequence_candidates_are_unique_and_preserve_vote_identity(self) -> None:
        candidates = sequence_candidate_indices(
            state_bits=64,
            preserved_bits=32,
            output_bits=32,
            arity=4,
            candidate_count=12,
            num_classes=2,
            sequence_length=4,
            seed=7,
        )
        self.assertEqual(candidates.shape, (32, 4, 12))
        self.assertGreaterEqual(int(candidates.min()), 0)
        self.assertLess(int(candidates.max()), 64)
        ordered = candidates.sort(dim=-1).values
        self.assertTrue(bool((ordered[..., 1:] != ordered[..., :-1]).all()))
        torch.testing.assert_close(candidates[:, 0, 0], torch.arange(32) + 32)

    def test_spatial_candidates_are_unique_and_preserve_vote_identity(self) -> None:
        candidates = spatial_candidate_indices(
            state_bits=64,
            preserved_bits=32,
            output_bits=32,
            arity=4,
            candidate_count=12,
            num_classes=2,
            image_shape=(2, 2, 1),
            seed=7,
        )
        self.assertEqual(candidates.shape, (32, 4, 12))
        self.assertGreaterEqual(int(candidates.min()), 0)
        self.assertLess(int(candidates.max()), 64)
        self.assertTrue(
            bool((candidates.sort(dim=-1).values[..., 1:] != candidates.sort(dim=-1).values[..., :-1]).all())
        )
        torch.testing.assert_close(candidates[:, 0, 0], torch.arange(32) + 32)

    def test_truth_support_counts_only_essential_inputs(self) -> None:
        parity4 = sum(
            (address.bit_count() & 1) << address for address in range(16)
        )
        projection0 = sum(
            ((address >> 0) & 1) << address for address in range(16)
        )
        self.assertEqual(essential_input_count(parity4, arity=4), 4)
        self.assertEqual(essential_input_count(projection0, arity=4), 1)

    def test_repeated_sources_reduce_the_realized_truth_table(self) -> None:
        unique_sources, reduced_truth = realized_truth(["x0", "x0"], 0b0110)
        self.assertEqual(unique_sources, ["x0"])
        self.assertEqual(reduced_truth, 0)
        self.assertEqual(
            essential_input_count(reduced_truth, arity=len(unique_sources)), 0
        )

    def test_two_input_bank_realizes_all_sixteen_boolean_functions(self) -> None:
        layer = LearnableLUTLayer(
            2, 16, arity=2, candidate_count=2, seed=3
        )
        with torch.no_grad():
            layer.frozen_sources[:, 0] = 0
            layer.frozen_sources[:, 1] = 1
            for function in range(16):
                for address in range(4):
                    layer.frozen_truth[function, address] = bool(
                        (function >> address) & 1
                    )
            layer.frozen_flag.fill_(True)
        inputs = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.bool
        )
        actual = layer.hard_forward(inputs)
        for function in range(16):
            expected = torch.tensor(
                [bool((function >> address) & 1) for address in range(4)]
            )
            torch.testing.assert_close(actual[:, function], expected)

    def test_two_three_and_four_input_luts_have_exact_hard_outputs(self) -> None:
        for arity in (2, 3, 4):
            width = arity
            layer = LearnableLUTLayer(
                width, 1, arity=arity, candidate_count=width, seed=arity
            )
            with torch.no_grad():
                layer.frozen_sources[0] = torch.arange(arity)
                # Odd parity over the local truth table.
                for address in range(1 << arity):
                    layer.frozen_truth[0, address] = bool(address.bit_count() & 1)
                layer.frozen_flag.fill_(True)
            inputs = torch.tensor(
                [
                    [(address >> bit) & 1 for bit in range(arity)]
                    for address in range(1 << arity)
                ],
                dtype=torch.bool,
            )
            expected = inputs.to(torch.int64).sum(dim=1).remainder(2).bool()
            torch.testing.assert_close(layer.hard_forward(inputs)[:, 0], expected)

    def test_hard_st_is_binary_and_reaches_wiring_and_truth_logits(self) -> None:
        torch.manual_seed(7)
        layer = LearnableLUTLayer(
            16, 12, arity=4, candidate_count=8, seed=5
        )
        inputs = torch.randint(0, 2, (9, 16), dtype=torch.float32)
        output = layer(inputs, mode="hard_st")
        self.assertTrue(bool(((output == 0) | (output == 1)).all()))
        output.square().mean().backward()
        for parameter in (layer.wiring_logits, layer.truth_logits):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_empirical_truth_refit_recovers_xor(self) -> None:
        layer = LearnableLUTLayer(
            4, 1, arity=2, candidate_count=4, seed=1
        )
        _select_source(layer, 0, 0)
        _select_source(layer, 1, 1)
        inputs = torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]],
            dtype=torch.bool,
        )
        target = (inputs[:, :1] ^ inputs[:, 1:2])
        metrics = layer.refit_truth_tables(inputs, target)
        self.assertEqual(metrics.bit_error, 0.0)
        self.assertEqual(metrics.address_coverage, 1.0)
        torch.testing.assert_close(layer.hard_forward(inputs), target)

    def test_greedy_wiring_refit_can_move_to_relevant_sources(self) -> None:
        layer = LearnableLUTLayer(
            4, 1, arity=2, candidate_count=4, seed=9
        )
        _select_source(layer, 0, 2)
        _select_source(layer, 1, 3)
        inputs = torch.tensor(
            [
                [(address >> bit) & 1 for bit in range(4)]
                for address in range(16)
            ],
            dtype=torch.bool,
        )
        target = (inputs[:, :1] ^ inputs[:, 1:2])
        metrics = layer.greedy_refit_wiring_and_truth(inputs, target, passes=2)
        self.assertEqual(metrics.bit_error, 0.0)
        self.assertGreater(metrics.changed_wiring_ratio, 0.0)
        torch.testing.assert_close(layer.hard_forward(inputs), target)


class StrictBitPlaneModelTest(unittest.TestCase):
    def test_strict_prefix_can_seed_a_deeper_model_exactly(self) -> None:
        common = {
            "input_symbols": 4,
            "state_bits": 48,
            "num_classes": 2,
            "arity": 4,
            "candidate_count": 8,
            "seed": 23,
            "candidate_policy": "sequence_causal",
            "input_shape": (4,),
        }
        source = BitPlaneLUTClassifier(blocks=2, **common)
        for block in source.blocks:
            block.freeze_hard()
        payload = source.hard_payload()
        target = BitPlaneLUTClassifier(blocks=4, **common)
        self.assertEqual(load_hard_prefix(target, payload), 2)
        self.assertTrue(all(block.is_frozen for block in target.blocks[:2]))
        self.assertTrue(all(not block.is_frozen for block in target.blocks[2:]))
        symbols = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
        torch.testing.assert_close(
            StrictBitPlaneLUTExecutor(payload).logits(symbols),
            target.hard_logits(symbols, block_count=2),
        )

    def _model(self) -> BitPlaneLUTClassifier:
        model = BitPlaneLUTClassifier(
            input_symbols=2,
            state_bits=32,
            num_classes=2,
            blocks=2,
            arity=2,
            candidate_count=4,
            seed=11,
        )
        for block in model.blocks:
            block.freeze_hard()
        return model

    def test_learned_parameters_are_only_wiring_and_truth_shadows(self) -> None:
        model = self._model()
        names = [name for name, _ in model.named_parameters()]
        self.assertTrue(names)
        self.assertTrue(
            all(name.endswith(("wiring_logits", "truth_logits")) for name in names)
        )
        self.assertFalse(any(isinstance(module, nn.Linear) for module in model.modules()))
        capacity = model.structural_diagnostics()
        self.assertEqual(capacity["learned_dense_integer_matrix_count"], 0)
        self.assertEqual(capacity["learned_numeric_weight_count"], 0)

    def test_payload_executor_is_exact_and_runtime_audit_has_no_real_tensor(self) -> None:
        model = self._model().cpu()
        symbols = torch.tensor(
            [[0, 0], [1, 2], [255, 17], [31, 128]], dtype=torch.uint8
        )
        expected = model.hard_logits(symbols)
        payload = model.hard_payload()
        validate_hard_payload(payload)
        executor = StrictBitPlaneLUTExecutor(payload)
        audit = BooleanRuntimeAudit()
        with audit:
            actual = executor.logits(symbols)
        self.assertGreater(audit.operations, 0)
        self.assertEqual(actual.dtype, torch.int32)
        torch.testing.assert_close(actual, expected)

    def test_raw_a8_planes_are_preserved_across_every_block(self) -> None:
        model = self._model()
        symbols = torch.tensor([[3, 17], [255, 0]], dtype=torch.uint8)
        encoded = model.encode(symbols, carrier=False)
        output = model.state_after(symbols, mode="hard")
        torch.testing.assert_close(
            output[:, :model.preserved_bits],
            encoded[:, :model.preserved_bits],
        )
        self.assertEqual(model.vote_bits, 16)

    def test_payload_validator_rejects_training_logits(self) -> None:
        payload = self._model().hard_payload()
        payload["forbidden_shadow"] = torch.zeros(1, dtype=torch.float32)
        with self.assertRaises(TypeError):
            validate_hard_payload(payload)

    def test_image_spatial_model_has_the_same_strict_payload_boundary(self) -> None:
        model = BitPlaneLUTClassifier(
            input_symbols=4,
            state_bits=48,
            num_classes=2,
            blocks=2,
            arity=4,
            candidate_count=8,
            seed=13,
            candidate_policy="image_spatial",
            input_shape=(2, 2, 1),
        )
        for block in model.blocks:
            block.freeze_hard()
        payload = model.hard_payload()
        validate_hard_payload(payload)
        self.assertEqual(payload["candidate_policy"], "image_spatial")
        symbols = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
        torch.testing.assert_close(
            StrictBitPlaneLUTExecutor(payload).logits(symbols),
            model.hard_logits(symbols),
        )

    def test_sequence_model_has_the_same_strict_payload_boundary(self) -> None:
        model = BitPlaneLUTClassifier(
            input_symbols=4,
            state_bits=48,
            num_classes=2,
            blocks=2,
            arity=4,
            candidate_count=8,
            seed=17,
            candidate_policy="sequence_causal",
            input_shape=(4,),
        )
        for block in model.blocks:
            block.freeze_hard()
        payload = model.hard_payload()
        validate_hard_payload(payload)
        self.assertEqual(payload["candidate_policy"], "sequence_causal")
        symbols = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
        torch.testing.assert_close(
            StrictBitPlaneLUTExecutor(payload).logits(symbols),
            model.hard_logits(symbols),
        )


if __name__ == "__main__":
    unittest.main()
