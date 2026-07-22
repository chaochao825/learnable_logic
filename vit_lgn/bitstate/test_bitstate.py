from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from vit_lgn.bitstate.blocks import BinaryTopKBlock, LocalBitLogicBlock
from vit_lgn.bitstate.encoder import RedundantPredicatePatchEncoder, ThermometerPatchEncoder
from vit_lgn.bitstate.gates import (
    HardSTGateLayer,
    TRUTH_TABLE,
    relaxed_gate_outputs,
    relaxed_lut_output,
)
from vit_lgn.bitstate.model import BitStateConfig, BitStateViT
from vit_lgn.bitstate.regularization import (
    collapse_regularization,
    entropy_unused_gate_ratio,
    gate_entropy_target_penalty,
)
from vit_lgn.bitstate.train_bitstate import (
    initialize_gate_logits,
    split_train_validation,
    supervised_distillation_loss,
)


def small_config() -> BitStateConfig:
    return BitStateConfig(
        image_size=4,
        patch_size=2,
        in_channels=1,
        threshold_levels=2,
        state_width=16,
        local_depth=1,
        global_depth=1,
        heads=2,
        qk_bits=4,
        topk=2,
        num_classes=2,
        votes_per_class=4,
        update_fraction=0.5,
        seed=7,
    )


def contains_float_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.is_floating_point()
    if isinstance(value, dict):
        return any(contains_float_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_float_tensor(item) for item in value)
    return isinstance(value, float)


class BitStateTest(unittest.TestCase):
    def test_gumbel_st_matches_pytorch_hard_gumbel_gradients(self) -> None:
        layer = HardSTGateLayer(
            3,
            2,
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            init_ops=torch.tensor([6, 7]),
            init_strength=0.3,
            surrogate_inputs=False,
        )
        actual_input = torch.tensor(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], requires_grad=True
        )
        reference_input = actual_input.detach().clone().requires_grad_()
        reference_logits = layer.logits.detach().clone().requires_grad_()

        torch.manual_seed(123)
        actual = layer(actual_input, mode="gumbel_st", tau=0.5)
        actual.sum().backward()

        torch.manual_seed(123)
        weights = F.gumbel_softmax(
            reference_logits,
            tau=0.5,
            hard=True,
            dim=-1,
        )
        truth = TRUTH_TABLE.to(dtype=reference_input.dtype)
        reference = relaxed_lut_output(
            reference_input[..., layer.indices_0],
            reference_input[..., layer.indices_1],
            weights @ truth,
        )
        reference.sum().backward()

        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_input.grad, reference_input.grad)
        torch.testing.assert_close(layer.logits.grad, reference_logits.grad)

    def test_normal_gate_initialization_is_seeded_and_paper_scaled(self) -> None:
        first = BitStateViT(small_config())
        second = BitStateViT(small_config())
        initialize_gate_logits(first, "normal", normal_std=1.0, seed=91)
        initialize_gate_logits(second, "normal", normal_std=1.0, seed=91)
        first_logits = torch.cat([layer.logits.flatten() for layer in first.gate_layers()])
        second_logits = torch.cat([layer.logits.flatten() for layer in second.gate_layers()])
        torch.testing.assert_close(first_logits, second_logits)
        self.assertLess(abs(float(first_logits.detach().mean())), 0.1)
        self.assertGreater(float(first_logits.detach().std()), 0.9)
        self.assertLess(float(first_logits.detach().std()), 1.1)

    def test_entropy_unused_metric_separates_uncommitted_and_sharp_gates(self) -> None:
        layer = HardSTGateLayer(
            2,
            3,
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 1, 1]),
            init_ops=torch.tensor([6, 7, 8]),
            init_strength=0.1,
        )
        self.assertEqual(entropy_unused_gate_ratio([layer]), 1.0)
        with torch.no_grad():
            selected = layer.logits.argmax(dim=-1, keepdim=True)
            layer.logits.fill_(-10.0)
            layer.logits.scatter_(1, selected, 10.0)
        self.assertEqual(entropy_unused_gate_ratio([layer]), 0.0)

    def test_train_validation_split_is_deterministic_and_disjoint(self) -> None:
        dataset = TensorDataset(torch.arange(20))
        train_a, validation_a = split_train_validation(dataset, dataset, 5, 17)
        train_b, validation_b = split_train_validation(dataset, dataset, 5, 17)
        self.assertIsNotNone(validation_a)
        self.assertIsNotNone(validation_b)
        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(validation_a.indices, validation_b.indices)
        self.assertFalse(set(train_a.indices) & set(validation_a.indices))

    def test_supervised_distillation_loss_backpropagates(self) -> None:
        student = torch.randn(6, 4, requires_grad=True)
        teacher = torch.randn(6, 4)
        labels = torch.tensor([0, 1, 2, 3, 0, 1])
        combined, supervised, distillation = supervised_distillation_loss(
            student,
            labels,
            label_smoothing=0.1,
            teacher_logits=teacher,
            teacher_alpha=0.4,
            teacher_temperature=2.0,
        )
        torch.testing.assert_close(combined, 0.6 * supervised + 0.4 * distillation)
        combined.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_compressed_lut_matches_16_function_mixture(self) -> None:
        torch.manual_seed(5)
        a = torch.rand(3, 7)
        b = torch.rand(3, 7)
        weights = torch.softmax(torch.randn(7, 16), dim=-1)
        expected = (relaxed_gate_outputs(a, b) * weights).sum(dim=-1)
        address_values = weights @ TRUTH_TABLE.to(torch.float32)
        actual = relaxed_lut_output(a, b, address_values)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_all_16_gates_match_truth_table(self) -> None:
        inputs = torch.tensor(
            [[False, False], [False, True], [True, False], [True, True]]
        )
        for op_id in range(16):
            layer = HardSTGateLayer(
                2,
                1,
                torch.tensor([0]),
                torch.tensor([1]),
                init_ops=torch.tensor([op_id]),
            )
            actual = layer.forward_bits(inputs).squeeze(-1)
            torch.testing.assert_close(actual, TRUTH_TABLE[op_id])

    def test_gate_hard_st_is_bit_exact_and_differentiable(self) -> None:
        layer = HardSTGateLayer(
            3,
            2,
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            init_ops=torch.tensor([6, 7]),
            surrogate_inputs=True,
        )
        bits = torch.tensor([[False, True, True], [True, False, True]])
        carrier = bits.to(torch.float32).requires_grad_()
        hard_st = layer(carrier, mode="hard_st")
        torch.testing.assert_close(hard_st.detach().bool(), layer.forward_bits(bits))
        hard_st.sum().backward()
        self.assertGreater(float(layer.logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(carrier.grad.abs().sum()), 0.0)

    def test_gumbel_st_has_boolean_forward_and_soft_gradient(self) -> None:
        torch.manual_seed(9)
        layer = HardSTGateLayer(
            3,
            2,
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            init_ops=torch.tensor([6, 7]),
            surrogate_inputs=True,
        )
        carrier = torch.tensor([[0.0, 1.0, 1.0]], requires_grad=True)
        output = layer(carrier, mode="gumbel_st", tau=0.7)
        self.assertTrue(bool(((output == 0) | (output == 1)).all()))
        output.sum().backward()
        self.assertGreater(float(layer.logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(carrier.grad.abs().sum()), 0.0)

    def test_encoder_emits_persistent_boolean_tokens(self) -> None:
        encoder = ThermometerPatchEncoder(
            image_size=4,
            patch_size=2,
            in_channels=1,
            threshold_levels=2,
            state_width=8,
        )
        output = encoder.forward_bits(torch.arange(16, dtype=torch.uint8).reshape(1, 1, 4, 4) * 16)
        self.assertEqual(output.dtype, torch.bool)
        self.assertEqual(output.shape, (1, 5, 8))

    def test_redundant_encoder_is_bit_exact_and_trainable(self) -> None:
        encoder = RedundantPredicatePatchEncoder(
            image_size=4,
            patch_size=2,
            in_channels=1,
            threshold_levels=2,
            state_width=16,
            identity_width=4,
            predicate_fanin=3,
            predicate_chunk_size=5,
            seed=13,
        )
        images = torch.arange(32, dtype=torch.uint8).reshape(2, 1, 4, 4) * 8
        carrier = encoder(images, mode="hard_st", tau=0.8)
        bits = encoder.forward_bits(images)
        torch.testing.assert_close(carrier.detach().bool(), bits)
        carrier.sum().backward()
        for parameter in (encoder.threshold_logits, encoder.polarity_logits):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        self.assertFalse(contains_float_tensor(encoder.deployment_payload()))

    def test_local_block_hard_carrier_matches_bits(self) -> None:
        block = LocalBitLogicBlock(state_width=8, grid_size=2, update_fraction=0.5, seed=2)
        bits = torch.randint(0, 2, (2, 5, 8), dtype=torch.bool)
        carrier = block(bits.float(), mode="hard")
        torch.testing.assert_close(carrier.bool(), block.forward_bits(bits))

    def test_binary_topk_is_stable_and_bit_exact(self) -> None:
        block = BinaryTopKBlock(
            state_width=8,
            num_tokens=4,
            heads=2,
            qk_bits=3,
            topk=2,
            seed=3,
        )
        tied = torch.zeros(1, 2, 4, 3, dtype=torch.bool)
        indices = block._topk_indices(tied, tied)
        torch.testing.assert_close(indices, torch.tensor([0, 1]).view(1, 1, 1, 2).expand_as(indices))

        bits = torch.randint(0, 2, (2, 4, 8), dtype=torch.bool)
        carrier = block(bits.float(), mode="hard")
        bit_output, bit_indices = block.forward_bits(bits, return_indices=True)
        torch.testing.assert_close(carrier.bool(), bit_output)
        torch.testing.assert_close(block._last_indices, bit_indices)

    def test_full_model_matches_integer_reference_at_every_boundary(self) -> None:
        model = BitStateViT(small_config()).eval()
        images = torch.randint(0, 256, (3, 1, 4, 4), dtype=torch.uint8)
        model.assert_bit_exact(images)
        logits, trace = model.forward_bits(images, return_trace=True)
        self.assertEqual(logits.dtype, torch.int32)
        self.assertEqual(logits.shape, (3, 2))
        self.assertTrue(all(item.dtype == torch.bool for item in trace))
        carrier_logits, carrier_trace = model(
            images,
            mode="hard_st",
            return_trace=True,
        )
        self.assertTrue(
            all(
                torch.equal(carrier.detach(), bits.to(carrier.dtype))
                for carrier, bits in zip(carrier_trace, trace)
            )
        )
        torch.testing.assert_close(carrier_logits.detach(), logits.to(torch.float32))

    def test_redundant_full_model_matches_integer_reference(self) -> None:
        config = small_config()
        config = BitStateConfig(
            **{
                **config.__dict__,
                "state_width": 24,
                "encoder_kind": "redundant_predicate",
                "predicate_fanin": 3,
                "encoder_identity_width": 4,
            }
        )
        model = BitStateViT(config).eval()
        images = torch.randint(0, 256, (3, 1, 4, 4), dtype=torch.uint8)
        model.assert_bit_exact(images)
        self.assertGreater(model.predicate_count(), 0)
        self.assertFalse(contains_float_tensor(model.deployment_payload()))

    def test_low_margin_initialization_preserves_hard_execution(self) -> None:
        config = small_config()
        config = BitStateConfig(**{**config.__dict__, "gate_init_strength": 0.1})
        model = BitStateViT(config).eval()
        images = torch.randint(0, 256, (2, 1, 4, 4), dtype=torch.uint8)
        model.assert_bit_exact(images)
        operations = [layer.hard_ops().clone() for layer in model.gate_layers()]
        confidence_before = torch.stack(
            [layer.confidence() for layer in model.gate_layers()]
        ).mean()
        self.assertTrue(
            all(
                float(layer.confidence().detach()) < 0.2
                for layer in model.gate_layers()
            )
        )
        model.scale_gate_logits(4.0)
        confidence_after = torch.stack(
            [layer.confidence() for layer in model.gate_layers()]
        ).mean()
        self.assertGreater(
            float(confidence_after.detach()),
            float(confidence_before.detach()),
        )
        for expected, layer in zip(operations, model.gate_layers()):
            torch.testing.assert_close(expected, layer.hard_ops())
        model.assert_bit_exact(images)

    def test_anti_collapse_losses_are_finite_and_differentiable(self) -> None:
        model = BitStateViT(small_config()).train()
        _logits, trace = model(
            torch.rand(4, 1, 4, 4),
            mode="hard_st",
            return_trace=True,
        )
        collapse_loss, metrics = collapse_regularization(
            trace,
            balance_weight=0.1,
            diversity_weight=0.1,
            flip_weight=0.1,
        )
        gate_loss, entropy, confidence = gate_entropy_target_penalty(
            model.gate_layers(),
            0.5,
        )
        loss = collapse_loss + gate_loss
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(value)) for value in metrics.values()))
        self.assertTrue(bool(torch.isfinite(entropy)))
        self.assertTrue(bool(torch.isfinite(confidence)))
        loss.backward()
        self.assertTrue(
            any(
                layer.logits.grad is not None
                and float(layer.logits.grad.abs().sum()) > 0.0
                for layer in model.gate_layers()
            )
        )

    def test_backward_reaches_every_gate_family(self) -> None:
        torch.manual_seed(123)
        model = BitStateViT(small_config()).train()
        labels = torch.tensor([0, 1, 0, 1])
        loss = sum(
            torch.nn.functional.cross_entropy(
                model(torch.rand(4, 1, 4, 4), mode="hard_st", tau=1.0),
                labels,
            )
            for _ in range(4)
        )
        loss.backward()
        for name, layer in model.named_modules():
            if isinstance(layer, HardSTGateLayer):
                self.assertIsNotNone(layer.logits.grad, name)
                self.assertTrue(bool(torch.isfinite(layer.logits.grad).all()), name)
                self.assertGreater(float(layer.logits.grad.abs().sum()), 0.0, name)

    def test_payload_contains_no_float_values_or_tensors(self) -> None:
        model = BitStateViT(small_config())
        payload = model.deployment_payload()
        self.assertFalse(contains_float_tensor(payload))
        self.assertEqual(payload["general_matrix_multipliers"], 0)
        self.assertEqual(payload["persistent_state_dtype"], "bool")

    def test_inactive_ratio_and_complexity_metrics_are_bounded(self) -> None:
        model = BitStateViT(small_config()).eval()
        batches = [torch.rand(4, 1, 4, 4), torch.rand(3, 1, 4, 4)]
        ratio = model.inactive_gate_ratio(batches)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)
        self.assertGreater(model.gate_count(), 0)
        self.assertGreater(model.logic_depth(), 0)
        self.assertGreater(model.fanout_max(), 0)


if __name__ == "__main__":
    unittest.main()
