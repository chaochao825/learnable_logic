from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
from vit_lgn.full_discrete.export_logic_payload import (
    SCHEMA_NAME,
    SCHEMA_TOP_LEVEL_KEYS,
    SCHEMA_VERSION,
    export_checkpoint,
    export_logic_payload,
    validate_logic_payload,
)
from vit_lgn.full_discrete.logic_backend import A8U4MagnitudeProductROM
from vit_lgn.full_discrete.model import FullDiscreteViT
from vit_lgn.full_discrete.shiftadd import ShiftAddLinear, _power_of_two_scale


def _walk(value, path="payload"):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


class LogicPayloadExportTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = FullDiscreteViT(
            dim=24,
            depth=1,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=7,
            activation_bits=8,
            qk_lanes=7,
        ).eval()

    def test_payload_is_integer_only_and_has_no_training_state(self) -> None:
        payload = export_logic_payload(
            self.model,
            requantize_magnitude_bits=4,
            checkpoint_metadata={
                "step": 50_000,
                "protocol_sha256": "abc123",
                "optimizer": {"forbidden": torch.tensor(1.5)},
                "torch_rng": torch.get_rng_state(),
            },
        )
        validate_logic_payload(payload)
        self.assertEqual(tuple(payload), SCHEMA_TOP_LEVEL_KEYS)
        self.assertEqual(payload["schema"]["name"], SCHEMA_NAME)
        self.assertEqual(payload["schema"]["version"], SCHEMA_VERSION)
        self.assertEqual(payload["source"]["checkpoint_step"], 50_000)
        for path, value in _walk(payload):
            lowered = path.lower()
            self.assertNotIn("optimizer", lowered)
            self.assertNotIn("rng", lowered)
            self.assertNotIn("latent", lowered)
            self.assertNotIn("shadow", lowered)
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.is_floating_point(), path)
                self.assertFalse(value.is_complex(), path)
                self.assertEqual(value.device.type, "cpu", path)
            self.assertFalse(isinstance(value, float), path)

    def test_w4_planes_reconstruct_every_integer_weight_matrix(self) -> None:
        payload = export_logic_payload(self.model, requantize_magnitude_bits=4)
        source_layers = {
            name: module
            for name, module in self.model.named_modules()
            if isinstance(module, ShiftAddLinear)
        }
        self.assertEqual(len(payload["shift_add_layers"]), len(source_layers))
        for layer_payload in payload["shift_add_layers"]:
            code = layer_payload["weight_code"].to(torch.int16)
            planes = layer_payload["weight_magnitude_planes_lsb_first"]
            magnitude = sum(
                planes[..., bit].to(torch.int16) << bit for bit in range(4)
            )
            reconstructed = torch.where(
                layer_payload["weight_sign"].bool(), -magnitude, magnitude
            )
            torch.testing.assert_close(reconstructed, code)
            self.assertGreaterEqual(int(code.min()), -15)
            self.assertLessEqual(int(code.max()), 15)
            self.assertEqual(layer_payload["target_magnitude_bits"], 4)
            self.assertEqual(layer_payload["requantized_from_bits"], 7)
            self.assertEqual(layer_payload["source_magnitude_bits"], 7)
            chunks = layer_payload["weight_magnitude_chunks_u4_lsb_first"]
            self.assertEqual(chunks.shape[-1], 1)
            torch.testing.assert_close(chunks[..., 0].to(torch.int16), magnitude)

            source = source_layers[layer_payload["name"]]
            scale = _power_of_two_scale(source.weight.detach().abs().amax(1, keepdim=True) / 15)
            exponent_scale = torch.pow(
                torch.tensor(2.0), layer_payload["weight_scale_exponent"].to(torch.float64)
            )
            expected = torch.round(source.weight.detach() / scale).clamp(-15, 15)
            torch.testing.assert_close(code, expected.to(torch.int16))
            torch.testing.assert_close(exponent_scale, scale.cpu().to(torch.float64))

    def test_default_preserves_w7_as_low4_high3_chunks(self) -> None:
        payload = export_logic_payload(self.model)
        rom = payload["arithmetic_contract"]["a8_by_u4_product_rom"]
        table = rom["product_rom_int16"]
        self.assertEqual(tuple(table.shape), (256, 16))
        oracle = torch.tensor(
            A8U4MagnitudeProductROM().payload, dtype=torch.int16
        )
        torch.testing.assert_close(table, oracle)
        self.assertEqual(int(table[0, 13]), 0)
        self.assertEqual(int(table[7, 13]), 91)
        self.assertEqual(int(table[127, 13]), 127 * 13)
        self.assertEqual(int(table[128, 13]), -128 * 13)
        self.assertEqual(int(table[249, 13]), -91)
        self.assertEqual(int(table[255, 13]), -13)
        for item in payload["shift_add_layers"]:
            self.assertEqual(item["source_magnitude_bits"], 7)
            self.assertEqual(item["target_magnitude_bits"], 7)
            self.assertEqual(item["requantized_from_bits"], 0)
            torch.testing.assert_close(
                item["magnitude_chunk_shift"], torch.tensor([0, 4], dtype=torch.uint8)
            )
            torch.testing.assert_close(
                item["magnitude_chunk_valid_bits"],
                torch.tensor([4, 3], dtype=torch.uint8),
            )
            chunks = item["weight_magnitude_chunks_u4_lsb_first"].to(torch.int16)
            reconstructed = chunks[..., 0] + (chunks[..., 1] << 4)
            torch.testing.assert_close(reconstructed, item["weight_code"].abs().to(torch.int16))

    def test_parameter_attention_and_rms_payloads_are_deployable(self) -> None:
        payload = export_logic_payload(self.model)
        for name, parameter in (
            ("cls_token", self.model.cls_token),
            ("position", self.model.position),
        ):
            item = payload["parameters"][name]
            self.assertEqual(tuple(item["code"].shape), tuple(parameter.shape))
            self.assertEqual(item["activation_bits"], 8)

        attention = payload["attention"][0]
        numerator = attention["threshold_fraction_numerator"].to(torch.float64)
        shift = attention["threshold_fraction_denominator_shift"].to(torch.float64)
        reconstructed = numerator / torch.pow(torch.tensor(2.0), shift)
        torch.testing.assert_close(
            reconstructed,
            self.model.blocks[0].attn.threshold_fractions.to(torch.float64),
            rtol=0,
            atol=0,
        )
        self.assertEqual(attention["topk"], 4)
        self.assertEqual(
            attention["topk_tie_rule"],
            "score_descending_then_key_index_ascending",
        )
        self.assertEqual(attention["gap"]["gap_right_shift"], 1)
        self.assertIn("x_code << denominator_shift", attention["threshold_compare"])
        # Cross multiplication preserves 1 >= 3/2 as false; multiplying and
        # right-shifting the RHS first would incorrectly make it true.
        self.assertFalse((1 << 1) >= 3 * 1)

        lut = payload["rms_luts"][0]
        table = lut["table_uint16_carried_as_int32"]
        self.assertEqual(int(table[0]), 1 << 15)
        self.assertEqual(int(table[1]), 1 << 15)
        self.assertEqual(int(table[4]), 1 << 14)
        self.assertEqual(lut["address_max"], 127 * 127)
        norm = payload["rms_norms"][0]
        self.assertEqual(norm["product_integer_bits"], 23)
        self.assertEqual(norm["product_exponent_delta"], -15)
        self.assertEqual(norm["pre_requant_right_shift"], 0)
        self.assertNotIn("product_right_shift", norm)

    def test_norm_variants_export_only_the_selected_integer_operator(self) -> None:
        expected_operator = {
            "rms_lut": "integer_rmsnorm_q15_lut",
            "shift_rms": "integer_shift_rms",
            "requant": "activation_requant_only",
            "none": "identity_wire",
        }
        for kind, operator in expected_operator.items():
            model = FullDiscreteViT(
                dim=24,
                depth=1,
                heads=3,
                topk=4,
                mlp_ratio=2.0,
                weight_bits=4,
                activation_bits=8,
                norm_kind=kind,
            ).eval()
            payload = export_logic_payload(model)
            self.assertEqual(payload["topology"]["norm_kind"], kind)
            self.assertTrue(payload["rms_norms"])
            self.assertEqual(
                {item["operator"] for item in payload["rms_norms"]}, {operator}
            )
            self.assertEqual(bool(payload["rms_luts"]), kind == "rms_lut")
            validate_logic_payload(payload)

    def test_schema_is_stable_across_base_and_enhanced_models(self) -> None:
        enhanced = EnhancedFullDiscreteViT(
            dim=24,
            depth=1,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=7,
            activation_bits=8,
            learned_gap=True,
            group_lut_groups=6,
            local_layers=1,
        ).eval()
        base_payload = export_logic_payload(self.model)
        enhanced_payload = export_logic_payload(enhanced)
        self.assertEqual(tuple(base_payload), tuple(enhanced_payload))
        self.assertEqual(tuple(enhanced_payload), SCHEMA_TOP_LEVEL_KEYS)
        self.assertEqual(len(enhanced_payload["activation_luts"]), 1)
        self.assertEqual(len(enhanced_payload["local_branches"]), 1)
        self.assertEqual(enhanced_payload["attention"][0]["gap"]["kind"],
                         "per_head_monotone_lut")
        self.assertEqual(
            tuple(enhanced_payload["activation_luts"][0]["table_int8"].shape),
            (6, 256),
        )

    def test_logic_tree_local_payload_is_fixed_routing_and_integer_only(self) -> None:
        model = EnhancedFullDiscreteViT(
            dim=24,
            depth=1,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=7,
            activation_bits=8,
            local_layers=1,
            local_operator="logic_tree3x3",
        ).eval()
        payload = export_logic_payload(model)
        self.assertEqual(len(payload["local_branches"]), 1)
        branch = payload["local_branches"][0]
        self.assertEqual(branch["operator"], "shared_logic_tree3x3_bitplane")
        self.assertEqual(tuple(branch["truth_table_00_01_10_11"].shape), (24, 8, 7, 4))
        self.assertEqual(tuple(branch["truth_nibble_lsb_address"].shape), (24, 8, 7))
        self.assertTrue(bool((branch["truth_nibble_lsb_address"] == 0xC).all()))
        self.assertEqual(tuple(branch["leaf_site"].shape), (24, 8, 8))
        self.assertTrue(bool((branch["leaf_site"][..., 0] == 0).all()))
        self.assertEqual(branch["output_projection"], "none")
        self.assertFalse(branch["learned_connections"])
        validate_logic_payload(payload)
        for path, value in _walk(branch, "local_branch"):
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.is_floating_point(), path)
            self.assertFalse(isinstance(value, float), path)

        corrupted = export_logic_payload(model)
        corrupted["local_branches"][0]["truth_table_00_01_10_11"][0, 0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "not binary"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["local_branches"][0]["leaf_site"][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "leaf zero"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["local_branches"][0]["leaf_site"][0, 0, 1] = 0
        with self.assertRaisesRegex(ValueError, "fixed leaf-map"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["local_branches"][0]["gate_children_node_index"][6, 0] = 11
        with self.assertRaisesRegex(ValueError, "gate topology"):
            validate_logic_payload(corrupted)

    def test_hadamard_global_mixer_exports_as_fixed_integer_topology(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=16,
            patch_size=4,
            dim=24,
            depth=2,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=7,
            activation_bits=8,
            global_mixer="hadamard",
            hadamard_group_size=8,
            hadamard_branch_shift=2,
            local_layers=1,
        ).eval()
        payload = export_logic_payload(model)
        self.assertEqual(payload["schema"]["version"], SCHEMA_VERSION)
        self.assertEqual(payload["attention"], [])
        self.assertEqual(len(payload["global_mixers"]), 2)
        for index, mixer in enumerate(payload["global_mixers"]):
            self.assertEqual(
                mixer["operator"],
                "fixed_hadamard_sign_hadamard_global_mixer",
            )
            self.assertEqual(mixer["patch_tokens"], 16)
            self.assertEqual(mixer["block_index"], index)
            self.assertEqual(mixer["normalization_right_shift"], 4)
            self.assertEqual(mixer["patch_add_sub_per_channel"], 128)
            self.assertEqual(mixer["input_code_signed_bits"], 8)
            self.assertEqual(mixer["first_butterfly_signed_bits"], 12)
            self.assertEqual(mixer["second_butterfly_signed_bits"], 16)
            self.assertEqual(mixer["patch_pre_branch_signed_bits"], 13)
            self.assertEqual(mixer["branch_output_accumulator_signed_bits"], 11)
            self.assertEqual(tuple(mixer["sign_mask_int8"].shape), (16,))
            self.assertEqual(set(mixer["sign_mask_int8"].tolist()), {-1, 1})
        validate_logic_payload(payload)

        args = {
            "image_size": 16, "patch_size": 4,
            "dim": 24, "depth": 2, "heads": 3, "topk": 4,
            "mlp_ratio": 2.0, "weight_magnitude_bits": 7,
            "activation_bits": 8, "qk_lanes": 7,
            "norm_kind": "rms_lut", "final_norm_kind": "same",
            "learned_gap": False, "group_lut_groups": 0,
            "local_layers": 1, "local_operator": "depthwise_shiftadd",
            "global_mixer": "hadamard", "hadamard_group_size": 8,
            "hadamard_branch_shift": 2,
            "logic_expert_width": 0, "logic_expert_count": 1,
            "state_control": "none", "state_expert_width": 0,
        }
        checkpoint = {
            "step": 50_000,
            "model": model.state_dict(),
            "args": args,
            "protocol_sha256": "hadamard-protocol",
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            output_path = Path(directory) / "payload.pt"
            torch.save(checkpoint, checkpoint_path)
            exported = export_checkpoint(checkpoint_path, output_path)
        self.assertEqual(len(exported["global_mixers"]), 2)
        self.assertEqual(exported["attention"], [])

    def test_hybrid_topology_finds_attention_after_fixed_first_block(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=16,
            patch_size=4,
            dim=24,
            depth=3,
            heads=3,
            topk=4,
            mlp_ratio=2.0,
            weight_bits=7,
            activation_bits=8,
            global_mixer="hybrid",
            hybrid_attention_period=3,
            hadamard_group_size=8,
        ).eval()
        payload = export_logic_payload(model)
        self.assertEqual(len(payload["global_mixers"]), 2)
        self.assertEqual(len(payload["attention"]), 1)
        self.assertEqual(payload["topology"]["heads"], 3)
        self.assertEqual(payload["topology"]["head_dim"], 8)
        self.assertEqual(payload["topology"]["topk"], 4)
        self.assertEqual(payload["topology"]["qk_lanes"], 7)
        validate_logic_payload(payload)

    def test_global_lut_tree_exports_all_learned_integer_roms(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=8,
            patch_size=4,
            dim=8,
            depth=1,
            heads=2,
            topk=2,
            mlp_ratio=2.0,
            weight_bits=7,
            activation_bits=8,
            qk_lanes=2,
            global_mixer="parallel_lut_tree",
            global_lut_group_size=4,
            global_lut_branch_shift=2,
        ).eval()
        payload = export_logic_payload(model)
        self.assertEqual(payload["schema"]["version"], 5)
        self.assertEqual(len(payload["attention"]), 1)
        self.assertEqual(len(payload["global_mixers"]), 1)
        mixer = payload["global_mixers"][0]
        self.assertEqual(
            mixer["operator"], "group_shared_a8_pair_lut_global_tree"
        )
        self.assertEqual(mixer["reduction_stages"], 2)
        self.assertEqual(mixer["runtime_scale_groups"], 2)
        self.assertEqual(mixer["tables_per_block"], 8)
        self.assertEqual(mixer["hard_payload_bits"], 4_194_304)
        self.assertEqual(mixer["rom_reads_per_image_per_block"], 72)
        self.assertEqual(mixer["rom_reads_per_group_per_block"], 36)
        self.assertEqual(
            mixer["single_port_rom_cycles_per_block_groups_parallel"], 36
        )
        self.assertEqual(
            mixer["group_size_ports_cycles_per_block_groups_parallel"], 9
        )
        self.assertEqual(mixer["ports_per_active_table_for_group_parallelism"], 4)
        self.assertEqual(
            mixer["input_requantization"]["maximum_reduction_axes_grouped"],
            [1, 3],
        )
        self.assertEqual(
            mixer["parallel_merge"]["intermediate_requantization"], "none"
        )
        self.assertEqual(
            mixer["outer_residual_boundary"]["output_scale_granularity"],
            "per_batch_per_token",
        )
        self.assertEqual(tuple(mixer["reduce_table_int8"].shape), (2, 2, 256, 256))
        self.assertEqual(tuple(mixer["context_table_int8"].shape), (2, 256, 256))
        self.assertEqual(tuple(mixer["broadcast_table_int8"].shape), (2, 256, 256))
        self.assertTrue(all(
            table.dtype == torch.int8
            for table in (
                mixer["reduce_table_int8"],
                mixer["context_table_int8"],
                mixer["broadcast_table_int8"],
            )
        ))
        validate_logic_payload(payload)

        corrupted = export_logic_payload(model)
        corrupted["global_mixers"][0]["broadcast_table_int8"][0, 0, 0] = -128
        with self.assertRaisesRegex(ValueError, "value range"):
            validate_logic_payload(corrupted)

        corrupted = export_logic_payload(model)
        corrupted["global_mixers"][0]["runtime_scale_groups"] = 1
        with self.assertRaisesRegex(ValueError, "cross-topology"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["global_mixers"][0]["block_index"] = 1
        with self.assertRaisesRegex(ValueError, "cross-topology"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["global_mixers"][0]["input_requantization"][
            "maximum_reduction_axes_grouped"
        ] = [3]
        with self.assertRaisesRegex(ValueError, "input requantization"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["global_mixers"][0]["parallel_merge"][
            "intermediate_requantization"
        ] = "A8"
        with self.assertRaisesRegex(ValueError, "parallel merge"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["global_mixers"].clear()
        with self.assertRaisesRegex(ValueError, "mode and per-block LUT"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["attention"].clear()
        with self.assertRaisesRegex(ValueError, "cross-topology"):
            validate_logic_payload(corrupted)
        corrupted = export_logic_payload(model)
        corrupted["attention"][0]["name"] = "blocks.0.attn.renamed"
        with self.assertRaisesRegex(ValueError, "cross-topology"):
            validate_logic_payload(corrupted)

        args = {
            "image_size": 8, "patch_size": 4,
            "dim": 8, "depth": 1, "heads": 2, "topk": 2,
            "mlp_ratio": 2.0, "weight_magnitude_bits": 7,
            "activation_bits": 8, "qk_lanes": 2,
            "norm_kind": "rms_lut", "final_norm_kind": "same",
            "learned_gap": False, "group_lut_groups": 0,
            "local_layers": 0, "local_operator": "depthwise_shiftadd",
            "global_mixer": "parallel_lut_tree",
            "global_lut_group_size": 4, "global_lut_branch_shift": 2,
            "logic_expert_width": 0, "logic_expert_count": 1,
            "state_control": "none", "state_expert_width": 0,
        }
        checkpoint = {
            "step": 1_000,
            "model": model.state_dict(),
            "args": args,
            "protocol_sha256": "global-lut-protocol",
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            output_path = Path(directory) / "payload.pt"
            torch.save(checkpoint, checkpoint_path)
            exported = export_checkpoint(checkpoint_path, output_path)
        self.assertEqual(len(exported["attention"]), 1)
        self.assertEqual(len(exported["global_mixers"]), 1)
        self.assertEqual(
            exported["global_mixers"][0]["operator"],
            "group_shared_a8_pair_lut_global_tree",
        )

    def test_checkpoint_reconstructs_logic_tree_operator_strictly(self) -> None:
        model = EnhancedFullDiscreteViT(
            dim=24, depth=1, heads=3, topk=4, mlp_ratio=2.0,
            weight_bits=7, activation_bits=8, local_layers=1,
            local_operator="logic_tree3x3",
        ).eval()
        args = {
            "dim": 24, "depth": 1, "heads": 3, "topk": 4,
            "mlp_ratio": 2.0, "weight_magnitude_bits": 7,
            "activation_bits": 8, "qk_lanes": 7,
            "norm_kind": "rms_lut", "final_norm_kind": "same",
            "learned_gap": False, "group_lut_groups": 0,
            "local_layers": 1, "local_operator": "logic_tree3x3",
            "logic_expert_width": 0, "logic_expert_count": 1,
            "state_control": "none", "state_expert_width": 0,
        }
        checkpoint = {
            "step": 50_000,
            "model": model.state_dict(),
            "args": args,
            "protocol_sha256": "logic-tree-protocol",
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            output_path = Path(directory) / "payload.pt"
            torch.save(checkpoint, checkpoint_path)
            exported = export_checkpoint(checkpoint_path, output_path)
        branch = exported["local_branches"][0]
        self.assertEqual(branch["operator"], "shared_logic_tree3x3_bitplane")
        self.assertTrue(bool((branch["truth_nibble_lsb_address"] == 0xC).all()))

    def test_checkpoint_export_drops_optimizer_and_rng(self) -> None:
        args = {
            "dim": 24,
            "depth": 1,
            "heads": 3,
            "topk": 4,
            "mlp_ratio": 2.0,
            "weight_magnitude_bits": 7,
            "activation_bits": 8,
            "qk_lanes": 7,
            "learned_gap": False,
            "group_lut_groups": 0,
            "local_layers": 0,
            "logic_expert_width": 0,
            "logic_expert_count": 1,
            "state_control": "none",
            "state_expert_width": 0,
        }
        checkpoint = {
            "step": 50_000,
            "model": self.model.state_dict(),
            "args": args,
            "protocol_sha256": "paired-run-hash",
            "optimizer": {"state": {0: {"exp_avg": torch.randn(2)}}},
            "scheduler": {"last_epoch": 50_000},
            "python_rng": (3, (1, 2, 3), None),
            "torch_rng": torch.get_rng_state(),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            output_path = Path(directory) / "logic_payload.pt"
            torch.save(checkpoint, checkpoint_path)
            exported = export_checkpoint(
                checkpoint_path, output_path, requantize_magnitude_bits=4
            )
            loaded = torch.load(output_path, map_location="cpu", weights_only=True)
        self.assertEqual(exported["source"]["checkpoint_step"], 50_000)
        self.assertEqual(loaded["source"]["protocol_sha256"], "paired-run-hash")
        validate_logic_payload(loaded)
        serialized_paths = [path.lower() for path, _ in _walk(loaded)]
        self.assertFalse(any("optimizer" in path for path in serialized_paths))
        self.assertFalse(any("rng" in path for path in serialized_paths))

    def test_validator_rejects_float_and_schema_drift(self) -> None:
        payload = export_logic_payload(self.model)
        payload["parameters"]["cls_token"]["code"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "floating tensor"):
            validate_logic_payload(payload)
        payload = export_logic_payload(self.model)
        payload["new_unversioned_field"] = 1
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_logic_payload(payload)
        payload = export_logic_payload(self.model)
        payload["schema"]["version"] = 999
        with self.assertRaisesRegex(ValueError, "schema identity"):
            validate_logic_payload(payload)
        payload = export_logic_payload(self.model)
        del payload["shift_add_layers"][0]["weight_code"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_logic_payload(payload)
        payload = export_logic_payload(self.model)
        del payload["attention"][0]["topk_tie_rule"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_logic_payload(payload)
        payload = export_logic_payload(self.model)
        payload["attention"][0]["topk_tie_rule"] = "unstable"
        with self.assertRaisesRegex(ValueError, "tie ABI"):
            validate_logic_payload(payload)
        payload = export_logic_payload(self.model)
        payload["arithmetic_contract"]["a8_by_u4_product_rom"][
            "product_rom_int16"
        ][128, 1] = 128
        with self.assertRaisesRegex(ValueError, "ROM payload"):
            validate_logic_payload(payload)

    def test_schema_rejects_non_a8_product_operands(self) -> None:
        model = FullDiscreteViT(
            dim=24, depth=1, heads=3, mlp_ratio=2.0, activation_bits=9
        ).eval()
        with self.assertRaisesRegex(ValueError, "A8-by-U4"):
            export_logic_payload(model)


if __name__ == "__main__":
    unittest.main()
