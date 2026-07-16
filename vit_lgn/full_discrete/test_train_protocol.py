from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from vit_lgn.full_discrete.enhancements_logic_tree import SharedLogicTreeConv3x3
from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
from vit_lgn.full_discrete.train_cifar import (
    CIFAR10_PAYLOAD_FILES,
    PROTOCOL_SOURCE_FILES,
    dataset_hashes,
    freeze_protocol_manifest,
    forced_projection_a,
    optimizer_parameter_groups,
    optimizer_role_gradient_l2,
    protocol,
    source_hashes,
)


class TrainProtocolTest(unittest.TestCase):
    @staticmethod
    def _fake_dataset(root: Path) -> None:
        extracted = root / "cifar-10-batches-py"
        extracted.mkdir()
        for index, name in enumerate(CIFAR10_PAYLOAD_FILES):
            (extracted / name).write_bytes(f"payload-{index}".encode())

    def test_extracted_dataset_is_hashed_when_archive_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "cifar-10-batches-py"
            extracted.mkdir()
            for index, name in enumerate(CIFAR10_PAYLOAD_FILES):
                (extracted / name).write_bytes(f"payload-{index}".encode())
            fingerprint = dataset_hashes(root)
        self.assertIsNone(fingerprint["archive_sha256"])
        self.assertEqual(
            set(fingerprint["extracted_files_sha256"]),
            set(CIFAR10_PAYLOAD_FILES),
        )
        self.assertTrue(all(
            len(value) == 64
            for value in fingerprint["extracted_files_sha256"].values()
        ))

    def test_incomplete_extracted_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cifar-10-batches-py").mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                dataset_hashes(root)

    def test_mismatched_manifest_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            original = {"sha256": "old", "sentinel": 7}
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "left untouched"):
                freeze_protocol_manifest(path, {"sha256": "new"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)

    def test_enhancement_sources_are_protocol_frozen(self) -> None:
        self.assertIn("enhancements_logic_tree.py", PROTOCOL_SOURCE_FILES)
        self.assertIn("enhancements_hadamard.py", PROTOCOL_SOURCE_FILES)
        self.assertIn("enhancements_global_lut.py", PROTOCOL_SOURCE_FILES)
        source_root = Path(__file__).resolve().parent
        hashes = source_hashes(source_root)
        self.assertEqual(set(hashes), set(PROTOCOL_SOURCE_FILES))
        self.assertEqual(len(hashes["enhancements_logic_tree.py"]), 64)
        self.assertEqual(len(hashes["enhancements_hadamard.py"]), 64)
        self.assertEqual(len(hashes["enhancements_global_lut.py"]), 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fake_dataset(root)
            common = {
                "data_root": root,
                "out_dir": root / "run",
                "valid_size": 5_000,
                "resume": False,
            }
            depthwise = protocol(argparse.Namespace(
                **common, local_operator="depthwise_shiftadd"
            ))
            logic_tree = protocol(argparse.Namespace(
                **common, local_operator="logic_tree3x3"
            ))
        self.assertNotEqual(depthwise["sha256"], logic_tree["sha256"])
        self.assertEqual(logic_tree["args"]["local_operator"], "logic_tree3x3")

    def test_forced_projection_a_is_exact_and_restores_shadow_logits(self) -> None:
        branch = SharedLogicTreeConv3x3(dim=2, grid_size=2)
        with torch.no_grad():
            branch.truth_table_logits[..., 0, :].copy_(
                torch.tensor([-2.0, 2.0, 2.0, -2.0])
            )
        original = branch.truth_table_logits.detach().clone()
        self.assertTrue(bool((branch.hard_truth_nibbles() != 0xC).any()))
        with forced_projection_a(branch):
            self.assertTrue(bool((branch.hard_truth_nibbles() == 0xC).all()))
        torch.testing.assert_close(branch.truth_table_logits, original)

    def test_global_lut_payload_is_excluded_from_adamw_weight_decay(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=8, patch_size=4, dim=8, depth=1, heads=2, topk=2,
            mlp_ratio=2.0, weight_bits=4, activation_bits=8, qk_lanes=2,
            global_mixer="parallel_lut_tree", global_lut_group_size=4,
        )
        groups = optimizer_parameter_groups(model, weight_decay=0.05)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["parameter_role"], "base_model")
        self.assertEqual(groups[0]["weight_decay"], 0.05)
        self.assertEqual(groups[1]["parameter_role"], "global_lut_payload")
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        grouped_parameters = [
            parameter for group in groups for parameter in group["params"]
        ]
        self.assertEqual(
            {id(parameter) for parameter in grouped_parameters},
            {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
        )
        self.assertEqual(len(grouped_parameters), len({id(p) for p in grouped_parameters}))

    def test_optimizer_role_gradient_norms_are_measured_before_shared_clip(self) -> None:
        model = EnhancedFullDiscreteViT(
            image_size=8, patch_size=4, dim=8, depth=1, heads=2, topk=2,
            mlp_ratio=2.0, weight_bits=4, activation_bits=8, qk_lanes=2,
            global_mixer="parallel_lut_tree", global_lut_group_size=4,
        )
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, weight_decay=0.05), lr=1e-3
        )
        base = optimizer.param_groups[0]["params"][0]
        lut = optimizer.param_groups[1]["params"][0]
        base.grad = torch.full_like(base, 3.0)
        lut.grad = torch.full_like(lut, 4.0)
        observed = optimizer_role_gradient_l2(optimizer)
        self.assertAlmostEqual(
            observed["base_model"], 3.0 * base.numel() ** 0.5, places=5
        )
        self.assertAlmostEqual(
            observed["global_lut_payload"], 4.0 * lut.numel() ** 0.5, places=4
        )
        global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        self.assertAlmostEqual(
            global_norm,
            (observed["base_model"] ** 2 + observed["global_lut_payload"] ** 2) ** 0.5,
            places=4,
        )

    def test_global_lut_pair_lock_precedes_shared_exit_sentinel(self) -> None:
        source = Path(__file__).with_name(
            "launch_global_lut_smoke_pair_210.sh"
        ).read_text(encoding="utf-8")
        pair_lock = source.index(
            'exec 8>"/tmp/codex_global_lut_smoke_pair_gpu${GPU_INDEX}.lock"'
        )
        duplicate_guard = source.index("flock -n 8", pair_lock)
        exit_sentinel = source.index('queue_exit="$ROOT/global_lut_smoke_pair.exit"')
        shared_lock = source.index(
            'exec 9>"/tmp/codex_lgn_vit_50k_gpu${GPU_INDEX}.lock"'
        )
        self.assertLess(pair_lock, duplicate_guard)
        self.assertLess(duplicate_guard, exit_sentinel)
        self.assertLess(exit_sentinel, shared_lock)


if __name__ == "__main__":
    unittest.main()
