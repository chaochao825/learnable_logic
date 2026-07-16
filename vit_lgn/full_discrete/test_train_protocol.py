from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from vit_lgn.full_discrete.enhancements_logic_tree import SharedLogicTreeConv3x3
from vit_lgn.full_discrete.train_cifar import (
    CIFAR10_PAYLOAD_FILES,
    PROTOCOL_SOURCE_FILES,
    dataset_hashes,
    freeze_protocol_manifest,
    forced_projection_a,
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

    def test_logic_tree_and_hadamard_sources_are_protocol_frozen(self) -> None:
        self.assertIn("enhancements_logic_tree.py", PROTOCOL_SOURCE_FILES)
        self.assertIn("enhancements_hadamard.py", PROTOCOL_SOURCE_FILES)
        source_root = Path(__file__).resolve().parent
        hashes = source_hashes(source_root)
        self.assertEqual(set(hashes), set(PROTOCOL_SOURCE_FILES))
        self.assertEqual(len(hashes["enhancements_logic_tree.py"]), 64)
        self.assertEqual(len(hashes["enhancements_hadamard.py"]), 64)

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


if __name__ == "__main__":
    unittest.main()
