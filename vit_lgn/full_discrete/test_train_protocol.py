from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vit_lgn.full_discrete.train_cifar import (
    CIFAR10_PAYLOAD_FILES,
    dataset_hashes,
    freeze_protocol_manifest,
)


class TrainProtocolTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
