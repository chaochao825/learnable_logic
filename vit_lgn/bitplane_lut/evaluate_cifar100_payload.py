"""Post-hoc strict training-set replay for CIFAR-100 hard payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from vit_lgn.bitplane_lut.executor import (
    BooleanRuntimeAudit,
    StrictBitPlaneLUTExecutor,
    validate_hard_payload,
)
from vit_lgn.bitplane_lut.train_cifar100 import NUM_CLASSES, load_splits
from vit_lgn.bitplane_lut.train_digits import atomic_json, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a CIFAR-100 hard payload on its fixed training split"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest_path = args.run_dir / "run_manifest.json"
    result_path = args.run_dir / "result.json"
    payload_path = args.run_dir / "hard_payload.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    training_args = manifest["args"]
    data_root = args.data_root or Path(training_args["data_root"])
    train_split, _, _, _, split_metadata = load_splits(
        data_root,
        download=False,
        split_seed=int(training_args["split_seed"]),
        max_train_samples=int(training_args["max_train_samples"]),
        calibration_samples=int(training_args["calibration_samples"]),
    )
    for key in (
        "train_size",
        "train_index_sha256",
        "official_train_image_sha256",
        "official_train_label_sha256",
    ):
        if split_metadata[key] != manifest["split"][key]:
            raise RuntimeError(f"training split provenance mismatch: {key}")

    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    validate_hard_payload(payload)
    executor = StrictBitPlaneLUTExecutor(payload)
    symbols, labels = train_split
    class_total = torch.bincount(labels, minlength=NUM_CLASSES).to(torch.int64)
    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    prediction_count = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    correct = 0
    positive_margin = 0
    margin_sum = 0
    audit = BooleanRuntimeAudit()
    started = time.perf_counter()
    for start in range(0, symbols.shape[0], args.batch_size):
        batch = symbols[start : start + args.batch_size]
        batch_labels = labels[start : start + args.batch_size]
        with audit:
            logits = executor.logits(batch)
        prediction = logits.argmax(dim=-1).to(torch.int64)
        matched = prediction == batch_labels
        correct += int(matched.sum(dtype=torch.int64).item())
        prediction_count += torch.bincount(
            prediction, minlength=NUM_CLASSES
        ).to(torch.int64)
        class_correct += torch.bincount(
            batch_labels[matched], minlength=NUM_CLASSES
        ).to(torch.int64)
        true_logit = logits.gather(1, batch_labels.view(-1, 1)).squeeze(1)
        competing = logits.clone()
        competing.scatter_(1, batch_labels.view(-1, 1), torch.iinfo(logits.dtype).min)
        margin = true_logit.to(torch.int64) - competing.max(dim=1).values.to(torch.int64)
        positive_margin += int((margin > 0).sum(dtype=torch.int64).item())
        margin_sum += int(margin.sum(dtype=torch.int64).item())

    total = int(labels.numel())
    output = {
        "dataset": "cifar100",
        "split": "fixed_training",
        "rows": total,
        "hard_correct": correct,
        "hard_acc": correct / total,
        "strict_positive_margin_rows": positive_margin,
        "strict_positive_margin_ratio": positive_margin / total,
        "integer_margin_mean": margin_sum / total,
        "class_total": class_total.tolist(),
        "class_correct": class_correct.tolist(),
        "prediction_count": prediction_count.tolist(),
        "predicted_class_count": int((prediction_count > 0).sum().item()),
        "audit_operations": audit.operations,
        "float_tensor_count": 0,
        "runtime_compliance": "operator_audited_bool_int",
        "elapsed_seconds": time.perf_counter() - started,
        "payload_sha256": file_sha256(payload_path),
        "payload_matches_result": (
            file_sha256(payload_path) == result["hard_payload_sha256"]
        ),
        "manifest_sha256": manifest["sha256"],
        "evaluator_sha256": file_sha256(Path(__file__)),
    }
    if not output["payload_matches_result"]:
        raise RuntimeError("payload hash differs from the recorded training result")
    atomic_json(args.run_dir / "train_strict_metrics.json", output)
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
