from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate_integer_checkpoint import (
    VALIDATION_ORDER_SEED,
    cifar_source_hashes,
    executor_source_hashes,
    file_sha256,
    load_cifar,
    payload_tensor_inventory,
)
from .export_logic_payload import export_logic_payload, load_checkpoint_model
from .integer_executor import IntegerRuntimeAudit, StrictIntegerExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strict integer inference with the QAT float carrier"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--valid-size", type=int, default=5_000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("limit, batch-size, and threads must be positive")
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint_model(args.checkpoint)
    payload = export_logic_payload(model, checkpoint_metadata=checkpoint)
    executor = StrictIntegerExecutor(payload)
    images, labels = load_cifar(args.data_root, args.split, args.valid_size)
    images = images[: args.limit]
    labels = labels[: args.limit]
    integer_predictions = []
    reference_predictions = []
    exact_logit_rows = 0
    maximum_absolute_logit_difference = 0
    audit = IntegerRuntimeAudit()
    with torch.inference_mode():
        for start in range(0, images.shape[0], args.batch_size):
            stop = min(start + args.batch_size, images.shape[0])
            batch = images[start:stop]
            with audit:
                integer_logits = executor.forward(batch)
            integer_value = torch.ldexp(
                integer_logits.code.to(torch.float32), integer_logits.exponent
            )
            reference_value = model(batch.to(torch.float32) / 255)
            difference = (integer_value - reference_value).abs()
            maximum_absolute_logit_difference = max(
                maximum_absolute_logit_difference,
                float(difference.max().item()),
            )
            exact_logit_rows += int((difference == 0).all(dim=-1).sum().item())
            integer_predictions.append(integer_logits.code.argmax(dim=-1))
            reference_predictions.append(reference_value.argmax(dim=-1))
    integer_prediction = torch.cat(integer_predictions)
    reference_prediction = torch.cat(reference_predictions)
    mismatch = int((integer_prediction != reference_prediction).sum().item())
    result = {
        "checkpoint_step": int(payload["source"]["checkpoint_step"]),
        "count": int(labels.numel()),
        "integer_correct": int((integer_prediction == labels).sum().item()),
        "reference_correct": int((reference_prediction == labels).sum().item()),
        "prediction_mismatches": mismatch,
        "exact_logit_rows": exact_logit_rows,
        "maximum_absolute_logit_difference": maximum_absolute_logit_difference,
        "strict_runtime_audit_operations": audit.operations,
        "strict_runtime_floating_tensor_count": 0,
        "float_reference_is_offline_verification_only": True,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "payload_tensor_element_inventory": payload_tensor_inventory(payload),
        "payload_contains_real_values": False,
        "executor_source_sha256": {
            **executor_source_hashes(),
            Path(__file__).name: file_sha256(Path(__file__)),
        },
        "cifar_source_sha256": cifar_source_hashes(args.data_root, args.split),
        "validation_order_seed": (
            VALIDATION_ORDER_SEED if args.split == "validation" else None
        ),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if mismatch:
        raise RuntimeError("strict integer and QAT carrier predictions diverged")


if __name__ == "__main__":
    main()
