from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Mapping

import torch

from .export_logic_payload import export_logic_payload, load_checkpoint_model
from .integer_executor import IntegerRuntimeAudit, StrictIntegerExecutor


VALIDATION_ORDER_SEED = 20260711


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executor_source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "enhanced_model.py",
        "enhancements_expert.py",
        "enhancements_global_lut.py",
        "enhancements_hadamard.py",
        "enhancements_logic_tree.py",
        "enhancements_lut.py",
        "enhancements_spatial.py",
        "evaluate_integer_checkpoint.py",
        "export_logic_payload.py",
        "integer_executor.py",
        "logic_backend.py",
        "model.py",
        "shiftadd.py",
    )
    return {name: file_sha256(directory / name) for name in names}


def payload_tensor_inventory(value: object) -> dict[str, int]:
    output: dict[str, int] = {}

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            name = str(item.dtype)
            if name.startswith("torch."):
                name = name[len("torch.") :]
            output[name] = output.get(name, 0) + item.numel()
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return output


def cifar_source_hashes(root: Path, split: str) -> dict[str, str]:
    directory = _cifar_directory(root)
    names = ["test_batch"] if split == "test" else [
        f"data_batch_{index}" for index in range(1, 6)
    ]
    names.append("batches.meta")
    return {name: file_sha256(directory / name) for name in names}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a hardened checkpoint with the strict integer executor"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--payload", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--valid-size", type=int, default=5_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--linear-backend", choices=["int_matmul", "lut"], default="int_matmul"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _cifar_directory(root: Path) -> Path:
    nested = root / "cifar-10-batches-py"
    directory = nested if nested.is_dir() else root
    if not (directory / "batches.meta").is_file():
        raise FileNotFoundError(f"CIFAR-10 Python payload not found under {root}")
    return directory


def _load_batch(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = torch.from_numpy(payload[b"data"]).reshape(-1, 3, 32, 32)
    labels = torch.tensor(payload[b"labels"], dtype=torch.int64)
    if data.dtype != torch.uint8:
        raise TypeError(f"{path} does not contain uint8 images")
    return data, labels


def load_cifar(
    root: Path, split: str, valid_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    directory = _cifar_directory(root)
    if split == "test":
        return _load_batch(directory / "test_batch")
    if not 0 < valid_size < 50_000:
        raise ValueError("valid-size must be in (0,50000)")
    batches = [_load_batch(directory / f"data_batch_{index}") for index in range(1, 6)]
    images = torch.cat([item[0] for item in batches], dim=0)
    labels = torch.cat([item[1] for item in batches], dim=0)
    generator = torch.Generator().manual_seed(VALIDATION_ORDER_SEED)
    order = torch.randperm(images.shape[0], generator=generator)[:valid_size]
    return images[order], labels[order]


def load_payload(args: argparse.Namespace) -> Mapping[str, object]:
    if args.payload is not None:
        payload = torch.load(args.payload, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("payload file must contain a mapping")
        return payload
    model, checkpoint = load_checkpoint_model(args.checkpoint)
    return export_logic_payload(model, checkpoint_metadata=checkpoint)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.limit < 0:
        raise ValueError("batch-size/threads must be positive and limit non-negative")
    torch.set_num_threads(args.threads)
    payload = load_payload(args)
    images, labels = load_cifar(args.data_root, args.split, args.valid_size)
    if args.limit:
        images = images[: args.limit]
        labels = labels[: args.limit]
    executor = StrictIntegerExecutor(payload, linear_backend=args.linear_backend)
    audit = IntegerRuntimeAudit()
    correct = 0
    started = time.perf_counter()
    with audit:
        for start in range(0, images.shape[0], args.batch_size):
            stop = min(start + args.batch_size, images.shape[0])
            prediction = executor.predict(images[start:stop])
            correct += int((prediction == labels[start:stop]).sum().item())
    elapsed = time.perf_counter() - started
    count = int(labels.numel())
    result = {
        "schema_version": int(payload["schema"]["version"]),
        "model_type": str(payload["source"]["model_type"]),
        "checkpoint_step": int(payload["source"]["checkpoint_step"]),
        "split": args.split,
        "correct": correct,
        "count": count,
        "accuracy": correct / count if count else 0,
        "elapsed_seconds": elapsed,
        "images_per_second": count / elapsed if elapsed else 0,
        "linear_backend": args.linear_backend,
        "input_dtype": str(images.dtype),
        "logit_code_dtype": "torch.int64",
        "logit_exponent_dtype": "torch.int32",
        "runtime_audit_operations": audit.operations,
        "runtime_floating_tensor_count": 0,
        "training_or_export_float_state_is_not_runtime": True,
        "payload_tensor_element_inventory": payload_tensor_inventory(payload),
        "payload_contains_real_values": False,
        "source_artifact_kind": "payload" if args.payload is not None else "checkpoint",
        "source_artifact_sha256": file_sha256(args.payload or args.checkpoint),
        "source_artifact_bytes": (args.payload or args.checkpoint).stat().st_size,
        "executor_source_sha256": executor_source_hashes(),
        "cifar_source_sha256": cifar_source_hashes(args.data_root, args.split),
        "validation_order_seed": (
            VALIDATION_ORDER_SEED if args.split == "validation" else None
        ),
    }
    encoded = json.dumps(result, sort_keys=True, indent=2)
    print(encoded, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
