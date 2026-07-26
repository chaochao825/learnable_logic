from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from hard_lgn_gap_proto.boolean_executor import BooleanRuntimeAudit
from vit_lgn.bitstate.boolean_executor import StrictBitStateExecutor
from vit_lgn.bitstate.model import BitStateConfig, BitStateViT


def _tensor_inventory(value: Any, output: dict[str, int]) -> None:
    if isinstance(value, torch.Tensor):
        name = str(value.dtype).removeprefix("torch.")
        output[name] = output.get(name, 0) + value.numel()
    elif isinstance(value, dict):
        for child in value.values():
            _tensor_inventory(child, output)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _tensor_inventory(child, output)


def export_payload(
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path | None,
    verify_samples: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = BitStateViT(BitStateConfig(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["model"])
    payload = model.deployment_payload()
    executor = StrictBitStateExecutor(payload)
    generator = torch.Generator().manual_seed(20260723)
    images = torch.randint(
        0,
        256,
        (
            verify_samples,
            model.config.in_channels,
            model.config.image_size,
            model.config.image_size,
        ),
        generator=generator,
        dtype=torch.uint8,
    )
    audit = BooleanRuntimeAudit()
    with audit:
        strict_logits = executor.logits(images)
    reference_logits = model.forward_bits(images)
    exact_logits = bool(torch.equal(strict_logits, reference_logits))
    if not exact_logits:
        raise AssertionError("exported payload differs from checkpoint hard path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=False)
    StrictBitStateExecutor(reloaded)
    inventory: dict[str, int] = {}
    _tensor_inventory(reloaded, inventory)
    result: dict[str, object] = {
        "path": output_path.name,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "bytes": output_path.stat().st_size,
        "source_checkpoint": checkpoint_path.name,
        "source_checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "exporter_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "contains_real_values": False,
        "tensor_element_inventory": inventory,
        "accepted_input_dtypes": ["uint8"],
        "persistent_state_dtype": "bool",
        "classifier_output_dtype": "int32",
        "verification": {
            "samples": verify_samples,
            "exact_logits": exact_logits,
            "operator_audit_operations": audit.operations,
            "floating_tensor_count": 0,
        },
    }
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.setdefault("artifacts", {})
        artifacts["deployment_payload"] = result
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a standalone Boolean/integer BitState artifact"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-samples", type=int, default=1)
    args = parser.parse_args()
    if args.verify_samples < 1:
        raise ValueError("verify-samples must be positive")
    output = args.output or args.checkpoint.with_name("deployment_payload.pt")
    result = export_payload(
        args.checkpoint,
        output,
        args.manifest,
        args.verify_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
