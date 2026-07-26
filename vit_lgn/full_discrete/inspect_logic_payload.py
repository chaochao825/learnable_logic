from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch

from .evaluate_integer_checkpoint import file_sha256
from .export_logic_payload import validate_logic_payload


DTYPE_BITS = {
    torch.bool: 1,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
}


def inspect_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    validate_logic_payload(payload)
    rows: list[dict[str, object]] = []

    def visit(value: object, location: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.dtype not in DTYPE_BITS:
                raise TypeError(f"unsupported tensor dtype at {location}: {value.dtype}")
            rows.append(
                {
                    "path": location,
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "shape": list(value.shape),
                    "elements": value.numel(),
                    "tensor_bits": value.numel() * DTYPE_BITS[value.dtype],
                }
            )
        elif isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")
        elif isinstance(value, float):
            raise TypeError(f"real scalar at {location}")

    visit(payload, "payload")
    by_top_level: dict[str, int] = defaultdict(int)
    by_leaf: dict[str, int] = defaultdict(int)
    by_dtype: dict[str, int] = defaultdict(int)
    for row in rows:
        components = str(row["path"]).split(".")
        top_level = components[1].split("[")[0] if len(components) > 1 else "payload"
        leaf = components[-1].split("[")[0]
        bits = int(row["tensor_bits"])
        by_top_level[top_level] += bits
        by_leaf[leaf] += bits
        by_dtype[str(row["dtype"])] += int(row["elements"])

    layers = payload["shift_add_layers"]
    logical_weight_bits = 0
    coefficient_count = 0
    dense_fanin_max = 0
    dense_fanout_max = 0
    for layer in layers:
        weight = layer["weight_code"]
        coefficient_count += weight.numel()
        logical_weight_bits += weight.numel() * (
            int(layer["target_magnitude_bits"]) + 1
        )
        dense_fanin_max = max(dense_fanin_max, int(weight.shape[1]))
        dense_fanout_max = max(
            dense_fanout_max,
            int(torch.count_nonzero(weight, dim=0).max()),
        )
    total_tensor_bits = sum(int(row["tensor_bits"]) for row in rows)
    return {
        "schema_version": int(payload["schema"]["version"]),
        "source": payload["source"],
        "topology": payload["topology"],
        "artifact": {
            "path": path.name,
            "sha256": file_sha256(path),
            "serialized_bytes": path.stat().st_size,
            "contains_real_values": False,
            "inspector_source_sha256": file_sha256(Path(__file__)),
        },
        "capacity": {
            "tensor_bits": total_tensor_bits,
            "tensor_bytes": (total_tensor_bits + 7) // 8,
            "tensor_element_inventory": dict(sorted(by_dtype.items())),
            "shift_add_layer_count": len(layers),
            "logical_weight_coefficients": coefficient_count,
            "logical_bitpacked_weight_bits": logical_weight_bits,
            "dense_fanin_max": dense_fanin_max,
            "dense_nonzero_fanout_max": dense_fanout_max,
            "transformer_block_depth": int(payload["topology"]["depth"]),
            "general_matrix_multipliers": int(
                payload["arithmetic_contract"]["general_learned_multipliers"]
            ),
        },
        "tensor_bits_by_top_level": dict(
            sorted(by_top_level.items(), key=lambda item: item[1], reverse=True)
        ),
        "tensor_bits_by_leaf_name": dict(
            sorted(by_leaf.items(), key=lambda item: item[1], reverse=True)
        ),
        "largest_tensors": sorted(
            rows,
            key=lambda row: int(row["tensor_bits"]),
            reverse=True,
        )[:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_payload(args.payload)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
