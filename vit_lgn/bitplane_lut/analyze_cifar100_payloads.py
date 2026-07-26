"""Structural diagnostics for scaled CIFAR-100 bit-plane LUT payloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import torch

from vit_lgn.bitplane_lut.executor import validate_hard_payload
from vit_lgn.bitplane_lut.layers import (
    deterministic_candidate_indices,
    sequence_candidate_indices,
    spatial_candidate_indices,
)
from vit_lgn.bitplane_lut.train_digits import atomic_json, file_sha256


RUN_PATTERN = re.compile(
    r"^(?:smoke|full)-(?:flat|spatial)-v(?:64|128|256)-d(?:2|4)-s\d+$"
)


def truth_as_int(table: torch.Tensor) -> int:
    value = 0
    for address, bit in enumerate(table.tolist()):
        value |= (int(bit) & 1) << address
    return value


def essential_slots(truth: int, arity: int) -> list[int]:
    return [
        slot
        for slot in range(arity)
        if any(
            ((truth >> address) & 1)
            != ((truth >> (address ^ (1 << slot))) & 1)
            for address in range(1 << arity)
        )
    ]


def realized_function(
    sources: list[int], truth: int
) -> tuple[list[int], int, list[int]]:
    unique: list[int] = []
    source_to_unique: list[int] = []
    for source in sources:
        if source not in unique:
            unique.append(source)
        source_to_unique.append(unique.index(source))
    reduced = 0
    for assignment in range(1 << len(unique)):
        address = 0
        for slot, unique_index in enumerate(source_to_unique):
            address |= ((assignment >> unique_index) & 1) << slot
        reduced |= ((truth >> address) & 1) << assignment
    return unique, reduced, essential_slots(reduced, len(unique))


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("empty percentile input")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def candidate_tensor(
    manifest_args: dict[str, object],
    *,
    input_bits: int,
    output_bits: int,
    preserved_bits: int,
    num_classes: int,
    input_shape: tuple[int, ...],
    block_index: int,
    layer_index: int,
) -> torch.Tensor:
    seed = (
        int(manifest_args["seed"])
        + block_index * 100_003
        + layer_index * 10_007
    )
    common = {
        "arity": int(manifest_args["arity"]),
        "candidate_count": int(manifest_args["candidate_count"]),
        "seed": seed,
    }
    if manifest_args["candidate_policy"] == "mixed":
        return deterministic_candidate_indices(
            input_bits,
            output_bits,
            identity_offset=preserved_bits,
            **common,
        )
    if manifest_args["candidate_policy"] == "image_spatial":
        if len(input_shape) != 3:
            raise ValueError("image payload is missing its three-dimensional shape")
        return spatial_candidate_indices(
            input_bits,
            preserved_bits,
            output_bits,
            num_classes=num_classes,
            image_shape=input_shape,
            **common,
        )
    if manifest_args["candidate_policy"] == "sequence_causal":
        if len(input_shape) != 1:
            raise ValueError("sequence payload is missing its context shape")
        return sequence_candidate_indices(
            input_bits,
            preserved_bits,
            output_bits,
            num_classes=num_classes,
            sequence_length=input_shape[0],
            **common,
        )
    raise ValueError("unsupported candidate policy")


def analyze_run(run_dir: Path) -> dict[str, object]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    payload_path = run_dir / "hard_payload.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    validate_hard_payload(payload)
    manifest_args = manifest["args"]
    input_bits = int(payload["input_bits"])
    state_bits = int(payload["state_bits"])
    preserved_bits = int(payload["preserved_bits"])
    vote_bits = int(payload["vote_bits"])
    num_classes = int(payload["num_classes"])
    input_shape = tuple(int(value) for value in payload.get("input_shape") or ())
    route = payload["encoder_route"].to(torch.int64).tolist()
    raw_masks = [1 << index for index in range(input_bits)]
    state_support = [raw_masks[index] for index in route]

    total_connections = 0
    raw_connections = 0
    vote_connections = 0
    identity_connections = 0
    nondefault_connections = 0
    truth_values: list[int] = []
    truth_support_histogram = [0, 0, 0, 0, 0]
    realized_support_histogram = [0, 0, 0, 0, 0]
    layer_rows: list[dict[str, object]] = []
    for block_index, block in enumerate(payload["blocks"]):
        for layer_index, layer in enumerate(block["layers"]):
            sources = layer["source_indices"].to(torch.int64)
            tables = layer["truth_table"].to(torch.uint8)
            candidates = candidate_tensor(
                manifest_args,
                input_bits=int(layer["input_bits"]),
                output_bits=int(layer["output_bits"]),
                preserved_bits=preserved_bits,
                num_classes=num_classes,
                input_shape=input_shape,
                block_index=block_index,
                layer_index=layer_index,
            )
            matches = sources.unsqueeze(-1).eq(candidates)
            if not bool(matches.any(dim=-1).all()) or bool(
                (matches.sum(dim=-1) != 1).any()
            ):
                raise RuntimeError("payload source is not in its fixed candidate pool")
            ranks = matches.to(torch.int64).argmax(dim=-1)
            layer_raw = int((sources < preserved_bits).sum().item())
            layer_vote = int((sources >= preserved_bits).sum().item())
            identity = preserved_bits + torch.arange(sources.shape[0])
            layer_identity = int((sources == identity[:, None]).sum().item())
            layer_nondefault = int((ranks != 0).sum().item())
            layer_support_counts: list[int] = []
            output_support: list[int] = []
            for gate, (source_row, table) in enumerate(zip(sources.tolist(), tables)):
                truth = truth_as_int(table)
                unique_sources, reduced_truth, realized_slots = realized_function(
                    source_row, truth
                )
                truth_values.append(truth)
                truth_support_histogram[len(essential_slots(truth, len(source_row)))] += 1
                realized_support_histogram[len(realized_slots)] += 1
                support = 0
                for slot in realized_slots:
                    support |= state_support[unique_sources[slot]]
                output_support.append(support)
                layer_support_counts.append(support.bit_count())
            state_support = [*state_support[:preserved_bits], *output_support]
            connections = int(sources.numel())
            total_connections += connections
            raw_connections += layer_raw
            vote_connections += layer_vote
            identity_connections += layer_identity
            nondefault_connections += layer_nondefault
            layer_rows.append(
                {
                    "block": block_index,
                    "layer": layer_index,
                    "raw_source_ratio": layer_raw / connections,
                    "vote_source_ratio": layer_vote / connections,
                    "identity_source_ratio": layer_identity / connections,
                    "nondefault_candidate_ratio": layer_nondefault / connections,
                    "candidate_rank_mean": float(ranks.to(torch.float64).mean().item()),
                    "structural_input_support_mean": sum(layer_support_counts)
                    / len(layer_support_counts),
                    "structural_input_support_p50": percentile(layer_support_counts, 0.5),
                    "structural_input_support_p90": percentile(layer_support_counts, 0.9),
                    "structural_input_support_max": max(layer_support_counts),
                    "zero_input_support_ratio": layer_support_counts.count(0)
                    / len(layer_support_counts),
                }
            )

    final_support = state_support[preserved_bits:]
    final_support_counts = [mask.bit_count() for mask in final_support]
    class_support = []
    for class_index in range(num_classes):
        mask = 0
        for vote_index in range(class_index, vote_bits, num_classes):
            mask |= final_support[vote_index]
        class_support.append(mask.bit_count())
    zero_support_classes = [
        class_index
        for class_index, support_count in enumerate(class_support)
        if support_count == 0
    ]
    vocabulary = manifest.get("vocabulary")
    zero_support_labels = (
        [str(vocabulary[index]) for index in zero_support_classes]
        if isinstance(vocabulary, list) and len(vocabulary) == num_classes
        else []
    )
    gate_count = int(result["structure"]["gate_count"])
    payload_sha256 = file_sha256(payload_path)
    if payload_sha256 != result["hard_payload_sha256"]:
        raise RuntimeError("payload hash differs from the recorded training result")
    unique_truth = len(set(truth_values))
    truth_dictionary_bits = unique_truth * 16 + gate_count * max(
        1, math.ceil(math.log2(max(1, unique_truth)))
    )
    source_index_bits = max(1, math.ceil(math.log2(state_bits)))
    learned_source_bits = gate_count * int(manifest_args["arity"]) * source_index_bits
    learned_truth_bits = gate_count * (1 << int(manifest_args["arity"]))
    output = {
        "run_id": run_dir.name,
        "candidate_policy": manifest_args["candidate_policy"],
        "votes_per_class": int(manifest_args["votes_per_class"]),
        "depth": int(result["structure"]["learned_logic_depth"]),
        "gate_count": gate_count,
        "connections": total_connections,
        "raw_source_ratio": raw_connections / total_connections,
        "vote_source_ratio": vote_connections / total_connections,
        "identity_source_ratio": identity_connections / total_connections,
        "nondefault_candidate_ratio": nondefault_connections / total_connections,
        "unused_gate_ratio": result["structure"]["unused_gate_ratio"],
        "truth_unique_count": unique_truth,
        "truth_dictionary_bits": truth_dictionary_bits,
        "truth_dictionary_ratio": truth_dictionary_bits / learned_truth_bits,
        "learned_truth_bits": learned_truth_bits,
        "learned_source_index_bits": learned_source_bits,
        "learned_logic_payload_bits": learned_truth_bits + learned_source_bits,
        "serialized_payload_bytes": payload_path.stat().st_size,
        "payload_sha256": payload_sha256,
        "analyzer_sha256": file_sha256(Path(__file__)),
        "truth_support_histogram": truth_support_histogram,
        "realized_truth_support_histogram": realized_support_histogram,
        "final_vote_input_support_mean": sum(final_support_counts)
        / len(final_support_counts),
        "final_vote_input_support_p50": percentile(final_support_counts, 0.5),
        "final_vote_input_support_p90": percentile(final_support_counts, 0.9),
        "final_vote_input_support_max": max(final_support_counts),
        "final_vote_zero_support_ratio": final_support_counts.count(0)
        / len(final_support_counts),
        "class_input_support_mean": sum(class_support) / len(class_support),
        "class_input_support_min": min(class_support),
        "class_input_support_max": max(class_support),
        "class_zero_support_count": len(zero_support_classes),
        "class_zero_support_ratio": len(zero_support_classes) / num_classes,
        "class_zero_support_indices": zero_support_classes,
        "class_zero_support_labels": zero_support_labels,
        "layers": layer_rows,
    }
    atomic_json(run_dir / "payload_diagnostics.json", output)
    return output


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "run_id",
        "candidate_policy",
        "votes_per_class",
        "depth",
        "gate_count",
        "unused_gate_ratio",
        "raw_source_ratio",
        "vote_source_ratio",
        "identity_source_ratio",
        "nondefault_candidate_ratio",
        "truth_unique_count",
        "truth_dictionary_ratio",
        "learned_logic_payload_bits",
        "serialized_payload_bytes",
        "final_vote_input_support_mean",
        "final_vote_input_support_p50",
        "final_vote_input_support_p90",
        "final_vote_input_support_max",
        "final_vote_zero_support_ratio",
        "class_input_support_mean",
        "class_input_support_min",
        "class_input_support_max",
        "class_zero_support_count",
        "class_zero_support_ratio",
        "payload_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        analyze_run(run_dir)
        for run_dir in sorted(args.runs_root.iterdir())
        if run_dir.is_dir()
        and RUN_PATTERN.fullmatch(run_dir.name)
        and (run_dir / "result.json").is_file()
    ]
    if not rows:
        raise RuntimeError("no completed CIFAR-100 payloads found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_summary(args.out, rows)
    print(json.dumps({"runs": len(rows), "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
