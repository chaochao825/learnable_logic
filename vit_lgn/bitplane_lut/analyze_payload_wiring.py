from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import torch

from .layers import deterministic_candidate_indices


RUN_PATTERN = re.compile(r"^v2_(argmax|truth|wiring)_w(672|832)_s([012])$")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(directory: Path, mode: str, state_bits: int, seed: int) -> dict[str, object]:
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    payload = torch.load(directory / "hard_payload.pt", map_location="cpu", weights_only=True)
    args = manifest["args"]
    if int(args["state_bits"]) != state_bits or int(args["seed"]) != seed:
        raise ValueError(f"run identity mismatch in {directory}")
    if payload["schema"] != "a8_bitplane_hard_lgn_v1":
        raise ValueError(f"unexpected payload schema in {directory}")
    if payload["learned_dense_integer_matrix_count"] != 0:
        raise ValueError(f"dense learned matrix in {directory}")

    preserved_bits = int(payload["preserved_bits"])
    candidate_count = int(args["candidate_count"])
    arity = int(args["arity"])
    changed_connections = 0
    total_connections = 0
    raw_connections = 0
    candidate_rank_sum = 0
    candidate_rank_max = 0
    changed_truth_bits = 0
    total_truth_bits = 0
    rank_counts = torch.zeros(candidate_count, dtype=torch.int64)
    layer_rows = []

    for block_index, block in enumerate(payload["blocks"]):
        for layer_index, layer in enumerate(block["layers"]):
            sources = layer["source_indices"].to(torch.int64)
            truth = layer["truth_table"].bool()
            layer_seed = seed + block_index * 100_003 + layer_index * 10_007
            candidates = deterministic_candidate_indices(
                input_bits=int(layer["input_bits"]),
                output_bits=int(layer["output_bits"]),
                arity=int(layer["arity"]),
                candidate_count=candidate_count,
                seed=layer_seed,
                identity_offset=preserved_bits,
            )
            matches = sources.unsqueeze(-1).eq(candidates)
            if not bool(matches.any(dim=-1).all()) or bool((matches.sum(dim=-1) != 1).any()):
                raise ValueError(f"exported source is not a unique candidate in {directory}")
            ranks = matches.to(torch.int64).argmax(dim=-1)
            projection = torch.bitwise_and(
                torch.arange(1 << arity, dtype=torch.int64), 1
            ).bool().expand_as(truth)

            layer_connections = int(ranks.numel())
            layer_changed = int((ranks != 0).sum().item())
            layer_truth_bits = int(truth.numel())
            layer_truth_changed = int((truth != projection).sum().item())
            changed_connections += layer_changed
            total_connections += layer_connections
            raw_connections += int((sources < preserved_bits).sum().item())
            candidate_rank_sum += int(ranks.sum().item())
            candidate_rank_max = max(candidate_rank_max, int(ranks.max().item()))
            changed_truth_bits += layer_truth_changed
            total_truth_bits += layer_truth_bits
            rank_counts += torch.bincount(ranks.flatten(), minlength=candidate_count)
            layer_rows.append({
                "block": block_index,
                "layer": layer_index,
                "nondefault_wiring_ratio": layer_changed / layer_connections,
                "raw_input_source_ratio": float((sources < preserved_bits).float().mean().item()),
                "candidate_rank_mean": float(ranks.float().mean().item()),
                "candidate_rank_max": int(ranks.max().item()),
                "truth_change_from_initial_ratio": layer_truth_changed / layer_truth_bits,
            })

    return {
        "mode": mode,
        "state_bits": state_bits,
        "vote_bits": state_bits - preserved_bits,
        "seed": seed,
        "connections": total_connections,
        "nondefault_wiring_ratio": changed_connections / total_connections,
        "raw_input_source_ratio": raw_connections / total_connections,
        "candidate_rank_mean": candidate_rank_sum / total_connections,
        "candidate_rank_max": candidate_rank_max,
        "truth_change_from_initial_ratio": changed_truth_bits / total_truth_bits,
        "candidate_rank_counts": json.dumps(rank_counts.tolist(), separators=(",", ":")),
        "layers": json.dumps(layer_rows, separators=(",", ":")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    seen = set()
    for directory in sorted(args.runs_root.iterdir()):
        match = RUN_PATTERN.match(directory.name)
        if not match:
            continue
        mode, width_text, seed_text = match.groups()
        key = (mode, int(width_text), int(seed_text))
        rows.append(analyze_run(directory, *key))
        seen.add(key)
    expected = {
        (mode, width, seed)
        for mode in ("argmax", "truth", "wiring")
        for width in (672, 832)
        for seed in range(3)
    }
    if seen != expected:
        raise ValueError(f"run matrix mismatch: missing={sorted(expected - seen)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    print(json.dumps({"runs": len(rows), "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
