#!/usr/bin/env python3
"""Extract real b-input local LUT cones from existing raw BLIF networks.

This supplements Goal 8 without retraining. It reads the raw network BLIF
artifacts produced by ``lightlogic_lut_minimization.py``, recursively expands
each internal node into a support-limited Boolean cone, exports the resulting
truth table, minimizes it with ABC, and aggregates the structural cost.

The current LightLogic pipeline trains two-input gates, but many layer-1/2
subcones collapse into real b>2 local Boolean functions. This script measures
those real local modules directly instead of relying only on synthetic b-bit
truth tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluate_abc_blif import BlifNetwork, parse_blif
from lut_minimization import LutMinimizer, pattern_for_index, raw_lut_estimate, write_csv


@dataclass(frozen=True)
class ConeFunction:
    support: tuple[str, ...] | None
    truth_bits: str | None
    internal_nodes: frozenset[str]
    depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="reports/goal8_large_scale_v1/combined_network_results.csv",
        help="CSV with raw_network_blif paths from prior Goal 8 runs.",
    )
    parser.add_argument("--abc-path", default="/home/spco/boolean_sat/abc/abc")
    parser.add_argument("--out-dir", default="runs/goal8_real_b_cones_v1")
    parser.add_argument("--min-b", type=int, default=3)
    parser.add_argument("--max-b", type=int, default=6)
    parser.add_argument("--limit-networks", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def eval_cover_scalar(cover: list[str], input_bits: list[int]) -> int:
    parsed_cover: list[tuple[str, str]] = []
    for line in cover:
        parts = line.split()
        if not input_bits:
            parsed_cover.append(("", parts[-1] if parts else "0"))
        elif len(parts) == 1:
            parsed_cover.append((parts[0], "1"))
        elif len(parts) == 2:
            parsed_cover.append((parts[0], parts[1]))
        else:
            raise ValueError(f"unsupported cover line: {line!r}")
    if not parsed_cover:
        return 0

    def matches(pattern: str) -> bool:
        if len(pattern) != len(input_bits):
            raise ValueError((pattern, input_bits))
        for want, got in zip(pattern, input_bits, strict=True):
            if want == "-":
                continue
            if want not in {"0", "1"}:
                raise ValueError(f"unsupported cover token {want!r}")
            if int(want) != int(got):
                return False
        return True

    output_values = {value for _, value in parsed_cover}
    if output_values == {"0"}:
        out = 1
        for pattern, _ in parsed_cover:
            if not input_bits or matches(pattern):
                out = 0
        return out
    if output_values == {"1"}:
        out = 0
        for pattern, _ in parsed_cover:
            if not input_bits or matches(pattern):
                out = 1
        return out
    out = 0
    for pattern, value in parsed_cover:
        if not input_bits or matches(pattern):
            if value not in {"0", "1"}:
                raise ValueError(f"unsupported BLIF output value {value!r}")
            out = int(value)
    return out


def ordered_union(items: list[tuple[str, ...]], order: dict[str, int]) -> tuple[str, ...]:
    merged = {name for support in items for name in support}
    return tuple(sorted(merged, key=lambda name: order[name]))


def truth_value(truth_bits: str, support: tuple[str, ...], assignment: dict[str, int]) -> int:
    if not support:
        return int(truth_bits[0] == "1")
    index_bits = "".join(str(int(assignment[name])) for name in support)
    return int(truth_bits[int(index_bits, 2)] == "1")


def assignment_pattern(index: int, width: int) -> str:
    if width == 0:
        return ""
    return pattern_for_index(index, width)


def extract_network_cones(
    network: BlifNetwork,
    max_b: int,
) -> dict[str, ConeFunction]:
    input_order = {name: idx for idx, name in enumerate(network.inputs)}
    node_by_output = {output_name: (input_names, output_name, cover) for input_names, output_name, cover in network.nodes}
    memo: dict[str, ConeFunction] = {}

    def build(signal: str) -> ConeFunction:
        cached = memo.get(signal)
        if cached is not None:
            return cached
        if signal in input_order:
            result = ConeFunction((signal,), "01", frozenset(), 0)
            memo[signal] = result
            return result
        node = node_by_output.get(signal)
        if node is None:
            result = ConeFunction(None, None, frozenset(), -1)
            memo[signal] = result
            return result

        input_names, output_name, cover = node
        child_functions = [build(name) for name in input_names]
        if any(child.support is None or child.truth_bits is None for child in child_functions):
            result = ConeFunction(None, None, frozenset(), -1)
            memo[signal] = result
            return result

        support = ordered_union([child.support for child in child_functions if child.support is not None], input_order)
        if len(support) > max_b:
            result = ConeFunction(None, None, frozenset(), -1)
            memo[signal] = result
            return result

        bits: list[str] = []
        for row_idx in range(1 << len(support)):
            assignment_bits = assignment_pattern(row_idx, len(support))
            assignment = {
                name: int(bit)
                for name, bit in zip(support, assignment_bits, strict=True)
            }
            node_inputs = [
                truth_value(child.truth_bits or "", child.support or (), assignment)
                for child in child_functions
            ]
            bits.append(str(eval_cover_scalar(cover, node_inputs)))

        internal_nodes = frozenset({output_name, *(name for child in child_functions for name in child.internal_nodes)})
        depth = 1 + max((child.depth for child in child_functions), default=0)
        result = ConeFunction(support, "".join(bits), internal_nodes, depth)
        memo[signal] = result
        return result

    for _, output_name, _ in network.nodes:
        build(output_name)
    return memo


def summarize_network(rows: list[dict[str, object]], meta: dict[str, str]) -> dict[str, object]:
    if not rows:
        return {
            "source_run": meta.get("source_run", ""),
            "dataset": meta.get("dataset", ""),
            "calibration_mode": meta.get("calibration_mode", meta.get("calibration_mode", "")),
            "seed": meta.get("seed", ""),
            "k": meta.get("k", ""),
            "raw_network_blif": meta.get("raw_network_blif", ""),
            "extractable_cones": 0,
            "unique_truths": 0,
            "min_b_seen": "",
            "max_b_seen": "",
            "mean_b": math.nan,
            "mean_raw_mux2": math.nan,
            "mean_abc_and": math.nan,
            "mean_ratio": math.nan,
            "mean_cone_gate_count": math.nan,
            "max_cone_depth": "",
            "all_equivalent": True,
        }

    unique_truths = {(int(row["b"]), str(row["truth_bits"])) for row in rows}
    ratios = [float(row["abc_and_count"]) / max(int(row["raw_mux2_count"]), 1) for row in rows]
    return {
        "source_run": meta.get("source_run", ""),
        "dataset": meta.get("dataset", ""),
        "calibration_mode": meta.get("calibration_mode", meta.get("calibration_mode", "")),
        "seed": meta.get("seed", ""),
        "k": meta.get("k", ""),
        "raw_network_blif": meta.get("raw_network_blif", ""),
        "extractable_cones": len(rows),
        "unique_truths": len(unique_truths),
        "min_b_seen": min(int(row["b"]) for row in rows),
        "max_b_seen": max(int(row["b"]) for row in rows),
        "mean_b": sum(int(row["b"]) for row in rows) / len(rows),
        "mean_raw_mux2": sum(int(row["raw_mux2_count"]) for row in rows) / len(rows),
        "mean_abc_and": sum(int(row["abc_and_count"]) for row in rows) / len(rows),
        "mean_ratio": sum(ratios) / len(ratios),
        "mean_cone_gate_count": sum(int(row["cone_gate_count"]) for row in rows) / len(rows),
        "max_cone_depth": max(int(row["cone_depth"]) for row in rows),
        "all_equivalent": all(bool(row["equivalent"]) and bool(row["source_equivalent"]) for row in rows),
    }


def build_markdown_report(
    instance_rows: list[dict[str, object]],
    network_rows: list[dict[str, object]],
    unique_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Goal 8 Real b-Input Cone LUT Report",
        "",
        "This run post-processes existing raw LightLogic BLIF networks, extracts real support-limited local cones with b > 2, and minimizes those cone truth tables with the same ABC flow used for Goal 8.",
        "",
        f"- input_csv: `{args.input_csv}`",
        f"- min_b: {args.min_b}",
        f"- max_b: {args.max_b}",
        f"- networks processed: {len(network_rows)}",
        f"- extracted cone instances: {len(instance_rows)}",
        f"- unique cone truth tables: {len(unique_rows)}",
        f"- all unique cones equivalent after ABC: {sum(1 for row in unique_rows if row.get('source_equivalent') and row.get('equivalent'))}/{len(unique_rows)}",
        "",
        "| b | cones | mean_raw_mux2 | mean_abc_and | mean_ratio | mean_cone_gate_count | max_abc_level |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_b: dict[int, list[dict[str, object]]] = {}
    for row in instance_rows:
        by_b.setdefault(int(row["b"]), []).append(row)
    for b in sorted(by_b):
        rows = by_b[b]
        mean_raw = sum(int(row["raw_mux2_count"]) for row in rows) / len(rows)
        mean_and = sum(int(row["abc_and_count"]) for row in rows) / len(rows)
        mean_ratio = sum(float(row["abc_and_count"]) / max(int(row["raw_mux2_count"]), 1) for row in rows) / len(rows)
        mean_cone = sum(int(row["cone_gate_count"]) for row in rows) / len(rows)
        max_level = max(int(row["abc_level"]) for row in rows)
        lines.append(
            f"| {b} | {len(rows)} | {mean_raw:.6g} | {mean_and:.6g} | {mean_ratio:.6g} | {mean_cone:.6g} | {max_level} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- `cone_gate_count` is the number of original 2-input BLIF nodes in the extracted subcone DAG.",
            "- `raw_mux2_count` is the direct b-input LUT mux-tree estimate `2^b - 1` for the same truth table.",
            "- `abc_and_count` / `abc_level` come from `read_blif; strash; dc2; print_stats` on each unique cone LUT.",
            "- This supplements, rather than replaces, the existing network-scale Goal 8 runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.quick:
        args.limit_networks = 8 if args.limit_networks <= 0 else min(args.limit_networks, 8)
    if args.min_b < 1 or args.max_b < args.min_b:
        raise ValueError((args.min_b, args.max_b))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_rows = read_csv_rows(Path(args.input_csv))
    if args.limit_networks > 0:
        input_rows = input_rows[: args.limit_networks]

    minimizer = LutMinimizer(Path(args.abc_path), out_dir)
    cone_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []

    for row_idx, meta in enumerate(input_rows, start=1):
        raw_path = Path(meta["raw_network_blif"])
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        network = parse_blif(raw_path)
        cones = extract_network_cones(network, args.max_b)
        local_rows: list[dict[str, object]] = []
        for node_id, (_, output_name, _) in enumerate(network.nodes):
            cone = cones.get(output_name)
            if cone is None or cone.support is None or cone.truth_bits is None:
                continue
            b = len(cone.support)
            if b < args.min_b or b > args.max_b:
                continue
            estimate = raw_lut_estimate(b, cone.truth_bits)
            result = minimizer.minimize(b, cone.truth_bits)
            instance = {
                "source_run": meta.get("source_run", ""),
                "dataset": meta.get("dataset", ""),
                "calibration_mode": meta.get("calibration_mode", meta.get("calibration_mode", "")),
                "seed": int(meta.get("seed", "0")),
                "k": int(meta.get("k", "0")),
                "network_row_index": row_idx - 1,
                "node_index": node_id,
                "node_output": output_name,
                "b": b,
                "support": " ".join(cone.support),
                "truth_bits": cone.truth_bits,
                "cone_gate_count": len(cone.internal_nodes),
                "cone_depth": cone.depth,
                "raw_lut_bits": estimate.raw_lut_bits,
                "raw_mux2_count": estimate.raw_mux2_count,
                "raw_sop_literals": estimate.raw_sop_literals,
                "raw_sop_not_estimate": estimate.raw_sop_not_estimate,
                "on_set_size": estimate.on_set_size,
                "truth_hash": result.truth_hash,
                "abc_status": result.abc_status,
                "abc_and_count": result.abc_and_count,
                "abc_level": result.abc_level,
                "abc_seconds": result.abc_seconds,
                "source_equivalent": result.source_equivalent,
                "equivalent": result.equivalent,
                "raw_blif_path": result.raw_blif_path,
                "optimized_blif_path": result.optimized_blif_path,
            }
            local_rows.append(instance)
        cone_rows.extend(local_rows)
        network_rows.append(summarize_network(local_rows, meta))
        print(
            "network={} dataset={} seed={} k={} cones={}".format(
                row_idx,
                meta.get("dataset", ""),
                meta.get("seed", ""),
                meta.get("k", ""),
                len(local_rows),
            ),
            flush=True,
        )

    unique_rows = [asdict(item) for item in minimizer.cache.values()]
    write_csv(out_dir / "real_b_cone_instances.csv", cone_rows)
    write_csv(out_dir / "real_b_network_summary.csv", network_rows)
    write_csv(out_dir / "real_b_unique_lut_minimization.csv", unique_rows)
    (out_dir / "real_b_cone_summary.md").write_text(
        build_markdown_report(cone_rows, network_rows, unique_rows, args),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "real_b_cone_instances.csv")
    print(out_dir / "real_b_network_summary.csv")
    print(out_dir / "real_b_cone_summary.md")


if __name__ == "__main__":
    main()
