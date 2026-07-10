#!/usr/bin/env python3
"""Build XOR-aware LUT reports from existing Goal 8 run artifacts.

This script does not retrain networks. It reads existing `lut_instances.csv`
and `network_minimization_results.csv` files, derives a canonical ANF/XAG
implementation for each unique truth table, and aggregates local
AND/XOR/NOT/depth estimates per network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_RUN_DIRS = [
    "runs/lightlogic_lut_min_goal8_bool_multiseed_fixed_v1",
    "runs/lightlogic_lut_min_goal8_bool_multiseed_per_layer_v1",
    "runs/lightlogic_lut_min_goal8_bool_multiseed_per_neuron_v1",
    "runs/lightlogic_lut_min_goal8_digits_multiseed_v1",
    "runs/lightlogic_lut_min_goal8_digits_per_layer_v1",
    "runs/lightlogic_lut_min_goal8_digits_per_neuron_v1",
    "runs/lightlogic_lut_min_goal8_mnist_scale_v1",
    "runs/lightlogic_lut_min_goal8_mnist_per_layer_v1",
    "runs/lightlogic_lut_min_goal8_cifar10_scale_v1",
    "runs/lightlogic_lut_min_goal8_cifar10_per_layer_v1",
    "runs/lightlogic_b_lut_min_goal7_bool_multiseed_v1",
    "runs/lightlogic_b_lut_min_goal7_digits_multiseed_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_seed0_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b3_w360_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b4_w360_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b3_w360_e150_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b3_w360_warmup_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b4_w360_warmup_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_matched_b3_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_matched_b4_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_matched_b3_multiseed_v1",
]


@dataclass
class XagTruthRow:
    b: int
    truth_bits: str
    truth_hash: str
    anf_nonzero_terms: int
    anf_product_terms: int
    anf_constant_term: int
    xag_and_count: int
    xag_xor_count: int
    xag_not_count: int
    xag_level_estimate: int
    xag_product_depth: int
    xag_xor_depth: int
    xag_analysis_seconds: float
    xag_backend: str = "anf_xag"
    xag_status: str = "ok"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def to_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--out-dir", default="reports/lut_xag_backend_v1")
    return parser.parse_args()


def infer_b(truth_bits: str) -> int:
    length = len(truth_bits)
    if length <= 0 or (length & (length - 1)):
        raise ValueError(f"truth table length must be power of two, got {length}")
    return int(math.log2(length))


def anf_coefficients(truth_bits: str) -> list[int]:
    coeffs = [1 if bit == "1" else 0 for bit in truth_bits]
    b = infer_b(truth_bits)
    for bit in range(b):
        step = 1 << bit
        for mask in range(1 << b):
            if mask & step:
                coeffs[mask] ^= coeffs[mask ^ step]
    return coeffs


def mask_degree(mask: int) -> int:
    return mask.bit_count()


@lru_cache(maxsize=None)
def monomial_plan(mask: int) -> tuple[frozenset[int], int]:
    degree = mask_degree(mask)
    if degree <= 1:
        return frozenset(), 0
    best_closure: frozenset[int] | None = None
    best_depth: int | None = None
    subset = (mask - 1) & mask
    while subset:
        other = mask ^ subset
        if other and subset <= other:
            left_closure, left_depth = monomial_plan(subset)
            right_closure, right_depth = monomial_plan(other)
            closure = left_closure | right_closure | frozenset({mask})
            depth = 1 + max(left_depth, right_depth)
            score = (len(closure), depth)
            if best_closure is None or score < (len(best_closure), best_depth if best_depth is not None else math.inf):
                best_closure = closure
                best_depth = depth
        subset = (subset - 1) & mask
    if best_closure is None or best_depth is None:
        raise ValueError(f"failed to plan mask={mask}")
    return best_closure, best_depth


def xag_from_truth_bits(truth_bits: str) -> XagTruthRow:
    started = time.perf_counter()
    b = infer_b(truth_bits)
    coeffs = anf_coefficients(truth_bits)
    nonzero_terms = [mask for mask, coeff in enumerate(coeffs) if coeff]
    product_terms = [mask for mask in nonzero_terms if mask_degree(mask) >= 2]
    product_closure: set[int] = set()
    product_depth = 0
    for mask in product_terms:
        closure, depth = monomial_plan(mask)
        product_closure.update(closure)
        product_depth = max(product_depth, depth)

    constant_term = 1 if coeffs[0] else 0
    nonconst_terms = [mask for mask in nonzero_terms if mask != 0]
    if constant_term and nonconst_terms:
        xag_not_count = 1
        xag_xor_count = max(len(nonconst_terms) - 1, 0)
        xor_depth = 0 if len(nonconst_terms) <= 1 else math.ceil(math.log2(len(nonconst_terms)))
        level = product_depth + xor_depth + 1
    else:
        xag_not_count = 0
        xag_xor_count = max(len(nonconst_terms) - 1, 0)
        xor_depth = 0 if len(nonconst_terms) <= 1 else math.ceil(math.log2(len(nonconst_terms)))
        level = product_depth + xor_depth
    if not nonconst_terms and constant_term:
        level = 0

    truth_hash = hashlib.sha256(f"{b}:{truth_bits}".encode("ascii")).hexdigest()
    return XagTruthRow(
        b=b,
        truth_bits=truth_bits,
        truth_hash=truth_hash,
        anf_nonzero_terms=len(nonzero_terms),
        anf_product_terms=len(product_terms),
        anf_constant_term=constant_term,
        xag_and_count=len(product_closure),
        xag_xor_count=xag_xor_count,
        xag_not_count=xag_not_count,
        xag_level_estimate=int(level),
        xag_product_depth=int(product_depth),
        xag_xor_depth=int(xor_depth),
        xag_analysis_seconds=time.perf_counter() - started,
    )


def identity_fields(network_rows: list[dict[str, str]], instance_rows: list[dict[str, str]]) -> list[str]:
    if not network_rows or not instance_rows:
        return []
    network_keys = set(network_rows[0].keys())
    instance_keys = set(instance_rows[0].keys())
    shared = network_keys & instance_keys
    candidates = ["dataset", "seed", "k", "calibration_mode", "b"]
    return [field for field in candidates if field in shared]


def group_key(row: dict[str, object], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def build_run_analysis(run_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    network_rows = read_csv(run_dir / "network_minimization_results.csv")
    instance_rows = read_csv(run_dir / "lut_instances.csv")
    id_fields = identity_fields(network_rows, instance_rows)

    xag_cache: dict[tuple[int, str], XagTruthRow] = {}
    enriched_instances: list[dict[str, object]] = []
    grouped_instances: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in instance_rows:
        b = int(row.get("b", infer_b(str(row["truth_bits"]))))
        truth_bits = str(row["truth_bits"])
        key = (b, truth_bits)
        if key not in xag_cache:
            xag_cache[key] = xag_from_truth_bits(truth_bits)
        xag = xag_cache[key]
        merged = dict(row)
        merged.update(asdict(xag))
        merged["run"] = run_dir.name
        enriched_instances.append(merged)
        grouped_instances.setdefault(group_key(merged, id_fields), []).append(merged)

    enriched_network_rows: list[dict[str, object]] = []
    for row in network_rows:
        merged = dict(row)
        merged["run"] = run_dir.name
        group_rows = grouped_instances.get(group_key(merged, id_fields), [])
        if not group_rows:
            merged.update(
                {
                    "xag_local_and_count": "",
                    "xag_local_xor_count": "",
                    "xag_local_not_count": "",
                    "xag_local_level_estimate": "",
                    "xag_unique_truths_seen": "",
                }
            )
            enriched_network_rows.append(merged)
            continue

        layer_levels: dict[int, int] = {}
        for item in group_rows:
            layer = int(item.get("layer", 0))
            layer_levels[layer] = max(layer_levels.get(layer, 0), int(item["xag_level_estimate"]))
        merged.update(
            {
                "xag_backend": "anf_xag",
                "xag_local_and_count": sum(int(item["xag_and_count"]) for item in group_rows),
                "xag_local_xor_count": sum(int(item["xag_xor_count"]) for item in group_rows),
                "xag_local_not_count": sum(int(item["xag_not_count"]) for item in group_rows),
                "xag_local_level_estimate": sum(layer_levels.values()),
                "xag_unique_truths_seen": len({(int(item["b"]), str(item["truth_bits"])) for item in group_rows}),
            }
        )
        if "abc_and_count" in merged:
            merged["xag_minus_aig_and"] = to_float(merged["xag_local_and_count"]) - to_float(merged["abc_and_count"])
        enriched_network_rows.append(merged)

    summary_rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in enriched_network_rows:
        groups.setdefault((run_dir.name, str(row.get("dataset", ""))), []).append(row)
    for (run_name, dataset), rows in sorted(groups.items()):
        summary_rows.append(
            {
                "run": run_name,
                "dataset": dataset,
                "rows": len(rows),
                "best_discrete_or_raw_acc": max(
                    max(to_float(row.get("discrete_acc", math.nan), -math.inf), to_float(row.get("raw_lut_acc", math.nan), -math.inf))
                    for row in rows
                ),
                "xag_and_mean": mean([to_float(row["xag_local_and_count"]) for row in rows]),
                "xag_xor_mean": mean([to_float(row["xag_local_xor_count"]) for row in rows]),
                "xag_not_mean": mean([to_float(row["xag_local_not_count"]) for row in rows]),
                "xag_level_mean": mean([to_float(row["xag_local_level_estimate"]) for row in rows]),
                "aig_and_mean": mean([to_float(row.get("abc_and_count", math.nan)) for row in rows]),
                "aig_level_mean": mean([to_float(row.get("abc_level", math.nan)) for row in rows]),
            }
        )
    return enriched_network_rows, enriched_instances, summary_rows


def build_markdown_report(
    combined_network_rows: list[dict[str, object]],
    combined_summary_rows: list[dict[str, object]],
    unique_truth_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# LUT XAG Backend Report",
        "",
        "This report adds a canonical ANF/XAG-style local backend on top of existing LUT artifacts. It complements the AIG/ABC `and/lev` numbers with local `AND/XOR/NOT` estimates.",
        "",
        f"- combined network rows: {len(combined_network_rows)}",
        f"- unique truth tables analyzed: {len(unique_truth_rows)}",
        "",
        "## Summary By Run/Dataset",
        "",
        "| run | dataset | rows | xag_and_mean | xag_xor_mean | xag_not_mean | xag_level_mean | aig_and_mean | aig_level_mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined_summary_rows:
        lines.append(
            "| {run} | {dataset} | {rows} | {xag_and_mean:.6g} | {xag_xor_mean:.6g} | {xag_not_mean:.6g} | {xag_level_mean:.6g} | {aig_and_mean:.6g} | {aig_level_mean:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Example Network Rows",
            "",
            "| run | dataset | seed | key | xag_local_and_count | xag_local_xor_count | xag_local_not_count | xag_local_level_estimate | abc_and_count | abc_level |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    sample_rows = combined_network_rows[: min(24, len(combined_network_rows))]
    for row in sample_rows:
        if "k" in row:
            key = f"k={row.get('k')}"
        elif "b" in row:
            key = f"b={row.get('b')}"
        else:
            key = ""
        lines.append(
            "| {run} | {dataset} | {seed} | {key} | {xag_local_and_count} | {xag_local_xor_count} | {xag_local_not_count} | {xag_local_level_estimate} | {abc_and_count} | {abc_level} |".format(
                run=row.get("run", ""),
                dataset=row.get("dataset", ""),
                seed=row.get("seed", ""),
                key=key,
                xag_local_and_count=row.get("xag_local_and_count", ""),
                xag_local_xor_count=row.get("xag_local_xor_count", ""),
                xag_local_not_count=row.get("xag_local_not_count", ""),
                xag_local_level_estimate=row.get("xag_local_level_estimate", ""),
                abc_and_count=row.get("abc_and_count", ""),
                abc_level=row.get("abc_level", ""),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `anf_xag` is an exact canonical ANF decomposition of each local LUT truth table, then a shared-product XAG estimate inside that LUT.",
            "- `xag_local_*` numbers are local per-LUT sums; they do not claim global cross-LUT XAG optimization.",
            "- `abc_and_count` / `abc_level` remain the global AIG/ABC numbers from the existing Goal 8 flow.",
            "- `xag_not_count` uses output inversion when the ANF has a constant-one bias term.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_network_rows: list[dict[str, object]] = []
    combined_instance_rows: list[dict[str, object]] = []
    combined_summary_rows: list[dict[str, object]] = []
    unique_truth_map: dict[tuple[int, str], dict[str, object]] = {}

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            continue
        network_rows, instance_rows, summary_rows = build_run_analysis(run_dir)
        combined_network_rows.extend(network_rows)
        combined_instance_rows.extend(instance_rows)
        combined_summary_rows.extend(summary_rows)
        for row in instance_rows:
            unique_truth_map[(int(row["b"]), str(row["truth_bits"]))] = {
                key: row[key]
                for key in [
                    "b",
                    "truth_bits",
                    "truth_hash",
                    "anf_nonzero_terms",
                    "anf_product_terms",
                    "anf_constant_term",
                    "xag_and_count",
                    "xag_xor_count",
                    "xag_not_count",
                    "xag_level_estimate",
                    "xag_product_depth",
                    "xag_xor_depth",
                    "xag_analysis_seconds",
                    "xag_backend",
                    "xag_status",
                ]
            }

    unique_truth_rows = list(unique_truth_map.values())
    write_csv(out_dir / "combined_network_xag_results.csv", combined_network_rows)
    write_csv(out_dir / "combined_lut_instance_xag_results.csv", combined_instance_rows)
    write_csv(out_dir / "summary_by_run_dataset.csv", combined_summary_rows)
    write_csv(out_dir / "unique_truth_xag_results.csv", unique_truth_rows)
    (out_dir / "lut_xag_report.md").write_text(
        build_markdown_report(combined_network_rows, combined_summary_rows, unique_truth_rows),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "lut_xag_report.md")


if __name__ == "__main__":
    main()
