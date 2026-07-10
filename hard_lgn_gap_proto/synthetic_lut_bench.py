#!/usr/bin/env python3
"""Synthetic b-bit LUT minimization sweep for Goal 8 coverage.

This script complements the network-scale Goal 8 runs. The current LightLogic
pipeline only exports 2-input local LUTs, so this synthetic sweep exercises
generic b-input truth tables directly through the same LUT minimization path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from lut_minimization import LutMinimizer


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bits_for_index(index: int, b: int) -> tuple[int, ...]:
    return tuple((index >> shift) & 1 for shift in range(b - 1, -1, -1))


def truth_bits_from_fn(b: int, fn) -> str:
    out = []
    for index in range(1 << b):
        bits = bits_for_index(index, b)
        out.append("1" if fn(bits) else "0")
    return "".join(out)


def deterministic_random_bit(bits: tuple[int, ...], salt: str, threshold: int) -> int:
    digest = hashlib.sha256((salt + "".join(str(bit) for bit in bits)).encode("ascii")).digest()
    return 1 if digest[0] < threshold else 0


def make_case_specs(b: int) -> list[tuple[str, callable]]:
    majority_threshold = (b + 1) // 2
    specs: list[tuple[str, callable]] = [
        ("const0", lambda bits: 0),
        ("const1", lambda bits: 1),
        ("proj0", lambda bits: bits[0]),
        ("proj_last", lambda bits: bits[-1]),
        ("and_all", lambda bits: int(all(bits))),
        ("or_all", lambda bits: int(any(bits))),
        ("parity", lambda bits: sum(bits) & 1),
        ("majority", lambda bits: int(sum(bits) >= majority_threshold)),
        ("exact_one", lambda bits: int(sum(bits) == 1)),
        ("sparse_hash", lambda bits: deterministic_random_bit(bits, f"sparse_b{b}:", 32)),
        ("dense_hash", lambda bits: deterministic_random_bit(bits, f"dense_b{b}:", 128)),
    ]
    if b >= 3:
        specs.append(("mux_last2", lambda bits: bits[-1] if bits[0] else bits[-2]))
    if b >= 6:
        specs.append(("selector4", lambda bits: bits[2 + ((bits[0] << 1) | bits[1])]))
    return specs


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6g}"


def build_report(rows: list[dict[str, object]], summary_rows: list[dict[str, object]], args: argparse.Namespace) -> str:
    lines = [
        "# Synthetic b-bit LUT Minimization Report",
        "",
        "This run supplements the network-scale Goal 8 evidence by exercising synthetic local LUTs with b > 2 through the same BLIF export and ABC minimization path.",
        "",
        f"- b values: {', '.join(str(b) for b in args.b_values)}",
        f"- cases per b: {len(make_case_specs(args.b_values[0]))} to {len(make_case_specs(args.b_values[-1]))}",
        f"- total rows: {len(rows)}",
        f"- all equivalence checks passed: {all(str(row['equivalent']) == 'True' and str(row['source_equivalent']) == 'True' for row in rows)}",
        "",
        "| b | rows | mean_raw_mux2 | mean_abc_and | best_case | best_raw_mux2 | best_abc_and | best_ratio |",
        "|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["b"]),
                    str(row["rows"]),
                    format_float(float(row["mean_raw_mux2"])),
                    format_float(float(row["mean_abc_and"])),
                    str(row["best_case"]),
                    str(row["best_raw_mux2"]),
                    str(row["best_abc_and"]),
                    format_float(float(row["best_ratio"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Selected cases:",
            "| b | case | raw_mux2 | abc_and | abc_level | ratio | equivalent |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    selected = [row for row in rows if row["case"] in {"parity", "majority", "mux_last2", "selector4", "dense_hash"}]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["b"]),
                    str(row["case"]),
                    str(row["raw_mux2_count"]),
                    str(row["abc_and_count"]),
                    str(row["abc_level"]),
                    format_float(float(row["abc_and_vs_raw_mux2_ratio"])),
                    str(row["equivalent"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `raw_mux2_count` is the direct LUT-as-mux-tree estimate `2^b - 1`.",
            "- `abc_and_count` and `abc_level` come from `read_blif; strash; dc2; print_stats`.",
            "- This experiment validates generic b-input LUT minimization behavior, independent of current LightLogic training producing only 2-input local LUTs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc-path", default="/home/spco/boolean_sat/abc/abc")
    parser.add_argument("--b-values", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--out-dir", default="runs/lut_minimization_synth_b3to6_v1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    minimizer = LutMinimizer(Path(args.abc_path), out_dir)
    rows: list[dict[str, object]] = []
    for b in args.b_values:
        for case_name, fn in make_case_specs(b):
            truth_bits = truth_bits_from_fn(b, fn)
            result = minimizer.minimize(b, truth_bits)
            row = {
                "b": b,
                "case": case_name,
                "truth_bits": truth_bits,
                "raw_lut_bits": result.raw_lut_bits,
                "raw_mux2_count": result.raw_mux2_count,
                "raw_sop_literals": result.raw_sop_literals,
                "raw_sop_not_estimate": result.raw_sop_not_estimate,
                "abc_status": result.abc_status,
                "abc_and_count": result.abc_and_count,
                "abc_level": result.abc_level,
                "abc_seconds": result.abc_seconds,
                "source_equivalent": result.source_equivalent,
                "equivalent": result.equivalent,
                "abc_and_vs_raw_mux2_ratio": safe_ratio(float(result.abc_and_count), float(result.raw_mux2_count)),
                "raw_blif_path": result.raw_blif_path,
                "optimized_blif_path": result.optimized_blif_path,
            }
            rows.append(row)

    summary_rows: list[dict[str, object]] = []
    for b in args.b_values:
        group = [row for row in rows if row["b"] == b]
        best = min(group, key=lambda row: (float(row["abc_and_count"]), float(row["abc_level"])))
        summary_rows.append(
            {
                "b": b,
                "rows": len(group),
                "mean_raw_mux2": sum(float(row["raw_mux2_count"]) for row in group) / len(group),
                "mean_abc_and": sum(float(row["abc_and_count"]) for row in group) / len(group),
                "best_case": best["case"],
                "best_raw_mux2": best["raw_mux2_count"],
                "best_abc_and": best["abc_and_count"],
                "best_ratio": best["abc_and_vs_raw_mux2_ratio"],
            }
        )

    write_csv(out_dir / "synthetic_lut_results.csv", rows)
    write_csv(out_dir / "synthetic_lut_summary_by_b.csv", summary_rows)
    (out_dir / "synthetic_lut_report.md").write_text(build_report(rows, summary_rows, args), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "synthetic_lut_results.csv")
    print(out_dir / "synthetic_lut_summary_by_b.csv")
    print(out_dir / "synthetic_lut_report.md")


if __name__ == "__main__":
    main()
