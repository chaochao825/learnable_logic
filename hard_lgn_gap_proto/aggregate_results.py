#!/usr/bin/env python3
"""Aggregate Hard-LGN benchmark CSVs across seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


METRIC_COLUMNS = [
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "path_soft_acc",
    "path_discrete_acc",
    "path_acc_gap",
    "path_soft_loss",
    "path_discrete_loss",
    "path_loss_gap",
    "train_time",
    "time_to_target",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "soft_inference_samples_per_sec",
    "discrete_inference_samples_per_sec",
    "inference_bench_repeats",
]

EPSILON_FOR_RATIO = 1e-2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.mean(finite) if finite else math.nan


def stdev(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.nan
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def aggregate_results(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(r["dataset"], r["method"]) for r in rows})
    for dataset, method in keys:
        group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "n": len(group),
        }
        for col in METRIC_COLUMNS:
            if col not in group[0]:
                continue
            vals = [safe_float(r[col]) for r in group]
            if col == "time_to_target":
                hit_vals = [v for v in vals if math.isfinite(v) and v > 0]
                row[f"{col}_mean"] = mean(hit_vals)
                row[f"{col}_std"] = stdev(hit_vals)
                row["n_target_hits"] = len(hit_vals)
                row["target_hit_rate"] = len(hit_vals) / len(group) if group else math.nan
                continue
            row[f"{col}_mean"] = mean(vals)
            row[f"{col}_std"] = stdev(vals)
        out.append(row)
    return out


def comparison_vs_baselines(agg_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_dataset_method = {(r["dataset"], r["method"]): r for r in agg_rows}
    out: list[dict[str, object]] = []
    for dataset in sorted({r["dataset"] for r in agg_rows}):
        dlgn = by_dataset_method.get((dataset, "dlgn"))
        gumbel = by_dataset_method.get((dataset, "gumbel_st"))
        if not dlgn:
            continue
        dlgn_gap = float(dlgn["acc_gap_mean"])
        dlgn_unused = float(dlgn["unused_gate_ratio_mean"])
        dlgn_time = float(dlgn["train_time_mean"])
        dlgn_time_to_target = float(dlgn.get("time_to_target_mean", math.nan))
        dlgn_target_hit_rate = float(dlgn.get("target_hit_rate", math.nan))
        for method_row in [r for r in agg_rows if r["dataset"] == dataset]:
            method = str(method_row["method"])
            gap = float(method_row["acc_gap_mean"])
            path_gap = float(method_row.get("path_acc_gap_mean", math.nan))
            unused = float(method_row["unused_gate_ratio_mean"])
            train_time = float(method_row["train_time_mean"])
            time_to_target = float(method_row.get("time_to_target_mean", math.nan))
            target_hit_rate = float(method_row.get("target_hit_rate", math.nan))
            row: dict[str, object] = {
                "dataset": dataset,
                "method": method,
                "acc_gap_mean": gap,
                "path_acc_gap_mean": path_gap,
                "discrete_acc_mean": method_row["discrete_acc_mean"],
                "unused_gate_ratio_mean": unused,
                "train_time_mean": train_time,
                "time_to_target_mean": time_to_target,
                "n_target_hits": method_row.get("n_target_hits", ""),
                "target_hit_rate": target_hit_rate,
                "dlgn_target_hit_rate": dlgn_target_hit_rate,
                "gap_delta_vs_dlgn": gap - dlgn_gap,
                "path_gap_delta_vs_dlgn": path_gap - dlgn_gap,
                "speedup_vs_dlgn": dlgn_time / train_time if train_time > 0 else math.nan,
                "target_speedup_vs_dlgn": (
                    dlgn_time_to_target / time_to_target
                    if (
                        dlgn_time_to_target > 0
                        and time_to_target > 0
                        and dlgn_target_hit_rate == 1.0
                        and target_hit_rate == 1.0
                    )
                    else math.nan
                ),
                "unused_delta_vs_dlgn": unused - dlgn_unused,
            }
            if dlgn_gap >= EPSILON_FOR_RATIO:
                row["gap_reduction_vs_dlgn"] = (dlgn_gap - gap) / dlgn_gap
                row["path_gap_reduction_vs_dlgn"] = (dlgn_gap - path_gap) / dlgn_gap
            else:
                row["gap_reduction_vs_dlgn"] = math.nan
                row["path_gap_reduction_vs_dlgn"] = math.nan
            if dlgn_unused > 0:
                row["unused_reduction_vs_dlgn"] = (dlgn_unused - unused) / dlgn_unused
            else:
                row["unused_reduction_vs_dlgn"] = math.nan
            if gumbel:
                gumbel_gap = float(gumbel["acc_gap_mean"])
                row["gap_delta_vs_gumbel"] = gap - gumbel_gap
                row["path_gap_delta_vs_gumbel"] = path_gap - gumbel_gap
            out.append(row)
    return out


def aggregate_block_diagnostics(
    rows: list[dict[str, str]],
    group_by_block: bool,
) -> list[dict[str, object]]:
    if not rows:
        return []
    out: list[dict[str, object]] = []
    if group_by_block:
        keys = sorted({(r["dataset"], r["method"], r["block"]) for r in rows})
    else:
        keys = sorted({(r["dataset"], r["method"]) for r in rows})
    metrics = [
        "path_acc_gap",
        "path_loss_gap",
        "hard_prefix_acc",
        "hard_prefix_loss",
        "elapsed_time",
        "prefix_reaches_target",
        "argmax_refit_mse",
        "truth_refit_mse",
        "refit_mse_delta",
        "op_change_ratio",
    ]
    for key in keys:
        if group_by_block:
            dataset, method, block = key
            group = [
                r
                for r in rows
                if r["dataset"] == dataset and r["method"] == method and r["block"] == block
            ]
        else:
            dataset, method = key
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "n_rows": len(group),
        }
        if group_by_block:
            row["block"] = int(block)
        for metric in metrics:
            if metric not in group[0]:
                continue
            vals = [safe_float(r[metric]) for r in group]
            row[f"{metric}_mean"] = mean(vals)
            row[f"{metric}_std"] = stdev(vals)
        out.append(row)
    return out


def aggregate_final_block_diagnostics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    if not rows:
        return []
    final_rows = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["dataset"] == dataset}):
            for seed in sorted({r["seed"] for r in rows if r["dataset"] == dataset and r["method"] == method}):
                group = [
                    r
                    for r in rows
                    if r["dataset"] == dataset and r["method"] == method and r["seed"] == seed
                ]
                if not group:
                    continue
                max_block = max(int(r["block"]) for r in group)
                final_rows.extend(r for r in group if int(r["block"]) == max_block)
    return aggregate_block_diagnostics(final_rows, group_by_block=False)


def aggregate_layer_diagnostics(
    rows: list[dict[str, str]],
    group_by_layer: bool,
) -> list[dict[str, object]]:
    if not rows:
        return []
    out: list[dict[str, object]] = []
    if group_by_layer:
        keys = sorted({(r["dataset"], r["method"], r["comparison"], r["layer"]) for r in rows})
    else:
        keys = sorted({(r["dataset"], r["method"], r["comparison"]) for r in rows})
    metrics = [
        "mean_abs_diff",
        "max_abs_diff",
        "mse",
        "binary_flip_ratio",
        "soft_inactive_ratio",
        "hard_inactive_ratio",
    ]
    for key in keys:
        if group_by_layer:
            dataset, method, comparison, layer = key
            group = [
                r
                for r in rows
                if (
                    r["dataset"] == dataset
                    and r["method"] == method
                    and r["comparison"] == comparison
                    and r["layer"] == layer
                )
            ]
        else:
            dataset, method, comparison = key
            group = [
                r
                for r in rows
                if r["dataset"] == dataset and r["method"] == method and r["comparison"] == comparison
            ]
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "comparison": comparison,
            "n_rows": len(group),
        }
        if group_by_layer:
            row["layer"] = int(layer)
            row["prefix_depth"] = int(group[0].get("prefix_depth", int(layer) + 1))
        for metric in metrics:
            if metric not in group[0]:
                continue
            vals = [safe_float(r[metric]) for r in group]
            row[f"{metric}_mean"] = mean(vals)
            row[f"{metric}_std"] = stdev(vals)
        out.append(row)
    return out


def aggregate_final_layer_diagnostics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    if not rows:
        return []
    final_rows = []
    keys = sorted({(r["dataset"], r["method"], r["comparison"], r["seed"]) for r in rows})
    for dataset, method, comparison, seed in keys:
        group = [
            r
            for r in rows
            if (
                r["dataset"] == dataset
                and r["method"] == method
                and r["comparison"] == comparison
                and r["seed"] == seed
            )
        ]
        if not group:
            continue
        max_layer = max(int(r["layer"]) for r in group)
        final_rows.extend(r for r in group if int(r["layer"]) == max_layer)
    return aggregate_layer_diagnostics(final_rows, group_by_layer=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--block-diagnostics", type=Path)
    parser.add_argument("--layer-diagnostics", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    result_rows = read_csv(args.results)
    agg = aggregate_results(result_rows)
    comp = comparison_vs_baselines(agg)
    write_csv(args.out_dir / "aggregate_by_method_dataset.csv", agg)
    write_csv(args.out_dir / "comparison_vs_baselines.csv", comp)

    if args.block_diagnostics and args.block_diagnostics.exists():
        block_rows = read_csv(args.block_diagnostics)
        all_block_agg = aggregate_block_diagnostics(block_rows, group_by_block=False)
        per_block_agg = aggregate_block_diagnostics(block_rows, group_by_block=True)
        final_block_agg = aggregate_final_block_diagnostics(block_rows)
        write_csv(args.out_dir / "block_diagnostics_all_blocks_aggregate.csv", all_block_agg)
        write_csv(args.out_dir / "block_diagnostics_by_block.csv", per_block_agg)
        write_csv(args.out_dir / "block_diagnostics_final_block.csv", final_block_agg)

    if args.layer_diagnostics and args.layer_diagnostics.exists():
        layer_rows = read_csv(args.layer_diagnostics)
        all_layer_agg = aggregate_layer_diagnostics(layer_rows, group_by_layer=False)
        per_layer_agg = aggregate_layer_diagnostics(layer_rows, group_by_layer=True)
        final_layer_agg = aggregate_final_layer_diagnostics(layer_rows)
        write_csv(args.out_dir / "layer_diagnostics_all_layers_aggregate.csv", all_layer_agg)
        write_csv(args.out_dir / "layer_diagnostics_by_layer.csv", per_layer_agg)
        write_csv(args.out_dir / "layer_diagnostics_final_layer.csv", final_layer_agg)

    print(args.out_dir / "aggregate_by_method_dataset.csv")
    print(args.out_dir / "comparison_vs_baselines.csv")


if __name__ == "__main__":
    main()
