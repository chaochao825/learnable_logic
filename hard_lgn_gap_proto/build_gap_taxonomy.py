#!/usr/bin/env python3
"""Build a three-view gap taxonomy report for block-wise Hard-LGN.

The goal is to keep three distinct measurements separate:

1. relaxed-full vs hard-full, the traditional DLGN / Mind-the-Gap metric.
2. method-native hard-prefix path vs final hard path, the block-wise objective.
3. relaxed block output vs fitted hard block output, the per-block refit error.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RUNS = [
    "bool_seeds012_layerdiag_v1",
    "digits_seeds012_layerdiag_v1",
    "mnist_smoke_layerdiag_v1",
    "cifar10_small_seed0_layerdiag_v1",
]

METHOD_ORDER = {
    "dlgn": 0,
    "dlgn_anneal": 1,
    "gumbel_st": 2,
    "hard_st": 3,
    "hard_st_cage": 4,
    "gumbel_st_cage": 5,
    "block_relaxed": 6,
    "block_hard_refit": 7,
    "block_hard_task_refit": 8,
}

ALL_METHOD_FIELDS = [
    "dataset",
    "method",
    "n",
    "full_soft_acc",
    "full_hard_acc",
    "full_acc_gap",
    "full_soft_loss",
    "full_hard_loss",
    "full_loss_gap",
    "native_soft_acc",
    "native_hard_acc",
    "native_acc_gap",
    "native_soft_loss",
    "native_hard_loss",
    "native_loss_gap",
    "full_minus_native_acc_gap",
    "full_minus_native_loss_gap",
    "unused_gate_ratio",
    "source_run",
]

SUMMARY_FIELDS = [
    "dataset",
    "n",
    "dlgn_full_acc_gap",
    "gumbel_full_acc_gap",
    "block_full_acc_gap",
    "block_native_acc_gap",
    "block_full_minus_native_acc_gap",
    "dlgn_full_loss_gap",
    "gumbel_full_loss_gap",
    "block_full_loss_gap",
    "block_native_loss_gap",
    "block_full_minus_native_loss_gap",
    "block_native_gap_better_than_full",
    "block_primary_full_gap_beats_dlgn",
    "block_native_gap_beats_dlgn",
    "block_primary_full_gap_competitive_with_gumbel",
    "block_native_gap_competitive_with_gumbel",
    "source_run",
]

BLOCK_FIELDS = [
    "dataset",
    "block",
    "n_rows",
    "fit_source",
    "fit_rows_mean",
    "path_acc_gap",
    "path_loss_gap",
    "hard_prefix_acc",
    "hard_prefix_loss",
    "argmax_refit_mse",
    "truth_refit_mse",
    "refit_mse_delta",
    "op_change_ratio",
    "refit_error_rank",
    "path_gap_rank",
    "source_run",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def fmt(value: object, digits: int = 8) -> str:
    numeric = to_float(value)
    if not math.isfinite(numeric):
        return "nan"
    if abs(numeric - round(numeric)) < 1e-12:
        return str(int(round(numeric)))
    return f"{numeric:.{digits}g}"


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def safe_delta(left: object, right: object) -> float:
    left_f = to_float(left)
    right_f = to_float(right)
    if not math.isfinite(left_f) or not math.isfinite(right_f):
        return math.nan
    return left_f - right_f


def mode_text(values: list[str]) -> str:
    values = [value for value in values if value]
    if not values:
        return ""
    counts = Counter(values)
    return ",".join(value for value, _ in counts.most_common())


def source_label(prefix: str, path: Path) -> str:
    clean_prefix = prefix.strip("/")
    text = path.as_posix()
    return f"{clean_prefix}/{text}" if clean_prefix else text


def method_row(agg: dict[str, str], run_name: str) -> dict[str, str]:
    full_acc_gap = to_float(agg.get("acc_gap_mean"))
    native_acc_gap = to_float(agg.get("path_acc_gap_mean"))
    full_loss_gap = to_float(agg.get("loss_gap_mean"))
    native_loss_gap = to_float(agg.get("path_loss_gap_mean"))
    row = {
        "dataset": agg["dataset"],
        "method": agg["method"],
        "n": agg.get("n", ""),
        "full_soft_acc": fmt(agg.get("soft_acc_mean")),
        "full_hard_acc": fmt(agg.get("discrete_acc_mean")),
        "full_acc_gap": fmt(full_acc_gap),
        "full_soft_loss": fmt(agg.get("soft_loss_mean")),
        "full_hard_loss": fmt(agg.get("discrete_loss_mean")),
        "full_loss_gap": fmt(full_loss_gap),
        "native_soft_acc": fmt(agg.get("path_soft_acc_mean")),
        "native_hard_acc": fmt(agg.get("path_discrete_acc_mean")),
        "native_acc_gap": fmt(native_acc_gap),
        "native_soft_loss": fmt(agg.get("path_soft_loss_mean")),
        "native_hard_loss": fmt(agg.get("path_discrete_loss_mean")),
        "native_loss_gap": fmt(native_loss_gap),
        "full_minus_native_acc_gap": fmt(full_acc_gap - native_acc_gap),
        "full_minus_native_loss_gap": fmt(full_loss_gap - native_loss_gap),
        "unused_gate_ratio": fmt(agg.get("unused_gate_ratio_mean")),
        "source_run": run_name,
    }
    return row


def build(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    all_method_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    block_rows: list[dict[str, str]] = []
    dataset_order: dict[str, int] = {}

    for run_index, run_name in enumerate(args.runs):
        aggregate_path = args.runs_root / run_name / "aggregate" / "aggregate_by_method_dataset.csv"
        block_agg_path = args.runs_root / run_name / "aggregate" / "block_diagnostics_by_block.csv"
        block_raw_path = args.runs_root / run_name / "block_diagnostics.csv"
        if not aggregate_path.exists():
            raise FileNotFoundError(f"missing aggregate file: {aggregate_path}")

        aggregate_rows = read_csv(aggregate_path)
        by_key = {(row["dataset"], row["method"]): row for row in aggregate_rows}
        for row in aggregate_rows:
            dataset_order.setdefault(row["dataset"], run_index * 1000 + len(dataset_order))
            all_method_rows.append(method_row(row, run_name))

        for dataset in sorted({row["dataset"] for row in aggregate_rows}):
            dlgn = by_key.get((dataset, "dlgn"))
            gumbel = by_key.get((dataset, "gumbel_st"))
            block = by_key.get((dataset, "block_hard_refit"))
            if not dlgn or not block:
                continue
            dlgn_full_gap = to_float(dlgn.get("acc_gap_mean"))
            gumbel_full_gap = to_float(gumbel.get("acc_gap_mean")) if gumbel else math.nan
            block_full_gap = to_float(block.get("acc_gap_mean"))
            block_native_gap = to_float(block.get("path_acc_gap_mean"))
            dlgn_loss_gap = to_float(dlgn.get("loss_gap_mean"))
            gumbel_loss_gap = to_float(gumbel.get("loss_gap_mean")) if gumbel else math.nan
            block_loss_gap = to_float(block.get("loss_gap_mean"))
            block_native_loss_gap = to_float(block.get("path_loss_gap_mean"))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "n": block.get("n", ""),
                    "dlgn_full_acc_gap": fmt(dlgn_full_gap),
                    "gumbel_full_acc_gap": fmt(gumbel_full_gap),
                    "block_full_acc_gap": fmt(block_full_gap),
                    "block_native_acc_gap": fmt(block_native_gap),
                    "block_full_minus_native_acc_gap": fmt(block_full_gap - block_native_gap),
                    "dlgn_full_loss_gap": fmt(dlgn_loss_gap),
                    "gumbel_full_loss_gap": fmt(gumbel_loss_gap),
                    "block_full_loss_gap": fmt(block_loss_gap),
                    "block_native_loss_gap": fmt(block_native_loss_gap),
                    "block_full_minus_native_loss_gap": fmt(block_loss_gap - block_native_loss_gap),
                    "block_native_gap_better_than_full": yes_no(block_native_gap < block_full_gap),
                    "block_primary_full_gap_beats_dlgn": yes_no(block_full_gap < dlgn_full_gap),
                    "block_native_gap_beats_dlgn": yes_no(block_native_gap < dlgn_full_gap),
                    "block_primary_full_gap_competitive_with_gumbel": yes_no(
                        math.isfinite(gumbel_full_gap) and block_full_gap <= gumbel_full_gap
                    ),
                    "block_native_gap_competitive_with_gumbel": yes_no(
                        math.isfinite(gumbel_full_gap) and block_native_gap <= gumbel_full_gap
                    ),
                    "source_run": run_name,
                }
            )

        if block_agg_path.exists() and block_raw_path.exists():
            raw_rows = read_csv(block_raw_path)
            raw_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for row in raw_rows:
                if row.get("method") == "block_hard_refit":
                    raw_by_key[(row["dataset"], row["block"])].append(row)
            block_agg_rows = [
                row for row in read_csv(block_agg_path) if row.get("method") == "block_hard_refit"
            ]
            refit_rank_source: dict[str, list[dict[str, str]]] = defaultdict(list)
            path_rank_source: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in block_agg_rows:
                refit_rank_source[row["dataset"]].append(row)
                path_rank_source[row["dataset"]].append(row)
            refit_ranks = rank_by_metric(refit_rank_source, "truth_refit_mse_mean")
            path_ranks = rank_by_metric(path_rank_source, "path_acc_gap_mean")
            for row in block_agg_rows:
                raw_group = raw_by_key.get((row["dataset"], row["block"]), [])
                fit_rows = [to_float(item.get("fit_rows")) for item in raw_group]
                block_rows.append(
                    {
                        "dataset": row["dataset"],
                        "block": row["block"],
                        "n_rows": row.get("n_rows", ""),
                        "fit_source": mode_text([item.get("fit_source", "") for item in raw_group]),
                        "fit_rows_mean": fmt(finite_mean(fit_rows)),
                        "path_acc_gap": fmt(row.get("path_acc_gap_mean")),
                        "path_loss_gap": fmt(row.get("path_loss_gap_mean")),
                        "hard_prefix_acc": fmt(row.get("hard_prefix_acc_mean")),
                        "hard_prefix_loss": fmt(row.get("hard_prefix_loss_mean")),
                        "argmax_refit_mse": fmt(row.get("argmax_refit_mse_mean")),
                        "truth_refit_mse": fmt(row.get("truth_refit_mse_mean")),
                        "refit_mse_delta": fmt(row.get("refit_mse_delta_mean")),
                        "op_change_ratio": fmt(row.get("op_change_ratio_mean")),
                        "refit_error_rank": str(refit_ranks.get((row["dataset"], row["block"]), "")),
                        "path_gap_rank": str(path_ranks.get((row["dataset"], row["block"]), "")),
                        "source_run": run_name,
                    }
                )

    def sort_key(row: dict[str, str]) -> tuple[int, str, int, str]:
        dataset = row["dataset"]
        return (dataset_order.get(dataset, 999999), dataset, METHOD_ORDER.get(row.get("method", ""), 99), row.get("block", ""), row.get("method", ""))

    all_method_rows.sort(key=sort_key)
    summary_rows.sort(key=lambda row: (dataset_order.get(row["dataset"], 999999), row["dataset"]))
    block_rows.sort(key=lambda row: (dataset_order.get(row["dataset"], 999999), row["dataset"], int(row["block"])))
    return all_method_rows, summary_rows, block_rows


def rank_by_metric(groups: dict[str, list[dict[str, str]]], metric: str) -> dict[tuple[str, str], int]:
    ranks: dict[tuple[str, str], int] = {}
    for dataset, rows in groups.items():
        sorted_rows = sorted(rows, key=lambda row: to_float(row.get(metric)), reverse=True)
        for idx, row in enumerate(sorted_rows, start=1):
            ranks[(dataset, row["block"])] = idx
    return ranks


def markdown_table(rows: list[dict[str, str]], fields: list[str], limit: int | None = None) -> list[str]:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return out


def report_markdown(
    all_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    block_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# Gap Taxonomy Report",
        "",
        "This report separates three gap definitions that should not be conflated:",
        "",
        "1. `full_*`: relaxed-full network vs hard-full network. This is the traditional DLGN / Mind-the-Gap gap.",
        "2. `native_*`: method-native training path vs final hard path. For `block_hard_refit`, this is the hard-prefix-trained path.",
        "3. `block_refit`: relaxed block output vs fitted hard block output, measured by per-block MSE and operation changes.",
        "",
        "## Block-Hard Gap Split Summary",
        "",
    ]
    summary_fields = [
        "dataset",
        "dlgn_full_acc_gap",
        "gumbel_full_acc_gap",
        "block_full_acc_gap",
        "block_native_acc_gap",
        "block_full_minus_native_acc_gap",
        "block_primary_full_gap_beats_dlgn",
        "block_native_gap_beats_dlgn",
        "block_primary_full_gap_competitive_with_gumbel",
        "block_native_gap_competitive_with_gumbel",
    ]
    lines.extend(markdown_table(summary_rows, summary_fields))
    lines.extend(
        [
            "",
            "## All Methods: Full vs Native Gap",
            "",
        ]
    )
    all_fields = [
        "dataset",
        "method",
        "full_acc_gap",
        "native_acc_gap",
        "full_minus_native_acc_gap",
        "full_loss_gap",
        "native_loss_gap",
        "full_minus_native_loss_gap",
        "unused_gate_ratio",
    ]
    lines.extend(markdown_table(all_rows, all_fields))
    lines.extend(
        [
            "",
            "## Block Refit Error By Block",
            "",
            "`refit_error_rank=1` is the largest `truth_refit_mse` within that dataset. `path_gap_rank=1` is the largest path accuracy gap.",
            "",
        ]
    )
    block_fields = [
        "dataset",
        "block",
        "fit_source",
        "fit_rows_mean",
        "path_acc_gap",
        "truth_refit_mse",
        "refit_mse_delta",
        "op_change_ratio",
        "refit_error_rank",
        "path_gap_rank",
    ]
    lines.extend(markdown_table(block_rows, block_fields))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current block-wise method often has a much smaller native gap than full gap. That supports the user's diagnosis: the optimized/evaluated path differs from the traditional relaxed-full DLGN path.",
            "- For Mind-the-Gap comparison, `full_acc_gap` remains the relevant primary metric. Under that metric, block-hard still fails except for the small CIFAR smoke case.",
            "- For a block-wise method claim, `native_acc_gap` and per-block refit diagnostics are the correct internal consistency metrics.",
            "- If the research target is to beat Mind-the-Gap directly, future training must make the relaxed-full path meaningful or stop using relaxed-full gap as the primary claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/reports_gap_taxonomy_v1"))
    parser.add_argument("--source-prefix", default="runs")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows, summary_rows, block_rows = build(args)
    write_csv(args.out_dir / "gap_taxonomy_all_methods.csv", all_rows, ALL_METHOD_FIELDS)
    write_csv(args.out_dir / "gap_taxonomy_block_hard_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(args.out_dir / "gap_taxonomy_block_refit_by_block.csv", block_rows, BLOCK_FIELDS)
    (args.out_dir / "gap_taxonomy_report.md").write_text(report_markdown(all_rows, summary_rows, block_rows))
    print(f"wrote {len(summary_rows)} summary rows, {len(all_rows)} method rows, {len(block_rows)} block rows to {args.out_dir}")


if __name__ == "__main__":
    main()
