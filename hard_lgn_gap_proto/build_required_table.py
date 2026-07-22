#!/usr/bin/env python3
"""Build the exact required Hard-LGN comparison metrics table.

The benchmark aggregates already contain mean values for the requested final
metrics. This script consolidates the latest boolean, digits, MNIST, and
CIFAR-small aggregate artifacts into one table with the user-requested columns.
It also writes a provenance companion that records source runs and target-hit
counts.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_RUNS = [
    "bool_seeds012_layerdiag_v1",
    "digits_seeds012_layerdiag_v1",
    "mnist_smoke_layerdiag_v1",
    "cifar10_small_seed0_layerdiag_v1",
]

REQUIRED_FIELDS = [
    "method",
    "dataset",
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "train_time",
    "epochs_to_target",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
]

PROVENANCE_FIELDS = [
    *REQUIRED_FIELDS,
    "n",
    "target_hits",
    "target_hit_rate",
    "source_run",
    "source_aggregate",
    "source_results",
]

MEAN_SOURCE_COLUMNS = {
    "soft_acc": "soft_acc_mean",
    "discrete_acc": "discrete_acc_mean",
    "acc_gap": "acc_gap_mean",
    "soft_loss": "soft_loss_mean",
    "discrete_loss": "discrete_loss_mean",
    "loss_gap": "loss_gap_mean",
    "train_time": "train_time_mean",
    "unused_gate_ratio": "unused_gate_ratio_mean",
    "gate_count": "gate_count_mean",
    "depth": "depth_mean",
    "fanout_max": "fanout_max_mean",
}

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


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


def fmt(value: object, digits: int = 8) -> str:
    numeric = to_float(value)
    if not math.isfinite(numeric):
        return "nan"
    if abs(numeric - round(numeric)) < 1e-12:
        return str(int(round(numeric)))
    return f"{numeric:.{digits}g}"


def source_label(prefix: str, path: Path) -> str:
    clean_prefix = prefix.strip("/")
    path_text = path.as_posix()
    return f"{clean_prefix}/{path_text}" if clean_prefix else path_text


def epochs_to_target_summary(result_rows: list[dict[str, str]], dataset: str, method: str) -> tuple[str, int, str]:
    rows = [row for row in result_rows if row["dataset"] == dataset and row["method"] == method]
    hits = [to_float(row.get("epochs_to_target")) for row in rows]
    hit_values = [value for value in hits if math.isfinite(value) and value > 0]
    hit_rate = len(hit_values) / len(rows) if rows else math.nan
    if not hit_values:
        return "-1", 0, fmt(hit_rate)
    return fmt(mean(hit_values)), len(hit_values), fmt(hit_rate)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    required_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []
    dataset_order: dict[str, int] = {}

    for run_index, run_name in enumerate(args.runs):
        aggregate_rel = Path(run_name) / "aggregate" / "aggregate_by_method_dataset.csv"
        results_rel = Path(run_name) / "results.csv"
        aggregate_path = args.runs_root / aggregate_rel
        results_path = args.runs_root / results_rel
        if not aggregate_path.exists():
            raise FileNotFoundError(f"missing aggregate file: {aggregate_path}")
        if not results_path.exists():
            raise FileNotFoundError(f"missing results file for epochs_to_target: {results_path}")

        aggregate_rows = read_csv(aggregate_path)
        result_rows = read_csv(results_path)
        for aggregate_row in aggregate_rows:
            dataset = aggregate_row["dataset"]
            method = aggregate_row["method"]
            dataset_order.setdefault(dataset, run_index * 1000 + len(dataset_order))

            required_row = {"method": method, "dataset": dataset}
            for output_col, source_col in MEAN_SOURCE_COLUMNS.items():
                if source_col not in aggregate_row:
                    raise KeyError(f"{aggregate_path} does not contain required column {source_col}")
                required_row[output_col] = fmt(aggregate_row[source_col])

            epochs_value, target_hits, target_hit_rate = epochs_to_target_summary(result_rows, dataset, method)
            required_row["epochs_to_target"] = epochs_value
            ordered_row = {field: required_row.get(field, "") for field in REQUIRED_FIELDS}
            required_rows.append(ordered_row)

            provenance_row = {field: ordered_row.get(field, "") for field in REQUIRED_FIELDS}
            provenance_row.update(
                {
                    "n": aggregate_row.get("n", ""),
                    "target_hits": str(target_hits),
                    "target_hit_rate": target_hit_rate,
                    "source_run": run_name,
                    "source_aggregate": source_label(args.source_prefix, aggregate_rel),
                    "source_results": source_label(args.source_prefix, results_rel),
                }
            )
            provenance_rows.append(provenance_row)

    def sort_key(row: dict[str, str]) -> tuple[int, str, int, str]:
        dataset = row["dataset"]
        method = row["method"]
        return (dataset_order.get(dataset, 999999), dataset, METHOD_ORDER.get(method, 99), method)

    required_rows.sort(key=sort_key)
    provenance_rows.sort(key=sort_key)
    return required_rows, provenance_rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Required Hard-LGN Metrics Table",
        "",
        "| " + " | ".join(REQUIRED_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(REQUIRED_FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in REQUIRED_FIELDS) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- Values are seed means from each run's aggregate output.",
            "- `epochs_to_target` is the mean first-hit epoch over seeds that reached the dataset target; `-1` means no seed reached target.",
            "- Provenance, seed count, target-hit count, and target-hit rate are in `required_metrics_table_with_provenance.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--source-prefix", default="runs")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/reports_required_table_v1"))
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_rows, provenance_rows = build_rows(args)
    write_csv(args.out_dir / "required_metrics_table.csv", required_rows, REQUIRED_FIELDS)
    write_csv(args.out_dir / "required_metrics_table_with_provenance.csv", provenance_rows, PROVENANCE_FIELDS)
    (args.out_dir / "required_metrics_table.md").write_text(markdown_table(required_rows))
    print(f"wrote {len(required_rows)} rows to {args.out_dir / 'required_metrics_table.csv'}")


if __name__ == "__main__":
    main()
