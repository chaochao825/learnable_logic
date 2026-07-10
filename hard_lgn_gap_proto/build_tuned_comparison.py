"""Build the seed-0 tuned Gumbel comparison artifact from source CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_METHODS = ["dlgn", "dlgn_anneal", "block_relaxed", "block_hard_refit"]
SOURCE_METRICS = [
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
    "epochs_to_target",
    "time_to_target",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "trial",
    "epochs_cfg",
    "lr_cfg",
    "gumbel_temp_start_cfg",
    "gumbel_temp_end_cfg",
    "trial_out_dir",
]
OUTPUT_FIELDS = [
    "method",
    "dataset",
    "seed",
    "comparison_method",
    "source_method",
    "source_run",
    "source_file",
    "selection_criterion",
    *SOURCE_METRICS,
]
DISPLAY_FIELDS = [
    "method",
    "dataset",
    "seed",
    "source_method",
    "source_run",
    "source_file",
    "selection_criterion",
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "train_time",
    "epochs_to_target",
    "time_to_target",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "trial",
    "lr_cfg",
    "gumbel_temp_start_cfg",
    "gumbel_temp_end_cfg",
    "trial_out_dir",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def source_row(
    row: dict[str, str],
    *,
    method: str,
    source_run: str,
    source_file: str,
    selection_criterion: str,
) -> dict[str, str]:
    out = {
        "method": method,
        "dataset": row["dataset"],
        "seed": row["seed"],
        "comparison_method": method,
        "source_method": row["method"],
        "source_run": source_run,
        "source_file": source_file,
        "selection_criterion": selection_criterion,
    }
    for field in SOURCE_METRICS:
        out[field] = row.get(field, "")
    return out


def source_label(source_prefix: str, path: Path) -> str:
    prefix = source_prefix.rstrip("/")
    if not prefix:
        return path.as_posix()
    return f"{prefix}/{path.as_posix()}"


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["# Tuned Gumbel Seed-0 Comparison", ""]
    lines.append("| " + " | ".join(DISPLAY_FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(DISPLAY_FIELDS)) + " |")
    for row in rows:
        values = [row.get(field, "") for field in DISPLAY_FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    notes = [
        "",
        "Notes:",
        "- `method` is the comparison label used in this table.",
        "- `source_method`, `source_run`, and `source_file` identify the exact source row.",
        "- Gumbel sweep rows use `selection_criterion=highest_discrete_acc_then_lowest_acc_gap_then_lowest_discrete_loss_then_lowest_train_time`.",
    ]
    if any(row["method"] == "gumbel_soft_best_discrete_acc_diag" for row in rows):
        notes.append(
            "- `gumbel_soft_best_discrete_acc_diag` is diagnostic only; it is not the straight-through Mind-the-Gap baseline."
        )
    lines.extend(notes)
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    base_file = Path(args.base_run) / "results.csv"
    for row in read_rows(args.runs_root / base_file):
        if row["method"] in args.base_methods:
            rows.append(
                source_row(
                    row,
                    method=row["method"],
                    source_run=args.base_run,
                    source_file=source_label(args.source_prefix, base_file),
                    selection_criterion="fixed_baseline",
                )
            )

    gumbel_specs = [
        (
            args.gumbel_st_run,
            "best_by_dataset.csv",
            "gumbel_st_best_discrete_acc",
            "highest_discrete_acc_then_lowest_acc_gap_then_lowest_discrete_loss_then_lowest_train_time",
        )
    ]
    if not args.skip_gumbel_soft:
        gumbel_specs.append(
            (
                args.gumbel_soft_run,
                "best_by_dataset.csv",
                "gumbel_soft_best_discrete_acc_diag",
                "highest_discrete_acc_then_lowest_acc_gap_then_lowest_discrete_loss_then_lowest_train_time",
            )
        )
    for run_name, file_name, method, selection_criterion in gumbel_specs:
        source_file = Path(run_name) / file_name
        for row in read_rows(args.runs_root / source_file):
            rows.append(
                source_row(
                    row,
                    method=method,
                    source_run=run_name,
                    source_file=source_label(args.source_prefix, source_file),
                    selection_criterion=selection_criterion,
                )
            )

    dataset_filter = set(args.datasets or [])
    if dataset_filter:
        rows = [row for row in rows if row["dataset"] in dataset_filter]
    seed_filter = {str(seed) for seed in args.seeds or []}
    if seed_filter:
        rows = [row for row in rows if row["seed"] in seed_filter]

    method_order = {
        "dlgn": 0,
        "dlgn_anneal": 1,
        "block_relaxed": 2,
        "block_hard_refit": 3,
        "gumbel_st_best_discrete_acc": 4,
        "gumbel_soft_best_discrete_acc_diag": 5,
    }
    rows.sort(key=lambda row: (row["dataset"], int(row["seed"]), method_order.get(row["method"], 99), row["method"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--source-prefix", default="runs")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gumbel_tuned_comparison_seed0"))
    parser.add_argument("--base-run", default="bool_seed0_cuda_abc")
    parser.add_argument("--gumbel-st-run", default="gumbel_sweep_bool_seed0")
    parser.add_argument("--gumbel-soft-run", default="gumbel_soft_sweep_bool_seed0")
    parser.add_argument("--skip-gumbel-soft", action="store_true")
    parser.add_argument("--base-methods", nargs="+", default=BASE_METHODS)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build(args)
    write_rows(args.out_dir / "comparison.csv", rows)
    with (args.out_dir / "summary.md").open("w", newline="\n") as handle:
        handle.write(markdown_table(rows))
    print(args.out_dir / "comparison.csv")
    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    main()
