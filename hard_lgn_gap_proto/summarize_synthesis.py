#!/usr/bin/env python3
"""Summarize optional ABC synthesis stats for Hard-LGN runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


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

JOIN_COLUMNS = [
    "run",
    "dataset",
    "method",
    "seed",
    "abc_status",
    "abc_runtime_seconds",
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "train_time",
    "time_to_target",
    "unused_gate_ratio",
    "pre_gate_count",
    "pre_depth",
    "pre_fanout_max",
    "abc_pre_nd",
    "abc_pre_edge",
    "abc_pre_cube",
    "abc_pre_lev",
    "abc_post_and",
    "abc_post_lev",
    "abc_and_reduction_vs_pre_nd",
    "abc_level_delta_vs_pre_lev",
    "abc_level_ratio_vs_pre_lev",
    "accuracy_recomputed_after_abc",
    "blif_path",
    "abc_log_path",
]

SUMMARY_NUMERIC_COLUMNS = [
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "train_time",
    "unused_gate_ratio",
    "abc_runtime_seconds",
    "pre_gate_count",
    "pre_depth",
    "pre_fanout_max",
    "abc_pre_nd",
    "abc_pre_edge",
    "abc_pre_cube",
    "abc_pre_lev",
    "abc_post_and",
    "abc_post_lev",
    "abc_and_reduction_vs_pre_nd",
    "abc_level_delta_vs_pre_lev",
    "abc_level_ratio_vs_pre_lev",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
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


def ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


def reduction(before: float, after: float) -> float:
    value = ratio(before - after, before)
    return value


def fmt(value: object, digits: int = 4) -> str:
    numeric = to_float(value)
    if math.isfinite(numeric):
        return f"{numeric:.{digits}g}"
    text = "" if value is None else str(value)
    return text if text and text.lower() != "nan" else "nan"


def pct(value: object) -> str:
    numeric = to_float(value)
    if not math.isfinite(numeric):
        return "nan"
    return f"{100.0 * numeric:.1f}%"


def sort_key(row: dict[str, object]) -> tuple[str, int, str, int]:
    method = str(row.get("method", ""))
    seed = int(to_float(row.get("seed", 0)) if math.isfinite(to_float(row.get("seed", 0))) else 0)
    return (str(row.get("dataset", "")), METHOD_ORDER.get(method, 99), method, seed)


def load_joined_rows(run_dir: Path, run_name: str) -> list[dict[str, object]]:
    results = read_csv(run_dir / "results.csv")
    synthesis = read_csv(run_dir / "synthesis_stats.csv")
    results_by_key = {
        (row["dataset"], row["method"], row["seed"]): row
        for row in results
    }
    joined: list[dict[str, object]] = []
    for synth_row in synthesis:
        key = (synth_row["dataset"], synth_row["method"], synth_row["seed"])
        result_row = results_by_key.get(key, {})
        pre_nd = to_float(synth_row.get("abc_pre_nd"))
        post_and = to_float(synth_row.get("abc_post_and"))
        pre_lev = to_float(synth_row.get("abc_pre_lev"))
        post_lev = to_float(synth_row.get("abc_post_lev"))
        row: dict[str, object] = {
            "run": run_name,
            "dataset": synth_row.get("dataset", ""),
            "method": synth_row.get("method", ""),
            "seed": synth_row.get("seed", ""),
            "abc_status": synth_row.get("abc_status", ""),
            "accuracy_recomputed_after_abc": "no",
            "blif_path": synth_row.get("blif_path", ""),
            "abc_log_path": synth_row.get("abc_log_path", ""),
        }
        for column in [
            "soft_acc",
            "discrete_acc",
            "acc_gap",
            "soft_loss",
            "discrete_loss",
            "loss_gap",
            "train_time",
            "time_to_target",
            "unused_gate_ratio",
        ]:
            row[column] = result_row.get(column, "")
        for column in [
            "pre_gate_count",
            "pre_depth",
            "pre_fanout_max",
            "abc_pre_nd",
            "abc_pre_edge",
            "abc_pre_cube",
            "abc_pre_lev",
            "abc_post_and",
            "abc_post_lev",
        ]:
            row[column] = synth_row.get(column, "")
        row["abc_and_reduction_vs_pre_nd"] = reduction(pre_nd, post_and)
        row["abc_level_delta_vs_pre_lev"] = post_lev - pre_lev if math.isfinite(post_lev) and math.isfinite(pre_lev) else math.nan
        row["abc_level_ratio_vs_pre_lev"] = ratio(post_lev, pre_lev)
        joined.append(row)
    return sorted(joined, key=sort_key)


def summarize_by_method_dataset(joined_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in joined_rows:
        grouped[(str(row["dataset"]), str(row["method"]))].append(row)

    out: list[dict[str, object]] = []
    for (dataset, method), group in sorted(grouped.items(), key=lambda item: sort_key(item[1][0])):
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "n": len(group),
            "n_abc_ok": sum(1 for item in group if item.get("abc_status") == "ok"),
            "accuracy_recomputed_after_abc": "no",
        }
        for column in SUMMARY_NUMERIC_COLUMNS:
            row[f"{column}_mean"] = finite_mean([to_float(item.get(column)) for item in group])
        out.append(row)
    return out


def write_markdown_report(
    path: Path,
    run_dir: Path,
    joined_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    ok_count = sum(1 for row in joined_rows if row.get("abc_status") == "ok")
    lines = [
        "# Hard-LGN ABC Synthesis Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "ABC is used here as a structural baseline only. The optimized networks are not imported back into PyTorch, so post-ABC accuracy and gap are not re-evaluated.",
        "",
        f"Rows: {len(joined_rows)} total, {ok_count} with `abc_status=ok`.",
        "",
        "## Method/Dataset Summary",
        "| dataset | method | discrete_acc | acc_gap | unused_gate_ratio | pre_gate_count | pre_depth | pre_fanout_max | abc_post_and | abc_post_lev | and_reduction | level_delta |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    fmt(row.get("discrete_acc_mean")),
                    fmt(row.get("acc_gap_mean")),
                    fmt(row.get("unused_gate_ratio_mean")),
                    fmt(row.get("pre_gate_count_mean")),
                    fmt(row.get("pre_depth_mean")),
                    fmt(row.get("pre_fanout_max_mean")),
                    fmt(row.get("abc_post_and_mean")),
                    fmt(row.get("abc_post_lev_mean")),
                    pct(row.get("abc_and_reduction_vs_pre_nd_mean")),
                    fmt(row.get("abc_level_delta_vs_pre_lev_mean")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `abc_post_and` is the post-`strash; dc2` ABC AND-node count, not a PyTorch gate-count replacement.",
            "- `and_reduction` is `(abc_pre_nd - abc_post_and) / abc_pre_nd`.",
            "- `level_delta` is `abc_post_lev - abc_pre_lev`; positive values mean ABC increased the reported logic level.",
            "- Accuracy columns are copied from `results.csv` before ABC optimization.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory containing results.csv and synthesis_stats.csv.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for synthesis summary outputs.")
    parser.add_argument("--run-name", default="", help="Optional label stored in synthesis_joined.csv.")
    args = parser.parse_args()

    run_name = args.run_name or args.run_dir.name
    joined_rows = load_joined_rows(args.run_dir, run_name)
    summary_rows = summarize_by_method_dataset(joined_rows)
    write_csv(args.out_dir / "synthesis_joined.csv", joined_rows, JOIN_COLUMNS)
    write_csv(args.out_dir / "synthesis_by_method_dataset.csv", summary_rows)
    write_markdown_report(args.out_dir / "synthesis_report.md", args.run_dir, joined_rows, summary_rows)
    print(args.out_dir / "synthesis_joined.csv")
    print(args.out_dir / "synthesis_by_method_dataset.csv")
    print(args.out_dir / "synthesis_report.md")


if __name__ == "__main__":
    main()
