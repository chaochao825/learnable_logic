#!/usr/bin/env python3
"""Build a consolidated Hard-LGN vs Mind-the-Gap-style report."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_RUNS = [
    "bool_seeds012_timing_v3",
    "digits_seeds012_timing_v3",
    "cifar10_small_seed0_timing_v3",
]
METHOD_ORDER = {
    "dlgn": 0,
    "dlgn_anneal": 1,
    "gumbel_st": 2,
    "block_relaxed": 3,
    "block_hard_refit": 4,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


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


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def run_label(name: str) -> str:
    if name.startswith("bool"):
        return "boolean"
    if name.startswith("digits"):
        return "digits"
    if name.startswith("cifar"):
        return "cifar10_small"
    return name


def load_run(runs_root: Path, run_name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    aggregate_path = runs_root / run_name / "aggregate" / "aggregate_by_method_dataset.csv"
    comparison_path = runs_root / run_name / "aggregate" / "comparison_vs_baselines.csv"
    return read_csv(aggregate_path), read_csv(comparison_path)


def read_optional_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def build_success_rows(runs_root: Path, run_names: list[str]) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    success_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, str]] = []
    for run_name in run_names:
        agg_rows, comp_rows = load_run(runs_root, run_name)
        label = run_label(run_name)
        by_key = {(row["dataset"], row["method"]): row for row in agg_rows}
        comp_by_key = {(row["dataset"], row["method"]): row for row in comp_rows}
        method_rows.extend(dict(row, run=label, source_run=run_name) for row in agg_rows)
        for dataset in sorted({row["dataset"] for row in agg_rows}):
            dlgn = by_key.get((dataset, "dlgn"))
            gumbel = by_key.get((dataset, "gumbel_st"))
            hard = by_key.get((dataset, "block_hard_refit"))
            hard_comp = comp_by_key.get((dataset, "block_hard_refit"), {})
            if not dlgn or not hard:
                continue
            dlgn_gap = to_float(dlgn.get("acc_gap_mean"))
            gumbel_gap = to_float(gumbel.get("acc_gap_mean")) if gumbel else math.nan
            hard_gap = to_float(hard.get("acc_gap_mean"))
            hard_path_gap = to_float(hard.get("path_acc_gap_mean"))
            dlgn_unused = to_float(dlgn.get("unused_gate_ratio_mean"))
            hard_unused = to_float(hard.get("unused_gate_ratio_mean"))
            row: dict[str, object] = {
                "run": label,
                "source_run": run_name,
                "dataset": dataset,
                "discrete_acc": hard.get("discrete_acc_mean", ""),
                "acc_gap": hard_gap,
                "path_acc_gap": hard_path_gap,
                "loss_gap": hard.get("loss_gap_mean", ""),
                "path_loss_gap": hard.get("path_loss_gap_mean", ""),
                "dlgn_acc_gap": dlgn_gap,
                "gumbel_acc_gap": gumbel_gap,
                "unused_gate_ratio": hard_unused,
                "dlgn_unused_gate_ratio": dlgn_unused,
                "train_speedup_vs_dlgn": hard_comp.get("speedup_vs_dlgn", ""),
                "time_to_target_mean": hard.get("time_to_target_mean", ""),
                "target_hit_rate": hard.get("target_hit_rate", ""),
                "primary_full_acc_gap_beats_dlgn": bool_text(
                    math.isfinite(hard_gap) and math.isfinite(dlgn_gap) and hard_gap < dlgn_gap
                ),
                "path_gap_diagnostic_less_than_dlgn": bool_text(
                    math.isfinite(hard_path_gap) and math.isfinite(dlgn_gap) and hard_path_gap < dlgn_gap
                ),
                "primary_full_acc_gap_competitive_with_gumbel": bool_text(
                    math.isfinite(hard_gap) and math.isfinite(gumbel_gap) and hard_gap <= gumbel_gap
                ),
                "unused_reduction_vs_dlgn": (
                    (dlgn_unused - hard_unused) / dlgn_unused if dlgn_unused > 0 and math.isfinite(hard_unused) else math.nan
                ),
            }
            success_rows.append(row)
    return success_rows, method_rows


def build_claim_rows(method_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    by_key = {(row["run"], row["dataset"], row["method"]): row for row in method_rows}
    for run, dataset in sorted({(row["run"], row["dataset"]) for row in method_rows}):
        dlgn = by_key.get((run, dataset, "dlgn"))
        gumbel = by_key.get((run, dataset, "gumbel_st"))
        if not dlgn or not gumbel:
            continue
        dlgn_gap = to_float(dlgn.get("acc_gap_mean"))
        gumbel_gap = to_float(gumbel.get("acc_gap_mean"))
        dlgn_unused = to_float(dlgn.get("unused_gate_ratio_mean"))
        gumbel_unused = to_float(gumbel.get("unused_gate_ratio_mean"))
        dlgn_time = to_float(dlgn.get("train_time_mean"))
        gumbel_time = to_float(gumbel.get("train_time_mean"))
        gap_reduction = (dlgn_gap - gumbel_gap) / dlgn_gap if dlgn_gap >= 1e-2 else math.nan
        unused_reduction = (dlgn_unused - gumbel_unused) / dlgn_unused if dlgn_unused > 0 else math.nan
        train_speedup = dlgn_time / gumbel_time if gumbel_time > 0 else math.nan
        out.append(
            {
                "run": run,
                "dataset": dataset,
                "dlgn_acc_gap": dlgn_gap,
                "gumbel_acc_gap": gumbel_gap,
                "gumbel_gap_reduction_vs_dlgn": gap_reduction,
                "mind_gap_98pct_gap_target_met": bool_text(math.isfinite(gap_reduction) and gap_reduction >= 0.98),
                "dlgn_unused_gate_ratio": dlgn_unused,
                "gumbel_unused_gate_ratio": gumbel_unused,
                "gumbel_unused_reduction_vs_dlgn": unused_reduction,
                "mind_gap_100pct_unused_target_met": bool_text(math.isfinite(unused_reduction) and unused_reduction >= 1.0),
                "train_time_speedup_vs_dlgn": train_speedup,
                "mind_gap_4p5x_train_time_target_met": bool_text(math.isfinite(train_speedup) and train_speedup >= 4.5),
                "dlgn_target_hit_rate": dlgn.get("target_hit_rate", ""),
                "gumbel_target_hit_rate": gumbel.get("target_hit_rate", ""),
            }
        )
    return out


def final_layer_row(rows: list[dict[str, str]], dataset: str, method: str, comparison: str) -> dict[str, str] | None:
    group = [
        row
        for row in rows
        if row["dataset"] == dataset and row["method"] == method and row["comparison"] == comparison
    ]
    if not group:
        return None
    return max(group, key=lambda row: int(row.get("layer", 0)))


def first_layer_row(rows: list[dict[str, str]], dataset: str, method: str, comparison: str) -> dict[str, str] | None:
    group = [
        row
        for row in rows
        if row["dataset"] == dataset and row["method"] == method and row["comparison"] == comparison
    ]
    if not group:
        return None
    return min(group, key=lambda row: int(row.get("layer", 0)))


def metric_value(row: dict[str, str] | None, metric: str) -> float:
    if not row:
        return math.nan
    return to_float(row.get(f"{metric}_mean", row.get(metric, "")))


def layer_growth(
    rows: list[dict[str, str]],
    dataset: str,
    method: str,
    comparison: str,
    metric: str,
) -> float:
    first = first_layer_row(rows, dataset, method, comparison)
    final = final_layer_row(rows, dataset, method, comparison)
    first_value = metric_value(first, metric)
    final_value = metric_value(final, metric)
    if not math.isfinite(first_value) or not math.isfinite(final_value):
        return math.nan
    return final_value - first_value


def build_depth_rows(runs_root: Path, run_names: list[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for run_name in run_names:
        layer_path = runs_root / run_name / "aggregate" / "layer_diagnostics_by_layer.csv"
        rows = read_optional_csv(layer_path)
        if not rows:
            continue
        label = run_label(run_name)
        for dataset in sorted({row["dataset"] for row in rows}):
            dlgn_final = final_layer_row(rows, dataset, "dlgn", "full_soft_vs_discrete")
            gumbel_final = final_layer_row(rows, dataset, "gumbel_st", "full_soft_vs_discrete")
            hard_full_final = final_layer_row(rows, dataset, "block_hard_refit", "full_soft_vs_discrete")
            hard_path_final = final_layer_row(rows, dataset, "block_hard_refit", "path_soft_vs_discrete")
            if not dlgn_final or not hard_full_final:
                continue
            dlgn_final_mad = metric_value(dlgn_final, "mean_abs_diff")
            gumbel_final_mad = metric_value(gumbel_final, "mean_abs_diff")
            hard_full_final_mad = metric_value(hard_full_final, "mean_abs_diff")
            hard_path_final_mad = metric_value(hard_path_final, "mean_abs_diff")
            dlgn_flip = metric_value(dlgn_final, "binary_flip_ratio")
            gumbel_flip = metric_value(gumbel_final, "binary_flip_ratio")
            hard_full_flip = metric_value(hard_full_final, "binary_flip_ratio")
            hard_path_flip = metric_value(hard_path_final, "binary_flip_ratio")
            row: dict[str, object] = {
                "run": label,
                "source_run": run_name,
                "dataset": dataset,
                "dlgn_final_mean_abs_diff": dlgn_final_mad,
                "gumbel_final_mean_abs_diff": gumbel_final_mad,
                "block_hard_full_final_mean_abs_diff": hard_full_final_mad,
                "block_hard_path_final_mean_abs_diff": hard_path_final_mad,
                "block_hard_full_vs_dlgn_final_delta": hard_full_final_mad - dlgn_final_mad,
                "block_hard_path_vs_dlgn_final_delta": hard_path_final_mad - dlgn_final_mad,
                "block_hard_full_final_less_than_dlgn": bool_text(
                    math.isfinite(hard_full_final_mad) and math.isfinite(dlgn_final_mad) and hard_full_final_mad < dlgn_final_mad
                ),
                "block_hard_path_final_less_than_dlgn": bool_text(
                    math.isfinite(hard_path_final_mad) and math.isfinite(dlgn_final_mad) and hard_path_final_mad < dlgn_final_mad
                ),
                "block_hard_path_final_competitive_with_gumbel": bool_text(
                    math.isfinite(hard_path_final_mad)
                    and math.isfinite(gumbel_final_mad)
                    and hard_path_final_mad <= gumbel_final_mad
                ),
                "dlgn_final_binary_flip_ratio": dlgn_flip,
                "gumbel_final_binary_flip_ratio": gumbel_flip,
                "block_hard_full_final_binary_flip_ratio": hard_full_flip,
                "block_hard_path_final_binary_flip_ratio": hard_path_flip,
                "dlgn_mean_abs_diff_growth": layer_growth(rows, dataset, "dlgn", "full_soft_vs_discrete", "mean_abs_diff"),
                "gumbel_mean_abs_diff_growth": layer_growth(rows, dataset, "gumbel_st", "full_soft_vs_discrete", "mean_abs_diff"),
                "block_hard_full_mean_abs_diff_growth": layer_growth(
                    rows, dataset, "block_hard_refit", "full_soft_vs_discrete", "mean_abs_diff"
                ),
                "block_hard_path_mean_abs_diff_growth": layer_growth(
                    rows, dataset, "block_hard_refit", "path_soft_vs_discrete", "mean_abs_diff"
                ),
            }
            out.append(row)
    return out


def build_tuned_seed0_checks(tuned_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build seed-0 checks where the swept Gumbel row is the comparator.

    The tuned comparison file is seed-0 only in the timing-v3 workflow, so this
    stays separate from the multi-seed aggregate success table above.
    """
    success_rows: list[dict[str, object]] = []
    claim_rows: list[dict[str, object]] = []
    by_dataset_method = {(row["dataset"], row["method"]): row for row in tuned_rows}
    datasets = sorted({row["dataset"] for row in tuned_rows})
    for dataset in datasets:
        dlgn = by_dataset_method.get((dataset, "dlgn"))
        hard = by_dataset_method.get((dataset, "block_hard_refit"))
        tuned_gumbel = by_dataset_method.get((dataset, "gumbel_st_best_discrete_acc"))
        if not dlgn or not hard or not tuned_gumbel:
            continue

        dlgn_gap = to_float(dlgn.get("acc_gap"))
        hard_gap = to_float(hard.get("acc_gap"))
        hard_path_gap = to_float(hard.get("path_acc_gap"))
        tuned_gap = to_float(tuned_gumbel.get("acc_gap"))
        dlgn_unused = to_float(dlgn.get("unused_gate_ratio"))
        hard_unused = to_float(hard.get("unused_gate_ratio"))
        tuned_unused = to_float(tuned_gumbel.get("unused_gate_ratio"))
        dlgn_train_time = to_float(dlgn.get("train_time"))
        tuned_train_time = to_float(tuned_gumbel.get("train_time"))
        dlgn_time_to_target = to_float(dlgn.get("time_to_target"))
        tuned_time_to_target = to_float(tuned_gumbel.get("time_to_target"))
        gap_reduction = (dlgn_gap - tuned_gap) / dlgn_gap if dlgn_gap >= 1e-2 else math.nan
        tuned_unused_reduction = (dlgn_unused - tuned_unused) / dlgn_unused if dlgn_unused > 0 else math.nan
        hard_unused_reduction = (dlgn_unused - hard_unused) / dlgn_unused if dlgn_unused > 0 else math.nan
        train_speedup = dlgn_train_time / tuned_train_time if tuned_train_time > 0 else math.nan
        time_to_target_speedup = (
            dlgn_time_to_target / tuned_time_to_target
            if dlgn_time_to_target > 0 and tuned_time_to_target > 0
            else math.nan
        )

        success_rows.append(
            {
                "comparison_scope": "seed0_tuned_gumbel",
                "dataset": dataset,
                "dlgn_discrete_acc": dlgn.get("discrete_acc", ""),
                "block_hard_discrete_acc": hard.get("discrete_acc", ""),
                "tuned_gumbel_discrete_acc": tuned_gumbel.get("discrete_acc", ""),
                "dlgn_acc_gap": dlgn_gap,
                "block_hard_acc_gap": hard_gap,
                "block_hard_path_acc_gap": hard_path_gap,
                "tuned_gumbel_acc_gap": tuned_gap,
                "block_hard_primary_full_acc_gap_beats_dlgn": bool_text(
                    math.isfinite(hard_gap) and math.isfinite(dlgn_gap) and hard_gap < dlgn_gap
                ),
                "block_hard_path_gap_diagnostic_less_than_dlgn": bool_text(
                    math.isfinite(hard_path_gap) and math.isfinite(dlgn_gap) and hard_path_gap < dlgn_gap
                ),
                "block_hard_primary_full_acc_gap_competitive_with_tuned_gumbel": bool_text(
                    math.isfinite(hard_gap) and math.isfinite(tuned_gap) and hard_gap <= tuned_gap
                ),
                "block_hard_path_gap_diagnostic_competitive_with_tuned_gumbel": bool_text(
                    math.isfinite(hard_path_gap) and math.isfinite(tuned_gap) and hard_path_gap <= tuned_gap
                ),
                "block_hard_unused_reduction_vs_dlgn": hard_unused_reduction,
                "tuned_gumbel_unused_reduction_vs_dlgn": tuned_unused_reduction,
                "tuned_gumbel_trial": tuned_gumbel.get("trial", ""),
                "tuned_gumbel_selection_criterion": tuned_gumbel.get("selection_criterion", ""),
                "tuned_gumbel_source_run": tuned_gumbel.get("source_run", ""),
            }
        )
        claim_rows.append(
            {
                "comparison_scope": "seed0_tuned_gumbel",
                "dataset": dataset,
                "dlgn_acc_gap": dlgn_gap,
                "tuned_gumbel_acc_gap": tuned_gap,
                "tuned_gumbel_gap_reduction_vs_dlgn": gap_reduction,
                "mind_gap_98pct_gap_target_met": bool_text(math.isfinite(gap_reduction) and gap_reduction >= 0.98),
                "dlgn_unused_gate_ratio": dlgn_unused,
                "tuned_gumbel_unused_gate_ratio": tuned_unused,
                "tuned_gumbel_unused_reduction_vs_dlgn": tuned_unused_reduction,
                "mind_gap_100pct_unused_target_met": bool_text(
                    math.isfinite(tuned_unused_reduction) and tuned_unused_reduction >= 1.0
                ),
                "train_time_speedup_vs_dlgn": train_speedup,
                "mind_gap_4p5x_train_time_target_met": bool_text(
                    math.isfinite(train_speedup) and train_speedup >= 4.5
                ),
                "dlgn_time_to_target": dlgn_time_to_target,
                "tuned_gumbel_time_to_target": tuned_time_to_target,
                "time_to_target_speedup_vs_dlgn": time_to_target_speedup,
                "mind_gap_4p5x_time_to_target_met": bool_text(
                    math.isfinite(time_to_target_speedup) and time_to_target_speedup >= 4.5
                ),
                "tuned_gumbel_trial": tuned_gumbel.get("trial", ""),
                "tuned_gumbel_selection_criterion": tuned_gumbel.get("selection_criterion", ""),
            }
        )
    return success_rows, claim_rows


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(
    path: Path,
    success_rows: list[dict[str, object]],
    claim_rows: list[dict[str, object]],
    depth_rows: list[dict[str, object]],
    tuned_success_rows: list[dict[str, object]],
    tuned_claim_rows: list[dict[str, object]],
    tuned_rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Hard-LGN Gap Report",
        "",
        "This report summarizes the refreshed timing-v3 prototype runs against the Mind-the-Gap-style metrics.",
        "Percent reductions are conservative: unstable ratios are reported as `nan` when the DLGN gap is below `1e-2` in aggregate or tuned seed-0 claim checks.",
        "",
        "## Proposed Method Success Checks",
        "",
        "Only `primary_full_acc_gap_*` columns are pass/fail success checks. `path_acc_gap` columns are diagnostics for the method-native hard-prefix training path and are not counted as primary success.",
    ]
    success_display = [
        {
            "run": row["run"],
            "dataset": row["dataset"],
            "discrete_acc": fmt(row["discrete_acc"]),
            "acc_gap": fmt(row["acc_gap"]),
            "path_acc_gap": fmt(row["path_acc_gap"]),
            "dlgn_acc_gap": fmt(row["dlgn_acc_gap"]),
            "gumbel_acc_gap": fmt(row["gumbel_acc_gap"]),
            "primary_full_acc_gap_beats_dlgn": row["primary_full_acc_gap_beats_dlgn"],
            "path_gap_diagnostic_less_than_dlgn": row["path_gap_diagnostic_less_than_dlgn"],
            "primary_full_acc_gap_competitive_with_gumbel": row[
                "primary_full_acc_gap_competitive_with_gumbel"
            ],
            "unused_reduction_vs_dlgn": pct(row["unused_reduction_vs_dlgn"]),
            "target_hit_rate": fmt(row["target_hit_rate"]),
        }
        for row in success_rows
    ]
    lines.extend(
        markdown_table(
            success_display,
            [
                "run",
                "dataset",
                "discrete_acc",
                "acc_gap",
                "path_acc_gap",
                "dlgn_acc_gap",
                "gumbel_acc_gap",
                "primary_full_acc_gap_beats_dlgn",
                "path_gap_diagnostic_less_than_dlgn",
                "primary_full_acc_gap_competitive_with_gumbel",
                "unused_reduction_vs_dlgn",
                "target_hit_rate",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Mind-the-Gap Claim Checks",
        ]
    )
    claim_display = [
        {
            "run": row["run"],
            "dataset": row["dataset"],
            "gumbel_gap_reduction_vs_dlgn": pct(row["gumbel_gap_reduction_vs_dlgn"]),
            "98pct_gap_target": row["mind_gap_98pct_gap_target_met"],
            "unused_reduction_vs_dlgn": pct(row["gumbel_unused_reduction_vs_dlgn"]),
            "100pct_unused_target": row["mind_gap_100pct_unused_target_met"],
            "train_time_speedup_vs_dlgn": fmt(row["train_time_speedup_vs_dlgn"]),
            "4p5x_time_target": row["mind_gap_4p5x_train_time_target_met"],
            "dlgn_target_hit_rate": row["dlgn_target_hit_rate"],
            "gumbel_target_hit_rate": row["gumbel_target_hit_rate"],
        }
        for row in claim_rows
    ]
    lines.extend(
        markdown_table(
            claim_display,
            [
                "run",
                "dataset",
                "gumbel_gap_reduction_vs_dlgn",
                "98pct_gap_target",
                "unused_reduction_vs_dlgn",
                "100pct_unused_target",
                "train_time_speedup_vs_dlgn",
                "4p5x_time_target",
                "dlgn_target_hit_rate",
                "gumbel_target_hit_rate",
            ],
        )
    )
    if depth_rows:
        lines.extend(
            [
                "",
                "## Depth-Wise Representation Gap Checks",
                "",
                "These rows use `layer_diagnostics_by_layer.csv`. `mean_abs_diff` is the relaxed-path versus hard-path representation mismatch at the final layer prefix; growth is final-layer minus first-layer mismatch.",
            ]
        )
        depth_display = [
            {
                "run": row["run"],
                "dataset": row["dataset"],
                "dlgn_final_mad": fmt(row["dlgn_final_mean_abs_diff"]),
                "gumbel_final_mad": fmt(row["gumbel_final_mean_abs_diff"]),
                "block_full_final_mad": fmt(row["block_hard_full_final_mean_abs_diff"]),
                "block_path_final_mad": fmt(row["block_hard_path_final_mean_abs_diff"]),
                "full_less_than_dlgn": row["block_hard_full_final_less_than_dlgn"],
                "path_less_than_dlgn": row["block_hard_path_final_less_than_dlgn"],
                "path_competitive_with_gumbel": row["block_hard_path_final_competitive_with_gumbel"],
                "dlgn_growth": fmt(row["dlgn_mean_abs_diff_growth"]),
                "gumbel_growth": fmt(row["gumbel_mean_abs_diff_growth"]),
                "block_full_growth": fmt(row["block_hard_full_mean_abs_diff_growth"]),
                "block_path_growth": fmt(row["block_hard_path_mean_abs_diff_growth"]),
            }
            for row in depth_rows
        ]
        lines.extend(
            markdown_table(
                depth_display,
                [
                    "run",
                    "dataset",
                    "dlgn_final_mad",
                    "gumbel_final_mad",
                    "block_full_final_mad",
                    "block_path_final_mad",
                    "full_less_than_dlgn",
                    "path_less_than_dlgn",
                    "path_competitive_with_gumbel",
                    "dlgn_growth",
                    "gumbel_growth",
                    "block_full_growth",
                    "block_path_growth",
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Tuned Seed-0 Gumbel Comparison",
        ]
    )
    if tuned_success_rows:
        lines.extend(
            [
                "",
                "The following checks compare seed-0 block-hard-refit directly against the swept `gumbel_st_best_discrete_acc` row. This is separate from the multi-seed aggregate table above.",
                "",
                "### Proposed Method vs Tuned Gumbel",
            ]
        )
        tuned_success_display = [
            {
                "dataset": row["dataset"],
                "block_hard_discrete_acc": fmt(row["block_hard_discrete_acc"]),
                "tuned_gumbel_discrete_acc": fmt(row["tuned_gumbel_discrete_acc"]),
                "block_hard_acc_gap": fmt(row["block_hard_acc_gap"]),
                "block_hard_path_acc_gap": fmt(row["block_hard_path_acc_gap"]),
                "dlgn_acc_gap": fmt(row["dlgn_acc_gap"]),
                "tuned_gumbel_acc_gap": fmt(row["tuned_gumbel_acc_gap"]),
                "primary_full_acc_gap_beats_dlgn": row["block_hard_primary_full_acc_gap_beats_dlgn"],
                "path_gap_diagnostic_less_than_dlgn": row[
                    "block_hard_path_gap_diagnostic_less_than_dlgn"
                ],
                "primary_full_acc_gap_competitive_with_tuned_gumbel": row[
                    "block_hard_primary_full_acc_gap_competitive_with_tuned_gumbel"
                ],
                "path_gap_diagnostic_competitive_with_tuned_gumbel": row[
                    "block_hard_path_gap_diagnostic_competitive_with_tuned_gumbel"
                ],
                "tuned_gumbel_trial": row["tuned_gumbel_trial"],
            }
            for row in tuned_success_rows
        ]
        lines.extend(
            markdown_table(
                tuned_success_display,
                [
                    "dataset",
                    "block_hard_discrete_acc",
                    "tuned_gumbel_discrete_acc",
                    "block_hard_acc_gap",
                    "block_hard_path_acc_gap",
                    "dlgn_acc_gap",
                    "tuned_gumbel_acc_gap",
                    "primary_full_acc_gap_beats_dlgn",
                    "path_gap_diagnostic_less_than_dlgn",
                    "primary_full_acc_gap_competitive_with_tuned_gumbel",
                    "path_gap_diagnostic_competitive_with_tuned_gumbel",
                    "tuned_gumbel_trial",
                ],
            )
        )
        lines.extend(["", "### Tuned Gumbel Claim Checks"])
        tuned_claim_display = [
            {
                "dataset": row["dataset"],
                "gap_reduction_vs_dlgn": pct(row["tuned_gumbel_gap_reduction_vs_dlgn"]),
                "98pct_gap_target": row["mind_gap_98pct_gap_target_met"],
                "unused_reduction_vs_dlgn": pct(row["tuned_gumbel_unused_reduction_vs_dlgn"]),
                "100pct_unused_target": row["mind_gap_100pct_unused_target_met"],
                "train_time_speedup_vs_dlgn": fmt(row["train_time_speedup_vs_dlgn"]),
                "4p5x_train_time_target": row["mind_gap_4p5x_train_time_target_met"],
                "time_to_target_speedup_vs_dlgn": fmt(row["time_to_target_speedup_vs_dlgn"]),
                "4p5x_time_to_target": row["mind_gap_4p5x_time_to_target_met"],
                "tuned_gumbel_trial": row["tuned_gumbel_trial"],
            }
            for row in tuned_claim_rows
        ]
        lines.extend(
            markdown_table(
                tuned_claim_display,
                [
                    "dataset",
                    "gap_reduction_vs_dlgn",
                    "98pct_gap_target",
                    "unused_reduction_vs_dlgn",
                    "100pct_unused_target",
                    "train_time_speedup_vs_dlgn",
                    "4p5x_train_time_target",
                    "time_to_target_speedup_vs_dlgn",
                    "4p5x_time_to_target",
                    "tuned_gumbel_trial",
                ],
            )
        )

    lines.extend(["", "### Source Rows"])
    tuned_display = []
    for row in tuned_rows:
        tuned_display.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "source_run": row["source_run"],
                "discrete_acc": fmt(row.get("discrete_acc")),
                "acc_gap": fmt(row.get("acc_gap")),
                "time_to_target": fmt(row.get("time_to_target")),
                "unused_gate_ratio": fmt(row.get("unused_gate_ratio")),
                "trial": row.get("trial", ""),
                "selection_criterion": row.get("selection_criterion", ""),
            }
        )
    lines.extend(
        markdown_table(
            tuned_display,
            [
                "dataset",
                "method",
                "source_run",
                "discrete_acc",
                "acc_gap",
                "time_to_target",
                "unused_gate_ratio",
                "trial",
                "selection_criterion",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `block_hard_refit` is interesting only where primary full `acc_gap` beats DLGN and is competitive with Gumbel-ST under the same run.",
            "- Depth-wise representation checks are diagnostic mismatch measures, not accuracy or loss measurements.",
            "- Tuned seed-0 Gumbel checks use the sweep-selected Gumbel row and should not be mixed with the multi-seed aggregate table.",
            "- `path_acc_gap` isolates the method-native hard-prefix training path and should not be conflated with full relaxed-network `acc_gap`.",
            "- Current timing-v3 evidence remains prototype-scale; it does not reproduce the full CIFAR-10 Mind-the-Gap setting.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--tuned-comparison", type=Path, default=Path("runs/gumbel_tuned_comparison_seed0_timing_v3/comparison.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/timing_v3"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success_rows, method_rows = build_success_rows(args.runs_root, args.runs)
    claim_rows = build_claim_rows(method_rows)
    depth_rows = build_depth_rows(args.runs_root, args.runs)
    tuned_rows = read_csv(args.tuned_comparison) if args.tuned_comparison.exists() else []
    tuned_success_rows, tuned_claim_rows = build_tuned_seed0_checks(tuned_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "hard_lgn_success_criteria.csv", success_rows)
    write_csv(args.out_dir / "mind_the_gap_claim_checks.csv", claim_rows)
    write_csv(args.out_dir / "depth_gap_accumulation_checks.csv", depth_rows)
    write_csv(args.out_dir / "strongest_seed0_gumbel_success_checks.csv", tuned_success_rows)
    write_csv(args.out_dir / "strongest_seed0_gumbel_claim_checks.csv", tuned_claim_rows)
    write_markdown(
        args.out_dir / "mind_the_gap_timing_v3_report.md",
        success_rows,
        claim_rows,
        depth_rows,
        tuned_success_rows,
        tuned_claim_rows,
        tuned_rows,
    )
    print(args.out_dir / "hard_lgn_success_criteria.csv")
    print(args.out_dir / "mind_the_gap_claim_checks.csv")
    print(args.out_dir / "depth_gap_accumulation_checks.csv")
    print(args.out_dir / "strongest_seed0_gumbel_success_checks.csv")
    print(args.out_dir / "strongest_seed0_gumbel_claim_checks.csv")
    print(args.out_dir / "mind_the_gap_timing_v3_report.md")


if __name__ == "__main__":
    main()
