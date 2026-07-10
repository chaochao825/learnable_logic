#!/usr/bin/env python3
"""Build current-evidence tables for the pasted Hard-LGN next-stage goal.

The output is deliberately an evidence audit, not a completion claim. It
collects the currently available Part A/B/C runs, writes the required tables
where evidence exists, and marks missing or smoke-only coverage explicitly.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


UNIFIED_FIELDS = [
    "axis",
    "source_run",
    "method",
    "item",
    "seed",
    "hard_acc",
    "soft_acc",
    "full_gap",
    "native_gap",
    "train_time",
    "unused_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "extra",
]

SUCCESS_FIELDS = [
    "criterion",
    "status",
    "evidence",
    "note",
]

COVERAGE_FIELDS = [
    "area",
    "requirement",
    "status",
    "evidence",
    "missing",
    "note",
]

STRONG_REDUNDANCY_FIELDS = [
    "dataset",
    "seed",
    "task_hardened_hard_acc",
    "task_hardened_soft_acc",
    "task_hardened_full_gap",
    "task_hardened_native_gap",
    "task_hardened_unused_gate_ratio",
    "task_hardened_gate_count",
    "task_hardened_factor",
    "task_hardened_hardening",
    "strong_baseline_method",
    "strong_baseline_hard_acc",
    "strong_baseline_soft_acc",
    "strong_baseline_full_gap",
    "strong_baseline_native_gap",
    "strong_baseline_unused_gate_ratio",
    "strong_baseline_gate_count",
    "strong_baseline_factor",
    "strong_baseline_hardening",
    "delta_hard_acc",
    "delta_full_gap",
    "delta_native_gap",
    "delta_unused_gate_ratio",
    "delta_gate_count",
    "status",
]

TRUTH_REFIT_FIELDS = [
    "dataset",
    "seed",
    "redundancy_factor",
    "truth_val_hard_acc",
    "truth_val_full_gap",
    "truth_refit_error",
    "truth_refit_delta",
    "truth_op_change_ratio",
    "truth_unused_gate_ratio",
    "truth_gate_count",
    "best_non_refit_candidate",
    "best_non_refit_val_hard_acc",
    "best_non_refit_val_full_gap",
    "best_non_refit_unused_gate_ratio",
    "selected_candidate",
    "selected_val_hard_acc",
    "selected_is_truth_refit",
    "delta_truth_vs_best_non_refit",
    "status",
    "note",
]

TRUTH_REFIT_FINAL_FIELDS = [
    "dataset",
    "seed",
    "truth_refit_hard_acc",
    "truth_refit_soft_acc",
    "truth_refit_full_gap",
    "truth_refit_native_gap",
    "truth_refit_unused_gate_ratio",
    "truth_refit_gate_count",
    "truth_refit_factor",
    "task_selected_hard_acc",
    "task_selected_hardening",
    "task_selected_factor",
    "strong_baseline_method",
    "strong_baseline_hard_acc",
    "strong_baseline_hardening",
    "strong_baseline_factor",
    "delta_vs_task_selected",
    "delta_vs_strong_baseline",
    "status_vs_task_selected",
    "status_vs_strong_baseline",
    "note",
]

MIND_GAP_DIRECT_FIELDS = [
    "comparison_scope",
    "source_run",
    "gumbel_source_run",
    "dataset",
    "n",
    "proposed_method",
    "gumbel_method",
    "dlgn_discrete_acc",
    "dlgn_soft_acc",
    "dlgn_full_gap",
    "dlgn_unused_gate_ratio",
    "dlgn_train_time",
    "dlgn_time_to_target",
    "block_hard_discrete_acc",
    "block_hard_soft_acc",
    "block_hard_full_gap",
    "block_hard_native_gap",
    "block_hard_unused_gate_ratio",
    "block_hard_train_time",
    "block_hard_time_to_target",
    "gumbel_discrete_acc",
    "gumbel_soft_acc",
    "gumbel_full_gap",
    "gumbel_unused_gate_ratio",
    "gumbel_train_time",
    "gumbel_time_to_target",
    "delta_hard_acc_vs_dlgn",
    "delta_hard_acc_vs_gumbel",
    "delta_full_gap_vs_dlgn",
    "delta_native_gap_vs_dlgn",
    "delta_full_gap_vs_gumbel",
    "delta_native_gap_vs_gumbel",
    "hard_acc_beats_dlgn",
    "hard_acc_beats_gumbel",
    "full_gap_beats_dlgn",
    "native_gap_beats_dlgn",
    "full_gap_competitive_with_gumbel",
    "native_gap_competitive_with_gumbel",
    "block_unused_reduction_vs_dlgn",
    "gumbel_gap_reduction_vs_dlgn",
    "gumbel_unused_reduction_vs_dlgn",
    "gumbel_train_speedup_vs_dlgn",
    "gumbel_time_to_target_speedup_vs_dlgn",
    "mind_gap_98pct_gap_target_met",
    "mind_gap_100pct_unused_target_met",
    "mind_gap_4p5x_train_time_target_met",
    "mind_gap_4p5x_time_to_target_target_met",
    "selection_criterion",
    "note",
]

SYNTHESIS_FIELDS = [
    "source_report",
    "run",
    "dataset",
    "method",
    "seed",
    "abc_status",
    "soft_acc",
    "discrete_acc",
    "acc_gap",
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
    "source_blif_acc",
    "post_abc_discrete_acc",
    "source_acc_delta_vs_results",
    "post_acc_delta_vs_pre_discrete",
    "post_abc_gap_vs_pre_soft",
    "post_gap_delta_vs_pre_gap",
    "source_blif_loss",
    "post_abc_discrete_loss",
    "source_loss_delta_vs_results",
    "post_loss_delta_vs_pre_discrete",
    "source_blif_fanout_max",
    "post_abc_fanout_max",
    "source_blif_unused_node_ratio",
    "post_abc_unused_node_ratio",
    "source_blif_unused_node_count",
    "post_abc_unused_node_count",
    "accuracy_change_after_abc",
    "gap_change_after_abc",
    "gate_count_change_after_abc",
    "depth_change_after_abc",
    "fanout_change_after_abc",
    "unused_gate_ratio_change_after_abc",
    "post_abc_eval_status",
    "post_abc_eval_report",
    "note",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_run_csvs(runs_root: Path, run_names: list[str], filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_name in run_names:
        for row in read_csv(runs_root / run_name / filename):
            row["source_run"] = run_name
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def union_fields(rows: list[dict[str, object]], preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or []
    seen = set()
    fields: list[str] = []
    for field in preferred:
        if field not in seen:
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def to_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def fmt(value: object) -> str:
    number = to_float(value)
    if math.isfinite(number):
        return f"{number:.6g}"
    return str(value)


def unique(rows: Iterable[dict[str, str]], key: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        value = row.get(key, "")
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def gumbel_sample_ids(candidate_names: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for name in candidate_names:
        if not name.startswith("gumbel_sample_"):
            continue
        try:
            ids.add(int(name.removeprefix("gumbel_sample_")))
        except ValueError:
            continue
    return ids


def dataset_categories(names: Iterable[str]) -> set[str]:
    categories: set[str] = set()
    for name in names:
        lower = name.lower()
        if lower.startswith("parity"):
            categories.add("parity")
        if lower.startswith("majority"):
            categories.add("majority")
        if lower.startswith("random_sparse"):
            categories.add("random_sparse")
        if "digits" in lower:
            categories.add("digits")
        if "mnist" in lower:
            categories.add("mnist")
        if "cifar" in lower:
            categories.add("cifar")
    return categories


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sort_seed_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):08d}")
    except ValueError:
        return (1, value)


def dataset_seed_evidence(rows: list[dict[str, str]], datasets: set[str]) -> str:
    seeds_by_dataset: dict[str, set[str]] = {dataset: set() for dataset in datasets}
    for row in rows:
        dataset = row.get("dataset", "")
        seed = row.get("seed", "")
        if dataset in seeds_by_dataset and seed:
            seeds_by_dataset[dataset].add(seed)
    parts = []
    for dataset in sorted(seeds_by_dataset):
        seeds = ",".join(sorted(seeds_by_dataset[dataset], key=sort_seed_key))
        parts.append(f"{dataset}:seeds={seeds or 'none'}")
    return "; ".join(parts)


def missing_dataset_seed_pairs(rows: list[dict[str, str]], datasets: set[str], seeds: set[str]) -> list[str]:
    observed = {(row.get("dataset", ""), row.get("seed", "")) for row in rows}
    return [
        f"{dataset}/seed{seed}"
        for dataset in sorted(datasets)
        for seed in sorted(seeds, key=sort_seed_key)
        if (dataset, seed) not in observed
    ]


def table_markdown(rows: list[dict[str, object]], fields: list[str], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    if limit is not None and len(rows) > limit:
        omitted = ["..."] + [f"{len(rows) - limit} more rows omitted"] + [""] * max(0, len(fields) - 2)
        lines.append("| " + " | ".join(omitted) + " |")
    return "\n".join(lines)


def unified_rows(
    wiring_rows: list[dict[str, str]],
    seq_rows: list[dict[str, str]],
    red_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in wiring_rows:
        rows.append(
            {
                "axis": "A_wiring",
                "source_run": row.get("source_run", ""),
                "method": row.get("method", ""),
                "item": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "hard_acc": row.get("hard_acc", ""),
                "soft_acc": row.get("soft_acc", ""),
                "full_gap": row.get("full_gap", ""),
                "native_gap": row.get("native_gap", ""),
                "train_time": row.get("train_time", ""),
                "unused_gate_ratio": row.get("unused_gate_ratio", ""),
                "gate_count": row.get("gate_count", ""),
                "depth": row.get("depth", ""),
                "fanout_max": row.get("fanout_max", ""),
                "extra": f"wiring_seed={row.get('wiring_seed', '')}",
            }
        )
    for row in seq_rows:
        rows.append(
            {
                "axis": "B_register",
                "source_run": row.get("source_run", ""),
                "method": row.get("method", ""),
                "item": row.get("task", ""),
                "seed": row.get("seed", ""),
                "hard_acc": row.get("hard_acc", ""),
                "soft_acc": row.get("soft_acc", ""),
                "full_gap": row.get("full_gap", ""),
                "native_gap": row.get("native_gap", ""),
                "train_time": row.get("train_time", ""),
                "unused_gate_ratio": "",
                "gate_count": row.get("gate_count", ""),
                "depth": row.get("depth", ""),
                "fanout_max": "",
                "extra": f"sequence_acc={row.get('sequence_acc', '')}; state_bits={row.get('state_bits', '')}; reg_util={row.get('register_utilization', '')}",
            }
        )
    for row in red_rows:
        rows.append(
            {
                "axis": "C_redundancy",
                "source_run": row.get("source_run", ""),
                "method": row.get("method", ""),
                "item": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "hard_acc": row.get("hard_acc", ""),
                "soft_acc": row.get("soft_acc", ""),
                "full_gap": row.get("full_gap", ""),
                "native_gap": row.get("native_gap", ""),
                "train_time": row.get("train_time", ""),
                "unused_gate_ratio": row.get("unused_gate_ratio", ""),
                "gate_count": row.get("gate_count", ""),
                "depth": row.get("depth", ""),
                "fanout_max": row.get("fanout_max", ""),
                "extra": f"redundancy_factor={row.get('redundancy_factor', '')}; hardening={row.get('hardening', '')}; checkpoint={row.get('checkpoint_source', '')}",
            }
        )
    return sorted(rows, key=lambda row: (-to_float(row.get("hard_acc")), str(row.get("axis")), str(row.get("item")), str(row.get("method"))))


def best_by(rows: list[dict[str, str]], key_fields: tuple[str, ...], score_field: str = "hard_acc") -> dict[tuple[str, ...], dict[str, str]]:
    best: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        if key not in best or to_float(row.get(score_field)) > to_float(best[key].get(score_field)):
            best[key] = row
    return best


def redundancy_strong_baseline_rows(red_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Compare task-aware hardening against the strongest non-task-aware redundant row."""
    task_rows = [row for row in red_rows if row.get("method") == "redundant_task_hardened"]
    baseline_rows = [row for row in red_rows if row.get("method") in {"more_gates_only", "redundant_regularized"}]
    best_task = best_by(task_rows, ("dataset", "seed"))
    best_baseline = best_by(baseline_rows, ("dataset", "seed"))
    rows: list[dict[str, object]] = []
    for key in sorted(set(best_task) | set(best_baseline), key=lambda item: (item[0], sort_seed_key(item[1]))):
        task = best_task.get(key)
        baseline = best_baseline.get(key)
        if not task or not baseline:
            rows.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "status": "MISSING",
                }
            )
            continue
        delta_hard_acc = to_float(task.get("hard_acc")) - to_float(baseline.get("hard_acc"))
        delta_full_gap = to_float(task.get("full_gap")) - to_float(baseline.get("full_gap"))
        delta_native_gap = to_float(task.get("native_gap")) - to_float(baseline.get("native_gap"))
        delta_unused = to_float(task.get("unused_gate_ratio")) - to_float(baseline.get("unused_gate_ratio"))
        delta_gate_count = to_float(task.get("gate_count")) - to_float(baseline.get("gate_count"))
        rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "task_hardened_hard_acc": task.get("hard_acc", ""),
                "task_hardened_soft_acc": task.get("soft_acc", ""),
                "task_hardened_full_gap": task.get("full_gap", ""),
                "task_hardened_native_gap": task.get("native_gap", ""),
                "task_hardened_unused_gate_ratio": task.get("unused_gate_ratio", ""),
                "task_hardened_gate_count": task.get("gate_count", ""),
                "task_hardened_factor": task.get("redundancy_factor", ""),
                "task_hardened_hardening": task.get("hardening", ""),
                "strong_baseline_method": baseline.get("method", ""),
                "strong_baseline_hard_acc": baseline.get("hard_acc", ""),
                "strong_baseline_soft_acc": baseline.get("soft_acc", ""),
                "strong_baseline_full_gap": baseline.get("full_gap", ""),
                "strong_baseline_native_gap": baseline.get("native_gap", ""),
                "strong_baseline_unused_gate_ratio": baseline.get("unused_gate_ratio", ""),
                "strong_baseline_gate_count": baseline.get("gate_count", ""),
                "strong_baseline_factor": baseline.get("redundancy_factor", ""),
                "strong_baseline_hardening": baseline.get("hardening", ""),
                "delta_hard_acc": delta_hard_acc,
                "delta_full_gap": delta_full_gap,
                "delta_native_gap": delta_native_gap,
                "delta_unused_gate_ratio": delta_unused,
                "delta_gate_count": delta_gate_count,
                "status": "WIN" if delta_hard_acc > 0 else ("TIE" if delta_hard_acc == 0 else "LOSS"),
            }
        )
    return rows


def truth_refit_candidate_rows(candidate_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Validation-level comparison of truth-table refit against non-refit hardening candidates."""
    task_candidates = [
        row
        for row in candidate_rows
        if row.get("method") == "redundant_task_hardened" and truthy(row.get("eligible_for_method"))
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in task_candidates:
        key = (row.get("dataset", ""), row.get("seed", ""), row.get("redundancy_factor", ""))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (item[0], sort_seed_key(item[1]), to_float(item[2]))):
        group = grouped[key]
        truth = next((row for row in group if row.get("candidate") == "truth_table_refit"), None)
        non_refits = [row for row in group if row.get("candidate") != "truth_table_refit"]
        selected = next((row for row in group if truthy(row.get("selected"))), None)
        best_non_refit = max(non_refits, key=lambda row: to_float(row.get("val_hard_acc")), default=None)
        if not truth or not best_non_refit:
            rows.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "redundancy_factor": key[2],
                    "status": "MISSING",
                    "note": "truth_table_refit or non-refit candidate missing",
                }
            )
            continue
        delta = to_float(truth.get("val_hard_acc")) - to_float(best_non_refit.get("val_hard_acc"))
        rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "redundancy_factor": key[2],
                "truth_val_hard_acc": truth.get("val_hard_acc", ""),
                "truth_val_full_gap": truth.get("val_full_gap", ""),
                "truth_refit_error": truth.get("refit_error", ""),
                "truth_refit_delta": truth.get("refit_delta", ""),
                "truth_op_change_ratio": truth.get("op_change_ratio", ""),
                "truth_unused_gate_ratio": truth.get("unused_gate_ratio", ""),
                "truth_gate_count": truth.get("gate_count", ""),
                "best_non_refit_candidate": best_non_refit.get("candidate", ""),
                "best_non_refit_val_hard_acc": best_non_refit.get("val_hard_acc", ""),
                "best_non_refit_val_full_gap": best_non_refit.get("val_full_gap", ""),
                "best_non_refit_unused_gate_ratio": best_non_refit.get("unused_gate_ratio", ""),
                "selected_candidate": selected.get("candidate", "") if selected else "",
                "selected_val_hard_acc": selected.get("val_hard_acc", "") if selected else "",
                "selected_is_truth_refit": int(selected is not None and selected.get("candidate") == "truth_table_refit"),
                "delta_truth_vs_best_non_refit": delta,
                "status": "WIN" if delta > 0 else ("TIE" if delta == 0 else "LOSS"),
                "note": "validation candidate comparison only; final test accuracy is reported for selected candidates in redundancy_scaling_curve.csv",
            }
        )
    return rows


def truth_refit_final_rows(red_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Held-out test comparison for the forced truth-table-refit method."""
    truth = best_by([row for row in red_rows if row.get("method") == "redundant_truth_refit_only"], ("dataset", "seed"))
    task = best_by([row for row in red_rows if row.get("method") == "redundant_task_hardened"], ("dataset", "seed"))
    strong = best_by(
        [row for row in red_rows if row.get("method") in {"more_gates_only", "redundant_regularized"}],
        ("dataset", "seed"),
    )
    rows: list[dict[str, object]] = []
    for key in sorted(set(truth), key=lambda item: (item[0], sort_seed_key(item[1]))):
        truth_row = truth.get(key)
        task_row = task.get(key)
        strong_row = strong.get(key)
        delta_task = to_float(truth_row.get("hard_acc")) - to_float(task_row.get("hard_acc")) if task_row else math.nan
        delta_strong = to_float(truth_row.get("hard_acc")) - to_float(strong_row.get("hard_acc")) if strong_row else math.nan
        rows.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "truth_refit_hard_acc": truth_row.get("hard_acc", ""),
                "truth_refit_soft_acc": truth_row.get("soft_acc", ""),
                "truth_refit_full_gap": truth_row.get("full_gap", ""),
                "truth_refit_native_gap": truth_row.get("native_gap", ""),
                "truth_refit_unused_gate_ratio": truth_row.get("unused_gate_ratio", ""),
                "truth_refit_gate_count": truth_row.get("gate_count", ""),
                "truth_refit_factor": truth_row.get("redundancy_factor", ""),
                "task_selected_hard_acc": task_row.get("hard_acc", "") if task_row else "",
                "task_selected_hardening": task_row.get("hardening", "") if task_row else "",
                "task_selected_factor": task_row.get("redundancy_factor", "") if task_row else "",
                "strong_baseline_method": strong_row.get("method", "") if strong_row else "",
                "strong_baseline_hard_acc": strong_row.get("hard_acc", "") if strong_row else "",
                "strong_baseline_hardening": strong_row.get("hardening", "") if strong_row else "",
                "strong_baseline_factor": strong_row.get("redundancy_factor", "") if strong_row else "",
                "delta_vs_task_selected": delta_task,
                "delta_vs_strong_baseline": delta_strong,
                "status_vs_task_selected": "WIN" if delta_task > 0 else ("TIE" if delta_task == 0 else ("LOSS" if math.isfinite(delta_task) else "MISSING")),
                "status_vs_strong_baseline": "WIN" if delta_strong > 0 else ("TIE" if delta_strong == 0 else ("LOSS" if math.isfinite(delta_strong) else "MISSING")),
                "note": "held-out test comparison; truth_refit row is forced truth_table_refit, task_selected row uses validation-selected hardening",
            }
        )
    return rows


def mean_metric(row: dict[str, str], metric: str) -> str:
    """Return aggregate mean column when available, otherwise a raw metric column."""
    return row.get(f"{metric}_mean", row.get(metric, ""))


def time_speedup(base_time: object, other_time: object) -> float:
    base = to_float(base_time)
    other = to_float(other_time)
    return base / other if base > 0 and other > 0 else math.nan


def reduction_ratio(base_value: object, other_value: object, min_base: float = 0.0) -> float:
    base = to_float(base_value)
    other = to_float(other_value)
    if not math.isfinite(base) or not math.isfinite(other) or base <= min_base:
        return math.nan
    return (base - other) / base


def mind_gap_direct_row(
    *,
    comparison_scope: str,
    source_run: str,
    dataset: str,
    n: str,
    dlgn: dict[str, str],
    block: dict[str, str],
    gumbel: dict[str, str],
    aggregate: bool,
    selection_criterion: str = "",
) -> dict[str, object]:
    def value(row: dict[str, str], metric: str) -> str:
        return mean_metric(row, metric) if aggregate else row.get(metric, "")

    dlgn_gap = to_float(value(dlgn, "acc_gap"))
    dlgn_hard_acc = to_float(value(dlgn, "discrete_acc"))
    block_gap = to_float(value(block, "acc_gap"))
    block_native_gap = to_float(value(block, "path_acc_gap"))
    block_hard_acc = to_float(value(block, "discrete_acc"))
    gumbel_gap = to_float(value(gumbel, "acc_gap"))
    gumbel_hard_acc = to_float(value(gumbel, "discrete_acc"))
    dlgn_unused = to_float(value(dlgn, "unused_gate_ratio"))
    block_unused = to_float(value(block, "unused_gate_ratio"))
    gumbel_unused = to_float(value(gumbel, "unused_gate_ratio"))
    gumbel_train_speedup = time_speedup(value(dlgn, "train_time"), value(gumbel, "train_time"))
    gumbel_time_to_target_speedup = time_speedup(value(dlgn, "time_to_target"), value(gumbel, "time_to_target"))
    gumbel_gap_reduction = reduction_ratio(dlgn_gap, gumbel_gap, min_base=1e-2)
    gumbel_unused_reduction = reduction_ratio(dlgn_unused, gumbel_unused)
    return {
        "comparison_scope": comparison_scope,
        "source_run": source_run,
        "gumbel_source_run": gumbel.get("source_run", source_run),
        "dataset": dataset,
        "n": n,
        "proposed_method": block.get("method", "block_hard_refit"),
        "gumbel_method": gumbel.get("method", "gumbel_st"),
        "dlgn_discrete_acc": value(dlgn, "discrete_acc"),
        "dlgn_soft_acc": value(dlgn, "soft_acc"),
        "dlgn_full_gap": dlgn_gap,
        "dlgn_unused_gate_ratio": dlgn_unused,
        "dlgn_train_time": value(dlgn, "train_time"),
        "dlgn_time_to_target": value(dlgn, "time_to_target"),
        "block_hard_discrete_acc": value(block, "discrete_acc"),
        "block_hard_soft_acc": value(block, "soft_acc"),
        "block_hard_full_gap": block_gap,
        "block_hard_native_gap": block_native_gap,
        "block_hard_unused_gate_ratio": block_unused,
        "block_hard_train_time": value(block, "train_time"),
        "block_hard_time_to_target": value(block, "time_to_target"),
        "gumbel_discrete_acc": value(gumbel, "discrete_acc"),
        "gumbel_soft_acc": value(gumbel, "soft_acc"),
        "gumbel_full_gap": gumbel_gap,
        "gumbel_unused_gate_ratio": gumbel_unused,
        "gumbel_train_time": value(gumbel, "train_time"),
        "gumbel_time_to_target": value(gumbel, "time_to_target"),
        "delta_hard_acc_vs_dlgn": block_hard_acc - dlgn_hard_acc,
        "delta_hard_acc_vs_gumbel": block_hard_acc - gumbel_hard_acc,
        "delta_full_gap_vs_dlgn": block_gap - dlgn_gap,
        "delta_native_gap_vs_dlgn": block_native_gap - dlgn_gap,
        "delta_full_gap_vs_gumbel": block_gap - gumbel_gap,
        "delta_native_gap_vs_gumbel": block_native_gap - gumbel_gap,
        "hard_acc_beats_dlgn": bool_text(math.isfinite(block_hard_acc) and math.isfinite(dlgn_hard_acc) and block_hard_acc > dlgn_hard_acc),
        "hard_acc_beats_gumbel": bool_text(math.isfinite(block_hard_acc) and math.isfinite(gumbel_hard_acc) and block_hard_acc > gumbel_hard_acc),
        "full_gap_beats_dlgn": bool_text(math.isfinite(block_gap) and math.isfinite(dlgn_gap) and block_gap < dlgn_gap),
        "native_gap_beats_dlgn": bool_text(math.isfinite(block_native_gap) and math.isfinite(dlgn_gap) and block_native_gap < dlgn_gap),
        "full_gap_competitive_with_gumbel": bool_text(math.isfinite(block_gap) and math.isfinite(gumbel_gap) and block_gap <= gumbel_gap),
        "native_gap_competitive_with_gumbel": bool_text(
            math.isfinite(block_native_gap) and math.isfinite(gumbel_gap) and block_native_gap <= gumbel_gap
        ),
        "block_unused_reduction_vs_dlgn": reduction_ratio(dlgn_unused, block_unused),
        "gumbel_gap_reduction_vs_dlgn": gumbel_gap_reduction,
        "gumbel_unused_reduction_vs_dlgn": gumbel_unused_reduction,
        "gumbel_train_speedup_vs_dlgn": gumbel_train_speedup,
        "gumbel_time_to_target_speedup_vs_dlgn": gumbel_time_to_target_speedup,
        "mind_gap_98pct_gap_target_met": bool_text(math.isfinite(gumbel_gap_reduction) and gumbel_gap_reduction >= 0.98),
        "mind_gap_100pct_unused_target_met": bool_text(math.isfinite(gumbel_unused_reduction) and gumbel_unused_reduction >= 1.0),
        "mind_gap_4p5x_train_time_target_met": bool_text(math.isfinite(gumbel_train_speedup) and gumbel_train_speedup >= 4.5),
        "mind_gap_4p5x_time_to_target_target_met": bool_text(
            math.isfinite(gumbel_time_to_target_speedup) and gumbel_time_to_target_speedup >= 4.5
        ),
        "selection_criterion": selection_criterion,
        "note": (
            "traditional full gap is block_hard_full_gap; method-native hard-prefix gap is block_hard_native_gap"
        ),
    }


def mind_gap_direct_rows(
    runs_root: Path,
    timing_runs: list[str],
    tuned_gumbel_comparisons: list[str],
) -> list[dict[str, object]]:
    """Build direct DLGN/Gumbel/block-hard rows without mixing full and native gap claims."""
    rows: list[dict[str, object]] = []
    for run_name in timing_runs:
        aggregate_path = runs_root / run_name / "aggregate" / "aggregate_by_method_dataset.csv"
        aggregate_rows = read_csv(aggregate_path)
        by_key = {(row.get("dataset", ""), row.get("method", "")): row for row in aggregate_rows}
        for dataset in sorted({row.get("dataset", "") for row in aggregate_rows}):
            dlgn = by_key.get((dataset, "dlgn"))
            block = by_key.get((dataset, "block_hard_refit"))
            gumbel = by_key.get((dataset, "gumbel_st"))
            if not dlgn or not block or not gumbel:
                continue
            rows.append(
                mind_gap_direct_row(
                    comparison_scope="multi_seed_fixed_gumbel",
                    source_run=run_name,
                    dataset=dataset,
                    n=block.get("n", ""),
                    dlgn=dlgn,
                    block=block,
                    gumbel=gumbel,
                    aggregate=True,
                    selection_criterion="fixed_gumbel_st_from_timing_run",
                )
            )

    seen_tuned: set[tuple[str, str, str, str]] = set()
    for tuned_comparison in tuned_gumbel_comparisons:
        tuned_path = Path(tuned_comparison)
        if not tuned_path.is_absolute():
            tuned_path = runs_root / tuned_path
        tuned_rows = read_csv(tuned_path)
        by_dataset_method = {(row.get("dataset", ""), row.get("method", "")): row for row in tuned_rows}
        for dataset in sorted({row.get("dataset", "") for row in tuned_rows}):
            dlgn = by_dataset_method.get((dataset, "dlgn"))
            block = by_dataset_method.get((dataset, "block_hard_refit"))
            gumbel = by_dataset_method.get((dataset, "gumbel_st_best_discrete_acc"))
            if not dlgn or not block or not gumbel:
                continue
            tuned_key = (
                dataset,
                block.get("seed", ""),
                block.get("source_run", ""),
                gumbel.get("source_run", ""),
            )
            if tuned_key in seen_tuned:
                continue
            seen_tuned.add(tuned_key)
            rows.append(
                mind_gap_direct_row(
                    comparison_scope="seed0_tuned_gumbel_best_discrete_acc",
                    source_run=block.get("source_run", ""),
                    dataset=dataset,
                    n="1",
                    dlgn=dlgn,
                    block=block,
                    gumbel=gumbel,
                    aggregate=False,
                    selection_criterion=gumbel.get("selection_criterion", ""),
                )
            )
    return rows


def summarize_status(rows: list[dict[str, object]], field: str) -> str:
    values = [str(row.get(field, "")) for row in rows]
    yes = sum(value == "yes" for value in values)
    no = sum(value == "no" for value in values)
    missing = len(values) - yes - no
    return f"yes={yes}; no={no}; missing={missing}"


def mind_gap_success_checks(mind_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not mind_rows:
        return [
            {
                "criterion": "Mind-the-Gap direct DLGN/Gumbel/block-hard comparison is available",
                "status": "MISSING",
                "evidence": "no timing-v3 direct comparison rows",
                "note": "Run or copy timing-v3 aggregate and tuned Gumbel comparison artifacts.",
            }
        ]
    full_yes = [row for row in mind_rows if row.get("full_gap_beats_dlgn") == "yes"]
    native_yes = [row for row in mind_rows if row.get("native_gap_beats_dlgn") == "yes"]
    full_gumbel_yes = [row for row in mind_rows if row.get("full_gap_competitive_with_gumbel") == "yes"]
    native_gumbel_yes = [row for row in mind_rows if row.get("native_gap_competitive_with_gumbel") == "yes"]
    hard_gumbel_yes = [row for row in mind_rows if row.get("hard_acc_beats_gumbel") == "yes"]
    claim_targets = [
        row
        for row in mind_rows
        if row.get("mind_gap_98pct_gap_target_met") == "yes"
        and row.get("mind_gap_100pct_unused_target_met") == "yes"
        and (
            row.get("mind_gap_4p5x_train_time_target_met") == "yes"
            or row.get("mind_gap_4p5x_time_to_target_target_met") == "yes"
        )
    ]
    return [
        {
            "criterion": "Mind-the-Gap direct comparison rows are present",
            "status": "PASS",
            "evidence": f"rows={len(mind_rows)}; scopes={','.join(sorted({str(row.get('comparison_scope', '')) for row in mind_rows}))}",
            "note": "Rows include traditional full gap and method-native hard-prefix gap as separate columns.",
        },
        {
            "criterion": "block-hard primary full gap beats DLGN in direct Mind-the-Gap comparison",
            "status": "PASS" if len(full_yes) == len(mind_rows) else ("PARTIAL" if full_yes else "FAIL"),
            "evidence": summarize_status(mind_rows, "full_gap_beats_dlgn"),
            "note": "This is the traditional relaxed-full vs hard-full metric and is the only primary Mind-the-Gap gap claim.",
        },
        {
            "criterion": "block-hard method-native gap beats DLGN in direct Mind-the-Gap comparison",
            "status": "PASS" if len(native_yes) == len(mind_rows) else ("PARTIAL" if native_yes else "FAIL"),
            "evidence": summarize_status(mind_rows, "native_gap_beats_dlgn"),
            "note": "This is the hard-prefix-trained path diagnostic; it supports method-native alignment but does not replace the traditional full-gap metric.",
        },
        {
            "criterion": "block-hard primary full gap is competitive with Gumbel in direct comparison",
            "status": "PASS" if len(full_gumbel_yes) == len(mind_rows) else ("PARTIAL" if full_gumbel_yes else "FAIL"),
            "evidence": summarize_status(mind_rows, "full_gap_competitive_with_gumbel"),
            "note": "Uses the fixed Gumbel row for multi-seed timing runs and the sweep-selected Gumbel row for seed-0 tuned checks.",
        },
        {
            "criterion": "block-hard method-native gap is competitive with Gumbel in direct comparison",
            "status": "PASS" if len(native_gumbel_yes) == len(mind_rows) else ("PARTIAL" if native_gumbel_yes else "FAIL"),
            "evidence": summarize_status(mind_rows, "native_gap_competitive_with_gumbel"),
            "note": "Diagnostic only; useful for the path-gap/full-gap split identified by the user.",
        },
        {
            "criterion": "block-hard hard accuracy beats tuned/fixed Gumbel in direct comparison",
            "status": "PASS" if len(hard_gumbel_yes) == len(mind_rows) else ("PARTIAL" if hard_gumbel_yes else "FAIL"),
            "evidence": summarize_status(mind_rows, "hard_acc_beats_gumbel"),
            "note": "Accuracy is reported separately from gap because small Gumbel gaps can coincide with weak hard accuracy in prototype runs.",
        },
        {
            "criterion": "local Gumbel baseline reproduces Mind-the-Gap headline targets",
            "status": "PASS" if len(claim_targets) == len(mind_rows) else ("PARTIAL" if claim_targets else "FAIL"),
            "evidence": f"rows meeting 98pct-gap + 100pct-unused + 4.5x-speed target={len(claim_targets)}/{len(mind_rows)}",
            "note": "Prototype-scale local runs should not be claimed as a full paper reproduction when this fails.",
        },
    ]


def mind_gap_coverage_rows(mind_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    datasets = {str(row.get("dataset", "")) for row in mind_rows if row.get("dataset")}
    scopes = {str(row.get("comparison_scope", "")) for row in mind_rows if row.get("comparison_scope")}
    required_boolean = {"parity8", "majority9", "random_sparse10"}
    required_image_smoke = {"digits", "binarized_mnist", "cifar10_small"}
    image_smoke = required_image_smoke & datasets
    return [
        {
            "area": "Mind_the_Gap_direct",
            "requirement": "direct DLGN vs Gumbel-ST vs block-hard-refit comparison reports both traditional full gap and method-native path gap",
            "status": "PASS" if mind_rows else "MISSING",
            "evidence": f"rows={len(mind_rows)}; scopes={','.join(sorted(scopes))}",
            "missing": "" if mind_rows else "timing_v3 aggregate/tuned comparison rows",
            "note": "This table is for Mind-the-Gap-style comparison and keeps path diagnostics separate from primary full-gap claims.",
        },
        {
            "area": "Mind_the_Gap_direct",
            "requirement": "Boolean direct comparison covers parity8, majority9, and random_sparse10",
            "status": "PASS" if required_boolean <= datasets else ("PARTIAL" if datasets else "MISSING"),
            "evidence": ",".join(sorted(required_boolean & datasets)),
            "missing": ",".join(sorted(required_boolean - datasets)),
            "note": "The tuned Gumbel comparison is seed-0 only; multi-seed timing rows use the fixed Gumbel baseline.",
        },
        {
            "area": "Mind_the_Gap_direct",
            "requirement": "image direct comparison includes sklearn digits, binarized MNIST smoke, and CIFAR-10 smoke when available",
            "status": "PASS" if required_image_smoke <= datasets else ("PARTIAL" if image_smoke else "MISSING"),
            "evidence": ",".join(sorted(image_smoke)),
            "missing": ",".join(sorted(required_image_smoke - datasets)),
            "note": "Digits, MNIST, and CIFAR rows are prototype/smoke scale; they are not full MNIST/CIFAR-10 paper-scale validation.",
        },
    ]


def synthesis_report_dirs(runs_root: Path, requested_dirs: list[str] | None) -> list[Path]:
    if requested_dirs:
        candidates: list[Path] = []
        for item in requested_dirs:
            path = Path(item)
            candidates.append(path if path.is_absolute() else Path(item))
            if not path.is_absolute():
                candidates.append(runs_root / path)
        return candidates
    return [
        runs_root / "reports_synthesis_timing_v3",
        Path("reports/synthesis_timing_v3"),
        Path("reports_synthesis_timing_v3"),
    ]


def post_abc_eval_dirs(runs_root: Path, requested_dirs: list[str] | None) -> list[Path]:
    if requested_dirs:
        candidates: list[Path] = []
        for item in requested_dirs:
            path = Path(item)
            candidates.append(path if path.is_absolute() else path)
            if not path.is_absolute():
                candidates.append(runs_root / path)
        return candidates
    return [
        runs_root / "reports_post_abc_eval_v3",
        Path("reports/post_abc_eval_v3"),
        Path("reports_post_abc_eval_v3"),
    ]


def read_post_abc_eval_rows(runs_root: Path, requested_dirs: list[str] | None) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    seen_paths: set[Path] = set()
    for report_dir in post_abc_eval_dirs(runs_root, requested_dirs):
        eval_path = report_dir / "post_abc_eval.csv"
        if not eval_path.exists():
            continue
        resolved = eval_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        for row in read_csv(eval_path):
            key = (row.get("dataset", ""), row.get("method", ""), row.get("seed", ""))
            row["post_abc_eval_report"] = str(report_dir)
            rows[key] = row
    return rows


def read_synthesis_rows(
    runs_root: Path,
    requested_dirs: list[str] | None,
    post_abc_eval: dict[tuple[str, str, str], dict[str, str]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    post_abc_eval = post_abc_eval or {}
    seen_paths: set[Path] = set()
    for report_dir in synthesis_report_dirs(runs_root, requested_dirs):
        joined_path = report_dir / "synthesis_joined.csv"
        if not joined_path.exists():
            continue
        resolved = joined_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        for row in read_csv(joined_path):
            pre_nd = to_float(row.get("abc_pre_nd"))
            post_and = to_float(row.get("abc_post_and"))
            pre_lev = to_float(row.get("abc_pre_lev"))
            post_lev = to_float(row.get("abc_post_lev"))
            key = (row.get("dataset", ""), row.get("method", ""), row.get("seed", ""))
            eval_row = post_abc_eval.get(key, {})
            eval_ok = eval_row.get("abc_eval_status") == "ok"
            accuracy_recomputed = "yes" if eval_ok else row.get("accuracy_recomputed_after_abc", "")
            source_fanout = to_float(eval_row.get("source_blif_fanout_max"))
            post_fanout = to_float(eval_row.get("post_abc_fanout_max"))
            source_unused = to_float(eval_row.get("source_blif_unused_node_ratio"))
            post_unused = to_float(eval_row.get("post_abc_unused_node_ratio"))
            fanout_change = post_fanout - source_fanout if math.isfinite(post_fanout) and math.isfinite(source_fanout) else "not_measured"
            unused_change = post_unused - source_unused if math.isfinite(post_unused) and math.isfinite(source_unused) else "not_measured"
            rows.append(
                {
                    "source_report": str(report_dir),
                    "run": row.get("run", ""),
                    "dataset": row.get("dataset", ""),
                    "method": row.get("method", ""),
                    "seed": row.get("seed", ""),
                    "abc_status": row.get("abc_status", ""),
                    "soft_acc": row.get("soft_acc", ""),
                    "discrete_acc": row.get("discrete_acc", ""),
                    "acc_gap": row.get("acc_gap", ""),
                    "unused_gate_ratio": row.get("unused_gate_ratio", ""),
                    "pre_gate_count": row.get("pre_gate_count", ""),
                    "pre_depth": row.get("pre_depth", ""),
                    "pre_fanout_max": row.get("pre_fanout_max", ""),
                    "abc_pre_nd": row.get("abc_pre_nd", ""),
                    "abc_pre_edge": row.get("abc_pre_edge", ""),
                    "abc_pre_cube": row.get("abc_pre_cube", ""),
                    "abc_pre_lev": row.get("abc_pre_lev", ""),
                    "abc_post_and": row.get("abc_post_and", ""),
                    "abc_post_lev": row.get("abc_post_lev", ""),
                    "abc_and_reduction_vs_pre_nd": row.get("abc_and_reduction_vs_pre_nd", ""),
                    "abc_level_delta_vs_pre_lev": row.get("abc_level_delta_vs_pre_lev", ""),
                    "abc_level_ratio_vs_pre_lev": row.get("abc_level_ratio_vs_pre_lev", ""),
                    "accuracy_recomputed_after_abc": accuracy_recomputed,
                    "source_blif_acc": eval_row.get("source_blif_acc", ""),
                    "post_abc_discrete_acc": eval_row.get("post_abc_discrete_acc", ""),
                    "source_acc_delta_vs_results": eval_row.get("source_acc_delta_vs_results", ""),
                    "post_acc_delta_vs_pre_discrete": eval_row.get("post_acc_delta_vs_pre_discrete", ""),
                    "post_abc_gap_vs_pre_soft": eval_row.get("post_abc_gap_vs_pre_soft", ""),
                    "post_gap_delta_vs_pre_gap": eval_row.get("post_gap_delta_vs_pre_gap", ""),
                    "source_blif_loss": eval_row.get("source_blif_loss", ""),
                    "post_abc_discrete_loss": eval_row.get("post_abc_discrete_loss", ""),
                    "source_loss_delta_vs_results": eval_row.get("source_loss_delta_vs_results", ""),
                    "post_loss_delta_vs_pre_discrete": eval_row.get("post_loss_delta_vs_pre_discrete", ""),
                    "source_blif_fanout_max": eval_row.get("source_blif_fanout_max", ""),
                    "post_abc_fanout_max": eval_row.get("post_abc_fanout_max", ""),
                    "source_blif_unused_node_ratio": eval_row.get("source_blif_unused_node_ratio", ""),
                    "post_abc_unused_node_ratio": eval_row.get("post_abc_unused_node_ratio", ""),
                    "source_blif_unused_node_count": eval_row.get("source_blif_unused_node_count", ""),
                    "post_abc_unused_node_count": eval_row.get("post_abc_unused_node_count", ""),
                    "accuracy_change_after_abc": eval_row.get("post_acc_delta_vs_pre_discrete", "") if eval_ok else "not_measured",
                    "gap_change_after_abc": eval_row.get("post_gap_delta_vs_pre_gap", "") if eval_ok else "not_measured",
                    "gate_count_change_after_abc": post_and - pre_nd if math.isfinite(post_and) and math.isfinite(pre_nd) else math.nan,
                    "depth_change_after_abc": post_lev - pre_lev if math.isfinite(post_lev) and math.isfinite(pre_lev) else math.nan,
                    "fanout_change_after_abc": fanout_change if eval_ok else "not_measured",
                    "unused_gate_ratio_change_after_abc": unused_change if eval_ok else "not_measured",
                    "post_abc_eval_status": eval_row.get("abc_eval_status", "missing"),
                    "post_abc_eval_report": eval_row.get("post_abc_eval_report", ""),
                    "note": (
                        "ABC optimized BLIF was re-evaluated; post_abc_gap_vs_pre_soft uses the original relaxed soft accuracy and optimized hard BLIF accuracy. Fanout/unused changes use BLIF graph metrics."
                        if eval_ok
                        else "ABC structural stats only; optimized BLIF is not imported back for post-synthesis accuracy/gap/fanout/unused evaluation."
                    ),
                }
            )
    return sorted(rows, key=lambda row: (str(row.get("dataset", "")), str(row.get("method", "")), sort_seed_key(str(row.get("seed", "")))))


def synthesis_success_checks(synthesis_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not synthesis_rows:
        return [
            {
                "criterion": "optional ABC synthesis structural baseline is available",
                "status": "MISSING",
                "evidence": "no synthesis_joined.csv rows found",
                "note": "Existing optional synthesis evidence must be generated or pointed to with --synthesis-report-dirs.",
            }
        ]
    ok_rows = [row for row in synthesis_rows if row.get("abc_status") == "ok"]
    post_accuracy_rows = [row for row in synthesis_rows if row.get("accuracy_recomputed_after_abc") == "yes"]
    reductions = [to_float(row.get("abc_and_reduction_vs_pre_nd")) for row in ok_rows]
    finite_reductions = [value for value in reductions if math.isfinite(value)]
    source_acc_deltas = [abs(to_float(row.get("source_acc_delta_vs_results"))) for row in post_accuracy_rows]
    post_acc_deltas = [abs(to_float(row.get("post_acc_delta_vs_pre_discrete"))) for row in post_accuracy_rows]
    source_loss_deltas = [abs(to_float(row.get("source_loss_delta_vs_results"))) for row in post_accuracy_rows]
    post_loss_deltas = [abs(to_float(row.get("post_loss_delta_vs_pre_discrete"))) for row in post_accuracy_rows]
    post_structure_rows = [
        row
        for row in post_accuracy_rows
        if math.isfinite(to_float(row.get("post_abc_fanout_max"))) and math.isfinite(to_float(row.get("post_abc_unused_node_ratio")))
    ]
    fanout_deltas = [to_float(row.get("fanout_change_after_abc")) for row in post_structure_rows]
    unused_deltas = [to_float(row.get("unused_gate_ratio_change_after_abc")) for row in post_structure_rows]
    finite_source_acc = [value for value in source_acc_deltas if math.isfinite(value)]
    finite_post_acc = [value for value in post_acc_deltas if math.isfinite(value)]
    finite_source_loss = [value for value in source_loss_deltas if math.isfinite(value)]
    finite_post_loss = [value for value in post_loss_deltas if math.isfinite(value)]
    finite_fanout_deltas = [value for value in fanout_deltas if math.isfinite(value)]
    finite_unused_deltas = [value for value in unused_deltas if math.isfinite(value)]
    return [
        {
            "criterion": "optional ABC synthesis structural baseline is available",
            "status": "PASS" if ok_rows else "FAIL",
            "evidence": f"rows={len(synthesis_rows)}; abc_ok={len(ok_rows)}; datasets={','.join(sorted({str(row.get('dataset', '')) for row in synthesis_rows}))}",
            "note": "ABC rows are structural baseline evidence, not post-synthesis accuracy evidence.",
        },
        {
            "criterion": "ABC synthesis reduces structural AND count",
            "status": "PASS" if finite_reductions and min(finite_reductions) > 0 else ("PARTIAL" if finite_reductions else "MISSING"),
            "evidence": f"min_reduction={min(finite_reductions):.6g}; max_reduction={max(finite_reductions):.6g}" if finite_reductions else "no finite ABC reduction ratios",
            "note": "`abc_post_and` is an ABC AIG structural count and is not a replacement PyTorch gate count.",
        },
        {
            "criterion": "post-ABC accuracy and gap are re-evaluated",
            "status": "PASS" if post_accuracy_rows else "MISSING",
            "evidence": f"post_accuracy_rows={len(post_accuracy_rows)}/{len(synthesis_rows)}",
            "note": "Optimized BLIF rows are evaluated through the standalone BLIF evaluator; post-ABC structure is checked separately." if post_accuracy_rows else "Current ABC evidence does not import optimized BLIF back into PyTorch, so accuracy/gap/fanout/unused effects remain unmeasured.",
        },
        {
            "criterion": "post-ABC fanout and unused-node ratio are re-evaluated",
            "status": "PASS" if len(post_structure_rows) == len(synthesis_rows) else ("PARTIAL" if post_structure_rows else "MISSING"),
            "evidence": (
                f"post_structure_rows={len(post_structure_rows)}/{len(synthesis_rows)}; "
                f"fanout_delta_range=[{min(finite_fanout_deltas):.6g},{max(finite_fanout_deltas):.6g}]; "
                f"unused_ratio_delta_range=[{min(finite_unused_deltas):.6g},{max(finite_unused_deltas):.6g}]"
                if finite_fanout_deltas and finite_unused_deltas
                else f"post_structure_rows={len(post_structure_rows)}/{len(synthesis_rows)}"
            ),
            "note": "Fanout and unused-node ratio are computed from source/optimized BLIF graph reachability; this is a structural metric, not the PyTorch inactive-neuron ratio.",
        },
        {
            "criterion": "ABC optimized BLIF preserves hard accuracy and loss",
            "status": (
                "PASS"
                if finite_source_acc
                and finite_post_acc
                and finite_source_loss
                and finite_post_loss
                and max(finite_source_acc) <= 1e-9
                and max(finite_post_acc) <= 1e-9
                and max(finite_source_loss) <= 1e-5
                and max(finite_post_loss) <= 1e-5
                else ("FAIL" if post_accuracy_rows else "MISSING")
            ),
            "evidence": (
                f"max_source_acc_delta={max(finite_source_acc):.6g}; "
                f"max_post_acc_delta={max(finite_post_acc):.6g}; "
                f"max_source_loss_delta={max(finite_source_loss):.6g}; "
                f"max_post_loss_delta={max(finite_post_loss):.6g}"
                if finite_source_acc and finite_post_acc and finite_source_loss and finite_post_loss
                else "no finite post-ABC equivalence deltas"
            ),
            "note": "Source BLIF first validates the evaluator against results.csv; optimized BLIF then checks ABC preservation.",
        },
    ]


def synthesis_coverage_rows(synthesis_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    datasets = {str(row.get("dataset", "")) for row in synthesis_rows if row.get("dataset")}
    methods = {str(row.get("method", "")) for row in synthesis_rows if row.get("method")}
    required_methods = {"dlgn", "dlgn_anneal", "gumbel_st", "block_relaxed", "block_hard_refit"}
    has_post_eval = any(row.get("accuracy_recomputed_after_abc") == "yes" for row in synthesis_rows)
    has_post_structure = any(
        math.isfinite(to_float(row.get("post_abc_fanout_max"))) and math.isfinite(to_float(row.get("post_abc_unused_node_ratio")))
        for row in synthesis_rows
    )
    measured_parts = ["structural_and_count", "structural_level"]
    missing_parts: list[str] = []
    if has_post_eval:
        measured_parts.extend(["post_abc_accuracy", "post_abc_gap"])
    else:
        missing_parts.extend(["post_abc_accuracy", "post_abc_gap"])
    if has_post_structure:
        measured_parts.extend(["post_abc_fanout", "post_abc_unused_node_ratio"])
    else:
        missing_parts.extend(["post_abc_fanout", "post_abc_unused_node_ratio"])
    complete_effect_coverage = bool(synthesis_rows and has_post_eval and has_post_structure)
    return [
        {
            "area": "D_synthesis",
            "requirement": "optional ABC/Yosys/Espresso-style synthesis baseline is integrated into the evidence audit",
            "status": "PASS" if synthesis_rows else "MISSING",
            "evidence": f"rows={len(synthesis_rows)}; datasets={','.join(sorted(datasets))}",
            "missing": "" if synthesis_rows else "synthesis_joined.csv",
            "note": "Current integrated optional baseline is ABC structural synthesis on the timing-v3 Boolean seed-0 run.",
        },
        {
            "area": "D_synthesis",
            "requirement": "synthesis rows cover DLGN, annealing, Gumbel-ST, block-relaxed, and block-hard-refit",
            "status": "PASS" if synthesis_rows and required_methods <= methods else ("PARTIAL" if methods else "MISSING"),
            "evidence": ",".join(sorted(methods)),
            "missing": ",".join(sorted(required_methods - methods)),
            "note": "This is method coverage for the existing Boolean seed-0 ABC report only.",
        },
        {
            "area": "D_synthesis",
            "requirement": "measure whether synthesis changes accuracy, gap, gate count, depth, fanout, and unused gate ratio",
            "status": "PASS" if complete_effect_coverage else ("PARTIAL" if synthesis_rows else "MISSING"),
            "evidence": ",".join(measured_parts) if synthesis_rows else "",
            "missing": ",".join(missing_parts),
            "note": (
                "Accuracy/gap/fanout/unused are post-ABC evaluated through optimized BLIF; fanout/unused use BLIF graph reachability metrics."
                if complete_effect_coverage
                else "Accuracy/gap are post-ABC evaluated, but fanout/unused are still not measured after synthesis."
                if has_post_eval
                else "Accuracy/gap/fanout/unused are not post-ABC re-evaluated; pre-ABC values are copied from results.csv."
            ),
        },
    ]


def success_checks(
    wiring_rows: list[dict[str, str]],
    seq_rows: list[dict[str, str]],
    red_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    matched = best_by([row for row in wiring_rows if row.get("method") == "fixed_random_matched_budget"], ("dataset", "seed"))
    search = best_by([row for row in wiring_rows if row.get("method") == "random_wiring_search_freeze"], ("dataset", "seed"))
    wins = []
    losses = []
    for key, search_row in search.items():
        base = matched.get(key)
        if not base:
            continue
        delta = to_float(search_row.get("hard_acc")) - to_float(base.get("hard_acc"))
        item = f"{key[0]}/seed{key[1]} delta={delta:.6g}"
        (wins if delta > 0 else losses).append(item)
    if wins and not losses:
        wiring_note = "Covered datasets favor random wiring search over the matched-budget fixed-random baseline; broader datasets/seeds are still needed."
    elif wins and losses:
        wiring_note = "Wiring search evidence is mixed; do not claim broad wiring-search success until more datasets/seeds pass."
    else:
        wiring_note = "Covered datasets do not show a hard-accuracy win over the matched-budget fixed-random baseline."
    checks.append(
        {
            "criterion": "random wiring search + freeze beats matched-budget fixed random",
            "status": "PARTIAL" if wins and losses else ("PASS" if wins else "FAIL"),
            "evidence": "; ".join(wins + losses),
            "note": wiring_note,
        }
    )

    candidate_wiring_rows = [row for row in wiring_rows if row.get("method") == "candidate_set_learnable_freeze"]
    matched_by_run = best_by(
        [row for row in wiring_rows if row.get("method") == "fixed_random_matched_budget"],
        ("source_run", "dataset", "seed"),
    )
    candidate_wins: list[str] = []
    candidate_losses: list[str] = []
    candidate_missing: list[str] = []
    for row in candidate_wiring_rows:
        key = (row.get("source_run", ""), row.get("dataset", ""), row.get("seed", ""))
        base = matched_by_run.get(key)
        if not base:
            candidate_missing.append(f"{key[0]}/{key[1]}/seed{key[2]}")
            continue
        delta = to_float(row.get("hard_acc")) - to_float(base.get("hard_acc"))
        item = f"{key[0]}/{key[1]}/seed{key[2]} delta={delta:.6g}"
        (candidate_wins if delta > 0 else candidate_losses).append(item)
    if candidate_wins and not candidate_losses and not candidate_missing:
        candidate_status = "PASS"
        candidate_note = "Candidate-set learnable connectivity beats the matched-budget fixed-random baseline on every covered same-run comparison."
    elif candidate_wiring_rows and candidate_wins:
        candidate_status = "PARTIAL"
        candidate_note = "Candidate-set learnable connectivity has mixed same-run hard-accuracy evidence; inspect wiring_search_comparison.csv before claiming a connectivity-learning win."
    elif candidate_wiring_rows:
        candidate_status = "FAIL"
        candidate_note = "Candidate-set learnable connectivity is implemented, but current same-run hard accuracy does not beat the matched-budget fixed-random baseline."
    else:
        candidate_status = "MISSING"
        candidate_note = "No candidate-set learnable connectivity rows are present in the merged wiring evidence."
    checks.append(
        {
            "criterion": "candidate-set learnable connectivity beats matched-budget fixed random",
            "status": candidate_status,
            "evidence": "; ".join(candidate_wins + candidate_losses + candidate_missing),
            "note": candidate_note,
        }
    )

    seq_by_task_seed_method = {
        (row.get("task", ""), row.get("seed", ""), row.get("method", "")): row
        for row in seq_rows
    }
    seq_solved_wins = []
    seq_non_solved = []
    for task, seed in sorted({(row.get("task", ""), row.get("seed", "")) for row in seq_rows}, key=lambda item: (item[0], sort_seed_key(item[1]))):
        ff = seq_by_task_seed_method.get((task, seed, "feedforward_lgn"))
        reg = seq_by_task_seed_method.get((task, seed, "register_lgn_soft"))
        if ff and reg:
            reg_acc = to_float(reg.get("sequence_acc"))
            ff_acc = to_float(ff.get("sequence_acc"))
            delta = reg_acc - ff_acc
            item = f"{task}/seed{seed} delta={delta:.6g} reg={fmt(reg.get('sequence_acc'))} ff={fmt(ff.get('sequence_acc'))}"
            if reg_acc >= 0.95 and delta >= 0.05:
                seq_solved_wins.append(f"SOLVED {item}")
            else:
                seq_non_solved.append(f"DIAGNOSTIC {item}")
    checks.append(
        {
            "criterion": "register LGN solves at least one stateful task that feedforward LGN fails",
            "status": "PASS" if seq_solved_wins else "FAIL",
            "evidence": "; ".join(seq_solved_wins + seq_non_solved),
            "note": "Solved evidence requires register_lgn_soft sequence_acc >= 0.95 and at least +0.05 over feedforward on the same task/seed; near-random sequence-parity deltas are diagnostic only.",
        }
    )

    red_by_dataset_seed_method = best_by(red_rows, ("dataset", "seed", "method"))
    red_wins = []
    red_losses = []
    red_keys = sorted({(row.get("dataset", ""), row.get("seed", "")) for row in red_rows}, key=lambda key: (key[0], sort_seed_key(key[1])))
    for dataset, seed in red_keys:
        base = red_by_dataset_seed_method.get((dataset, seed, "no_redundancy"))
        hardened = red_by_dataset_seed_method.get((dataset, seed, "redundant_task_hardened"))
        if base and hardened:
            delta = to_float(hardened.get("hard_acc")) - to_float(base.get("hard_acc"))
            item = (
                f"{dataset}/seed{seed} delta={delta:.6g} "
                f"hardened={fmt(hardened.get('hard_acc'))} base={fmt(base.get('hard_acc'))} "
                f"hardened_factor={hardened.get('redundancy_factor', '')} base_factor={base.get('redundancy_factor', '')}"
            )
            (red_wins if delta > 0 else red_losses).append(item)
    red_categories = dataset_categories(unique(red_rows, "dataset"))
    if "mnist" in red_categories and "digits" in red_categories:
        red_win_note = "Covered Boolean, sklearn digits, and binarized MNIST dataset/seed pairs favor the best redundant_task_hardened factor over no_redundancy; this does not prove superiority over more_gates_only/redundant_regularized, larger MNIST budgets, or image-dataset multi-seed runs."
    elif "digits" in red_categories:
        red_win_note = "Covered Boolean plus sklearn digits dataset/seed pairs favor the best redundant_task_hardened factor over no_redundancy; this does not prove superiority over more_gates_only/redundant_regularized or missing binarized MNIST/image-dataset multi-seed runs."
    else:
        red_win_note = "Covered dataset/seed pairs favor the best redundant_task_hardened factor over no_redundancy; this does not prove superiority over more_gates_only/redundant_regularized or missing digits/MNIST/image-dataset seeds."
    checks.append(
        {
            "criterion": "redundant task-aware hardening improves hard accuracy over no redundancy",
            "status": "PARTIAL" if red_wins and red_losses else ("PASS" if red_wins else "FAIL"),
            "evidence": "; ".join(red_wins + red_losses),
            "note": red_win_note if red_wins and not red_losses else "Current evidence does not uniformly support the redundancy-hardening success claim.",
        }
    )

    strong_rows = redundancy_strong_baseline_rows(red_rows)
    strong_wins = []
    strong_ties = []
    strong_losses = []
    strong_missing = []
    for row in strong_rows:
        dataset = row.get("dataset", "")
        seed = row.get("seed", "")
        status = row.get("status", "")
        if status == "MISSING":
            strong_missing.append(f"{dataset}/seed{seed}")
            continue
        item = (
            f"{dataset}/seed{seed} delta={fmt(row.get('delta_hard_acc'))} "
            f"task={fmt(row.get('task_hardened_hard_acc'))} "
            f"baseline={fmt(row.get('strong_baseline_hard_acc'))} "
            f"baseline_method={row.get('strong_baseline_method', '')}"
        )
        if status == "WIN":
            strong_wins.append(item)
        elif status == "TIE":
            strong_ties.append(item)
        else:
            strong_losses.append(item)
    strong_counts = f"wins={len(strong_wins)}; ties={len(strong_ties)}; losses={len(strong_losses)}; missing={len(strong_missing)}"
    if strong_wins and not strong_ties and not strong_losses and not strong_missing:
        strong_status = "PASS"
        strong_note = "Task-aware hardening beats the strongest non-task-aware redundant baseline for every covered dataset/seed pair."
    elif strong_wins:
        strong_status = "PARTIAL"
        strong_note = "Task-aware hardening is not uniformly better than extra gates or redundancy regularization; inspect redundancy_strong_baseline_comparison.csv."
    else:
        strong_status = "FAIL" if strong_rows else "MISSING"
        strong_note = "Current evidence does not show a hard-accuracy advantage over the strongest non-task-aware redundant baseline."
    checks.append(
        {
            "criterion": "redundant task-aware hardening beats strongest non-task-aware redundant baseline",
            "status": strong_status,
            "evidence": "; ".join([strong_counts] + strong_wins + strong_ties + strong_losses + strong_missing),
            "note": strong_note,
        }
    )

    truth_rows = truth_refit_candidate_rows(candidate_rows)
    truth_wins = []
    truth_ties = []
    truth_losses = []
    truth_missing = []
    for row in truth_rows:
        dataset = row.get("dataset", "")
        seed = row.get("seed", "")
        factor = row.get("redundancy_factor", "")
        status = row.get("status", "")
        if status == "MISSING":
            truth_missing.append(f"{dataset}/seed{seed}/factor{factor}")
            continue
        item = (
            f"{dataset}/seed{seed}/factor{factor} delta={fmt(row.get('delta_truth_vs_best_non_refit'))} "
            f"truth={fmt(row.get('truth_val_hard_acc'))} "
            f"best_non_refit={fmt(row.get('best_non_refit_val_hard_acc'))} "
            f"best_non_refit_candidate={row.get('best_non_refit_candidate', '')} "
            f"selected={row.get('selected_candidate', '')}"
        )
        if status == "WIN":
            truth_wins.append(item)
        elif status == "TIE":
            truth_ties.append(item)
        else:
            truth_losses.append(item)
    truth_counts = f"wins={len(truth_wins)}; ties={len(truth_ties)}; losses={len(truth_losses)}; missing={len(truth_missing)}"
    if truth_wins and not truth_ties and not truth_losses and not truth_missing:
        truth_status = "PASS"
        truth_note = "Truth-table refit is the best validation candidate for every covered redundant_task_hardened dataset/seed/factor row."
    elif truth_wins:
        truth_status = "PARTIAL"
        truth_note = "Truth-table refit is sometimes best, but not uniformly better than argmax/Gumbel candidates; this is validation-level evidence only."
    else:
        truth_status = "FAIL" if truth_rows else "MISSING"
        truth_note = "Current validation evidence does not show truth-table refit beating the best non-refit hardening candidates."
    checks.append(
        {
            "criterion": "truth-table refit beats best non-refit hardening candidate within redundant_task_hardened",
            "status": truth_status,
            "evidence": "; ".join([truth_counts] + truth_wins + truth_ties + truth_losses + truth_missing),
            "note": truth_note,
        }
    )

    truth_final = truth_refit_final_rows(red_rows)
    final_wins = []
    final_ties = []
    final_losses = []
    final_missing = []
    for row in truth_final:
        dataset = row.get("dataset", "")
        seed = row.get("seed", "")
        status = row.get("status_vs_task_selected", "")
        if status == "MISSING":
            final_missing.append(f"{dataset}/seed{seed}")
            continue
        item = (
            f"{dataset}/seed{seed} delta={fmt(row.get('delta_vs_task_selected'))} "
            f"truth={fmt(row.get('truth_refit_hard_acc'))} "
            f"task={fmt(row.get('task_selected_hard_acc'))} "
            f"task_hardening={row.get('task_selected_hardening', '')}"
        )
        if status == "WIN":
            final_wins.append(item)
        elif status == "TIE":
            final_ties.append(item)
        else:
            final_losses.append(item)
    final_counts = f"wins={len(final_wins)}; ties={len(final_ties)}; losses={len(final_losses)}; missing={len(final_missing)}"
    if final_wins and not final_ties and not final_losses and not final_missing:
        final_status = "PASS"
        final_note = "Forced truth-table refit beats the validation-selected task-aware hardening row on every covered held-out test dataset/seed pair."
    elif final_wins:
        final_status = "PARTIAL"
        final_note = "Forced truth-table refit has mixed held-out test results versus validation-selected task-aware hardening; inspect truth_refit_final_test_comparison.csv."
    else:
        final_status = "FAIL" if truth_final else "MISSING"
        final_note = "Current held-out test evidence does not show forced truth-table refit beating validation-selected task-aware hardening."
    checks.append(
        {
            "criterion": "truth-table refit-only final test beats validation-selected task-aware hardening",
            "status": final_status,
            "evidence": "; ".join([final_counts] + final_wins + final_ties + final_losses + final_missing),
            "note": final_note,
        }
    )

    ckpt_wins = [row for row in red_rows if to_float(row.get("best_hard_ckpt_hard_val_acc")) > to_float(row.get("best_soft_ckpt_hard_val_acc"))]
    checks.append(
        {
            "criterion": "best-hard-accuracy checkpoint beats best-soft-loss checkpoint",
            "status": "PARTIAL" if ckpt_wins else "FAIL",
            "evidence": f"{len(ckpt_wins)}/{len(red_rows)} redundancy final rows have best_hard_val_acc > best_soft_ckpt_hard_val_acc",
            "note": "This is a checkpointing signal, not proof that final redundant hardening wins.",
        }
    )

    hardened_rows = [row for row in red_rows if row.get("method") == "redundant_task_hardened"]
    gaps = [to_float(row.get("native_gap")) for row in hardened_rows if math.isfinite(to_float(row.get("native_gap")))]
    checks.append(
        {
            "criterion": "native_acc_gap remains small after hardening",
            "status": "PARTIAL" if gaps and max(gaps) <= 0.1 else ("FAIL" if gaps else "MISSING"),
            "evidence": f"max redundant_task_hardened native_gap={max(gaps):.6g}" if gaps else "no redundant_task_hardened rows",
            "note": "Threshold is diagnostic only; full claim needs broader redundancy sweep.",
        }
    )
    return checks


def coverage_audit(
    wiring_rows: list[dict[str, str]],
    seq_rows: list[dict[str, str]],
    red_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    synthesis_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    synthesis_rows = synthesis_rows or []

    required_a = {"parity", "majority", "random_sparse", "digits", "mnist"}
    observed_a = dataset_categories(unique(wiring_rows, "dataset"))
    if {"digits", "mnist", "random_sparse"} <= observed_a:
        a_dataset_note = "Boolean, sklearn digits, and binarized MNIST coverage is present; MNIST evidence is smoke-scale."
    elif "digits" in observed_a and "random_sparse" in observed_a:
        a_dataset_note = "Boolean plus sklearn digits coverage is present; binarized MNIST remains missing."
    elif "random_sparse" in observed_a:
        a_dataset_note = "Boolean coverage now includes random_sparse; digits/MNIST remain missing."
    else:
        a_dataset_note = "Current available run is smoke-scale."
    rows.append(
        {
            "area": "A_wiring",
            "requirement": "run on parity, majority, random sparse Boolean functions, sklearn digits, binarized MNIST",
            "status": "PASS" if required_a <= observed_a else ("PARTIAL" if observed_a else "MISSING"),
            "evidence": ",".join(sorted(observed_a)),
            "missing": ",".join(sorted(required_a - observed_a)),
            "note": a_dataset_note,
        }
    )

    required_a_methods = {"fixed_random", "fixed_random_matched_budget", "random_wiring_search_freeze"}
    learnable_a_methods = {"candidate_set_learnable_freeze", "learnable_connectivity", "candidate_refresh_connectivity"}
    observed_a_methods = set(unique(wiring_rows, "method"))
    observed_learnable_a = observed_a_methods & learnable_a_methods
    missing_a_methods = sorted(required_a_methods - observed_a_methods)
    if not observed_learnable_a:
        missing_a_methods.append("candidate_set_or_learnable_connectivity")
    rows.append(
        {
            "area": "A_wiring",
            "requirement": "compare fixed random wiring, matched budget, random wiring search + freeze, optional learnable connectivity",
            "status": "PASS" if required_a_methods <= observed_a_methods and observed_learnable_a else ("PARTIAL" if observed_a_methods else "MISSING"),
            "evidence": ",".join(sorted(observed_a_methods)),
            "missing": ",".join(missing_a_methods),
            "note": (
                "Candidate-set learnable connectivity is present and freezes back to ordinary fixed wiring for final held-out evaluation; inspect per-dataset rows before claiming a win."
                if observed_learnable_a
                else "Optional learnable/candidate connectivity remains unimplemented unless reliable source code is integrated."
            ),
        }
    )

    boolean_datasets = {"parity8", "majority9", "random_sparse10"}
    required_boolean_seeds = {"0", "1", "2"}
    missing_a_seed_pairs = missing_dataset_seed_pairs(wiring_rows, boolean_datasets, required_boolean_seeds)
    rows.append(
        {
            "area": "A_wiring",
            "requirement": "Boolean wiring-search evidence covers seeds 0, 1, and 2 for parity8, majority9, random_sparse10",
            "status": "PASS" if not missing_a_seed_pairs else "PARTIAL",
            "evidence": dataset_seed_evidence(wiring_rows, boolean_datasets),
            "missing": ",".join(missing_a_seed_pairs),
            "note": "Boolean multi-seed coverage is present; sklearn digits and binarized MNIST remain seed-0/smoke evidence.",
        }
    )

    required_tasks = {"sequence_parity", "delayed_copy", "temporal_majority", "fsm_endswith101"}
    observed_tasks = set(unique(seq_rows, "task"))
    observed_seq_seeds = set(unique(seq_rows, "seed"))
    rows.append(
        {
            "area": "B_register",
            "requirement": "run delayed copy, sequence parity, temporal majority, finite-state-machine recognition",
            "status": "PASS" if required_tasks <= observed_tasks else "PARTIAL",
            "evidence": ",".join(sorted(observed_tasks)),
            "missing": ",".join(sorted(required_tasks - observed_tasks)),
            "note": f"Required-task run covers seeds={','.join(sorted(observed_seq_seeds, key=sort_seed_key)) or 'none'}.",
        }
    )

    required_seq_methods = {"feedforward_lgn", "register_lgn_soft", "register_lgn_st", "rnn_small", "gru_small"}
    observed_seq_methods = set(unique(seq_rows, "method"))
    rows.append(
        {
            "area": "B_register",
            "requirement": "compare feedforward LGN, relaxed register LGN, hard/ST register LGN, RNN, GRU",
            "status": "PASS" if required_seq_methods <= observed_seq_methods else "PARTIAL",
            "evidence": ",".join(sorted(observed_seq_methods)),
            "missing": ",".join(sorted(required_seq_methods - observed_seq_methods)),
            "note": "RDDLGN-style external baseline remains optional/unintegrated.",
        }
    )

    required_c = {"parity", "majority", "random_sparse", "digits", "mnist"}
    observed_c = dataset_categories(unique(red_rows, "dataset"))
    if {"digits", "mnist", "random_sparse"} <= observed_c:
        c_dataset_note = "Boolean, sklearn digits, and binarized MNIST coverage is present; MNIST evidence is smoke-scale."
    elif "digits" in observed_c and "random_sparse" in observed_c:
        c_dataset_note = "Boolean plus sklearn digits coverage is present; binarized MNIST remains missing."
    elif "random_sparse" in observed_c:
        c_dataset_note = "Boolean coverage now includes random_sparse; digits/MNIST remain missing."
    else:
        c_dataset_note = "Current available run is quick smoke only."
    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "run redundancy hardening on Boolean tasks, digits, binarized MNIST",
            "status": "PASS" if required_c <= observed_c else ("PARTIAL" if observed_c else "MISSING"),
            "evidence": ",".join(sorted(observed_c)),
            "missing": ",".join(sorted(required_c - observed_c)),
            "note": c_dataset_note,
        }
    )

    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "run CIFAR-10 small smoke redundancy hardening when feasible",
            "status": "PASS" if "cifar" in observed_c else "MISSING",
            "evidence": "cifar" if "cifar" in observed_c else "",
            "missing": "" if "cifar" in observed_c else "cifar10_small",
            "note": "CIFAR evidence is small smoke-scale path coverage, not the Mind-the-Gap 61M-gate CIFAR setting.",
        }
    )

    missing_c_seed_pairs = missing_dataset_seed_pairs(red_rows, boolean_datasets, required_boolean_seeds)
    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "Boolean redundancy-hardening evidence covers seeds 0, 1, and 2 for parity8, majority9, random_sparse10",
            "status": "PASS" if not missing_c_seed_pairs else "PARTIAL",
            "evidence": dataset_seed_evidence(red_rows, boolean_datasets),
            "missing": ",".join(missing_c_seed_pairs),
            "note": "Boolean multi-seed coverage is present; sklearn digits and binarized MNIST remain seed-0/smoke evidence.",
        }
    )

    required_factors = {"1", "2", "4", "8"}
    observed_factors = set(unique(red_rows, "redundancy_factor"))
    factor_status = "PASS" if required_factors <= observed_factors else ("PARTIAL" if observed_factors else "MISSING")
    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "sweep redundancy factors 1x, 2x, 4x, 8x",
            "status": factor_status,
            "evidence": ",".join(sorted(observed_factors)),
            "missing": ",".join(sorted(required_factors - observed_factors)),
            "note": "Factor coverage is complete across the merged redundancy evidence; per-dataset completeness is not implied." if factor_status == "PASS" else "Quick smoke or partial run does not cover all requested factors.",
        }
    )

    observed_candidates = set(unique(candidate_rows, "candidate"))
    has_gumbel = any(name.startswith("gumbel_sample_") for name in observed_candidates)
    gumbel_ids = gumbel_sample_ids(observed_candidates)
    best_of_8_observed = set(range(8)) <= gumbel_ids
    best_of_32_observed = set(range(32)) <= gumbel_ids
    required_candidate_notes = []
    if "truth_table_refit" not in observed_candidates:
        required_candidate_notes.append("truth_table_refit")
    if "argmax_best_soft" not in observed_candidates or "argmax_best_hard" not in observed_candidates:
        required_candidate_notes.append("argmax")
    if not has_gumbel:
        required_candidate_notes.append("gumbel_samples")
    if not best_of_8_observed:
        required_candidate_notes.append("best_of_8_full_sweep")
    if not best_of_32_observed:
        required_candidate_notes.append("best_of_32_optional")
    synthesis_has_post_eval = any(row.get("accuracy_recomputed_after_abc") == "yes" for row in synthesis_rows)
    synthesis_has_post_structure = any(
        math.isfinite(to_float(row.get("post_abc_fanout_max"))) and math.isfinite(to_float(row.get("post_abc_unused_node_ratio")))
        for row in synthesis_rows
    )
    if synthesis_rows and not synthesis_has_post_eval:
        required_candidate_notes.append("abc_post_accuracy_reimport_optional")
    elif synthesis_has_post_eval and not synthesis_has_post_structure:
        required_candidate_notes.append("abc_post_fanout_unused_optional")
    elif not synthesis_rows:
        required_candidate_notes.append("abc_optional")
    candidate_status = "PASS" if observed_candidates and not required_candidate_notes else ("PARTIAL" if observed_candidates else "MISSING")
    candidate_note_parts = []
    if best_of_32_observed:
        candidate_note_parts.append("Best-of-32 candidates are present in the merged evidence.")
    elif best_of_8_observed:
        candidate_note_parts.append("Best-of-8 candidates are present in the merged evidence; best-of-32 remains optional/missing.")
    else:
        candidate_note_parts.append("This run does not expose the full gumbel_sample_0..7 set.")
    if synthesis_has_post_structure:
        candidate_note_parts.append("ABC structural stats plus post-ABC accuracy/gap/fanout/unused evaluation are integrated separately in abc_synthesis_comparison.csv.")
    elif synthesis_has_post_eval:
        candidate_note_parts.append("ABC structural stats and post-ABC accuracy/gap evaluation are integrated separately in abc_synthesis_comparison.csv, but post-ABC fanout/unused are not available.")
    elif synthesis_rows:
        candidate_note_parts.append("ABC structural stats are integrated separately in abc_synthesis_comparison.csv, but post-ABC accuracy import is not available.")
    else:
        candidate_note_parts.append("ABC remains optional/missing for this merged evidence set.")
    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "compare argmax, best-of-N Gumbel hard samples, truth-table hardening, optional ABC",
            "status": candidate_status,
            "evidence": ",".join(sorted(observed_candidates)),
            "missing": ",".join(required_candidate_notes),
            "note": " ".join(candidate_note_parts),
        }
    )

    selected_candidates = [row for row in candidate_rows if truthy(row.get("selected"))]
    selected_ineligible = [row for row in selected_candidates if not truthy(row.get("eligible_for_method"))]
    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "selected hardening candidates are eligible for their method-specific selection policy",
            "status": "PASS" if selected_candidates and not selected_ineligible else ("FAIL" if selected_ineligible else "MISSING"),
            "evidence": f"selected={len(selected_candidates)}; ineligible_selected={len(selected_ineligible)}",
            "missing": "" if selected_candidates and not selected_ineligible else "eligible selected candidates",
            "note": "This guards against reporting a task-aware hardening candidate in a baseline row that was not allowed to select it.",
        }
    )

    truth_final_rows = [row for row in red_rows if row.get("method") == "redundant_truth_refit_only"]
    boolean_truth_final_rows = [row for row in truth_final_rows if row.get("dataset") in {"parity8", "majority9", "random_sparse10"}]
    boolean_truth_final_bad = [row for row in boolean_truth_final_rows if row.get("hardening") != "truth_table_refit"]
    truth_final_missing = missing_dataset_seed_pairs(boolean_truth_final_rows, {"parity8", "majority9", "random_sparse10"}, {"0", "1", "2"})
    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "final-test truth-table-refit-only rows cover Boolean seeds 0, 1, and 2",
            "status": "PASS" if boolean_truth_final_rows and not truth_final_missing and not boolean_truth_final_bad else ("PARTIAL" if boolean_truth_final_rows else "MISSING"),
            "evidence": f"rows={len(boolean_truth_final_rows)}; non_truth_hardening={len(boolean_truth_final_bad)}; {dataset_seed_evidence(boolean_truth_final_rows, {'parity8', 'majority9', 'random_sparse10'})}",
            "missing": ",".join(truth_final_missing),
            "note": "These are held-out test rows for forced truth_table_refit, not validation-only candidate rows.",
        }
    )

    truth_image_missing = missing_dataset_seed_pairs(truth_final_rows, {"digits", "binarized_mnist", "cifar10_small"}, {"0"})
    rows.append(
        {
            "area": "C_redundancy",
            "requirement": "final-test truth-table-refit-only rows cover sklearn digits, binarized MNIST, and CIFAR-10 smoke seed 0",
            "status": "PASS" if truth_final_rows and not truth_image_missing else ("PARTIAL" if truth_final_rows else "MISSING"),
            "evidence": dataset_seed_evidence(truth_final_rows, {"digits", "binarized_mnist", "cifar10_small"}),
            "missing": ",".join(truth_image_missing),
            "note": "Digits is seed-0 coverage; binarized MNIST and CIFAR-10 remain small smoke runs, not full-scale validation.",
        }
    )

    return rows


def selected_candidate_refits(candidate_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str], str]:
    refits: dict[tuple[str, str, str, str, str], str] = {}
    for row in candidate_rows:
        if not truthy(row.get("selected")):
            continue
        key = (
            row.get("method", ""),
            row.get("dataset", ""),
            row.get("seed", ""),
            row.get("redundancy_factor", ""),
            row.get("candidate", ""),
        )
        refits[key] = row.get("refit_error", "")
    return refits


def gap_taxonomy_rows(
    wiring_rows: list[dict[str, str]],
    seq_rows: list[dict[str, str]],
    red_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in wiring_rows:
        rows.append(
            {
                "axis": "A_wiring",
                "source_run": row.get("source_run", ""),
                "method": row.get("method", ""),
                "item": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "hard_acc": row.get("hard_acc", ""),
                "soft_acc": row.get("soft_acc", ""),
                "full_gap": row.get("full_gap", ""),
                "native_gap": row.get("native_gap", ""),
                "block_refit_error": "not_applicable",
                "note": f"wiring_seed={row.get('wiring_seed', '')}",
            }
        )
    for row in seq_rows:
        rows.append(
            {
                "axis": "B_register",
                "source_run": row.get("source_run", ""),
                "method": row.get("method", ""),
                "item": row.get("task", ""),
                "seed": row.get("seed", ""),
                "hard_acc": row.get("hard_acc", ""),
                "soft_acc": row.get("soft_acc", ""),
                "full_gap": row.get("full_gap", ""),
                "native_gap": row.get("native_gap", ""),
                "block_refit_error": "not_applicable",
                "note": f"sequence_acc={row.get('sequence_acc', '')}; state_bits={row.get('state_bits', '')}",
            }
        )
    selected_refits = selected_candidate_refits(candidate_rows)
    for row in red_rows:
        key = (
            row.get("method", ""),
            row.get("dataset", ""),
            row.get("seed", ""),
            row.get("redundancy_factor", ""),
            row.get("hardening", ""),
        )
        rows.append(
            {
                "axis": "C_redundancy",
                "source_run": row.get("source_run", ""),
                "method": row.get("method", ""),
                "item": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "hard_acc": row.get("hard_acc", ""),
                "soft_acc": row.get("soft_acc", ""),
                "full_gap": row.get("full_gap", ""),
                "native_gap": row.get("native_gap", ""),
                "block_refit_error": selected_refits.get(key, "missing_selected_candidate"),
                "note": f"selected_hardening={row.get('hardening', '')}; candidate-level refit evidence is in hardening_method_comparison.csv",
            }
        )
    return sorted(rows, key=lambda row: (str(row.get("axis")), str(row.get("item")), str(row.get("method")), str(row.get("seed"))))


def markdown_report(
    unified: list[dict[str, object]],
    success: list[dict[str, object]],
    coverage: list[dict[str, object]],
    mind_gap_direct: list[dict[str, object]],
    synthesis_rows: list[dict[str, object]],
    wiring_rows: list[dict[str, str]],
    seq_rows: list[dict[str, str]],
    red_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> str:
    success_by_criterion = {str(row.get("criterion", "")): str(row.get("status", "")) for row in success}
    candidate_names = {str(row.get("candidate", "")) for row in candidate_rows}
    best_of_32_observed = set(range(32)) <= gumbel_sample_ids(candidate_names)
    synthesis_has_post_eval = any(row.get("accuracy_recomputed_after_abc") == "yes" for row in synthesis_rows)
    synthesis_has_post_structure = any(
        math.isfinite(to_float(row.get("post_abc_fanout_max"))) and math.isfinite(to_float(row.get("post_abc_unused_node_ratio")))
        for row in synthesis_rows
    )

    def part_c_limitations(red_categories: set[str]) -> str:
        limitations = ["image-dataset multi-seed runs", "stronger redundant-baseline comparisons"]
        if "digits" not in red_categories:
            limitations.append("sklearn digits")
        if "mnist" not in red_categories:
            limitations.append("binarized MNIST")
        else:
            limitations.append("MNIST remains smoke-scale")
        if not best_of_32_observed:
            limitations.append("best-of-32")
        if not synthesis_rows:
            limitations.append("ABC")
        elif not synthesis_has_post_eval:
            limitations.append("post-ABC accuracy/gap recomputation")
        elif not synthesis_has_post_structure:
            limitations.append("post-ABC fanout/unused metrics")
        return ", ".join(limitations)

    lines = [
        "# Next-Stage Hard-LGN Evidence Report",
        "",
        "This report is generated from current run artifacts. It is an evidence audit, not a claim that the pasted-text goal is complete.",
        "",
        "Primary objective tracked here: maximize final deployable hard-path accuracy under fixed budgets, while keeping full-gap, native-gap, and local hardening evidence separate.",
        "",
        "## Success Checks",
        "",
        table_markdown(success, SUCCESS_FIELDS),
        "",
        "## Coverage Audit",
        "",
        table_markdown(coverage, COVERAGE_FIELDS),
        "",
        "## Mind-the-Gap Direct Comparison",
        "",
        "This section keeps the traditional `relaxed-full vs hard-full` gap separate from the method-native hard-prefix path gap. Only `full_gap_*` columns should be used for the primary Mind-the-Gap comparison.",
        "",
        table_markdown(
            mind_gap_direct,
            [
                "comparison_scope",
                "dataset",
                "block_hard_discrete_acc",
                "gumbel_discrete_acc",
                "delta_hard_acc_vs_gumbel",
                "block_hard_full_gap",
                "block_hard_native_gap",
                "dlgn_full_gap",
                "gumbel_full_gap",
                "hard_acc_beats_gumbel",
                "full_gap_beats_dlgn",
                "native_gap_beats_dlgn",
                "full_gap_competitive_with_gumbel",
                "native_gap_competitive_with_gumbel",
            ],
        )
        if mind_gap_direct
        else "No direct Mind-the-Gap comparison rows were available.",
        "",
        "## Unified Hard-Accuracy Leaderboard",
        "",
        table_markdown(unified, UNIFIED_FIELDS, limit=30),
        "",
        "## ABC Synthesis Structural Baseline",
        "",
        "ABC is integrated as a structural baseline plus post-ABC BLIF accuracy/gap and graph-structure evaluation when `accuracy_recomputed_after_abc=yes`. Post-ABC fanout and unused ratios are BLIF graph metrics, not PyTorch inactive-neuron ratios.",
        "",
        table_markdown(
            synthesis_rows,
            [
                "dataset",
                "method",
                "seed",
                "abc_status",
                "discrete_acc",
                "acc_gap",
                "post_abc_discrete_acc",
                "post_acc_delta_vs_pre_discrete",
                "post_abc_gap_vs_pre_soft",
                "post_gap_delta_vs_pre_gap",
                "unused_gate_ratio",
                "source_blif_fanout_max",
                "post_abc_fanout_max",
                "fanout_change_after_abc",
                "source_blif_unused_node_ratio",
                "post_abc_unused_node_ratio",
                "unused_gate_ratio_change_after_abc",
                "pre_gate_count",
                "pre_depth",
                "pre_fanout_max",
                "abc_post_and",
                "abc_post_lev",
                "abc_and_reduction_vs_pre_nd",
                "accuracy_recomputed_after_abc",
            ],
        )
        if synthesis_rows
        else "No ABC synthesis rows were available.",
        "",
        "## Key Current Observations",
        "",
    ]

    def seq_mean(task: str, method: str, metric: str) -> tuple[float, int]:
        values = [
            to_float(row.get(metric))
            for row in seq_rows
            if row.get("task") == task and row.get("method") == method and math.isfinite(to_float(row.get(metric)))
        ]
        return (sum(values) / len(values), len(values)) if values else (math.nan, 0)

    delayed_ff_mean, delayed_ff_n = seq_mean("delayed_copy", "feedforward_lgn", "sequence_acc")
    delayed_reg_mean, delayed_reg_n = seq_mean("delayed_copy", "register_lgn_soft", "sequence_acc")
    if delayed_ff_n and delayed_reg_n:
        lines.append(
            f"- Delayed copy: `register_lgn_soft` mean sequence_acc={delayed_reg_mean:.6g} over {delayed_reg_n} seeds, "
            f"feedforward_lgn mean sequence_acc={delayed_ff_mean:.6g} over {delayed_ff_n} seeds."
        )
    fsm_reg_mean, fsm_reg_n = seq_mean("fsm_endswith101", "register_lgn_soft", "hard_acc")
    if fsm_reg_n:
        lines.append(f"- FSM recognition: `register_lgn_soft` mean hard_acc={fsm_reg_mean:.6g} over {fsm_reg_n} seeds.")

    wiring_status = success_by_criterion.get("random wiring search + freeze beats matched-budget fixed random", "")
    candidate_wiring_status = success_by_criterion.get("candidate-set learnable connectivity beats matched-budget fixed random", "")
    if wiring_status == "PASS":
        lines.append("- On the covered wiring-search dataset/seed pairs, `random_wiring_search_freeze` beats the matched-budget fixed-random baseline; image-dataset multi-seed evidence is still needed.")
    elif wiring_status == "PARTIAL":
        lines.append("- Wiring search evidence is mixed; inspect `wiring_search_comparison.csv` before making any search-win claim.")
    elif wiring_status == "FAIL":
        lines.append("- Wiring search does not beat the matched-budget fixed-random baseline on the covered run; treat the Part A hypothesis as unsupported by current evidence.")
    if candidate_wiring_status == "PASS":
        lines.append("- Candidate-set learnable connectivity beats the same-run matched-budget fixed-random baseline on all covered comparisons and still freezes to ordinary fixed wiring before final training.")
    elif candidate_wiring_status == "PARTIAL":
        lines.append("- Candidate-set learnable connectivity is implemented but mixed against the same-run matched-budget fixed-random baseline; treat it as a controlled comparator, not a win claim.")
    elif candidate_wiring_status == "FAIL":
        lines.append("- Candidate-set learnable connectivity is implemented, but current same-run evidence does not beat matched-budget fixed random.")
    if red_rows:
        red_status = success_by_criterion.get("redundant task-aware hardening improves hard accuracy over no redundancy", "")
        strong_red_status = success_by_criterion.get("redundant task-aware hardening beats strongest non-task-aware redundant baseline", "")
        truth_refit_status = success_by_criterion.get("truth-table refit beats best non-refit hardening candidate within redundant_task_hardened", "")
        truth_final_status = success_by_criterion.get("truth-table refit-only final test beats validation-selected task-aware hardening", "")
        red_categories = dataset_categories(unique(red_rows, "dataset"))
        if red_status == "PASS":
            if "mnist" in red_categories and "digits" in red_categories:
                lines.append(f"- On the covered Boolean, sklearn digits, and binarized MNIST dataset/seed pairs, the best `redundant_task_hardened` factor beats `no_redundancy` on hard accuracy; remaining Part C limitations: {part_c_limitations(red_categories)}.")
            elif "digits" in red_categories:
                lines.append(f"- On the covered Boolean plus sklearn digits dataset/seed pairs, the best `redundant_task_hardened` factor beats `no_redundancy` on hard accuracy; remaining Part C limitations: {part_c_limitations(red_categories)}.")
            else:
                lines.append(f"- On the covered Boolean dataset/seed pairs, the best `redundant_task_hardened` factor beats `no_redundancy` on hard accuracy; remaining Part C limitations: {part_c_limitations(red_categories)}.")
        else:
            lines.append("- Current redundancy hardening evidence does not uniformly support a hard-accuracy win over `no_redundancy`; treat Part C as incomplete.")
        if strong_red_status == "PASS":
            lines.append("- `redundant_task_hardened` also beats the strongest non-task-aware redundant baseline on every covered dataset/seed pair.")
        elif strong_red_status == "PARTIAL":
            lines.append("- Against the strongest non-task-aware redundant baseline, `redundant_task_hardened` is mixed; this separates task-aware hardening/selection from simply adding more gates.")
        elif strong_red_status == "FAIL":
            lines.append("- Against the strongest non-task-aware redundant baseline, current evidence does not support a task-aware hardening advantage.")
        if truth_refit_status == "PASS":
            lines.append("- Within `redundant_task_hardened`, `truth_table_refit` is the best validation hardening candidate on every covered dataset/seed/factor row.")
        elif truth_refit_status == "PARTIAL":
            lines.append("- Within `redundant_task_hardened`, `truth_table_refit` is mixed against argmax/Gumbel candidates; this is validation-level evidence, not final test accuracy.")
        elif truth_refit_status == "FAIL":
            lines.append("- Within `redundant_task_hardened`, current validation evidence does not support `truth_table_refit` over the best non-refit candidates.")
        if truth_final_status == "PASS":
            lines.append("- Forced `redundant_truth_refit_only` beats validation-selected `redundant_task_hardened` on every covered held-out test dataset/seed pair.")
        elif truth_final_status == "PARTIAL":
            lines.append("- Forced `redundant_truth_refit_only` has mixed held-out test results versus validation-selected `redundant_task_hardened`; this is the direct final-test truth-refit-only check.")
        elif truth_final_status == "FAIL":
            lines.append("- Forced `redundant_truth_refit_only` does not beat validation-selected `redundant_task_hardened` on the covered held-out test pairs.")
        if best_of_32_observed:
            lines.append("- Best-of-32 Gumbel hardening candidates are present in the merged Part C evidence (`gumbel_sample_0..31`).")
        lines.append("- `soft_hard_gap_taxonomy_current.csv` reports the selected final hardening candidate's `block_refit_error`; all candidate-level refit errors are in `hardening_method_comparison.csv`.")
    if mind_gap_direct:
        full_status = success_by_criterion.get("block-hard primary full gap beats DLGN in direct Mind-the-Gap comparison", "")
        native_status = success_by_criterion.get("block-hard method-native gap beats DLGN in direct Mind-the-Gap comparison", "")
        gumbel_status = success_by_criterion.get("block-hard primary full gap is competitive with Gumbel in direct comparison", "")
        hard_gumbel_status = success_by_criterion.get("block-hard hard accuracy beats tuned/fixed Gumbel in direct comparison", "")
        if full_status in {"FAIL", "PARTIAL"} and native_status in {"PASS", "PARTIAL"}:
            lines.append("- Direct Mind-the-Gap rows show the expected split: method-native hard-prefix gap can improve while the traditional relaxed-full gap remains weak.")
        if gumbel_status == "FAIL":
            lines.append("- Under the primary full-gap metric, current block-hard-refit evidence is not competitive with the available Gumbel baselines; do not present the native-path gap as a Mind-the-Gap win.")
        if hard_gumbel_status in {"PASS", "PARTIAL"}:
            lines.append("- Direct Gumbel comparisons now report hard accuracy separately from gap; inspect `delta_hard_acc_vs_gumbel` because low Gumbel gap can coincide with low hard accuracy in these prototype-scale runs.")
    if synthesis_rows:
        ok_count = sum(1 for row in synthesis_rows if row.get("abc_status") == "ok")
        post_accuracy_count = sum(1 for row in synthesis_rows if row.get("accuracy_recomputed_after_abc") == "yes")
        post_acc_deltas = [abs(to_float(row.get("post_acc_delta_vs_pre_discrete"))) for row in synthesis_rows if row.get("accuracy_recomputed_after_abc") == "yes"]
        finite_post_acc = [value for value in post_acc_deltas if math.isfinite(value)]
        post_structure_count = sum(
            1
            for row in synthesis_rows
            if math.isfinite(to_float(row.get("post_abc_fanout_max"))) and math.isfinite(to_float(row.get("post_abc_unused_node_ratio")))
        )
        fanout_deltas = [to_float(row.get("fanout_change_after_abc")) for row in synthesis_rows]
        unused_deltas = [to_float(row.get("unused_gate_ratio_change_after_abc")) for row in synthesis_rows]
        finite_fanout_deltas = [value for value in fanout_deltas if math.isfinite(value)]
        finite_unused_deltas = [value for value in unused_deltas if math.isfinite(value)]
        if finite_post_acc:
            lines.append(
                f"- ABC synthesis evidence is integrated for {ok_count}/{len(synthesis_rows)} rows; "
                f"post-ABC accuracy/gap recomputation is available for {post_accuracy_count}/{len(synthesis_rows)} rows, "
                f"with max hard-accuracy delta={max(finite_post_acc):.6g}."
            )
        else:
            lines.append(f"- ABC synthesis structural evidence is now integrated for {ok_count}/{len(synthesis_rows)} rows; post-ABC accuracy/gap recomputation is available for {post_accuracy_count}/{len(synthesis_rows)} rows.")
        if finite_fanout_deltas and finite_unused_deltas:
            lines.append(
                f"- Post-ABC BLIF graph metrics are available for {post_structure_count}/{len(synthesis_rows)} rows; "
                f"fanout delta range=[{min(finite_fanout_deltas):.6g},{max(finite_fanout_deltas):.6g}], "
                f"unused-node-ratio delta range=[{min(finite_unused_deltas):.6g},{max(finite_unused_deltas):.6g}]."
            )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `unified_hard_accuracy_leaderboard.csv`",
            "- `wiring_search_comparison.csv`",
            "- `register_sequential_comparison.csv`",
            "- `redundancy_scaling_curve.csv`",
            "- `redundancy_strong_baseline_comparison.csv`",
            "- `truth_table_refit_candidate_comparison.csv`",
            "- `truth_refit_final_test_comparison.csv`",
            "- `mind_gap_direct_comparison.csv`",
            "- `abc_synthesis_comparison.csv`",
            "- `hardening_method_comparison.csv`",
            "- `soft_hard_gap_taxonomy_current.csv`",
            "- `success_checks.csv`",
            "- `coverage_audit.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="remote_runs")
    parser.add_argument("--wiring-run", default=None, help="Legacy single wiring run. Prefer --wiring-runs.")
    parser.add_argument("--wiring-runs", nargs="+", default=None)
    parser.add_argument("--sequential-run", default=None, help="Legacy single sequential run. Prefer --sequential-runs.")
    parser.add_argument("--sequential-runs", nargs="+", default=None)
    parser.add_argument("--redundancy-run", default=None, help="Legacy single redundancy run. Prefer --redundancy-runs.")
    parser.add_argument("--redundancy-runs", nargs="+", default=None)
    parser.add_argument(
        "--mind-gap-runs",
        nargs="+",
        default=["bool_seeds012_timing_v3", "digits_seeds012_timing_v3", "cifar10_small_seed0_timing_v3"],
        help="Timing-aware runs with aggregate_by_method_dataset.csv for direct Mind-the-Gap comparison.",
    )
    parser.add_argument(
        "--tuned-gumbel-comparison",
        default=None,
        help="Legacy single CSV path for seed-0 tuned Gumbel comparison; ignored when --tuned-gumbel-comparisons is set.",
    )
    parser.add_argument(
        "--tuned-gumbel-comparisons",
        nargs="+",
        default=None,
        help="CSV paths for seed-0 tuned Gumbel comparisons; relative paths are resolved under --runs-root.",
    )
    parser.add_argument(
        "--synthesis-report-dirs",
        nargs="*",
        default=None,
        help="Optional directories containing synthesis_joined.csv. Defaults search runs-root/reports_synthesis_timing_v3 and reports/synthesis_timing_v3.",
    )
    parser.add_argument(
        "--post-abc-eval-dirs",
        nargs="*",
        default=None,
        help="Optional directories containing post_abc_eval.csv. Defaults search runs-root/reports_post_abc_eval_v3 and reports/post_abc_eval_v3.",
    )
    parser.add_argument("--out-dir", default="remote_runs/reports_next_stage_v15")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    wiring_runs = args.wiring_runs or (
        [args.wiring_run]
        if args.wiring_run
        else [
            "wiring_search_bool_sweep_v1",
            "wiring_search_bool_seeds12_v1",
            "wiring_search_digits_v2",
            "wiring_search_mnist_smoke_v1",
        ]
    )
    redundancy_runs = args.redundancy_runs or (
        [args.redundancy_run]
        if args.redundancy_run
        else [
            "redundant_hardening_bool_sweep_v1",
            "redundant_hardening_bool_seeds12_v1",
            "redundant_truth_refit_bool_seeds012_v1",
            "redundant_hardening_digits_v1",
            "redundant_truth_refit_digits_v1",
            "redundant_hardening_mnist_smoke_v1",
            "redundant_truth_refit_mnist_smoke_v1",
            "redundant_hardening_cifar10_smoke_v1",
        ]
    )
    sequential_runs = args.sequential_runs or (
        [args.sequential_run]
        if args.sequential_run
        else ["sequential_register_required_tasks_v1"]
    )
    wiring_rows = read_run_csvs(runs_root, wiring_runs, "wiring_search_results.csv")
    seq_rows = read_run_csvs(runs_root, sequential_runs, "sequential_results.csv")
    red_rows = read_run_csvs(runs_root, redundancy_runs, "redundant_hardening_results.csv")
    candidate_rows = read_run_csvs(runs_root, redundancy_runs, "hardening_candidates.csv")
    if args.tuned_gumbel_comparisons:
        tuned_gumbel_comparisons = args.tuned_gumbel_comparisons
    elif args.tuned_gumbel_comparison:
        tuned_gumbel_comparisons = [args.tuned_gumbel_comparison]
    else:
        tuned_gumbel_comparisons = [
            "gumbel_tuned_comparison_seed0_timing_v3/comparison.csv",
            "gumbel_tuned_comparison_seed0_digits_layerdiag_v2/comparison.csv",
            "gumbel_tuned_comparison_seed0_cifar_layerdiag_v2/comparison.csv",
            "gumbel_tuned_comparison_seed0_mnist_layerdiag_v2/comparison.csv",
        ]
    mind_gap_rows = mind_gap_direct_rows(runs_root, args.mind_gap_runs, tuned_gumbel_comparisons)
    post_abc_eval = read_post_abc_eval_rows(runs_root, args.post_abc_eval_dirs)
    synthesis_rows = read_synthesis_rows(runs_root, args.synthesis_report_dirs, post_abc_eval)

    unified = unified_rows(wiring_rows, seq_rows, red_rows)
    strong_redundancy = redundancy_strong_baseline_rows(red_rows)
    truth_refit_candidates = truth_refit_candidate_rows(candidate_rows)
    truth_refit_final = truth_refit_final_rows(red_rows)
    success = (
        success_checks(wiring_rows, seq_rows, red_rows, candidate_rows)
        + mind_gap_success_checks(mind_gap_rows)
        + synthesis_success_checks(synthesis_rows)
    )
    coverage = (
        coverage_audit(wiring_rows, seq_rows, red_rows, candidate_rows, synthesis_rows)
        + mind_gap_coverage_rows(mind_gap_rows)
        + synthesis_coverage_rows(synthesis_rows)
    )

    write_csv(out_dir / "unified_hard_accuracy_leaderboard.csv", unified, UNIFIED_FIELDS)
    write_csv(out_dir / "wiring_search_comparison.csv", wiring_rows, union_fields(wiring_rows))
    write_csv(out_dir / "register_sequential_comparison.csv", seq_rows, union_fields(seq_rows))
    write_csv(
        out_dir / "redundancy_scaling_curve.csv",
        red_rows,
        union_fields(
            red_rows,
            [
                "method",
                "dataset",
                "seed",
                "redundancy_factor",
                "train_estimator",
                "hardening",
                "checkpoint_source",
                "hard_acc",
                "soft_acc",
                "full_gap",
                "native_gap",
                "hard_loss",
                "soft_loss",
                "train_time",
                "hardening_time",
                "train_valid",
                "invalid_reason",
                "unused_gate_ratio",
                "gate_count",
                "depth",
                "fanout_max",
                "source_run",
            ],
        ),
    )
    write_csv(out_dir / "redundancy_strong_baseline_comparison.csv", strong_redundancy, STRONG_REDUNDANCY_FIELDS)
    write_csv(out_dir / "truth_table_refit_candidate_comparison.csv", truth_refit_candidates, TRUTH_REFIT_FIELDS)
    write_csv(out_dir / "truth_refit_final_test_comparison.csv", truth_refit_final, TRUTH_REFIT_FINAL_FIELDS)
    write_csv(out_dir / "mind_gap_direct_comparison.csv", mind_gap_rows, MIND_GAP_DIRECT_FIELDS)
    write_csv(out_dir / "abc_synthesis_comparison.csv", synthesis_rows, SYNTHESIS_FIELDS)
    write_csv(
        out_dir / "hardening_method_comparison.csv",
        candidate_rows,
        union_fields(
            candidate_rows,
            [
                "method",
                "dataset",
                "seed",
                "redundancy_factor",
                "train_estimator",
                "eligible_for_method",
                "selection_policy",
                "candidate",
                "checkpoint_source",
                "val_hard_acc",
                "val_hard_loss",
                "val_soft_acc",
                "val_soft_loss",
                "val_full_gap",
                "source_run",
            ],
        ),
    )
    gap_rows = gap_taxonomy_rows(wiring_rows, seq_rows, red_rows, candidate_rows)
    gap_fields = ["axis", "source_run", "method", "item", "seed", "hard_acc", "soft_acc", "full_gap", "native_gap", "block_refit_error", "note"]
    write_csv(out_dir / "soft_hard_gap_taxonomy_current.csv", gap_rows, gap_fields)
    write_csv(out_dir / "success_checks.csv", success, SUCCESS_FIELDS)
    write_csv(out_dir / "coverage_audit.csv", coverage, COVERAGE_FIELDS)
    (out_dir / "next_stage_evidence_report.md").write_text(
        markdown_report(unified, success, coverage, mind_gap_rows, synthesis_rows, wiring_rows, seq_rows, red_rows, candidate_rows),
        encoding="utf-8",
    )
    print(f"wrote next-stage evidence tables to {out_dir}")


if __name__ == "__main__":
    main()
