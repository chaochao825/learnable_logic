#!/usr/bin/env python3
"""Merge LightLogic K-expanded student-distillation runs.

The report separates three questions that otherwise get conflated:

1. Does learning deployable student thresholds improve the hard K-expanded net?
2. Does teacher-logit KL help beyond CE-only threshold tuning?
3. Does data-weighted K rounding actually change the hard truth-table choice?

Under the current unconstrained per-entry K expansion, item 3 is expected to be
degenerate: each truth-table entry can choose its own integer r_ab, so weighted
and uniform rounding choose the same r_ab whenever the empirical weight is
positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
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


def format_float(value: object) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.6g}"


CONFIG_FIELDS = [
    "width",
    "layers",
    "epochs",
    "lr",
    "batch_size",
    "eval_batch_size",
    "group_tau",
    "estimator",
    "init",
    "init_strength",
    "threshold",
    "temp_start",
    "temp_end",
    "temp_eval",
    "anneal_train",
    "distill_epochs",
    "student_lr",
    "student_temp",
    "adapter_modes",
    "adapter_init",
    "adapter_anchor_weight",
    "data_dir",
    "download_data",
    "image_max_train",
    "image_max_test",
]


def merge_distill_runs(runs_root: Path, run_names: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_name in run_names:
        config_path = runs_root / run_name / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        for row in read_csv(runs_root / run_name / "distill_results.csv"):
            row["source_run"] = run_name
            row.setdefault("adapter_mode", "none")
            row.setdefault("final_local_mae", row.get("local_mae", ""))
            row.setdefault("final_weighted_local_mse", row.get("weighted_local_mse", ""))
            row.setdefault("final_uniform_local_mse", row.get("uniform_local_mse", ""))
            row.setdefault("truth_table_changed_ratio", "0.0")
            row.setdefault("truth_table_l1_from_initial", "0.0")
            for field in CONFIG_FIELDS:
                row.setdefault(field, str(config.get(field, row.get(field, ""))))
            rows.append(row)
    return rows


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("seed", ""),
        row.get("k", ""),
        row.get("init_mode", ""),
        row.get("adapter_mode", "none") or "none",
    )


def dataset_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("dataset", ""), row.get("seed", "")


def best_by_group(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    out: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        best = max(group, key=lambda row: to_float(row.get("distilled_hard_acc")))
        initial_acc = to_float(best.get("initial_hard_acc"))
        best_acc = to_float(best.get("distilled_hard_acc"))
        ce_rows = [row for row in group if abs(to_float(row.get("alpha"))) < 1e-12]
        kl_rows = [row for row in group if to_float(row.get("alpha")) > 0]
        best_ce = max(ce_rows, key=lambda row: to_float(row.get("distilled_hard_acc"))) if ce_rows else None
        best_kl = max(kl_rows, key=lambda row: to_float(row.get("distilled_hard_acc"))) if kl_rows else None
        best_ce_acc = to_float(best_ce.get("distilled_hard_acc")) if best_ce else math.nan
        best_kl_acc = to_float(best_kl.get("distilled_hard_acc")) if best_kl else math.nan
        out.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "k": key[2],
                "init_mode": key[3],
                "adapter_mode": key[4],
                "teacher_acc": best.get("teacher_acc", ""),
                "initial_hard_acc": best.get("initial_hard_acc", ""),
                "best_distilled_hard_acc": best.get("distilled_hard_acc", ""),
                "best_delta_vs_initial": best_acc - initial_acc,
                "best_alpha": best.get("alpha", ""),
                "best_tau": best.get("distill_tau", ""),
                "best_hard_gap_vs_teacher": best.get("hard_gap_vs_teacher", ""),
                "best_ce_hard_acc": best_ce.get("distilled_hard_acc", "") if best_ce else "",
                "best_ce_alpha": best_ce.get("alpha", "") if best_ce else "",
                "best_ce_tau": best_ce.get("distill_tau", "") if best_ce else "",
                "best_kl_hard_acc": best_kl.get("distilled_hard_acc", "") if best_kl else "",
                "best_kl_alpha": best_kl.get("alpha", "") if best_kl else "",
                "best_kl_tau": best_kl.get("distill_tau", "") if best_kl else "",
                "kl_delta_vs_ce": best_kl_acc - best_ce_acc,
                "weighted_rounding_changed_ratio": best.get("weighted_rounding_changed_ratio", ""),
                "local_mae": best.get("local_mae", ""),
                "final_local_mae": best.get("final_local_mae", ""),
                "truth_table_changed_ratio": best.get("truth_table_changed_ratio", ""),
                "truth_table_l1_from_initial": best.get("truth_table_l1_from_initial", ""),
                "expanded_gate_count": best.get("expanded_gate_count", ""),
                "train_time": best.get("train_time", ""),
                "distill_time": best.get("distill_time", ""),
                "source_run": best.get("source_run", ""),
            }
        )
    return out


def best_by_dataset(group_best: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in group_best:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("seed", ""))), []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        best = max(group, key=lambda row: to_float(row.get("best_distilled_hard_acc")))
        out.append(dict(best))
    return out


def load_k_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def compare_against_k(group_best: list[dict[str, object]], k_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_exact = {
        (row.get("dataset", ""), row.get("seed", ""), row.get("k", ""), row.get("calibration_mode", "")): row
        for row in k_rows
    }
    best_dataset: dict[tuple[str, str], dict[str, str]] = {}
    for row in k_rows:
        key = (row.get("dataset", ""), row.get("seed", ""))
        if key not in best_dataset or to_float(row.get("expanded_acc")) > to_float(best_dataset[key].get("expanded_acc")):
            best_dataset[key] = row

    out = []
    for row in group_best:
        exact = by_exact.get(
            (
                str(row.get("dataset", "")),
                str(row.get("seed", "")),
                str(row.get("k", "")),
                str(row.get("init_mode", "")),
            )
        )
        dataset_best = best_dataset.get((str(row.get("dataset", "")), str(row.get("seed", ""))))
        distill_acc = to_float(row.get("best_distilled_hard_acc"))
        exact_delta: object = ""
        dataset_delta: object = ""
        if exact:
            exact_delta = distill_acc - to_float(exact.get("expanded_acc"))
        if dataset_best:
            dataset_delta = distill_acc - to_float(dataset_best.get("expanded_acc"))
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "k": row.get("k", ""),
                "init_mode": row.get("init_mode", ""),
                "adapter_mode": row.get("adapter_mode", ""),
                "distilled_hard_acc": row.get("best_distilled_hard_acc", ""),
                "k_exact_expanded_acc": exact.get("expanded_acc", "") if exact else "",
                "delta_vs_k_exact": exact_delta,
                "k_best_mode": dataset_best.get("calibration_mode", "") if dataset_best else "",
                "k_best_k": dataset_best.get("k", "") if dataset_best else "",
                "k_best_expanded_acc": dataset_best.get("expanded_acc", "") if dataset_best else "",
                "delta_vs_k_dataset_best": dataset_delta,
            }
        )
    return out


def compare_against_light(best_dataset: list[dict[str, object]], light_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    best_method: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in light_rows:
        key = (row.get("dataset", ""), row.get("seed", ""), row.get("method", ""))
        if key not in best_method or to_float(row.get("discrete_acc")) > to_float(best_method[key].get("discrete_acc")):
            best_method[key] = row

    methods = sorted({key[2] for key in best_method})
    out = []
    for row in best_dataset:
        base_key = (str(row.get("dataset", "")), str(row.get("seed", "")))
        distill_acc = to_float(row.get("best_distilled_hard_acc"))
        out_row: dict[str, object] = {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "distill_k": row.get("k", ""),
            "distill_init": row.get("init_mode", ""),
            "distill_adapter": row.get("adapter_mode", ""),
            "distilled_hard_acc": row.get("best_distilled_hard_acc", ""),
        }
        for method in methods:
            baseline = best_method.get((*base_key, method))
            if not baseline:
                continue
            out_row[f"{method}_hard_acc"] = baseline.get("discrete_acc", "")
            out_row[f"delta_vs_{method}"] = distill_acc - to_float(baseline.get("discrete_acc"))
        out.append(out_row)
    return out


def adapter_comparison(group_best: list[dict[str, object]]) -> list[dict[str, object]]:
    none_by_key = {
        (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("k", "")),
            str(row.get("init_mode", "")),
        ): row
        for row in group_best
        if str(row.get("adapter_mode", "none")) == "none"
    }
    out = []
    for row in group_best:
        adapter_mode = str(row.get("adapter_mode", "none"))
        if adapter_mode == "none":
            continue
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("k", "")),
            str(row.get("init_mode", "")),
        )
        baseline = none_by_key.get(key)
        adapter_acc = to_float(row.get("best_distilled_hard_acc"))
        baseline_acc = to_float(baseline.get("best_distilled_hard_acc")) if baseline else math.nan
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "k": row.get("k", ""),
                "init_mode": row.get("init_mode", ""),
                "adapter_mode": adapter_mode,
                "none_best_hard_acc": baseline.get("best_distilled_hard_acc", "") if baseline else "",
                "adapter_best_hard_acc": row.get("best_distilled_hard_acc", ""),
                "delta_vs_none": adapter_acc - baseline_acc if baseline else "",
                "adapter_best_alpha": row.get("best_alpha", ""),
                "adapter_best_tau": row.get("best_tau", ""),
                "truth_table_changed_ratio": row.get("truth_table_changed_ratio", ""),
                "truth_table_l1_from_initial": row.get("truth_table_l1_from_initial", ""),
                "final_local_mae": row.get("final_local_mae", ""),
            }
        )
    return out


def goal_checks(
    rows: list[dict[str, str]],
    group_best: list[dict[str, object]],
    k_compare: list[dict[str, object]],
    adapter_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    alpha_numbers = sorted({to_float(row.get("alpha")) for row in rows if math.isfinite(to_float(row.get("alpha")))})
    tau_numbers = sorted({to_float(row.get("distill_tau")) for row in rows if math.isfinite(to_float(row.get("distill_tau")))})
    alpha_values = [f"{value:.6g}" for value in alpha_numbers]
    tau_values = [f"{value:.6g}" for value in tau_numbers]
    k_values = sorted({row.get("k", "") for row in rows if row.get("k", "")}, key=lambda value: int(float(value)))
    init_modes = sorted({row.get("init_mode", "") for row in rows if row.get("init_mode", "")})
    adapter_modes = sorted({row.get("adapter_mode", "none") or "none" for row in rows})
    dataset_seed_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        dataset_seed_groups.setdefault((row.get("dataset", ""), row.get("seed", "")), []).append(row)
    full_sweep_dataset_seeds = 0
    confirmation_dataset_seeds = 0
    expected_group_keys = {("2", "fixed"), ("2", "per_neuron"), ("4", "fixed"), ("4", "per_neuron")}
    expected_alpha_tau = {(0.0, 1.0), (0.0, 2.0), (0.5, 1.0), (0.5, 2.0), (2.0, 1.0), (2.0, 2.0)}
    for group_rows in dataset_seed_groups.values():
        group_keys = {(row.get("k", ""), row.get("init_mode", "")) for row in group_rows}
        if expected_group_keys <= group_keys:
            complete = True
            for k_value, init_mode in expected_group_keys:
                combos = {
                    (to_float(row.get("alpha")), to_float(row.get("distill_tau")))
                    for row in group_rows
                    if row.get("k", "") == k_value and row.get("init_mode", "") == init_mode
                }
                if not expected_alpha_tau <= combos:
                    complete = False
                    break
            if complete:
                full_sweep_dataset_seeds += 1
        if {("4", "fixed")} <= group_keys:
            confirmation_dataset_seeds += 1
    improved = [row for row in group_best if to_float(row.get("best_delta_vs_initial")) > 0]
    worsened = [row for row in group_best if to_float(row.get("best_delta_vs_initial")) < 0]
    kl_wins = [row for row in group_best if to_float(row.get("kl_delta_vs_ce")) > 0]
    kl_losses = [row for row in group_best if to_float(row.get("kl_delta_vs_ce")) < 0]
    weighted_values = [
        to_float(row.get("weighted_rounding_changed_ratio"))
        for row in rows
        if math.isfinite(to_float(row.get("weighted_rounding_changed_ratio")))
    ]
    exact_compared = [row for row in k_compare if math.isfinite(to_float(row.get("delta_vs_k_exact")))]
    dataset_compared = [row for row in k_compare if math.isfinite(to_float(row.get("delta_vs_k_dataset_best")))]
    exact_k_wins = [row for row in exact_compared if to_float(row.get("delta_vs_k_exact")) > 0]
    dataset_k_wins = [row for row in dataset_compared if to_float(row.get("delta_vs_k_dataset_best")) > 0]
    adapter_compared = [row for row in adapter_rows if math.isfinite(to_float(row.get("delta_vs_none")))]
    adapter_wins = [row for row in adapter_compared if to_float(row.get("delta_vs_none")) > 0]
    adapter_losses = [row for row in adapter_compared if to_float(row.get("delta_vs_none")) < 0]
    changed_adapter_rows = [
        row
        for row in adapter_compared
        if to_float(row.get("truth_table_changed_ratio")) > 0
    ]
    return [
        {
            "goal": "Goal 6 seed0 sweep plus seed1/2 confirmation is implemented",
            "status": "PASS" if rows and full_sweep_dataset_seeds > 0 else ("PARTIAL" if rows else "MISSING"),
            "evidence": f"rows={len(rows)}; dataset_seed_count={len(dataset_seed_groups)}; full_sweep_dataset_seeds={full_sweep_dataset_seeds}; k4_fixed_confirmation_dataset_seeds={confirmation_dataset_seeds}; k_values={','.join(k_values)}; init_modes={','.join(init_modes)}; adapter_modes={','.join(adapter_modes)}; alphas={','.join(alpha_values)}; taus={','.join(tau_values)}",
            "note": "Teacher is continuous LightLogic; student is K-expanded hard network with learnable thresholds. Only dataset/seeds counted as full_sweep have the full K/init/alpha/tau grid.",
        },
        {
            "goal": "Goal 6+ truth-table adapter improves over threshold-only distillation",
            "status": "PASS" if adapter_wins and not adapter_losses else ("PARTIAL" if adapter_wins else ("FAIL" if adapter_compared else "MISSING")),
            "evidence": f"adapter_wins={len(adapter_wins)}/{len(adapter_compared)}; adapter_losses={len(adapter_losses)}/{len(adapter_compared)}; changed_truth_table_rows={len(changed_adapter_rows)}/{len(adapter_compared)}",
            "note": "Compared at the same dataset/seed/K/init against the best alpha/tau row with adapter_mode=none.",
        },
        {
            "goal": "Distillation/tuning improves deployable hard student accuracy",
            "status": "PASS" if improved and not worsened else ("PARTIAL" if improved else ("FAIL" if group_best else "MISSING")),
            "evidence": f"improved_groups={len(improved)}/{len(group_best)}; worsened_groups={len(worsened)}/{len(group_best)}",
            "note": "Best row per dataset/seed/K/init is compared against the same group's initial hard student.",
        },
        {
            "goal": "Teacher-KL improves beyond CE-only threshold tuning",
            "status": "PASS" if kl_wins and not kl_losses else ("PARTIAL" if kl_wins else ("FAIL" if group_best else "MISSING")),
            "evidence": f"kl_wins={len(kl_wins)}/{len(group_best)}; kl_losses={len(kl_losses)}/{len(group_best)}",
            "note": "Alpha=0 rows are CE-only; alpha>0 rows use teacher-logit KL.",
        },
        {
            "goal": "Goal 5 data-weighted K rounding changes q to r/K under current constraints",
            "status": "DEGENERATE" if weighted_values and max(abs(value) for value in weighted_values) == 0.0 else ("PARTIAL" if weighted_values else "MISSING"),
            "evidence": (
                f"weighted_rounding_changed_ratio_range=[{min(weighted_values):.6g},{max(weighted_values):.6g}]"
                if weighted_values
                else "no weighted_rounding_changed_ratio values"
            ),
            "note": "With independent per-entry integer choices, data weights do not change nearest rounding; Goal 5 needs coupled or task-aware constraints.",
        },
        {
            "goal": "Best distilled student beats the matching K-expanded/calibrated hard baseline",
            "status": (
                "PASS"
                if exact_k_wins and len(exact_k_wins) == len(exact_compared) and len(exact_compared) == len(k_compare)
                else ("PARTIAL" if exact_k_wins else ("FAIL" if exact_compared else "MISSING"))
            ),
            "evidence": f"exact_k_wins={len(exact_k_wins)}/{len(exact_compared)}; dataset_best_wins={len(dataset_k_wins)}/{len(dataset_compared)}; missing_exact={len(k_compare)-len(exact_compared)}",
            "note": "Exact comparison matches dataset/seed/K/init; dataset-best comparison is stricter.",
        },
    ]


def report(
    checks: list[dict[str, object]],
    best_dataset: list[dict[str, object]],
    group_best: list[dict[str, object]],
    k_compare: list[dict[str, object]],
    light_compare: list[dict[str, object]],
    adapter_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# LightLogic Distilled K-Expansion Evidence",
        "",
        "## Goal Checks",
        "",
        "| goal | status | evidence | note |",
        "| --- | --- | --- | --- |",
    ]
    for row in checks:
        lines.append(f"| {row.get('goal')} | {row.get('status')} | {row.get('evidence')} | {row.get('note')} |")

    lines.extend(
        [
            "",
            "## Best Per Dataset",
            "",
            "| dataset | seed | K | init | adapter | teacher_acc | initial_hard_acc | distilled_hard_acc | delta_vs_initial | alpha | tau | hard_gap_vs_teacher | tt_changed | expanded_gate_count |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in best_dataset:
        lines.append(
            "| {dataset} | {seed} | {k} | {init_mode} | {adapter_mode} | {teacher_acc} | {initial_hard_acc} | {best_distilled_hard_acc} | {delta} | {best_alpha} | {best_tau} | {best_hard_gap_vs_teacher} | {tt_changed} | {expanded_gate_count} |".format(
                dataset=row.get("dataset", ""),
                seed=row.get("seed", ""),
                k=row.get("k", ""),
                init_mode=row.get("init_mode", ""),
                adapter_mode=row.get("adapter_mode", ""),
                teacher_acc=row.get("teacher_acc", ""),
                initial_hard_acc=row.get("initial_hard_acc", ""),
                best_distilled_hard_acc=row.get("best_distilled_hard_acc", ""),
                delta=format_float(row.get("best_delta_vs_initial")),
                best_alpha=row.get("best_alpha", ""),
                best_tau=row.get("best_tau", ""),
                best_hard_gap_vs_teacher=row.get("best_hard_gap_vs_teacher", ""),
                tt_changed=format_float(row.get("truth_table_changed_ratio")),
                expanded_gate_count=row.get("expanded_gate_count", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Best Per K/Init",
            "",
            "| dataset | seed | K | init | adapter | initial_hard_acc | best_distilled_hard_acc | delta | best_alpha | best_tau | CE_best | KL_best | KL_delta_vs_CE | tt_changed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in group_best:
        lines.append(
            "| {dataset} | {seed} | {k} | {init_mode} | {adapter_mode} | {initial_hard_acc} | {best_distilled_hard_acc} | {delta} | {best_alpha} | {best_tau} | {ce} | {kl} | {kl_delta} | {tt_changed} |".format(
                dataset=row.get("dataset", ""),
                seed=row.get("seed", ""),
                k=row.get("k", ""),
                init_mode=row.get("init_mode", ""),
                adapter_mode=row.get("adapter_mode", ""),
                initial_hard_acc=row.get("initial_hard_acc", ""),
                best_distilled_hard_acc=row.get("best_distilled_hard_acc", ""),
                delta=format_float(row.get("best_delta_vs_initial")),
                best_alpha=row.get("best_alpha", ""),
                best_tau=row.get("best_tau", ""),
                ce=row.get("best_ce_hard_acc", ""),
                kl=row.get("best_kl_hard_acc", ""),
                kl_delta=format_float(row.get("kl_delta_vs_ce")),
                tt_changed=format_float(row.get("truth_table_changed_ratio")),
            )
        )

    if adapter_rows:
        lines.extend(
            [
                "",
                "## Adapter Comparison",
                "",
                "| dataset | seed | K | init | adapter | none_best | adapter_best | delta_vs_none | alpha | tau | tt_changed | tt_l1 | final_local_mae |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in adapter_rows:
            lines.append(
                "| {dataset} | {seed} | {k} | {init_mode} | {adapter_mode} | {none_best_hard_acc} | {adapter_best_hard_acc} | {delta} | {adapter_best_alpha} | {adapter_best_tau} | {tt_changed} | {tt_l1} | {final_local_mae} |".format(
                    dataset=row.get("dataset", ""),
                    seed=row.get("seed", ""),
                    k=row.get("k", ""),
                    init_mode=row.get("init_mode", ""),
                    adapter_mode=row.get("adapter_mode", ""),
                    none_best_hard_acc=row.get("none_best_hard_acc", ""),
                    adapter_best_hard_acc=row.get("adapter_best_hard_acc", ""),
                    delta=format_float(row.get("delta_vs_none")),
                    adapter_best_alpha=row.get("adapter_best_alpha", ""),
                    adapter_best_tau=row.get("adapter_best_tau", ""),
                    tt_changed=format_float(row.get("truth_table_changed_ratio")),
                    tt_l1=format_float(row.get("truth_table_l1_from_initial")),
                    final_local_mae=row.get("final_local_mae", ""),
                )
            )

    lines.extend(
        [
            "",
            "## K-Baseline Comparison",
            "",
            "| dataset | seed | K | init | adapter | distilled_hard_acc | K_exact_acc | delta_exact | K_dataset_best | delta_dataset_best |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in k_compare:
        k_best = ""
        if row.get("k_best_expanded_acc", "") != "":
            k_best = f"{row.get('k_best_k', '')}/{row.get('k_best_mode', '')}:{row.get('k_best_expanded_acc', '')}"
        lines.append(
            "| {dataset} | {seed} | {k} | {init_mode} | {adapter_mode} | {distilled_hard_acc} | {k_exact_expanded_acc} | {delta_exact} | {best} | {delta_best} |".format(
                dataset=row.get("dataset", ""),
                seed=row.get("seed", ""),
                k=row.get("k", ""),
                init_mode=row.get("init_mode", ""),
                adapter_mode=row.get("adapter_mode", ""),
                distilled_hard_acc=row.get("distilled_hard_acc", ""),
                k_exact_expanded_acc=row.get("k_exact_expanded_acc", ""),
                delta_exact=format_float(row.get("delta_vs_k_exact")),
                best=k_best,
                delta_best=format_float(row.get("delta_vs_k_dataset_best")),
            )
        )

    if light_compare:
        method_fields = sorted(
            {
                field
                for row in light_compare
                for field in row
                if (field.endswith("_hard_acc") and field != "distilled_hard_acc") or field.startswith("delta_vs_")
            }
        )
        acc_fields = [field for field in method_fields if field.endswith("_hard_acc")]
        delta_fields = [field for field in method_fields if field.startswith("delta_vs_")]
        method_fields = sorted(acc_fields) + sorted(delta_fields)
        method_fields = list(dict.fromkeys(method_fields))
        coverage = {
            field: sum(1 for row in light_compare if row.get(field, "") != "")
            for field in acc_fields
        }
        lines.extend(
            [
                "",
                "## LightLogic/OP Baseline Comparison",
                "",
                "Baseline coverage: "
                + "; ".join(f"{field}={count}/{len(light_compare)}" for field, count in sorted(coverage.items())),
                "",
                "| dataset | seed | distilled_hard_acc | " + " | ".join(method_fields) + " |",
                "| --- | --- | --- | " + " | ".join("---" for _ in method_fields) + " |",
            ]
        )
        for row in light_compare:
            values = [format_float(row.get(field)) for field in method_fields]
            lines.append(
                f"| {row.get('dataset', '')} | {row.get('seed', '')} | {row.get('distilled_hard_acc', '')} | "
                + " | ".join(values)
                + " |"
            )

    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "- `alpha=0` is CE-only student threshold tuning; only `alpha>0` uses teacher-logit KL.",
            "- `adapter_mode=none` keeps K-expanded truth tables fixed and only learns thresholds; `truth_table_st` learns quantized truth-table entries plus thresholds.",
            "- `weighted_rounding_changed_ratio=0` means Goal 5 is not an effective intervention under the current independent rounding formulation.",
            "- The full K/init/alpha/tau sweep is seed-0 unless more full-sweep runs are supplied; seed 1/2 rows, when present, are scoped K=4/fixed confirmation runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--k-merged", default="runs/lightlogic_k_report_goal34_v4/k_expansion_merged_results.csv")
    parser.add_argument("--lightlogic-merged", default="runs/lightlogic_report_goal012_v5/lightlogic_merged_results.csv")
    parser.add_argument("--out-dir", default="runs/lightlogic_distill_report_v1")
    args = parser.parse_args()

    rows = merge_distill_runs(Path(args.runs_root), args.runs)
    group_best = best_by_group(rows)
    best_dataset = best_by_dataset(group_best)
    k_compare = compare_against_k(group_best, load_k_rows(Path(args.k_merged)))
    light_compare = compare_against_light(best_dataset, read_csv(Path(args.lightlogic_merged)))
    adapter_rows = adapter_comparison(group_best)
    checks = goal_checks(rows, group_best, k_compare, adapter_rows)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "distill_merged_results.csv", rows)
    write_csv(out_dir / "distill_best_by_group.csv", group_best)
    write_csv(out_dir / "distill_best_by_dataset.csv", best_dataset)
    write_csv(out_dir / "distill_adapter_comparison.csv", adapter_rows)
    write_csv(out_dir / "distill_k_baseline_comparison.csv", k_compare)
    write_csv(out_dir / "distill_lightlogic_baseline_comparison.csv", light_compare)
    write_csv(out_dir / "distill_goal_checks.csv", checks, ["goal", "status", "evidence", "note"])
    (out_dir / "distill_report.md").write_text(
        report(checks, best_dataset, group_best, k_compare, light_compare, adapter_rows),
        encoding="utf-8",
    )
    print(f"wrote distillation report to {out_dir}")


if __name__ == "__main__":
    main()
