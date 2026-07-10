#!/usr/bin/env python3
"""Merge LightLogic K-expansion and calibration runs."""

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


CONFIG_FIELDS = [
    "width",
    "layers",
    "epochs",
    "lr",
    "weight_decay",
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
    "data_dir",
    "download_data",
    "threshold_levels",
    "image_max_train",
    "image_max_test",
]


def config_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in CONFIG_FIELDS)


def comparison_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("dataset", ""),
        row.get("seed", ""),
        row.get("k", ""),
        *config_signature(row),
    )


def trend_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("dataset", ""),
        row.get("seed", ""),
        row.get("calibration_mode", ""),
        *config_signature(row),
    )


def merge_runs(runs_root: Path, run_names: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_name in run_names:
        config_path = runs_root / run_name / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        for row in read_csv(runs_root / run_name / "k_expansion_results.csv"):
            row["source_run"] = run_name
            if not row.get("calibration_mode"):
                row["calibration_mode"] = "fixed"
            for field in CONFIG_FIELDS:
                row.setdefault(field, str(config.get(field, row.get(field, ""))))
            rows.append(row)
    return rows


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row.get("dataset", ""), row.get("seed", "")), []).append(row)
    out = []
    for key, group in sorted(grouped.items()):
        best = max(group, key=lambda row: to_float(row.get("expanded_acc")))
        teacher_acc = to_float(best.get("teacher_acc"))
        expanded_acc = to_float(best.get("expanded_acc"))
        out.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "best_k": best.get("k", ""),
                "best_calibration_mode": best.get("calibration_mode", ""),
                "teacher_acc": best.get("teacher_acc", ""),
                "expanded_acc": best.get("expanded_acc", ""),
                "acc_gap": best.get("acc_gap", ""),
                "local_mae": best.get("local_mae", ""),
                "expanded_gate_count": best.get("expanded_gate_count", ""),
                "source_run": best.get("source_run", ""),
                "delta_expanded_vs_teacher": expanded_acc - teacher_acc,
            }
        )
    return out


def calibration_comparison(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fixed = {
        comparison_key(row): row
        for row in rows
        if row.get("calibration_mode") == "fixed"
    }
    out = []
    for row in rows:
        mode = row.get("calibration_mode", "")
        if mode == "fixed":
            continue
        base = fixed.get(comparison_key(row))
        if not base:
            continue
        delta_acc = to_float(row.get("expanded_acc")) - to_float(base.get("expanded_acc"))
        delta_gap = to_float(row.get("acc_gap")) - to_float(base.get("acc_gap"))
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "k": row.get("k", ""),
                "calibration_mode": mode,
                "fixed_acc": base.get("expanded_acc", ""),
                "calibrated_acc": row.get("expanded_acc", ""),
                "delta_acc": delta_acc,
                "fixed_gap": base.get("acc_gap", ""),
                "calibrated_gap": row.get("acc_gap", ""),
                "delta_gap": delta_gap,
                "status": "WIN" if delta_acc > 0 else ("TIE" if delta_acc == 0 else "LOSS"),
            }
        )
    return sorted(out, key=lambda row: (str(row["dataset"]), str(row["seed"]), int(str(row["k"])), str(row["calibration_mode"])))


def local_error_trend(rows: list[dict[str, str]]) -> tuple[str, str]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(trend_key(row), []).append(row)
    violations = []
    duplicates = []
    checked = 0
    for group_key, group in sorted(grouped.items()):
        seen_k = set()
        duplicate_k = set()
        for row in group:
            k_value = row.get("k", "")
            if k_value in seen_k:
                duplicate_k.add(k_value)
            seen_k.add(k_value)
        if duplicate_k:
            duplicates.append(f"{group_key[0]}/seed{group_key[1]}/{group_key[2]}:K={','.join(sorted(duplicate_k, key=int))}")
            continue
        ordered = sorted(group, key=lambda row: int(row.get("k", "0")))
        values = [to_float(row.get("local_mae")) for row in ordered]
        if not values or any(not math.isfinite(value) for value in values):
            continue
        checked += 1
        for prev, cur in zip(values, values[1:], strict=False):
            if cur > prev + 1e-9:
                violations.append(f"{group_key[0]}/seed{group_key[1]}/{group_key[2]}")
                break
    if not checked:
        return "MISSING", "no finite grouped local_mae trends"
    if duplicates:
        return "PARTIAL", f"checked_groups={checked}; duplicate_k_groups={';'.join(duplicates)}"
    if violations:
        return "PARTIAL", f"checked_groups={checked}; nonmonotonic={','.join(violations)}"
    return "PASS", f"checked_groups={checked}; all local_mae trends are non-increasing with K"


def goal_checks(rows: list[dict[str, str]], calib_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    k_values = sorted({row.get("k", "") for row in rows if row.get("k", "")}, key=lambda item: int(item))
    modes = sorted({row.get("calibration_mode", "") for row in rows if row.get("calibration_mode", "")})
    local_errors = [to_float(row.get("local_mae")) for row in rows if math.isfinite(to_float(row.get("local_mae")))]
    wins = [row for row in calib_rows if row.get("status") == "WIN"]
    trend_status, trend_evidence = local_error_trend(rows)
    return [
        {
            "goal": "Goal 3 K-gate truth-table expansion is implemented and swept",
            "status": "PASS" if rows and {"2", "4", "8", "16", "32"} <= set(k_values) else ("PARTIAL" if rows else "MISSING"),
            "evidence": f"rows={len(rows)}; k_values={','.join(k_values)}; calibration_modes={','.join(modes)}",
            "note": "Local q vs r/K error is reported separately from full-network thresholded accuracy.",
        },
        {
            "goal": "K expansion reduces local truth-table approximation error as K grows",
            "status": trend_status,
            "evidence": (
                f"{trend_evidence}; local_mae_range=[{min(local_errors):.6g},{max(local_errors):.6g}]"
                if local_errors
                else trend_evidence
            ),
            "note": "Lower local error does not imply full-network accuracy improvement after thresholding.",
        },
        {
            "goal": "Goal 4 layer-wise/per-neuron calibration improves fixed-threshold expansion",
            "status": "PASS" if wins and len(wins) == len(calib_rows) else ("PARTIAL" if wins else ("FAIL" if calib_rows else "MISSING")),
            "evidence": f"wins={len(wins)}/{len(calib_rows)}",
            "note": "Calibration is useful only where delta_acc is positive under the same dataset/seed/K.",
        },
    ]


def report(rows: list[dict[str, str]], best: list[dict[str, object]], checks: list[dict[str, object]], calib: list[dict[str, object]]) -> str:
    lines = [
        "# LightLogic K-Expansion Evidence",
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
            "## Best Expanded Rows",
            "",
            "| dataset | seed | best_k | best_calibration_mode | teacher_acc | expanded_acc | acc_gap | local_mae | expanded_gate_count | source_run |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in best:
        lines.append(
            "| {dataset} | {seed} | {best_k} | {best_calibration_mode} | {teacher_acc} | {expanded_acc} | {acc_gap} | {local_mae} | {expanded_gate_count} | {source_run} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Calibration Deltas",
            "",
            "| dataset | seed | K | mode | fixed_acc | calibrated_acc | delta_acc | status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in calib:
        lines.append(
            "| {dataset} | {seed} | {k} | {calibration_mode} | {fixed_acc} | {calibrated_acc} | {delta_acc:.6g} | {status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "- The current K-expansion student still thresholds back to one bit after each layer.",
            "- Accuracy can fall even when local truth-table error improves, so use `expanded_acc` and `acc_gap` for claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/lightlogic_k_report_v1")
    args = parser.parse_args()
    rows = merge_runs(Path(args.runs_root), args.runs)
    best = best_rows(rows)
    calib = calibration_comparison(rows)
    checks = goal_checks(rows, calib)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "k_expansion_merged_results.csv", rows)
    write_csv(out_dir / "k_expansion_best_by_dataset.csv", best)
    write_csv(out_dir / "k_expansion_calibration_comparison.csv", calib)
    write_csv(out_dir / "k_expansion_goal_checks.csv", checks, ["goal", "status", "evidence", "note"])
    (out_dir / "k_expansion_report.md").write_text(report(rows, best, checks, calib), encoding="utf-8")
    print(f"wrote K-expansion report to {out_dir}")


if __name__ == "__main__":
    main()
