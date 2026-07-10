#!/usr/bin/env python3
"""Merge LightLogic-first experiment runs into goal-level evidence tables."""

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


def key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("dataset", ""), row.get("seed", "")


def comparison_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("source_run", ""),
        row.get("dataset", ""),
        row.get("seed", ""),
        row.get("width", ""),
        row.get("layers", ""),
        row.get("epochs", ""),
        row.get("lr", ""),
        row.get("estimator", ""),
        row.get("init", ""),
    )


def best_by_hard(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        row_key = key(row)
        if row_key not in best or to_float(row.get("discrete_acc")) > to_float(best[row_key].get("discrete_acc")):
            best[row_key] = row
    return best


def best_by_hard_config(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    best: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        row_key = comparison_key(row)
        if row_key not in best or to_float(row.get("discrete_acc")) > to_float(best[row_key].get("discrete_acc")):
            best[row_key] = row
    return best


def merge_runs(runs_root: Path, run_names: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    config_fields = [
        "width",
        "layers",
        "epochs",
        "lr",
        "batch_size",
        "estimator",
        "init",
        "temp_start",
        "temp_end",
        "entropy_weight",
    ]
    for run_name in run_names:
        config_path = runs_root / run_name / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        for row in read_csv(runs_root / run_name / "lightlogic_results.csv"):
            row["source_run"] = run_name
            for field in config_fields:
                row.setdefault(field, str(config.get(field, row.get(field, ""))))
            rows.append(row)
    return rows


def numeric_fields_ok(row: dict[str, str], fields: list[str]) -> list[str]:
    bad = []
    for field in fields:
        value = to_float(row.get(field))
        if not math.isfinite(value):
            bad.append(field)
    return bad


def no_utilization_collapse(candidate: dict[str, str], baseline: dict[str, str]) -> bool:
    candidate_util = to_float(candidate.get("gate_utilization"))
    baseline_util = to_float(baseline.get("gate_utilization"))
    if not math.isfinite(candidate_util) or not math.isfinite(baseline_util):
        return False
    return candidate_util >= 0.5 and candidate_util >= baseline_util - 0.10


def goal_checks(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    light = [row for row in rows if row.get("method") == "light_iwp"]
    op = [row for row in rows if row.get("method") == "dlgn_op"]
    anneal = [row for row in rows if row.get("method") == "light_iwp_anneal"]
    st_rows = [row for row in rows if row.get("method") in {"light_iwp_st", "light_iwp_gumbel_st"}]
    required_goal0_numeric = [
        "continuous_acc",
        "discrete_acc",
        "acc_gap",
        "gate_utilization",
        "unused_gate_ratio",
        "gate_count",
        "inference_gate_count",
        "parameter_count",
        "train_time",
    ]
    malformed_light = {
        f"{row.get('source_run')}/{row.get('dataset')}/seed{row.get('seed')}": numeric_fields_ok(row, required_goal0_numeric)
        for row in light
        if numeric_fields_ok(row, required_goal0_numeric)
    }

    checks.append(
        {
            "goal": "Goal 0 LightLogic baseline",
            "status": "PASS" if light and not malformed_light else ("PARTIAL" if light else "MISSING"),
            "evidence": f"light_iwp_rows={len(light)}; malformed_rows={len(malformed_light)}",
            "note": "Rows report continuous_acc, discrete_acc, acc_gap, utilization, gate count, parameter count, and train time.",
        }
    )

    paired_param = []
    op_by_key = {comparison_key(row): row for row in op}
    for row in light:
        baseline = op_by_key.get(comparison_key(row))
        if baseline:
            light_params = to_float(row.get("parameter_count"))
            op_params = to_float(baseline.get("parameter_count"))
            ratio = light_params / op_params if op_params else math.nan
            paired_param.append((row, baseline, ratio))
    param_evidence = "; ".join(
        f"{row.get('source_run')}/{row.get('dataset')}/seed{row.get('seed')} param_ratio={ratio:.6g}"
        for row, _baseline, ratio in paired_param
        if math.isfinite(ratio)
    )
    checks.append(
        {
            "goal": "LightLogic parameter reduction vs OP/DLGN",
            "status": "PASS" if paired_param and all(abs(ratio - 0.25) < 1e-6 for _row, _baseline, ratio in paired_param) else ("PARTIAL" if paired_param else "MISSING"),
            "evidence": param_evidence,
            "note": "For two-input gates, IWP uses four truth-table parameters versus sixteen OP parameters.",
        }
    )

    anneal_evidence = []
    light_by_config = {comparison_key(row): row for row in light}
    anneal_wins = 0
    for row in anneal:
        baseline = light_by_config.get(comparison_key(row))
        if not baseline:
            continue
        delta_gap = to_float(row.get("acc_gap")) - to_float(baseline.get("acc_gap"))
        delta_hard = to_float(row.get("discrete_acc")) - to_float(baseline.get("discrete_acc"))
        delta_util = to_float(row.get("gate_utilization")) - to_float(baseline.get("gate_utilization"))
        util_ok = no_utilization_collapse(row, baseline)
        if delta_gap < 0 and delta_hard >= 0 and util_ok:
            anneal_wins += 1
        anneal_evidence.append(
            f"{row.get('source_run')}/{row.get('dataset')}/seed{row.get('seed')}/{row.get('estimator')} delta_gap={delta_gap:.6g} delta_hard={delta_hard:.6g} delta_util={delta_util:.6g} util_ok={util_ok}"
        )
    checks.append(
        {
            "goal": "Goal 1 annealing+entropy improves LightLogic gap without hurting hard accuracy",
            "status": "PASS" if anneal_evidence and anneal_wins == len(anneal_evidence) else ("PARTIAL" if anneal_evidence else "MISSING"),
            "evidence": "; ".join(anneal_evidence),
            "note": "Current sigmoid annealing evidence must be checked for collapse through gate_utilization.",
        }
    )

    best_st = best_by_hard_config(st_rows)
    st_evidence = []
    st_wins = 0
    for row_key, row in best_st.items():
        baseline = light_by_config.get(row_key)
        if not baseline:
            continue
        delta_hard = to_float(row.get("discrete_acc")) - to_float(baseline.get("discrete_acc"))
        delta_gap = to_float(row.get("acc_gap")) - to_float(baseline.get("acc_gap"))
        delta_util = to_float(row.get("gate_utilization")) - to_float(baseline.get("gate_utilization"))
        util_ok = no_utilization_collapse(row, baseline)
        if delta_hard > 0 and util_ok:
            st_wins += 1
        st_evidence.append(
            f"{row.get('source_run')}/{row.get('dataset')}/seed{row.get('seed')}/{row.get('estimator')} best_st={row.get('method')} delta_hard={delta_hard:.6g} delta_gap={delta_gap:.6g} delta_util={delta_util:.6g} util_ok={util_ok}"
        )
    checks.append(
        {
            "goal": "Goal 2 ST/Gumbel-ST improves final hard LightLogic accuracy",
            "status": "PASS" if st_evidence and st_wins == len(st_evidence) else ("PARTIAL" if st_evidence else "MISSING"),
            "evidence": "; ".join(st_evidence),
            "note": "This is same-architecture evidence against plain Light IWP, not a claim against tuned OP/DLGN.",
        }
    )

    best_lightish = best_by_hard_config([row for row in rows if row.get("method", "").startswith("light_iwp")])
    op_evidence = []
    op_wins = 0
    for row_key, row in best_lightish.items():
        baseline = op_by_key.get(row_key)
        if not baseline:
            continue
        delta_hard = to_float(row.get("discrete_acc")) - to_float(baseline.get("discrete_acc"))
        if delta_hard >= 0:
            op_wins += 1
        op_evidence.append(
            f"{row.get('source_run')}/{row.get('dataset')}/seed{row.get('seed')}/{row.get('estimator')} best_light={row.get('method')} delta_hard_vs_op={delta_hard:.6g}"
        )
    checks.append(
        {
            "goal": "Best current LightLogic-family hard accuracy is competitive with OP/DLGN",
            "status": "PASS" if op_evidence and op_wins == len(op_evidence) else ("PARTIAL" if op_evidence else "MISSING"),
            "evidence": "; ".join(op_evidence),
            "note": "A lower parameter count alone is not an accuracy win; inspect discrete_acc and acc_gap.",
        }
    )
    return checks


def report(rows: list[dict[str, str]], checks: list[dict[str, object]], run_names: list[str]) -> str:
    top = sorted(rows, key=lambda row: (-to_float(row.get("discrete_acc")), row.get("dataset", ""), row.get("method", "")))
    lines = [
        "# LightLogic Goal Evidence",
        "",
        "Merged runs: " + ", ".join(run_names),
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
            "## Top Rows By Discrete Accuracy",
            "",
            "| method | dataset | seed | continuous_acc | discrete_acc | acc_gap | gate_utilization | gate_count | parameter_count | train_time | source_run |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in top[:30]:
        lines.append(
            "| {method} | {dataset} | {seed} | {continuous_acc} | {discrete_acc} | {acc_gap} | {gate_utilization} | {gate_count} | {parameter_count} | {train_time} | {source_run} |".format(
                method=row.get("method", ""),
                dataset=row.get("dataset", ""),
                seed=row.get("seed", ""),
                continuous_acc=row.get("continuous_acc", ""),
                discrete_acc=row.get("discrete_acc", ""),
                acc_gap=row.get("acc_gap", ""),
                gate_utilization=row.get("gate_utilization", ""),
                gate_count=row.get("gate_count", ""),
                parameter_count=row.get("parameter_count", ""),
                train_time=row.get("train_time", ""),
                source_run=row.get("source_run", ""),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "- Goal 0 is satisfied when plain `light_iwp` rows exist with continuous/discrete/gap/utilization/time/gate metrics.",
            "- Goal 1/2 are only supported where the same dataset/seed comparison improves hard accuracy or gap without collapsing utilization.",
            "- OP/DLGN remains the direct 16-parameter-per-gate baseline; LightLogic parameter savings do not imply an accuracy win.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/lightlogic_report_v1")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    rows = merge_runs(runs_root, args.runs)
    checks = goal_checks(rows)
    write_csv(out_dir / "lightlogic_merged_results.csv", rows)
    write_csv(out_dir / "lightlogic_goal_checks.csv", checks, ["goal", "status", "evidence", "note"])
    (out_dir / "lightlogic_goal_report.md").write_text(report(rows, checks, args.runs), encoding="utf-8")
    print(f"wrote LightLogic report to {out_dir}")


if __name__ == "__main__":
    main()
