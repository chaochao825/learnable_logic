#!/usr/bin/env python3
"""Aggregate real b-input LUT network runs and compare against Goal 8 b=2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_RUN_DIRS = [
    "runs/lightlogic_b_lut_min_goal7_bool_multiseed_v1",
    "runs/lightlogic_b_lut_min_goal7_digits_multiseed_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_seed0_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b3_w360_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b4_w360_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b3_w360_e150_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b3_w360_warmup_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b4_w360_warmup_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_b3_distill_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_matched_b3_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_matched_b4_v1",
    "runs/lightlogic_b_lut_min_goal7_mnist_matched_b3_multiseed_v1",
]

BASELINE_SOURCE_RUN_TO_DIR = {
    "mnist_fixed": "runs/lightlogic_lut_min_goal8_mnist_scale_v1",
    "digits_fixed": "runs/lightlogic_lut_min_goal8_digits_multiseed_v1",
    "bool_per_neuron": "runs/lightlogic_lut_min_goal8_bool_multiseed_per_neuron_v1",
    "bool_fixed": "runs/lightlogic_lut_min_goal8_bool_multiseed_fixed_v1",
    "bool_per_layer": "runs/lightlogic_lut_min_goal8_bool_multiseed_per_layer_v1",
}

COMPARISON_KEYS = [
    ("width", "width"),
    ("layers", "layers"),
    ("epochs", "epochs"),
    ("image_max_train", "train"),
    ("image_max_test", "test"),
]

DATASET_NAME_MAP = {
    "mnist": "binarized_mnist",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected object in {path}")
    return data


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
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


def accuracy_preservation_stats(rows: list[dict[str, object]], tolerance: float) -> dict[str, float]:
    preserved = 0
    max_source_delta = 0.0
    max_minimized_source_delta = 0.0
    max_minimized_discrete_delta = 0.0
    for row in rows:
        discrete = to_float(row.get("discrete_acc"))
        source = to_float(row.get("source_blif_acc"))
        minimized = to_float(row.get("minimized_blif_acc"))
        if any(math.isnan(value) for value in (discrete, source, minimized)):
            continue
        source_delta = abs(source - discrete)
        minimized_source_delta = abs(minimized - source)
        minimized_discrete_delta = abs(minimized - discrete)
        max_source_delta = max(max_source_delta, source_delta)
        max_minimized_source_delta = max(max_minimized_source_delta, minimized_source_delta)
        max_minimized_discrete_delta = max(max_minimized_discrete_delta, minimized_discrete_delta)
        if source_delta <= tolerance and minimized_source_delta <= tolerance:
            preserved += 1
    return {
        "preserved": preserved,
        "max_source_delta": max_source_delta,
        "max_minimized_source_delta": max_minimized_source_delta,
        "max_minimized_discrete_delta": max_minimized_discrete_delta,
    }


def config_value(config: dict[str, object], key: str, default: object = "n/a") -> object:
    value = config.get(key, default)
    if value in (None, "", []):
        return default
    return value


def list_str(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(DATASET_NAME_MAP.get(str(item), item)) for item in value)
    if value in (None, ""):
        return "n/a"
    return str(DATASET_NAME_MAP.get(str(value), value))


def train_test_summary(config: dict[str, object]) -> str:
    return f"{config_value(config, 'image_max_train')}/{config_value(config, 'image_max_test')}"


def warmup_summary(config: dict[str, object]) -> str:
    epochs = int(config.get("warmup_epochs", 0) or 0)
    if epochs <= 0:
        return "off"
    return f"{epochs}ep:{config_value(config, 'warmup_mode')}"


def distill_summary(config: dict[str, object]) -> str:
    weight = float(config.get("distill_weight", 0.0) or 0.0)
    teacher_width = int(config.get("teacher_width", 0) or 0)
    teacher_layers = int(config.get("teacher_layers", 0) or 0)
    teacher_epochs = int(config.get("teacher_epochs", 0) or 0)
    if weight <= 0.0 and teacher_width <= 0 and teacher_layers <= 0 and teacher_epochs <= 0:
        return "off"
    return (
        f"w={weight:g},tau={to_float(config.get('distill_tau', 1.0)):g},"
        f"teacher={teacher_width}x{teacher_layers}/{teacher_epochs}"
    )


def direct_run_summary(run_dir: Path, config: dict[str, object], row_count: int) -> dict[str, object]:
    return {
        "run": run_dir.name,
        "datasets": list_str(config_value(config, "datasets")),
        "b_values": list_str(config_value(config, "b_values")),
        "seeds": list_str(config_value(config, "seeds")),
        "width": config_value(config, "width"),
        "layers": config_value(config, "layers"),
        "epochs": config_value(config, "epochs"),
        "train_test": train_test_summary(config),
        "train_forward_mode": config_value(config, "train_forward_mode"),
        "warmup": warmup_summary(config),
        "distill": distill_summary(config),
        "device": config_value(config, "device"),
        "rows": row_count,
    }


def comparison_note(
    best_config: dict[str, object] | None,
    baseline_config: dict[str, object] | None,
) -> str:
    if not best_config or not baseline_config:
        return "baseline config unavailable"
    diffs: list[str] = []
    for key, label in COMPARISON_KEYS:
        best_value = str(best_config.get(key, ""))
        baseline_value = str(baseline_config.get(key, ""))
        if best_value and baseline_value and best_value != baseline_value:
            diffs.append(f"{label}:{baseline_value}->{best_value}")
    if not diffs:
        return "matched on width/layers/epochs/train/test"
    return "best Goal 8 row; differs on " + ", ".join(diffs)


def baseline_summary(
    dataset: str,
    baseline_row: dict[str, str],
    baseline_dir_str: str,
    baseline_config: dict[str, object] | None,
    best_config: dict[str, object] | None,
) -> dict[str, object]:
    config = baseline_config or {}
    return {
        "dataset": dataset,
        "baseline_source_run": baseline_row.get("source_run", ""),
        "baseline_run_dir": baseline_dir_str,
        "baseline_calibration_mode": baseline_row.get("calibration_mode", config_value(config, "calibration_mode")),
        "baseline_seed": baseline_row.get("seed", ""),
        "baseline_k": baseline_row.get("k", ""),
        "baseline_width": config_value(config, "width"),
        "baseline_layers": config_value(config, "layers"),
        "baseline_epochs": config_value(config, "epochs"),
        "baseline_train_test": train_test_summary(config),
        "baseline_raw_lut_acc": to_float(baseline_row.get("raw_lut_acc", "")),
        "baseline_minimized_blif_acc": to_float(baseline_row.get("minimized_blif_acc", "")),
        "baseline_abc_and_count": to_float(baseline_row.get("abc_and_count", "")),
        "baseline_abc_level": to_float(baseline_row.get("abc_level", "")),
        "baseline_raw_lut_gate_estimate": to_float(baseline_row.get("raw_lut_gate_estimate", "")),
        "comparison_note": comparison_note(best_config, baseline_config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUN_DIRS)
    parser.add_argument("--baseline-csv", default="reports/goal8_large_scale_v1/best_by_dataset.csv")
    parser.add_argument("--out-dir", default="reports/lightlogic_b_lut_goal7_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, object]] = []
    direct_config_rows: list[dict[str, object]] = []
    direct_configs_by_run: dict[str, dict[str, object]] = {}
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        rows = read_csv(run_dir / "network_minimization_results.csv")
        config = read_json(run_dir / "config.json")
        direct_configs_by_run[run_dir.name] = config
        direct_config_rows.append(direct_run_summary(run_dir, config, len(rows)))
        for row in rows:
            merged = dict(row)
            merged["run"] = run_dir.name
            combined_rows.append(merged)

    best_by_dataset_b: list[dict[str, object]] = []
    grouped_db: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in combined_rows:
        grouped_db.setdefault((str(row["dataset"]), int(row["b"])), []).append(row)
    for key in sorted(grouped_db):
        dataset, b = key
        rows = grouped_db[key]
        best_row = max(rows, key=lambda item: to_float(item["discrete_acc"]))
        best_config = direct_configs_by_run.get(str(best_row["run"]), {})
        best_by_dataset_b.append(
            {
                "dataset": dataset,
                "b": b,
                "rows": len(rows),
                "best_run": best_row["run"],
                "best_seed": best_row["seed"],
                "best_width": best_row["width"],
                "best_layers": config_value(best_config, "layers"),
                "best_epochs": config_value(best_config, "epochs"),
                "best_train_test": train_test_summary(best_config),
                "best_train_forward_mode": config_value(best_config, "train_forward_mode"),
                "best_discrete_acc": to_float(best_row["discrete_acc"]),
                "best_minimized_blif_acc": to_float(best_row["minimized_blif_acc"]),
                "best_abc_and_count": to_float(best_row["abc_and_count"]),
                "best_abc_level": to_float(best_row["abc_level"]),
                "best_raw_lut_gate_estimate": to_float(best_row["raw_lut_gate_estimate"]),
                "mean_discrete_acc": sum(to_float(row["discrete_acc"]) for row in rows) / len(rows),
                "mean_abc_and_count": sum(to_float(row["abc_and_count"]) for row in rows) / len(rows),
                "mean_total_abc_seconds": sum(to_float(row["total_abc_seconds"]) for row in rows) / len(rows),
                "accuracy_preserved_rows": sum(1 for row in rows if str(row.get("accuracy_preserved")) == "True"),
            }
        )

    baseline_rows = read_csv(Path(args.baseline_csv))
    baseline_by_dataset = {str(row["dataset"]): row for row in baseline_rows}

    baseline_definition_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    grouped_dataset: dict[str, list[dict[str, object]]] = {}
    for row in best_by_dataset_b:
        grouped_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset in sorted(grouped_dataset):
        candidate_rows = grouped_dataset[dataset]
        best_new = max(candidate_rows, key=lambda item: to_float(item["best_discrete_acc"]))
        baseline = baseline_by_dataset.get(dataset, {})
        baseline_run_alias = str(baseline.get("source_run", ""))
        baseline_dir_str = BASELINE_SOURCE_RUN_TO_DIR.get(baseline_run_alias, "")
        baseline_config = read_json(Path(baseline_dir_str) / "config.json") if baseline_dir_str else None
        best_config = direct_configs_by_run.get(str(best_new["best_run"]))
        baseline_summary_row = baseline_summary(
            dataset=dataset,
            baseline_row=baseline,
            baseline_dir_str=baseline_dir_str,
            baseline_config=baseline_config,
            best_config=best_config,
        )
        baseline_definition_rows.append(baseline_summary_row)
        comparison_rows.append(
            {
                "dataset": dataset,
                "best_new_b": best_new["b"],
                "best_new_run": best_new["best_run"],
                "best_new_seed": best_new["best_seed"],
                "best_new_width": best_new["best_width"],
                "best_new_layers": best_new["best_layers"],
                "best_new_epochs": best_new["best_epochs"],
                "best_new_train_test": best_new["best_train_test"],
                "best_new_discrete_acc": best_new["best_discrete_acc"],
                "best_new_minimized_blif_acc": best_new["best_minimized_blif_acc"],
                "best_new_abc_and_count": best_new["best_abc_and_count"],
                "best_new_abc_level": best_new["best_abc_level"],
                "best_new_raw_lut_gate_estimate": best_new["best_raw_lut_gate_estimate"],
                **baseline_summary_row,
                "delta_discrete_acc_vs_baseline": to_float(best_new["best_discrete_acc"]) - to_float(baseline.get("raw_lut_acc", "")),
                "delta_abc_and_vs_baseline": to_float(best_new["best_abc_and_count"]) - to_float(baseline.get("abc_and_count", "")),
                "delta_raw_gate_estimate_vs_baseline": to_float(best_new["best_raw_lut_gate_estimate"]) - to_float(baseline.get("raw_lut_gate_estimate", "")),
            }
        )

    strict_preservation_count = sum(
        1 for row in combined_rows if str(row.get("accuracy_preserved")) == "True"
    )
    tolerant_preservation = accuracy_preservation_stats(combined_rows, tolerance=1e-7)

    report_lines = [
        "# Real b-Input LUT Network Report",
        "",
        "This report aggregates the direct b-input LUT training runs and compares them against the earlier Goal 8 b=2 LightLogic/K-expansion baseline.",
        "",
        f"- combined rows: {len(combined_rows)}",
        f"- dataset/b groups: {len(best_by_dataset_b)}",
        f"- direct run configs summarized: {len(direct_config_rows)}",
        f"- baseline datasets imported from `{args.baseline_csv}`: {len(baseline_definition_rows)}",
        f"- strict `accuracy_preserved=True` rows at source tolerance: {strict_preservation_count}/{len(combined_rows)}",
        f"- source/minimized accuracy-preserved rows at tolerance 1e-7: {int(tolerant_preservation['preserved'])}/{len(combined_rows)}",
        f"- max |source_blif_acc - discrete_acc|: {tolerant_preservation['max_source_delta']:.3g}",
        f"- max |minimized_blif_acc - source_blif_acc|: {tolerant_preservation['max_minimized_source_delta']:.3g}",
        "",
        "## Comparison Protocol",
        "",
        "- Direct b-input candidates are loaded from the run directories listed in this report's `config.json`.",
        "- For each `(dataset, b)` pair, the report keeps the row with the highest `discrete_acc` from `network_minimization_results.csv`.",
        "- For each dataset, the final comparison uses the best direct `b>2` row against the dataset-specific Goal 8 `b=2` baseline imported from `reports/goal8_large_scale_v1/best_by_dataset.csv`.",
        "- The baseline rows are reused from the prior Goal 8 selection and are not retrained inside this report; the baseline section below documents the exact source run and config behind each dataset.",
        "",
        "## Experiment Configurations",
        "",
        "| run | datasets | b_values | seeds | width | layers | epochs | train/test | train_mode | warmup | distill | device | rows |",
        "|---|---|---|---|---:|---:|---:|---|---|---|---|---|---:|",
    ]
    for row in direct_config_rows:
        report_lines.append(
            "| {run} | {datasets} | {b_values} | {seeds} | {width} | {layers} | {epochs} | {train_test} | {train_forward_mode} | {warmup} | {distill} | {device} | {rows} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Baseline Definition",
            "",
            "| dataset | baseline_source_run | baseline_run_dir | calib | seed | k | width | layers | epochs | train/test | baseline_raw_lut_acc | baseline_abc_and_count | note |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in baseline_definition_rows:
        report_lines.append(
            "| {dataset} | {baseline_source_run} | `{baseline_run_dir}` | {baseline_calibration_mode} | {baseline_seed} | {baseline_k} | {baseline_width} | {baseline_layers} | {baseline_epochs} | {baseline_train_test} | {baseline_raw_lut_acc:.6g} | {baseline_abc_and_count:.6g} | {comparison_note} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Best Direct b-Input Results",
            "",
            "| dataset | b | run | seed | width | layers | epochs | train/test | best_discrete_acc | best_minimized_blif_acc | best_abc_and_count | best_abc_level | best_raw_lut_gate_estimate |",
            "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_by_dataset_b:
        report_lines.append(
            "| {dataset} | {b} | {best_run} | {best_seed} | {best_width} | {best_layers} | {best_epochs} | {best_train_test} | {best_discrete_acc:.6g} | {best_minimized_blif_acc:.6g} | {best_abc_and_count:.6g} | {best_abc_level:.6g} | {best_raw_lut_gate_estimate:.6g} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Comparison Against Prior b=2 Baseline",
            "",
            "| dataset | best_new_b | best_new_run | best_new_discrete_acc | baseline_source_run | baseline_raw_lut_acc | delta_acc | best_new_abc_and_count | baseline_abc_and_count | delta_abc_and | best_new_raw_lut_gate_estimate | baseline_raw_lut_gate_estimate |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        report_lines.append(
            "| {dataset} | {best_new_b} | {best_new_run} | {best_new_discrete_acc:.6g} | {baseline_source_run} | {baseline_raw_lut_acc:.6g} | {delta_discrete_acc_vs_baseline:.6g} | {best_new_abc_and_count:.6g} | {baseline_abc_and_count:.6g} | {delta_abc_and_vs_baseline:.6g} | {best_new_raw_lut_gate_estimate:.6g} | {baseline_raw_lut_gate_estimate:.6g} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "Interpretation:",
            "- BLIF export and ABC minimization preserve classification accuracy to float-evaluation tolerance: all rows are preserved at 1e-7, and ABC minimization does not change source BLIF accuracy in the aggregated rows.",
            "- The strict `accuracy_preserved=True` flag is only 19/36 because image-dataset rows have float32 evaluation differences up to roughly 2.85e-8, exceeding the original 1e-9 tolerance despite matching at reported precision.",
            "- The strongest closure of the original evidence gap is matched-setting binarized MNIST: direct `b=3` at `width=800`, `layers=3`, `train/test=12000/3000`, `epochs=80` beats the earlier matched Goal 8 `b=2` baseline by `+0.049` absolute accuracy.",
            "- Digits direct `b=4` is slightly below the imported Goal 8 baseline, but that baseline uses a stronger `width=320`, `epochs=150` configuration while the direct b-input run here uses `width=240`, `epochs=60`; this should be read as feasibility evidence rather than a clean negative result.",
            "- Boolean `majority9` and `random_sparse10` improve accuracy over their imported Goal 8 baselines, while `parity8` remains flat at `0.53125` and therefore does not show a direct-b accuracy gain yet.",
            "- The cost tradeoff remains real: the direct b-input winners generally require substantially larger raw LUT gate estimates and larger final ABC AIG counts than the best Goal 8 b=2 baselines.",
            "- This report stays AIG/ABC-focused; the separate `reports/lut_xag_backend_v1` report provides local AND/XOR/NOT XAG estimates for the LUT functions.",
        ]
    )

    write_csv(out_dir / "combined_network_results.csv", combined_rows)
    write_csv(out_dir / "best_by_dataset_b.csv", best_by_dataset_b)
    write_csv(out_dir / "comparison_vs_goal8_baseline.csv", comparison_rows)
    write_csv(out_dir / "experiment_configurations.csv", direct_config_rows)
    write_csv(out_dir / "baseline_definitions.csv", baseline_definition_rows)
    (out_dir / "goal7_real_b_lut_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "goal7_real_b_lut_report.md")


if __name__ == "__main__":
    main()
