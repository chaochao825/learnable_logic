#!/usr/bin/env python3
"""Aggregate and validate the strict H200 matched CIFAR-10 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPORT_TARGET_ACCURACY = 0.20

METHOD_INPUTS = {
    "dlgn": {
        "summary_method": "bitstate_soft_argmax",
        "source_commits": ["5aaee9d"] * 3,
        "summaries": [HERE / f"h200_dlgn_seed{seed}_summary.json" for seed in range(3)],
        "diagnostics": [
            HERE / f"h200_dlgn_seed{seed}_posthoc_diagnostics.json"
            for seed in range(3)
        ],
    },
    "annealing": {
        "summary_method": "bitstate_anneal_argmax",
        "source_commits": ["5aaee9d"] * 3,
        "summaries": [
            HERE / f"h200_anneal_seed{seed}_summary.json" for seed in range(3)
        ],
        "diagnostics": [
            HERE / f"h200_anneal_seed{seed}_posthoc_diagnostics.json"
            for seed in range(3)
        ],
    },
    "gumbel_st": {
        "summary_method": "bitstate_gumbel_st",
        "source_commits": ["7cc6b24", "5aaee9d", "5aaee9d"],
        "summaries": [
            HERE.parent
            / "bitstate_h200_long_cifar_20260723"
            / "gumbel_matched_summary.json",
            HERE / "h200_gumbel_seed1_summary.json",
            HERE / "h200_gumbel_seed2_summary.json",
        ],
        "diagnostics": [
            HERE.parent
            / "bitstate_h200_long_cifar_20260723"
            / "gumbel_matched_posthoc.json",
            HERE / "h200_gumbel_seed1_posthoc_diagnostics.json",
            HERE / "h200_gumbel_seed2_posthoc_diagnostics.json",
        ],
        "structural": [
            (
                HERE.parent
                / "bitstate_h200_long_cifar_20260723"
                / "gate_function_metrics.json",
                ("runs", "matched_gumbel"),
            ),
            (HERE / "h200_gumbel_seed1_posthoc_diagnostics.json", ()),
            (HERE / "h200_gumbel_seed2_posthoc_diagnostics.json", ()),
        ],
    },
    "progressive_hard_st": {
        "summary_method": "bitstate_progressive_hard_st",
        "source_commits": ["7cc6b24"] * 3,
        "summaries": [
            HERE.parent / "bitstate_h200_long_cifar_20260723" / "hard_scale16_summary.json",
            HERE.parent
            / "bitstate_multiseed_cifar_20260723"
            / "proposed_seed1_summary.json",
            HERE.parent
            / "bitstate_multiseed_cifar_20260723"
            / "proposed_seed2_summary.json",
        ],
        "diagnostics": [
            HERE.parent / "bitstate_h200_long_cifar_20260723" / "hard_scale16_posthoc.json",
            HERE.parent
            / "bitstate_multiseed_cifar_20260723"
            / "proposed_seed1_gate_function_metrics.json",
            HERE.parent
            / "bitstate_multiseed_cifar_20260723"
            / "proposed_seed2_gate_function_metrics.json",
        ],
        "structural": [
            (
                HERE.parent
                / "bitstate_h200_long_cifar_20260723"
                / "gate_function_metrics.json",
                ("runs", "progressive_scale16"),
            ),
            (
                HERE.parent
                / "bitstate_multiseed_cifar_20260723"
                / "proposed_seed1_gate_function_metrics.json",
                (),
            ),
            (
                HERE.parent
                / "bitstate_multiseed_cifar_20260723"
                / "proposed_seed2_gate_function_metrics.json",
                (),
            ),
        ],
    },
}

MODEL_PROTOCOL_FIELDS = (
    "image_size",
    "in_channels",
    "num_classes",
    "patch_size",
    "state_width",
    "threshold_levels",
    "encoder_kind",
    "encoder_identity_width",
    "predicate_fanin",
    "predicate_chunk_size",
    "local_depth",
    "global_depth",
    "heads",
    "qk_bits",
    "topk",
    "exclude_self",
    "update_fraction",
    "votes_per_class",
    "attention_temperature",
    "predicate_temperature",
    "gate_init_strength",
)

TRAINING_PROTOCOL_FIELDS = (
    "dataset",
    "train_limit",
    "validation_size",
    "eval_limit",
    "epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "weight_decay",
    "lr_schedule",
    "warmup_epochs",
    "min_learning_rate",
    "amp_bfloat16",
    "augment",
    "label_smoothing",
    "group_sum_temperature",
    "state_balance_weight",
    "state_diversity_weight",
    "state_diversity_pairs",
    "state_flip_weight",
    "state_minimum_flip",
    "state_maximum_flip",
    "state_maximum_similarity",
    "gate_entropy_target_start",
    "gate_entropy_target_end",
    "gate_entropy_weight_start",
    "gate_entropy_weight_end",
    "gate_init_strength",
    "inactive_batches",
    "restore_best",
)

RUN_FIELDS = (
    "method",
    "dataset",
    "seed",
    "hardware",
    "source_commit",
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "train_time",
    "epoch_to_20pct",
    "time_to_20pct",
    "unused_gate_ratio",
    "activation_inactive_gate_ratio",
    "layer_gap_max_flip_ratio",
    "layer_gap_final_flip_ratio",
    "constant_gate_ratio",
    "literal_gate_ratio",
    "nontrivial_gate_ratio",
    "gate_count",
    "depth",
    "fanout_max",
    "bit_exact_verified",
)

AGGREGATE_FIELDS = tuple(
    field
    for field in RUN_FIELDS
    if field
    not in {
        "method",
        "dataset",
        "seed",
        "hardware",
        "source_commit",
        "bit_exact_verified",
    }
)

DELTA_FIELDS = (
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "train_time",
    "epoch_to_20pct",
    "time_to_20pct",
    "unused_gate_ratio",
    "activation_inactive_gate_ratio",
    "layer_gap_max_flip_ratio",
    "layer_gap_final_flip_ratio",
)

REQUIRED_FIELDS = (
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
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing experiment artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def descend(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        value = value[key]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def first_target_time(summary: dict[str, Any]) -> tuple[int, float | None]:
    for epoch in summary["history"]:
        if float(epoch["discrete_acc"]) >= REPORT_TARGET_ACCURACY:
            return int(epoch["epoch"]), float(epoch["elapsed"])
    return -1, None


def overlay(summary: dict[str, Any], diagnostic: dict[str, Any], key: str) -> Any:
    if key == "unused_gate_ratio" and key in diagnostic:
        return diagnostic[key]
    if key in summary:
        return summary[key]
    return diagnostic[key]


def build_rows() -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}

    for method, inputs in METHOD_INPUTS.items():
        structural_inputs = inputs.get("structural")
        for seed, (summary_path, diagnostic_path) in enumerate(
            zip(inputs["summaries"], inputs["diagnostics"], strict=True)
        ):
            summary = read_json(summary_path)
            diagnostic = read_json(diagnostic_path)
            structural = diagnostic
            if structural_inputs is not None:
                structural_path, keys = structural_inputs[seed]
                structural = descend(read_json(structural_path), keys)
                input_hashes[
                    structural_path.relative_to(HERE.parent.parent).as_posix()
                ] = sha256(structural_path)

            input_hashes[summary_path.relative_to(HERE.parent.parent).as_posix()] = sha256(
                summary_path
            )
            input_hashes[
                diagnostic_path.relative_to(HERE.parent.parent).as_posix()
            ] = sha256(diagnostic_path)

            if int(summary["seed"]) != seed:
                raise AssertionError(f"seed mismatch in {summary_path}")
            if summary["method"] != inputs["summary_method"]:
                raise AssertionError(f"method mismatch in {summary_path}")
            if not close(
                summary["acc_gap"],
                abs(float(summary["soft_acc"]) - float(summary["discrete_acc"])),
            ):
                raise AssertionError(f"accuracy-gap invariant failed in {summary_path}")
            if not close(
                summary["loss_gap"],
                abs(float(summary["soft_loss"]) - float(summary["discrete_loss"])),
            ):
                raise AssertionError(f"loss-gap invariant failed in {summary_path}")

            epoch_to_target, time_to_target = first_target_time(summary)
            if close(
                float(summary["training_args"]["target_accuracy"]),
                REPORT_TARGET_ACCURACY,
            ):
                stored_epoch = summary["epochs_to_target"]
                stored_epoch = -1 if stored_epoch in (None, "") else int(stored_epoch)
                if stored_epoch != epoch_to_target:
                    raise AssertionError(f"target-epoch invariant failed in {summary_path}")

            bit_exact = bool(overlay(summary, diagnostic, "bit_exact_verified"))
            if not bit_exact:
                raise AssertionError(f"hard path is not bit-exact in {summary_path}")

            constant = float(structural["constant_gate_ratio"])
            literal = float(structural["literal_gate_ratio"])
            nontrivial = float(structural["nontrivial_gate_ratio"])
            if not close(constant + literal + nontrivial, 1.0):
                raise AssertionError(f"gate-function partition failed in {summary_path}")

            rows.append(
                {
                    "method": method,
                    "dataset": summary["dataset"],
                    "seed": seed,
                    "hardware": "NVIDIA H200 NVL",
                    "source_commit": inputs["source_commits"][seed],
                    "soft_acc": float(summary["soft_acc"]),
                    "discrete_acc": float(summary["discrete_acc"]),
                    "acc_gap": float(summary["acc_gap"]),
                    "soft_loss": float(summary["soft_loss"]),
                    "discrete_loss": float(summary["discrete_loss"]),
                    "loss_gap": float(summary["loss_gap"]),
                    "train_time": float(summary["train_time"]),
                    "epoch_to_20pct": epoch_to_target,
                    "time_to_20pct": time_to_target,
                    "unused_gate_ratio": float(
                        overlay(summary, diagnostic, "unused_gate_ratio")
                    ),
                    "activation_inactive_gate_ratio": float(
                        overlay(summary, diagnostic, "activation_inactive_gate_ratio")
                    ),
                    "layer_gap_max_flip_ratio": float(
                        overlay(summary, diagnostic, "layer_gap_max_flip_ratio")
                    ),
                    "layer_gap_final_flip_ratio": float(
                        overlay(summary, diagnostic, "layer_gap_final_flip_ratio")
                    ),
                    "constant_gate_ratio": constant,
                    "literal_gate_ratio": literal,
                    "nontrivial_gate_ratio": nontrivial,
                    "gate_count": int(summary["gate_count"]),
                    "depth": int(summary["depth"]),
                    "fanout_max": int(summary["fanout_max"]),
                    "bit_exact_verified": bit_exact,
                    "_model_protocol": {
                        key: summary["model_config"][key]
                        for key in MODEL_PROTOCOL_FIELDS
                    },
                    "_training_protocol": {
                        key: summary["training_args"][key]
                        for key in TRAINING_PROTOCOL_FIELDS
                    },
                    "_layer_gap_diagnostics": (
                        summary["layer_gap_diagnostics"]
                        if "layer_gap_diagnostics" in summary
                        else diagnostic["layer_gap_diagnostics"]
                    ),
                }
            )

    return rows, dict(sorted(input_hashes.items()))


def validate_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_runs = {(method, seed) for method in METHOD_INPUTS for seed in range(3)}
    observed_runs = {(str(row["method"]), int(row["seed"])) for row in rows}
    if observed_runs != expected_runs or len(rows) != len(expected_runs):
        raise AssertionError(
            f"incomplete method/seed matrix: expected {sorted(expected_runs)}, "
            f"observed {sorted(observed_runs)}"
        )

    reference_model = rows[0]["_model_protocol"]
    reference_training = rows[0]["_training_protocol"]
    for row in rows[1:]:
        if row["_model_protocol"] != reference_model:
            raise AssertionError(f"model protocol differs for {row['method']} seed {row['seed']}")
        if row["_training_protocol"] != reference_training:
            raise AssertionError(
                f"training protocol differs for {row['method']} seed {row['seed']}"
            )

    topology: dict[str, dict[str, int]] = {}
    for seed in range(3):
        seed_rows = [row for row in rows if row["seed"] == seed]
        signatures = {
            (row["gate_count"], row["depth"], row["fanout_max"])
            for row in seed_rows
        }
        if len(signatures) != 1:
            raise AssertionError(f"topology differs across methods for seed {seed}")
        gate_count, depth, fanout_max = signatures.pop()
        topology[str(seed)] = {
            "gate_count": gate_count,
            "depth": depth,
            "fanout_max": fanout_max,
        }

    layer_signature = [
        (int(item["layer"]), str(item["name"]))
        for item in rows[0]["_layer_gap_diagnostics"]
    ]
    for row in rows[1:]:
        candidate = [
            (int(item["layer"]), str(item["name"]))
            for item in row["_layer_gap_diagnostics"]
        ]
        if candidate != layer_signature:
            raise AssertionError(
                f"layer diagnostics differ for {row['method']} seed {row['seed']}"
            )

    return {
        "status": "passed",
        "run_count": len(rows),
        "method_count": len(METHOD_INPUTS),
        "seed_count": 3,
        "hardware": "NVIDIA H200 NVL",
        "uniform_reporting_target_accuracy": REPORT_TARGET_ACCURACY,
        "metric_invariants": [
            "acc_gap == abs(soft_acc - discrete_acc)",
            "loss_gap == abs(soft_loss - discrete_loss)",
            "20% convergence epoch/time is recomputed from per-epoch hard validation history",
            "constant + literal + nontrivial gate ratios == 1",
            "bit_exact_verified is true",
        ],
        "common_model_protocol": reference_model,
        "common_training_protocol": reference_training,
        "per_seed_topology": topology,
        "layer_gap_signature": [
            {"layer": layer, "name": name} for layer, name in layer_signature
        ],
        "method_specific_fields_excluded_from_protocol_check": [
            "method",
            "tau/tau schedule",
            "soft warmup and progressive hardening schedule",
            "stored target_accuracy used only for logging, normalized to 20% here",
            "output directory",
            "seed",
        ],
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in RUN_FIELDS}


def write_csv(path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHOD_INPUTS:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate: dict[str, Any] = {
            "method": method,
            "dataset": method_rows[0]["dataset"],
            "n": len(method_rows),
            "hardware": method_rows[0]["hardware"],
            "source_commit": ";".join(
                sorted({str(row["source_commit"]) for row in method_rows})
            ),
            "target_hit_count": sum(row["epoch_to_20pct"] >= 0 for row in method_rows),
            "target_hit_rate": statistics.mean(
                row["epoch_to_20pct"] >= 0 for row in method_rows
            ),
        }
        for field in AGGREGATE_FIELDS:
            values = [
                float(row[field])
                for row in method_rows
                if row[field] is not None
                and not (field == "epoch_to_20pct" and row[field] < 0)
            ]
            aggregate[f"{field}_mean"] = statistics.mean(values) if values else None
            aggregate[f"{field}_sd"] = (
                statistics.stdev(values) if len(values) >= 2 else None
            )
        aggregate["bit_exact_all"] = all(
            bool(row["bit_exact_verified"]) for row in method_rows
        )
        output.append(aggregate)
    return output


def paired_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method_seed = {(row["method"], row["seed"]): row for row in rows}
    comparisons = (
        ("progressive_hard_st_minus_dlgn", "dlgn"),
        ("progressive_hard_st_minus_annealing", "annealing"),
        ("progressive_hard_st_minus_gumbel_st", "gumbel_st"),
    )
    output: list[dict[str, Any]] = []
    for comparison, baseline in comparisons:
        per_seed: list[dict[str, Any]] = []
        for seed in range(3):
            proposed = by_method_seed[("progressive_hard_st", seed)]
            control = by_method_seed[(baseline, seed)]
            item: dict[str, Any] = {"comparison": comparison, "seed": seed}
            for field in DELTA_FIELDS:
                if (
                    proposed[field] is None
                    or control[field] is None
                    or (
                        field in {"epoch_to_20pct", "time_to_20pct"}
                        and (
                            proposed["epoch_to_20pct"] < 0
                            or control["epoch_to_20pct"] < 0
                        )
                    )
                ):
                    item[f"{field}_delta"] = None
                else:
                    item[f"{field}_delta"] = float(proposed[field]) - float(
                        control[field]
                    )
            per_seed.append(item)
            output.append(item)
        for statistic_name, statistic_fn in (
            ("mean", statistics.mean),
            ("sd", statistics.stdev),
        ):
            item = {"comparison": comparison, "seed": statistic_name}
            for field in DELTA_FIELDS:
                values = [
                    float(row[f"{field}_delta"])
                    for row in per_seed
                    if row[f"{field}_delta"] is not None
                ]
                if statistic_name == "sd" and len(values) < 2:
                    item[f"{field}_delta"] = None
                else:
                    item[f"{field}_delta"] = statistic_fn(values) if values else None
            output.append(item)
    return output


def required_metric_rows(
    rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in aggregate:
        method_rows = [row for row in rows if row["method"] == item["method"]]
        output.append(
            {
                "method": item["method"],
                "dataset": "cifar10_full_bitstate",
                "soft_acc": item["soft_acc_mean"],
                "discrete_acc": item["discrete_acc_mean"],
                "acc_gap": item["acc_gap_mean"],
                "soft_loss": item["soft_loss_mean"],
                "discrete_loss": item["discrete_loss_mean"],
                "loss_gap": item["loss_gap_mean"],
                "train_time": item["train_time_mean"],
                "epochs_to_target": (
                    item["epoch_to_20pct_mean"]
                    if item["target_hit_count"] > 0
                    else -1
                ),
                "unused_gate_ratio": item["unused_gate_ratio_mean"],
                "gate_count": int(method_rows[0]["gate_count"]),
                "depth": int(method_rows[0]["depth"]),
                "fanout_max": max(int(row["fanout_max"]) for row in method_rows),
            }
        )
    return output


def layer_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for item in row["_layer_gap_diagnostics"]:
            output.append(
                {
                    "method": row["method"],
                    "seed": row["seed"],
                    "layer": int(item["layer"]),
                    "name": str(item["name"]),
                    "mae": float(item["mae"]),
                    "flip_ratio": float(item["flip_ratio"]),
                    "elements": int(item["elements"]),
                }
            )
    return output


def aggregate_layer_gap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHOD_INPUTS:
        method_rows = [row for row in rows if row["method"] == method]
        for layer in sorted({int(row["layer"]) for row in method_rows}):
            layer_rows = [row for row in method_rows if row["layer"] == layer]
            mae = [float(row["mae"]) for row in layer_rows]
            flips = [float(row["flip_ratio"]) for row in layer_rows]
            output.append(
                {
                    "method": method,
                    "layer": layer,
                    "name": layer_rows[0]["name"],
                    "n": len(layer_rows),
                    "mae_mean": statistics.mean(mae),
                    "mae_sd": statistics.stdev(mae),
                    "flip_ratio_mean": statistics.mean(flips),
                    "flip_ratio_sd": statistics.stdev(flips),
                }
            )
    return output


def main() -> None:
    rows, input_hashes = build_rows()
    validation = validate_protocol(rows)
    validation["input_sha256"] = input_hashes

    write_csv(HERE / "h200_matched_runs.csv", RUN_FIELDS, [public_row(row) for row in rows])

    aggregate = aggregate_rows(rows)
    aggregate_fields = list(aggregate[0])
    write_csv(HERE / "h200_matched_aggregate.csv", aggregate_fields, aggregate)

    required = required_metric_rows(rows, aggregate)
    write_csv(HERE / "h200_required_metrics_table.csv", REQUIRED_FIELDS, required)

    layer_runs = layer_gap_rows(rows)
    write_csv(HERE / "h200_layer_gap_runs.csv", list(layer_runs[0]), layer_runs)
    layer_aggregate = aggregate_layer_gap(layer_runs)
    write_csv(
        HERE / "h200_layer_gap_aggregate.csv",
        list(layer_aggregate[0]),
        layer_aggregate,
    )

    paired = paired_delta_rows(rows)
    paired_fields = list(paired[0])
    write_csv(HERE / "h200_matched_paired_deltas.csv", paired_fields, paired)

    with (HERE / "h200_matched_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("validated 12 strict H200 runs across 4 methods and 3 seeds")


if __name__ == "__main__":
    main()
