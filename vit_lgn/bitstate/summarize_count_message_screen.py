from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable


RUNS = {
    "bitstate_majority_control": "majority",
    "bitstate_count_message_v0": "count",
}
SEEDS = (0, 1, 2)


def _average(rows: Iterable[dict[str, object]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def _run_row(root: Path, method_id: str, prefix: str, seed: int) -> dict[str, object]:
    run = root / f"{prefix}_seed{seed}"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (run / "count_message_diagnostics.json").read_text(encoding="utf-8")
    )
    deployment = manifest["artifacts"]["deployment_payload"]
    if deployment["contains_real_values"]:
        raise ValueError(f"real-valued deployment artifact: {run}")
    if deployment["verification"]["floating_tensor_count"] != 0:
        raise ValueError(f"floating runtime tensor observed: {run}")
    if not deployment["verification"]["exact_logits"]:
        raise ValueError(f"deployment logits mismatch: {run}")
    best_epoch = int(summary["best_epoch"])
    validation = summary["history"][best_epoch - 1]
    finite_curve = all(
        math.isfinite(value)
        for epoch in summary["history"]
        for value in epoch.values()
        if isinstance(value, float)
    )
    merge = diagnostics["merge_gate_usage"]["aggregate"]
    blocks = diagnostics["blocks"]
    return {
        "method_id": method_id,
        "seed": seed,
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "best_epoch": best_epoch,
        "finite_curve": finite_curve,
        "late_validation_drop": float(summary["best_discrete_acc"])
        - float(summary["history"][-1]["discrete_acc"]),
        "validation_soft_acc": validation["soft_acc"],
        "validation_hard_acc": validation["discrete_acc"],
        "validation_acc_gap": validation["acc_gap"],
        "validation_soft_loss": validation["soft_loss"],
        "validation_hard_loss": validation["discrete_loss"],
        "validation_loss_gap": abs(
            float(validation["soft_loss"]) - float(validation["discrete_loss"])
        ),
        "test_soft_acc": summary["soft_acc"],
        "test_hard_acc": summary["discrete_acc"],
        "test_acc_gap": summary["acc_gap"],
        "test_soft_loss": summary["soft_loss"],
        "test_hard_loss": summary["discrete_loss"],
        "test_loss_gap": summary["loss_gap"],
        "train_time_s": summary["train_time"],
        "epochs_to_target": summary["epochs_to_target"],
        "unused_gate_ratio": summary["unused_gate_ratio"],
        "inactive_neuron_ratio": summary["activation_inactive_gate_ratio"],
        "literal_gate_ratio": summary["literal_gate_ratio"],
        "nontrivial_gate_ratio": summary["nontrivial_gate_ratio"],
        "state_entropy": summary["state_entropy"],
        "state_constant_ratio": summary["state_constant_ratio"],
        "state_duplicate_ratio": summary["state_duplicate_ratio"],
        "state_flip_rate": summary["state_flip_rate"],
        "max_layer_flip_ratio": summary["layer_gap_max_flip_ratio"],
        "merge_uses_message_ratio": merge["uses_message_ratio"],
        "merge_identity_state_ratio": merge["identity_state_ratio"],
        "message_change_vs_majority": mean(
            float(block["message_change_vs_majority"]) for block in blocks
        ),
        "gate_count": summary["gate_count"],
        "depth": summary["depth"],
        "fanout_max": summary["fanout_max"],
        "trainable_parameters": summary["trainable_parameters"],
        "strict_runtime_audit_operations": summary[
            "strict_runtime_audit_operations"
        ],
        "strict_runtime_floating_tensor_count": summary[
            "strict_runtime_floating_tensor_count"
        ],
        "strict_runtime_exact_logits": summary["strict_runtime_exact_logits"],
        "deployment_payload_bytes": deployment["bytes"],
        "deployment_payload_sha256": deployment["sha256"],
        "deployment_tensor_inventory": json.dumps(
            deployment["tensor_element_inventory"], sort_keys=True
        ),
        "source_files_sha256": json.dumps(
            manifest["source"]["files_sha256"], sort_keys=True
        ),
    }


def summarize(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        _run_row(root, method_id, prefix, seed)
        for method_id, prefix in RUNS.items()
        for seed in SEEDS
    ]
    protocol_hashes = {str(row["protocol_sha256"]) for row in rows}
    source_hashes = {str(row["source_files_sha256"]) for row in rows}
    if len(protocol_hashes) != 1:
        raise ValueError(f"unmatched protocol hashes: {protocol_hashes}")
    if len(source_hashes) != 1:
        raise ValueError("source hashes differ across paired runs")

    numeric_metrics = (
        "validation_soft_acc",
        "validation_hard_acc",
        "validation_acc_gap",
        "validation_soft_loss",
        "validation_hard_loss",
        "validation_loss_gap",
        "test_soft_acc",
        "test_hard_acc",
        "test_acc_gap",
        "test_soft_loss",
        "test_hard_loss",
        "test_loss_gap",
        "train_time_s",
        "unused_gate_ratio",
        "inactive_neuron_ratio",
        "literal_gate_ratio",
        "nontrivial_gate_ratio",
        "state_entropy",
        "state_constant_ratio",
        "state_duplicate_ratio",
        "state_flip_rate",
        "max_layer_flip_ratio",
        "merge_uses_message_ratio",
        "merge_identity_state_ratio",
        "message_change_vs_majority",
        "late_validation_drop",
    )
    grouped = {
        method_id: [row for row in rows if row["method_id"] == method_id]
        for method_id in RUNS
    }
    means = {
        method_id: {
            metric: _average(method_rows, metric) for metric in numeric_metrics
        }
        for method_id, method_rows in grouped.items()
    }
    baseline = {int(row["seed"]): row for row in grouped["bitstate_majority_control"]}
    candidate = {int(row["seed"]): row for row in grouped["bitstate_count_message_v0"]}
    paired_metrics = (
        "validation_hard_acc",
        "test_hard_acc",
        "test_acc_gap",
        "unused_gate_ratio",
        "inactive_neuron_ratio",
        "merge_uses_message_ratio",
        "merge_identity_state_ratio",
    )
    paired = {
        metric: [
            float(candidate[seed][metric]) - float(baseline[seed][metric])
            for seed in SEEDS
        ]
        for metric in paired_metrics
    }
    validation_deltas = paired["validation_hard_acc"]
    promotion = {
        "passed": mean(validation_deltas) > 0.0
        and sum(delta > 0.0 for delta in validation_deltas) >= 2,
        "selection_metric": "best validation hard accuracy",
        "mean_validation_hard_acc_delta": mean(validation_deltas),
        "validation_seed_wins": sum(delta > 0.0 for delta in validation_deltas),
        "test_metrics_used_for_selection": False,
    }
    if not promotion["passed"]:
        promotion["decision"] = "reject_full_scale_and_retain_as_diagnostic"
    return rows, {
        "schema_version": 1,
        "protocol_sha256": next(iter(protocol_hashes)),
        "source_files_sha256": json.loads(next(iter(source_hashes))),
        "means": means,
        "paired_candidate_minus_control": {
            metric: {"per_seed": values, "mean": mean(values)}
            for metric, values in paired.items()
        },
        "promotion": promotion,
        "capacity": json.loads(
            (
                root
                / "count_seed0"
                / "count_message_diagnostics.json"
            ).read_text(encoding="utf-8")
        )["capacity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows, aggregate = summarize(args.root)
    csv_path = args.root / "screen_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate["promotion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
