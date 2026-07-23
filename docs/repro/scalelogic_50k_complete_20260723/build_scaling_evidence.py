#!/usr/bin/env python3
"""Validate and summarize the completed ScaleLogic 50k runs."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {
    "d6_reference": (HERE / "d6_result.json", HERE / "d6_protocol.json"),
    "d12e192_depth_reference": (
        HERE / "d12e192_result.json",
        HERE / "d12e192_protocol.json",
    ),
    "d12_local0": (HERE / "local0_result.json", HERE / "local0_protocol.json"),
    "d12_local4": (HERE / "local4_result.json", HERE / "local4_protocol.json"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def saturation(row: dict[str, Any]) -> tuple[float | None, float | None]:
    stats = row.get("normalization_statistics")
    if not stats:
        return None, None
    final = stats["norm"]
    final_ratio = final["saturated_code_count"] / final["code_count"]
    all_ratio = sum(item["saturated_code_count"] for item in stats.values()) / sum(
        item["code_count"] for item in stats.values()
    )
    return final_ratio, all_ratio


def validate_pair(
    left_result: dict[str, Any],
    left_protocol: dict[str, Any],
    right_result: dict[str, Any],
    right_protocol: dict[str, Any],
) -> None:
    assert left_result["protocol_sha256"] == left_protocol["sha256"]
    assert right_result["protocol_sha256"] == right_protocol["sha256"]
    assert left_result["source_sha256"] == left_protocol["sources"]
    assert right_result["source_sha256"] == right_protocol["sources"]
    assert left_protocol["sources"] == right_protocol["sources"]
    assert left_protocol["dataset"] == right_protocol["dataset"]
    assert left_protocol["split"] == right_protocol["split"]

    differences = {
        key
        for key in left_protocol["args"] | right_protocol["args"]
        if left_protocol["args"].get(key) != right_protocol["args"].get(key)
    }
    assert differences == {"local_layers", "out_dir"}, differences


def main() -> None:
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    for name, (result_path, protocol_path) in RUNS.items():
        result = read_json(result_path)
        protocol = read_json(protocol_path)
        assert result["protocol_sha256"] == protocol["sha256"]
        assert result["history"][-1]["step"] == protocol["args"]["steps"] == 50_000
        assert result["final_validation_accuracy"] == result["history"][-1][
            "validation_accuracy"
        ]
        loaded[name] = result, protocol
        input_hashes[result_path.name] = sha256(result_path)
        input_hashes[protocol_path.name] = sha256(protocol_path)

    validate_pair(*loaded["d12_local0"], *loaded["d12_local4"])
    d6_result, d6_protocol = loaded["d6_reference"]
    depth_result, depth_protocol = loaded["d12e192_depth_reference"]
    assert d6_result["source_sha256"] == depth_result["source_sha256"]
    assert d6_protocol["sources"] == depth_protocol["sources"]
    assert d6_protocol["dataset_archive_sha256"] == depth_protocol[
        "dataset_archive_sha256"
    ]
    depth_differences = {
        key
        for key in d6_protocol["args"] | depth_protocol["args"]
        if d6_protocol["args"].get(key) != depth_protocol["args"].get(key)
    }
    assert depth_differences == {"depth", "data_root", "out_dir"}

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for name, (result, protocol) in loaded.items():
        args = protocol["args"]
        for row in result["history"]:
            final_saturation, all_saturation = saturation(row)
            curve_rows.append(
                {
                    "variant": name,
                    "step": row["step"],
                    "validation_accuracy": row["validation_accuracy"],
                    "train_loss": row["train_loss"],
                    "next_learning_rate": row["next_learning_rate"],
                    "final_norm_saturation_ratio": final_saturation,
                    "all_norm_saturation_ratio": all_saturation,
                    "unclipped_global_gradient_l2": row.get(
                        "unclipped_global_gradient_l2"
                    ),
                    "gradient_clip_coefficient": row.get(
                        "gradient_clip_coefficient"
                    ),
                }
            )

        best = max(result["history"], key=lambda item: item["validation_accuracy"])
        final_saturation, all_saturation = saturation(result["history"][-1])
        clip_values = [
            row["gradient_clip_coefficient"]
            for row in result["history"]
            if "gradient_clip_coefficient" in row
        ]
        summary_rows.append(
            {
                "variant": name,
                "dim": args["dim"],
                "depth": args["depth"],
                "heads": args["heads"],
                "head_dim": args["dim"] // args["heads"],
                "local_layers": args["local_layers"],
                "parameters": result["parameters"],
                "steps": args["steps"],
                "batch_size": args["batch_size"],
                "training_examples_seen": args["steps"] * args["batch_size"],
                "final_validation_accuracy": result["final_validation_accuracy"],
                "best_validation_accuracy": best["validation_accuracy"],
                "best_step": best["step"],
                "elapsed_seconds": result["elapsed_seconds"],
                "final_norm_saturation_ratio": final_saturation,
                "all_norm_saturation_ratio": all_saturation,
                "mean_gradient_clip_coefficient": (
                    statistics.mean(clip_values) if clip_values else None
                ),
                "protocol_sha256": protocol["sha256"],
                "source_set": (
                    "legacy_scale_source"
                    if name in {"d6_reference", "d12e192_depth_reference"}
                    else "matched_d12_source"
                ),
            }
        )

    attention_rows = read_json(
        ROOT
        / "vit_lgn"
        / "attention_clean"
        / "artifacts"
        / "accuracy_summary_final_20260710.json"
    )
    attention = next(
        row
        for row in attention_rows
        if row["name"] == "h6_k8_defaulttemp_d6e192_aug_const_200k_20260630.log"
    )
    reference_rows = [
        {
            "family": "attention_clean_augmented_k8",
            "step": step,
            "validation_eval_accuracy": eval_accuracy,
            "validation_train_mode_accuracy": train_accuracy,
        }
        for step, eval_accuracy, train_accuracy in attention["eval_curve"]
    ]

    write_csv(HERE / "completed_50k_curves.csv", curve_rows)
    write_csv(HERE / "completed_50k_summary.csv", summary_rows)
    write_csv(HERE / "attention_clean_augmented_200k_curve.csv", reference_rows)

    validation = {
        "status": "passed",
        "strict_pair": ["d12_local0", "d12_local4"],
        "strict_pair_differences": ["local_layers", "out_dir"],
        "depth_reference": {
            "runs": ["d6_reference", "d12e192_depth_reference"],
            "argument_differences": sorted(depth_differences),
            "source_files_identical": True,
            "dataset_archive_identical": True,
            "runtime_identical": False,
        },
        "d6_to_d12e384_status": "historical reference; source set differs",
        "input_sha256": dict(sorted(input_hashes.items())),
        "attention_reference": {
            "name": attention["name"],
            "best_validation_accuracy": attention["best_valid_eval"],
            "best_step": max(
                attention["eval_curve"], key=lambda item: item[1]
            )[0],
            "final_test": attention["final_test"],
        },
    }
    (HERE / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("validated completed ScaleLogic 50k pair and training-length reference")


if __name__ == "__main__":
    main()
