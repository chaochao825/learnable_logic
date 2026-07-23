from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_summary(result: dict[str, object]) -> dict[str, object]:
    history = result["history"]
    best = max(history, key=lambda row: float(row["validation_accuracy"]))
    finite = all(
        math.isfinite(value)
        for row in history
        for value in row.values()
        if isinstance(value, float)
    )

    def first_step(target: float) -> int | None:
        return next(
            (
                int(row["step"])
                for row in history
                if float(row["validation_accuracy"]) >= target
            ),
            None,
        )

    final = float(result["final_validation_accuracy"])
    return {
        "parameters": int(result["parameters"]),
        "train_time_s": float(result["elapsed_seconds"]),
        "final_validation_accuracy": final,
        "best_validation_accuracy": float(best["validation_accuracy"]),
        "best_step": int(best["step"]),
        "late_validation_drop": float(best["validation_accuracy"]) - final,
        "steps_to_75pct": first_step(0.75),
        "steps_to_76pct": first_step(0.76),
        "finite_curve": finite,
        "protocol_sha256": result["protocol_sha256"],
    }


def summarize(
    d6_result_path: Path,
    d6_strict_path: Path,
    d12_result_path: Path,
    d12_strict_path: Path,
    d12_capacity_path: Path,
) -> dict[str, object]:
    d6_result = _load(d6_result_path)
    d12_result = _load(d12_result_path)
    d6_strict = _load(d6_strict_path)
    d12_strict = _load(d12_strict_path)
    d12_capacity = _load(d12_capacity_path)
    d6_training = _training_summary(d6_result)
    d12_training = _training_summary(d12_result)
    if float(d6_strict["accuracy"]) != d6_training["final_validation_accuracy"]:
        raise ValueError("d6 strict replay does not reproduce stored final accuracy")
    if float(d12_strict["accuracy"]) != d12_training["final_validation_accuracy"]:
        raise ValueError("d12 strict replay does not reproduce stored final accuracy")
    for name, replay in (("d6", d6_strict), ("d12", d12_strict)):
        if int(replay["count"]) != 5000:
            raise ValueError(f"{name} replay is not full validation")
        if int(replay["runtime_floating_tensor_count"]) != 0:
            raise ValueError(f"{name} replay observed a real runtime tensor")
    return {
        "schema_version": 1,
        "d6": {"training": d6_training, "strict_replay": d6_strict},
        "d12": {
            "training": d12_training,
            "strict_replay": d12_strict,
            "payload_capacity": d12_capacity,
        },
        "d12_minus_d6": {
            "strict_accuracy": float(d12_strict["accuracy"])
            - float(d6_strict["accuracy"]),
            "best_validation_accuracy": d12_training["best_validation_accuracy"]
            - d6_training["best_validation_accuracy"],
            "parameters": d12_training["parameters"] - d6_training["parameters"],
            "parameter_ratio": d12_training["parameters"]
            / d6_training["parameters"],
            "train_time_s": d12_training["train_time_s"]
            - d6_training["train_time_s"],
            "train_time_ratio": d12_training["train_time_s"]
            / d6_training["train_time_s"],
        },
        "interpretation": {
            "strict_depth_gain_reproduced": True,
            "faster_step_convergence": (
                d12_training["steps_to_75pct"] is not None
                and d6_training["steps_to_75pct"] is not None
                and d12_training["steps_to_75pct"]
                < d6_training["steps_to_75pct"]
            ),
            "multi_seed_scale_claim": False,
            "reason": "one historical seed proves a stronger strict checkpoint, not a robust scaling law",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d6-result", type=Path, required=True)
    parser.add_argument("--d6-strict", type=Path, required=True)
    parser.add_argument("--d12-result", type=Path, required=True)
    parser.add_argument("--d12-strict", type=Path, required=True)
    parser.add_argument("--d12-capacity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.d6_result,
        args.d6_strict,
        args.d12_result,
        args.d12_strict,
        args.d12_capacity,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "parameters",
                "train_time_s",
                "best_validation_accuracy",
                "best_step",
                "final_validation_accuracy",
                "strict_accuracy",
                "strict_runtime_s",
                "strict_audit_ops",
                "runtime_floating_tensor_count",
                "steps_to_75pct",
                "steps_to_76pct",
            ),
        )
        writer.writeheader()
        for model in ("d6", "d12"):
            training = result[model]["training"]
            replay = result[model]["strict_replay"]
            writer.writerow(
                {
                    "model": model,
                    "parameters": training["parameters"],
                    "train_time_s": training["train_time_s"],
                    "best_validation_accuracy": training[
                        "best_validation_accuracy"
                    ],
                    "best_step": training["best_step"],
                    "final_validation_accuracy": training[
                        "final_validation_accuracy"
                    ],
                    "strict_accuracy": replay["accuracy"],
                    "strict_runtime_s": replay["elapsed_seconds"],
                    "strict_audit_ops": replay["runtime_audit_operations"],
                    "runtime_floating_tensor_count": replay[
                        "runtime_floating_tensor_count"
                    ],
                    "steps_to_75pct": training["steps_to_75pct"],
                    "steps_to_76pct": training["steps_to_76pct"],
                }
            )
    print(json.dumps(result["d12_minus_d6"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
