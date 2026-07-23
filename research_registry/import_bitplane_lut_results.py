from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from research_registry.manifest import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research_registry"
PROTOCOL_ID = "bitplane_lut_digits_scale_s012_v2"
METHOD = {
    "argmax": "bitplane_lut_argmax",
    "truth": "bitplane_lut_refit",
    "wiring": "bitplane_lut_wiring_refit",
}


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def final_block_best_hard(run_dir: Path) -> float:
    _, curve = load_csv(run_dir / "per_epoch.csv")
    block_counts: dict[int, int] = {}
    for row in curve:
        block = int(row["block"])
        block_counts[block] = block_counts.get(block, 0) + 1
    if block_counts != {0: 40, 1: 40}:
        raise ValueError(f"incomplete registered curve in {run_dir}: {block_counts}")
    final_block = max(int(row["block"]) for row in curve)
    values = [
        float(row["validation_hard_acc"])
        for row in curve
        if int(row["block"]) == final_block
    ]
    if not values:
        raise ValueError(f"missing final block curve in {run_dir}")
    return max(values)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def build_result(
    source: dict[str, str],
    runs_root: Path,
    protocol_sha256: str,
) -> dict[str, str]:
    mode = source["mode"]
    width = int(source["state_bits"])
    seed = int(source["seed"])
    run_name = f"v2_{mode}_w{width}_s{seed}"
    run_dir = runs_root / run_name
    best_hard = final_block_best_hard(run_dir)
    final_hard = float(source["discrete_acc"])
    late_drop = best_hard - final_hard
    if late_drop > 0.02 + 1e-12:
        health = "late_validation_drop"
    else:
        health = "pass"
    if int(source["audit_ops"]) <= 0:
        raise ValueError(f"missing strict runtime audit in {run_name}")
    if len(source["payload_sha256"]) != 64:
        raise ValueError(f"missing deployment hash in {run_name}")
    return {
        "result_id": f"bitplane_{mode}_w{width}_s{seed}",
        "method_id": METHOD[mode],
        "dataset": "sklearn_digits",
        "protocol_id": PROTOCOL_ID,
        "seed": str(seed),
        "seeds": "3",
        "aggregation": "paired seed",
        "selection_split": "validation",
        "soft_acc": source["soft_acc"],
        "hard_acc": source["discrete_acc"],
        "acc_gap": source["acc_gap"],
        "final_hard_acc": source["discrete_acc"],
        "best_hard_acc": format_value(best_hard),
        "train_time_s": source["train_time_s"],
        "budget": "2x40 block epochs",
        "unused_gate_ratio": source["unused_gate_ratio"],
        "gate_count": source["gate_count"],
        "depth": source["depth"],
        "fanout_max": source["fanout_max"],
        "runtime_compliance": "operator_audited_bool_int",
        "float_tensor_count": "0",
        "audit_ops": source["audit_ops"],
        "training_health": health,
        "decision": (
            "promote" if mode == "argmax" and width == 832
            else "baseline" if mode == "argmax"
            else "reject"
        ),
        "evidence": (
            "../docs/repro/bitplane_lut_digits_v2_20260724/"
            f"runs/{run_name}/result.json"
        ),
        "soft_loss": source["soft_loss"],
        "hard_loss": source["discrete_loss"],
        "loss_gap": source["loss_gap"],
        "test_soft_acc": source["test_soft_acc"],
        "test_hard_acc": source["test_discrete_acc"],
        "test_acc_gap": source["test_acc_gap"],
        "inactive_neuron_ratio": source["inactive_vote_ratio"],
        "protocol_sha256": protocol_sha256,
        "deployment_payload_sha256": source["payload_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()
    protocols = json.loads((REGISTRY / "protocols.json").read_text(encoding="utf-8"))[
        "protocols"
    ]
    protocol_sha256 = canonical_sha256(protocols[PROTOCOL_ID])
    _, source_rows = load_csv(args.summary_dir / "required_results.csv")
    if len(source_rows) != 18:
        raise ValueError(f"expected 18 source rows, found {len(source_rows)}")

    results_path = REGISTRY / "results.csv"
    fields, existing = load_csv(results_path)
    generated = [
        build_result(row, args.summary_dir / "runs", protocol_sha256)
        for row in source_rows
    ]
    expected_fields = set(fields)
    for row in generated:
        if set(row) != expected_fields:
            raise ValueError(
                f"registry schema mismatch for {row['result_id']}: "
                f"missing={sorted(expected_fields - set(row))}, "
                f"extra={sorted(set(row) - expected_fields)}"
            )
    generated_ids = {row["result_id"] for row in generated}
    output = [row for row in existing if row["result_id"] not in generated_ids]
    output.extend(generated)
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"generated": len(generated), "protocol_sha256": protocol_sha256}))


if __name__ == "__main__":
    main()
