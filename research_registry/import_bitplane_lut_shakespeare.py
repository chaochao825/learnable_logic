from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from research_registry.manifest import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research_registry"
PROTOCOL_ID = "bitplane_lut_shakespeare_char_scale_s0_v1"
VARIANTS = {
    "smoke-mixed-v32-d2": {
        "result_id": "bitplane_shakespeare_smoke_mixed_v32_d2_s0",
        "method_id": "bitplane_lut_sequence_mixed_argmax",
        "config_id": "shakespeare_mixed_v32_d2",
        "decision": "baseline",
    },
    "smoke-causal-v32-d2": {
        "result_id": "bitplane_shakespeare_smoke_causal_v32_d2_s0",
        "method_id": "bitplane_lut_sequence_causal_argmax",
        "config_id": "shakespeare_causal_v32_d2",
        "decision": "screen",
    },
    "causal-v32-d2": {
        "result_id": "bitplane_shakespeare_causal_v32_d2_s0",
        "method_id": "bitplane_lut_sequence_causal_argmax",
        "config_id": "shakespeare_causal_v32_d2",
        "decision": "screen",
    },
    "causal-v64-d2": {
        "result_id": "bitplane_shakespeare_causal_v64_d2_s0",
        "method_id": "bitplane_lut_sequence_causal_argmax",
        "config_id": "shakespeare_causal_v64_d2",
        "decision": "screen",
    },
    "causal-v64-d4": {
        "result_id": "bitplane_shakespeare_causal_v64_d4_s0",
        "method_id": "bitplane_lut_sequence_causal_argmax",
        "config_id": "shakespeare_causal_v64_d4",
        "decision": "screen",
    },
    "causal-v128-d4": {
        "result_id": "bitplane_shakespeare_causal_v128_d4_s0",
        "method_id": "bitplane_lut_sequence_causal_argmax",
        "config_id": "shakespeare_causal_v128_d4",
        "decision": "screen",
    },
}


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_capacity_csv(
    path: Path, fields: list[str], rows: list[dict[str, str]]
) -> None:
    if fields[-1] != "capacity_note":
        raise ValueError("capacity_note must remain the final capacity field")
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            note = '"' + row["capacity_note"].replace('"', '""') + '"'
            values = [row[field] for field in fields[:-1]]
            handle.write(",".join([*values, note]) + "\n")


def build_result(
    source: dict[str, str], protocol_sha256: str
) -> dict[str, str]:
    variant = source["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"unexpected character-model variant: {variant}")
    identity = VARIANTS[variant]
    if source["method"] != identity["method_id"]:
        raise ValueError(f"method mismatch for {variant}")
    sample_budget = "20000" if variant.startswith("smoke-") else "100000"
    return {
        "result_id": identity["result_id"],
        "method_id": identity["method_id"],
        "dataset": "tiny_shakespeare_char",
        "protocol_id": PROTOCOL_ID,
        "seed": source["seed"],
        "seeds": "1",
        "aggregation": "single pre-registered feasibility seed",
        "selection_split": "validation",
        "soft_acc": source["soft_acc"],
        "hard_acc": source["discrete_acc"],
        "acc_gap": source["acc_gap"],
        "final_hard_acc": source["discrete_acc"],
        "best_hard_acc": source["discrete_acc"],
        "train_time_s": source["train_time_s"],
        "budget": (
            f"{source['epochs_ran']} block epochs on {sample_budget} windows"
        ),
        "unused_gate_ratio": source["unused_gate_ratio"],
        "gate_count": source["gate_count"],
        "depth": source["depth"],
        "fanout_max": source["fanout_max"],
        "runtime_compliance": "operator_audited_bool_int",
        "float_tensor_count": "0",
        "audit_ops": source["audit_ops"],
        "training_health": "pass",
        "decision": identity["decision"],
        "evidence": (
            "../docs/repro/bitplane_lut_shakespeare_char_20260724/"
            f"runs/{source['run_id']}/result.json"
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


def update_capacity(source_rows: list[dict[str, str]]) -> None:
    fields, rows = load_csv(REGISTRY / "capacity.csv")
    by_key = {(row["method_id"], row["config_id"]): row for row in rows}
    for source in source_rows:
        identity = VARIANTS[source["variant"]]
        key = (identity["method_id"], identity["config_id"])
        if key not in by_key:
            raise ValueError(f"missing pre-registered capacity row: {key}")
        capacity = by_key[key]
        for capacity_key, result_key in (
            ("gate_count", "gate_count"),
            ("logic_depth", "depth"),
            ("hard_payload_bits", "hard_payload_bits"),
        ):
            if int(capacity[capacity_key]) != int(source[result_key]):
                raise ValueError(
                    f"capacity mismatch for {key}: {capacity_key}"
                )
        capacity["fanout_max"] = source["fanout_max"]
    write_capacity_csv(REGISTRY / "capacity.csv", fields, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()
    protocols = json.loads(
        (REGISTRY / "protocols.json").read_text(encoding="utf-8")
    )["protocols"]
    protocol_sha256 = canonical_sha256(protocols[PROTOCOL_ID])
    _, source_rows = load_csv(args.summary_dir / "required_results.csv")
    if not source_rows:
        raise ValueError("character-model summary is empty")
    if len({row["variant"] for row in source_rows}) != len(source_rows):
        raise ValueError("duplicate character-model variant")
    unexpected = {row["variant"] for row in source_rows} - set(VARIANTS)
    if unexpected:
        raise ValueError(f"unexpected variants: {sorted(unexpected)}")

    results_path = REGISTRY / "results.csv"
    fields, existing = load_csv(results_path)
    generated = [build_result(row, protocol_sha256) for row in source_rows]
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
    update_capacity(source_rows)
    print(
        json.dumps(
            {"generated": len(generated), "protocol_sha256": protocol_sha256},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
