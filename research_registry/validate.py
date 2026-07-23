from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from research_registry.manifest import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research_registry"

METHOD_STATUSES = {
    "reference",
    "candidate",
    "screen",
    "rejected",
    "external_ceiling",
    "unvalidated",
}
RESULT_DECISIONS = {"baseline", "ceiling", "reference", "reject", "screen", "promote"}


def _load_csv(name: str) -> list[dict[str, str]]:
    with (REGISTRY / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{name} is empty")
    return rows


def _unique(rows: Iterable[Mapping[str, str]], key: str, name: str) -> None:
    seen: set[str] = set()
    for row in rows:
        value = row.get(key, "")
        if not value:
            raise ValueError(f"{name} has an empty {key}")
        if value in seen:
            raise ValueError(f"{name} duplicates {key}={value}")
        seen.add(value)


def _evidence_exists(value: str) -> bool:
    if value.startswith(("https://", "http://")):
        return True
    return (REGISTRY / value).resolve().exists()


def _optional_rate(row: Mapping[str, str], key: str) -> None:
    value = row.get(key, "")
    if value and not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{row.get('result_id')} has invalid {key}={value}")


def validate() -> dict[str, int]:
    branches = _load_csv("branches.csv")
    methods = _load_csv("methods.csv")
    results = _load_csv("results.csv")
    synthesis = _load_csv("synthesis.csv")
    capacity = _load_csv("capacity.csv")
    protocols = json.loads((REGISTRY / "protocols.json").read_text(encoding="utf-8"))[
        "protocols"
    ]
    policy = json.loads(
        (REGISTRY / "promotion_policy.json").read_text(encoding="utf-8")
    )

    _unique(methods, "method_id", "methods.csv")
    _unique(results, "result_id", "results.csv")
    _unique(synthesis, "synthesis_id", "synthesis.csv")
    method_ids = {row["method_id"] for row in methods}
    capacity_ids = {row["method_id"] for row in capacity}
    missing_capacity = method_ids - capacity_ids
    if missing_capacity:
        raise ValueError(f"methods missing capacity rows: {sorted(missing_capacity)}")

    branch_keys: set[tuple[str, str]] = set()
    for branch in branches:
        key = (branch["repository"], branch["branch"])
        if key in branch_keys:
            raise ValueError(f"duplicate branch {key}")
        branch_keys.add(key)
        if len(branch["head"]) != 40:
            raise ValueError(f"branch {key} does not use a full commit hash")
        if not _evidence_exists(branch["evidence"]):
            raise ValueError(f"missing branch evidence: {branch['evidence']}")

    for method in methods:
        if method["status"] not in METHOD_STATUSES:
            raise ValueError(f"invalid method status: {method['status']}")
        if not method["source_ref"]:
            raise ValueError(f"{method['method_id']} has no source_ref")
        if not _evidence_exists(method["evidence"]):
            raise ValueError(f"missing method evidence: {method['evidence']}")

    for row in capacity:
        if row["method_id"] not in method_ids:
            raise ValueError(f"unknown capacity method: {row['method_id']}")
        for key in (
            "input_patch_bits",
            "state_width",
            "state_bits_per_value",
            "state_storage_bits_per_token",
            "tokens",
            "heads",
            "qk_bits",
            "topk",
            "trainable_parameters",
            "hard_payload_bits",
            "gate_count",
            "logic_depth",
            "fanout_max",
        ):
            value = row.get(key, "")
            if value and int(value) < 0:
                raise ValueError(f"negative capacity field {key}={value}")

    for result in results:
        if any(value is None for value in result.values()):
            raise ValueError(f"{result.get('result_id')} has a short CSV row")
        if result["method_id"] not in method_ids:
            raise ValueError(f"unknown result method: {result['method_id']}")
        if result["protocol_id"] not in protocols:
            raise ValueError(f"unknown protocol: {result['protocol_id']}")
        if result["decision"] not in RESULT_DECISIONS:
            raise ValueError(f"invalid decision: {result['decision']}")
        for key in (
            "soft_acc",
            "hard_acc",
            "acc_gap",
            "unused_gate_ratio",
            "test_soft_acc",
            "test_hard_acc",
            "test_acc_gap",
            "inactive_neuron_ratio",
        ):
            _optional_rate(result, key)
        for key in ("protocol_sha256", "deployment_payload_sha256"):
            value = result.get(key, "")
            if value and len(value) != 64:
                raise ValueError(f"{result['result_id']} has invalid {key}")
        protocol_hash = result.get("protocol_sha256", "")
        if protocol_hash and protocol_hash != canonical_sha256(
            protocols[result["protocol_id"]]
        ):
            raise ValueError(
                f"{result['result_id']} protocol hash does not match registry"
            )
        if not _evidence_exists(result["evidence"]):
            raise ValueError(f"missing result evidence: {result['evidence']}")
        if result["runtime_compliance"] == "operator_audited_bool_int":
            if result["float_tensor_count"] != "0":
                raise ValueError(f"{result['result_id']} lacks zero-float proof")
            if int(result["audit_ops"] or 0) <= 0:
                raise ValueError(f"{result['result_id']} lacks audited operations")
        if result["decision"] in {"promote", "reference"}:
            if (
                result["decision"] == "promote"
                and int(result["seeds"]) < int(policy["minimum_seeds"])
            ):
                raise ValueError(f"{result['result_id']} is promoted with too few seeds")
            if result["runtime_compliance"] != policy["required_runtime_compliance"]:
                raise ValueError(
                    f"{result['result_id']} is a headline result without strict runtime"
                )
            if result["training_health"] != "pass":
                raise ValueError(
                    f"{result['result_id']} is a headline result after unhealthy training"
                )
            if len(result["protocol_sha256"]) != 64:
                raise ValueError(
                    f"{result['result_id']} is a headline result without protocol hash"
                )
            if policy.get("require_deployment_payload_hash") and len(
                result["deployment_payload_sha256"]
            ) != 64:
                raise ValueError(
                    f"{result['result_id']} is a headline result without deployment artifact"
                )

    result_ids = {row["result_id"] for row in results}
    for row in synthesis:
        if any(value is None for value in row.values()):
            raise ValueError(f"{row.get('synthesis_id')} has a short synthesis row")
        if row["method_id"] not in method_ids:
            raise ValueError(f"unknown synthesis method: {row['method_id']}")
        if row["source_result_id"] not in result_ids:
            raise ValueError(f"unknown synthesis source result: {row['source_result_id']}")
        if row["protocol_id"] not in protocols:
            raise ValueError(f"unknown synthesis protocol: {row['protocol_id']}")
        if row["protocol_sha256"] != canonical_sha256(protocols[row["protocol_id"]]):
            raise ValueError(f"{row['synthesis_id']} protocol hash mismatch")
        for key in (
            "source_hard_acc",
            "post_synthesis_hard_acc",
            "source_test_hard_acc",
            "post_synthesis_test_hard_acc",
            "gate_reduction",
            "payload_reduction",
        ):
            _optional_rate(row, key)
        for key in (
            "source_gate_count",
            "mapped_gate_count",
            "source_depth",
            "mapped_depth",
            "source_learned_payload_bits",
            "mapped_payload_bits",
            "fanout_max",
            "export_vectors",
        ):
            if int(row[key]) < 0:
                raise ValueError(f"negative synthesis metric {key} in {row['synthesis_id']}")
        if float(row["synthesis_runtime_s"]) < 0:
            raise ValueError(f"negative synthesis runtime in {row['synthesis_id']}")
        if row["export_exact"] != "true":
            raise ValueError(f"{row['synthesis_id']} failed export replay")
        if row["equivalence"] != "abc_cec_equivalent":
            raise ValueError(f"{row['synthesis_id']} lacks formal equivalence")
        if row["source_hard_acc"] != row["post_synthesis_hard_acc"]:
            raise ValueError(f"{row['synthesis_id']} changed validation accuracy")
        if row["source_test_hard_acc"] != row["post_synthesis_test_hard_acc"]:
            raise ValueError(f"{row['synthesis_id']} changed test accuracy")
        if not _evidence_exists(row["evidence"]) or not _evidence_exists(row["artifact"]):
            raise ValueError(f"missing synthesis evidence for {row['synthesis_id']}")
        artifact = (REGISTRY / row["artifact"]).resolve()
        actual_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_sha256 != row["artifact_sha256"]:
            raise ValueError(f"artifact hash mismatch for {row['synthesis_id']}")

    return {
        "branches": len(branches),
        "methods": len(methods),
        "results": len(results),
        "synthesis_rows": len(synthesis),
        "capacity_rows": len(capacity),
        "protocols": len(protocols),
    }


def main() -> None:
    summary = validate()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
