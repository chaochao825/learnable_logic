from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPECTED_PROTOCOL_SHA256 = (
    "72c4f07bbdd533906e0b026d213ecb20768de65ddcf087a1a5fc8d4fec05bd53"
)
EXPECTED_PAYLOAD_SHA256 = (
    "51ccd7bbfab9877c9cefdc070d836620be0fbc2b985fb517c7a7ada16f28a60a"
)


def load_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_file_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in {
        ".csv",
        ".exit",
        ".json",
        ".log",
        ".md",
        ".py",
    }:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def main() -> None:
    protocol = load_json("protocol.json")
    strict = load_json("raw/d12/strict_integer_validation5000.json")
    d6_strict = load_json("raw/d6/strict_integer_validation5000.json")
    exact = load_json("raw/d12/integer_vs_carrier_limit20_full_source.json")
    d6_capacity = load_json("raw/d6/payload_capacity.json")
    d12_capacity = load_json("raw/d12/payload_capacity.json")
    d6_training = load_json("raw/training/d6_result.json")
    d12_training = load_json("raw/training/d12_result.json")
    remote_manifest = load_json("raw/d12/run_manifest.json")

    assert canonical_sha256(protocol) == EXPECTED_PROTOCOL_SHA256
    assert protocol["deployment_payload_sha256"] == EXPECTED_PAYLOAD_SHA256
    assert strict["source_artifact_sha256"] == EXPECTED_PAYLOAD_SHA256
    assert strict["source_artifact_kind"] == "payload"
    assert strict["count"] == 5000 and strict["correct"] == 3805
    assert strict["accuracy"] == 0.761
    assert strict["runtime_audit_operations"] > 0
    assert strict["runtime_floating_tensor_count"] == 0
    assert strict["payload_contains_real_values"] is False
    assert set(strict["payload_tensor_element_inventory"]) <= {
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
    }
    assert d6_strict["count"] == 5000 and d6_strict["correct"] == 3765
    assert d6_strict["accuracy"] == 0.753
    assert d6_strict["runtime_audit_operations"] > 0
    assert d6_strict["runtime_floating_tensor_count"] == 0

    assert exact["count"] == exact["exact_logit_rows"] == 20
    assert exact["maximum_absolute_logit_difference"] == 0
    assert exact["prediction_mismatches"] == 0
    assert exact["strict_runtime_floating_tensor_count"] == 0
    assert exact["payload_contains_real_values"] is False

    d6 = d6_capacity["capacity"]
    d12 = d12_capacity["capacity"]
    assert d6_capacity["artifact"]["contains_real_values"] is False
    assert d12_capacity["artifact"]["contains_real_values"] is False
    assert d12_capacity["artifact"]["sha256"] == EXPECTED_PAYLOAD_SHA256
    assert d6["transformer_block_depth"] == 6
    assert d12["transformer_block_depth"] == 12
    assert d12["general_matrix_multipliers"] == 0
    assert d12["logical_weight_coefficients"] > d6["logical_weight_coefficients"]
    assert d12["tensor_bits"] > d12["logical_bitpacked_weight_bits"]

    assert d6_training["final_validation_accuracy"] == 0.753
    assert d12_training["final_validation_accuracy"] == strict["accuracy"]
    assert d12_training["parameters"] == 7_101_696
    assert len(d6_training["history"]) == len(d12_training["history"]) == 10
    assert d6_training["history"][-1]["step"] == 50_000
    assert d12_training["history"][-1]["step"] == 50_000

    for name, metadata in remote_manifest["artifacts"].items():
        local = ROOT / "raw" / "d12" / name
        assert local.is_file(), f"missing remote artifact: {name}"
        assert local.stat().st_size == metadata["bytes"]
        assert file_sha256(local) == metadata["sha256"]

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "validation.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        repository_bytes = repository_file_bytes(path)
        files[relative] = {
            "bytes": len(repository_bytes),
            "sha256": hashlib.sha256(repository_bytes).hexdigest(),
        }

    output = {
        "schema_version": 1,
        "status": "pass",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "deployment_payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "assertions": {
            "strict_accuracy": strict["accuracy"],
            "strict_count": strict["count"],
            "strict_correct": strict["correct"],
            "runtime_audit_operations": strict["runtime_audit_operations"],
            "runtime_floating_tensor_count": strict[
                "runtime_floating_tensor_count"
            ],
            "exact_logit_rows": exact["exact_logit_rows"],
            "maximum_absolute_logit_difference": exact[
                "maximum_absolute_logit_difference"
            ],
            "d6_final_validation_accuracy": d6_training[
                "final_validation_accuracy"
            ],
            "d6_strict_accuracy": d6_strict["accuracy"],
            "d12_final_validation_accuracy": d12_training[
                "final_validation_accuracy"
            ],
        },
        "files": files,
    }
    (ROOT / "validation.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "files": len(files),
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
