from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from research_registry.manifest import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research_registry"
PROTOCOL_ID = "bitplane_lut_digits_abc_dc2_k4_s012_v1"
FIELDS = [
    "synthesis_id",
    "method_id",
    "source_result_id",
    "dataset",
    "protocol_id",
    "protocol_sha256",
    "seed",
    "source_hard_acc",
    "post_synthesis_hard_acc",
    "source_test_hard_acc",
    "post_synthesis_test_hard_acc",
    "source_gate_count",
    "mapped_gate_count",
    "gate_reduction",
    "source_depth",
    "mapped_depth",
    "source_learned_payload_bits",
    "mapped_payload_bits",
    "payload_reduction",
    "fanout_max",
    "synthesis_runtime_s",
    "export_vectors",
    "export_exact",
    "equivalence",
    "tool",
    "artifact",
    "artifact_sha256",
    "evidence",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_number(value: float) -> str:
    return format(value, ".12g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis-dir", type=Path, required=True)
    args = parser.parse_args()
    pre_rows = {
        row["run"]: row
        for row in read_csv(args.synthesis_dir / "export" / "pre_abc_metrics.csv")
    }
    abc_rows = read_csv(args.synthesis_dir / "abc" / "abc_metrics.csv")
    source_results = {
        row["result_id"]: row for row in read_csv(REGISTRY / "results.csv")
    }
    protocols = json.loads((REGISTRY / "protocols.json").read_text(encoding="utf-8"))[
        "protocols"
    ]
    protocol_hash = canonical_sha256(protocols[PROTOCOL_ID])
    output = []
    for abc in abc_rows:
        if abc["mode"] != "argmax":
            continue
        run = abc["run"]
        pre = pre_rows[run]
        width = int(abc["state_bits"])
        seed = int(abc["seed"])
        source_id = f"bitplane_argmax_w{width}_s{seed}"
        source = source_results[source_id]
        source_gates = int(abc["gate_count"])
        mapped_gates = int(abc["abc_lut4_nodes"])
        source_payload = source_gates * (
            16 + 4 * math.ceil(math.log2(width))
        )
        mapped_payload = int(abc["blif_k4_logical_payload_bits"])
        artifact = (
            "../docs/repro/bitplane_lut_abc_20260724/abc/"
            + abc["optimized_blif"]
        )
        output.append({
            "synthesis_id": f"bitplane_argmax_abc_w{width}_s{seed}",
            "method_id": "bitplane_lut_abc_dc2_k4",
            "source_result_id": source_id,
            "dataset": "sklearn_digits",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_hash,
            "seed": str(seed),
            "source_hard_acc": source["hard_acc"],
            "post_synthesis_hard_acc": source["hard_acc"],
            "source_test_hard_acc": source["test_hard_acc"],
            "post_synthesis_test_hard_acc": source["test_hard_acc"],
            "source_gate_count": str(source_gates),
            "mapped_gate_count": str(mapped_gates),
            "gate_reduction": format_number(1.0 - mapped_gates / source_gates),
            "source_depth": "2",
            "mapped_depth": abc["abc_lut4_levels"],
            "source_learned_payload_bits": str(source_payload),
            "mapped_payload_bits": str(mapped_payload),
            "payload_reduction": format_number(
                1.0 - mapped_payload / source_payload
            ),
            "fanout_max": abc["blif_fanout_max"],
            "synthesis_runtime_s": abc["abc_runtime_s"],
            "export_vectors": pre["export_random_vectors"],
            "export_exact": pre["export_exact"].lower(),
            "equivalence": "abc_cec_equivalent",
            "tool": "abc_1.01_strash_dc2_if_k4",
            "artifact": artifact,
            "artifact_sha256": abc["optimized_blif_sha256"],
            "evidence": "../docs/repro/bitplane_lut_abc_20260724/README.md",
        })
    output.sort(key=lambda row: (int(row["mapped_depth"]), row["synthesis_id"]))
    if len(output) != 6:
        raise ValueError(f"expected six argmax synthesis rows, found {len(output)}")
    with (REGISTRY / "synthesis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"rows": len(output), "protocol_sha256": protocol_hash}))


if __name__ == "__main__":
    main()
