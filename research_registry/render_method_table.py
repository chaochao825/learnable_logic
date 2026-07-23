from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research_registry"
OUTPUT = REGISTRY / "METHOD_TABLE.md"


def _rows(name: str) -> list[dict[str, str]]:
    with (REGISTRY / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render() -> str:
    methods = _rows("methods.csv")
    results = _rows("results.csv")
    synthesis = _rows("synthesis.csv")
    capacity = _rows("capacity.csv")
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in results:
        by_method.setdefault(row["method_id"], []).append(row)
    synthesis_by_method: dict[str, list[dict[str, str]]] = {}
    for row in synthesis:
        synthesis_by_method.setdefault(row["method_id"], []).append(row)
    lines = [
        "# Unified method table",
        "",
        "Generated from the machine-readable research registry. Hard accuracy is",
        "the primary outcome; gap is diagnostic and never sufficient by itself.",
        "",
        "| method | status | architecture | state boundary | training | hard runtime | best recorded hard result | conclusion |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method in methods:
        candidates = [row for row in by_method.get(method["method_id"], []) if row["hard_acc"]]
        if candidates:
            reference = [row for row in candidates if row["decision"] == "reference"]
            best = max(reference or candidates, key=lambda row: float(row["hard_acc"]))
            result = (
                f"{best['dataset']} {100 * float(best['hard_acc']):.2f}% "
                f"on {best['selection_split']} ({best['decision']})"
            )
            if best.get("test_hard_acc"):
                result += f"; test {100 * float(best['test_hard_acc']):.2f}%"
        elif synthesis_by_method.get(method["method_id"]):
            synthesized = synthesis_by_method[method["method_id"]]
            best = max(
                synthesized,
                key=lambda row: float(row["post_synthesis_hard_acc"]),
            )
            result = (
                f"{best['dataset']} "
                f"{100 * float(best['post_synthesis_hard_acc']):.2f}% "
                f"CEC-equivalent; {best['source_gate_count']}->"
                f"{best['mapped_gate_count']} K4"
            )
        else:
            result = "no compatible result"
        values = (
            method["method_id"],
            method["status"],
            method["architecture"],
            method["state_boundary"],
            method["training"],
            method["hard_runtime"],
            result,
            method["conclusion"],
        )
        escaped = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(
        [
            "",
            "## Registered synthesis",
            "",
            "Synthesis accuracy is inherited only after exact export replay and",
            "formal source/optimized netlist equivalence. GroupSum/argmax scope",
            "limitations remain explicit in the method and protocol rows.",
            "",
            "| synthesis | source result | seed | hard acc | test hard | gates | reduction | depth | payload bits | payload reduction | fanout | runtime (s) | equivalence |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in synthesis:
        values = (
            row["synthesis_id"],
            row["source_result_id"],
            row["seed"],
            f"{100 * float(row['post_synthesis_hard_acc']):.2f}%",
            f"{100 * float(row['post_synthesis_test_hard_acc']):.2f}%",
            f"{row['source_gate_count']}->{row['mapped_gate_count']}",
            f"{100 * float(row['gate_reduction']):.2f}%",
            f"{row['source_depth']}->{row['mapped_depth']}",
            f"{row['source_learned_payload_bits']}->{row['mapped_payload_bits']}",
            f"{100 * float(row['payload_reduction']):.2f}%",
            row["fanout_max"],
            row["synthesis_runtime_s"],
            row["equivalence"],
        )
        escaped = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(
        [
            "",
            "## Registered results",
            "",
            "`hard_acc` is measured on `selection_split`. The explicit test column",
            "is never used by the promotion script.",
            "",
            "| result | method | dataset | protocol | seed(s) | split | soft acc | hard acc | gap | soft loss | hard loss | loss gap | test hard | time (s) | unused | inactive | gates | depth | fanout | runtime | decision |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in results:
        percent = lambda key: (
            f"{100 * float(row[key]):.2f}%" if row.get(key) else ""
        )
        values = (
            row["result_id"],
            row["method_id"],
            row["dataset"],
            row["protocol_id"],
            row["seed"] or row["seeds"],
            row["selection_split"],
            percent("soft_acc"),
            percent("hard_acc"),
            percent("acc_gap"),
            row.get("soft_loss", ""),
            row.get("hard_loss", ""),
            row.get("loss_gap", ""),
            percent("test_hard_acc"),
            row["train_time_s"],
            percent("unused_gate_ratio"),
            percent("inactive_neuron_ratio"),
            row["gate_count"],
            row["depth"],
            row["fanout_max"],
            row["runtime_compliance"],
            row["decision"],
        )
        escaped = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(
        [
            "",
            "## Capacity configurations",
            "",
            "Payload size counts serialized hard tensors, not training shadows. Empty",
            "cells mean that the historical branch did not preserve the measurement.",
            "",
            "| method | config | input bits | direct preserved bits | state bits/token | parameters | hard payload bits | gates | depth | fanout max | capacity diagnosis |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in capacity:
        values = (
            row["method_id"],
            row["config_id"],
            row["input_patch_bits"],
            row["direct_preserved_input_bits"],
            row["state_storage_bits_per_token"],
            row["trainable_parameters"],
            row["hard_payload_bits"],
            row["gate_count"],
            row["logic_depth"],
            row["fanout_max"],
            row["capacity_note"],
        )
        escaped = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend(
        [
            "",
            "## Mainline decision",
            "",
            "`full_discrete_a8` remains the 76.10% strict integer accuracy reference,",
            "but its dense learned Wmag7 projections make it an integer baseline, not",
            "the logic-native mainline. `bitplane_lut_argmax` is the current",
            "logic-native candidate: doubling learned vote planes from 160 to 320",
            "improves mean validation hard accuracy by 3.46 pp and test hard accuracy",
            "by 4.44 pp with 3/3 paired validation wins, zero inactive vote planes,",
            "and a zero-real standalone runtime. Independent truth refit is rejected",
            "at -1.67 pp mean validation hard accuracy versus argmax. The greedy",
            "wiring refitter changes no post-training source even though training",
            "itself moves 65.27%-73.67% of connections away from default routes.",
            "ABC confirms all truth tables are logic-synthesizable, but the current",
            "functions are only weakly K4-compressible: 2.7%-6.9% fewer mapped",
            "LUTs, with mapped depth increasing from two to three. All 18 optimized",
            "networks pass CEC; GroupSum and argmax are outside that synthesis scope.",
            "Next work must use joint task-margin-aware hard fitting and matched",
            "vote-width/spatial scaling without learned dense numeric matrices.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != content:
            raise SystemExit("METHOD_TABLE.md is stale; run the renderer")
        return
    OUTPUT.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
