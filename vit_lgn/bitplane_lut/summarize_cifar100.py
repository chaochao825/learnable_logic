"""Validate and summarize the pre-registered CIFAR-100 scale ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CONFIGS = (
    ("full-flat-v64-d2-s0", "bitplane_lut_argmax", "mixed-v64-d2"),
    (
        "full-spatial-v64-d2-s0",
        "bitplane_lut_spatial_argmax",
        "spatial-v64-d2",
    ),
    (
        "full-spatial-v128-d2-s0",
        "bitplane_lut_spatial_argmax",
        "spatial-v128-d2",
    ),
    (
        "full-spatial-v128-d4-s0",
        "bitplane_lut_spatial_argmax",
        "spatial-v128-d4",
    ),
    (
        "full-spatial-v256-d4-s0",
        "bitplane_lut_spatial_argmax",
        "spatial-v256-d4",
    ),
)
COMPARISONS = (
    ("routing_spatial_vs_mixed", "mixed-v64-d2", "spatial-v64-d2"),
    ("width_128_vs_64_d2", "spatial-v64-d2", "spatial-v128-d2"),
    ("depth_4_vs_2_v128", "spatial-v128-d2", "spatial-v128-d4"),
    ("width_256_vs_128_d4", "spatial-v128-d4", "spatial-v256-d4"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_run(
    runs_root: Path,
    run_id: str,
    method: str,
    variant: str,
) -> dict[str, object]:
    run_dir = runs_root / run_id
    result = load_json(run_dir / "result.json")
    manifest = load_json(run_dir / "run_manifest.json")
    replay = load_json(run_dir / "train_strict_metrics.json")
    diagnostic = load_json(run_dir / "payload_diagnostics.json")
    payload_path = run_dir / "hard_payload.pt"
    payload_sha256 = sha256_file(payload_path)
    strict = result["strict_runtime"]
    structure = result["structure"]
    blocks = result["block_results"]
    if payload_sha256 != result["hard_payload_sha256"]:
        raise RuntimeError(f"payload hash mismatch: {run_id}")
    if payload_sha256 != replay["payload_sha256"]:
        raise RuntimeError(f"training replay payload mismatch: {run_id}")
    if payload_sha256 != diagnostic["payload_sha256"]:
        raise RuntimeError(f"diagnostic payload mismatch: {run_id}")
    if strict["compliance"] != "operator_audited_bool_int":
        raise RuntimeError(f"non-strict deployment runtime: {run_id}")
    if int(strict["float_tensor_count"]) != 0:
        raise RuntimeError(f"deployment contains a real tensor: {run_id}")
    if int(strict["audit_operations"]) <= 0 or not strict[
        "exact_carrier_logit_match"
    ]:
        raise RuntimeError(f"incomplete deployment replay: {run_id}")
    if replay["runtime_compliance"] != "operator_audited_bool_int":
        raise RuntimeError(f"non-strict training replay: {run_id}")
    if int(replay["float_tensor_count"]) != 0 or int(
        replay["audit_operations"]
    ) <= 0:
        raise RuntimeError(f"invalid training replay audit: {run_id}")
    if int(structure["learned_dense_integer_matrix_count"]) != 0:
        raise RuntimeError(f"learned dense integer matrix leaked into {run_id}")
    if int(structure["learned_numeric_weight_count"]) != 0:
        raise RuntimeError(f"learned numeric weight leaked into {run_id}")
    if not result["training_health"]["finite_gradients"]:
        raise RuntimeError(f"non-finite training state: {run_id}")
    if len(blocks) != int(manifest["args"]["blocks"]):
        raise RuntimeError(f"incomplete block history: {run_id}")
    final_vote = blocks[-1]["vote_state_diagnostics"]
    return {
        "run_id": run_id,
        "method": method,
        "variant": variant,
        "dataset": "cifar100",
        "seed": int(result["seed"]),
        "candidate_policy": result["candidate_policy"],
        "votes_per_class": int(result["votes_per_class"]),
        "soft_acc": float(result["soft_acc"]),
        "discrete_acc": float(result["hard_acc"]),
        "acc_gap": float(result["acc_gap"]),
        "soft_loss": float(result["soft_loss"]),
        "discrete_loss": float(result["hard_loss"]),
        "loss_gap": float(result["loss_gap"]),
        "train_discrete_acc": float(replay["hard_acc"]),
        "test_soft_acc": float(result["test_soft_acc"]),
        "test_discrete_acc": float(result["test_hard_acc"]),
        "test_acc_gap": float(result["test_acc_gap"]),
        "train_time_s": float(result["train_time_s"]),
        "epochs_to_target": int(result["epochs_to_target"]),
        "epochs_ran": "+".join(str(block["epochs_ran"]) for block in blocks),
        "unused_gate_ratio": float(structure["unused_gate_ratio"]),
        "inactive_vote_ratio": float(final_vote["inactive_bit_ratio"]),
        "gate_count": int(structure["gate_count"]),
        "depth": int(structure["learned_logic_depth"]),
        "fanout_max": int(structure["fanout_max"]),
        "hard_payload_bits": int(structure["total_logical_payload_bits"]),
        "learned_logic_payload_bits": int(
            diagnostic["learned_logic_payload_bits"]
        ),
        "hard_payload_bytes": int(result["hard_payload_bytes"]),
        "raw_source_ratio": float(diagnostic["raw_source_ratio"]),
        "vote_source_ratio": float(diagnostic["vote_source_ratio"]),
        "identity_source_ratio": float(diagnostic["identity_source_ratio"]),
        "truth_dictionary_ratio": float(diagnostic["truth_dictionary_ratio"]),
        "final_vote_input_support_mean": float(
            diagnostic["final_vote_input_support_mean"]
        ),
        "class_input_support_mean": float(
            diagnostic["class_input_support_mean"]
        ),
        "audit_ops": int(strict["audit_operations"])
        + int(replay["audit_operations"]),
        "payload_sha256": payload_sha256,
        "protocol_sha256": manifest["protocol_sha256"],
        "source_bundle_sha256": canonical_sha256(manifest["source_sha256"]),
        "train_split_sha256": manifest["split"]["train_index_sha256"],
        "validation_split_sha256": manifest["split"][
            "validation_index_sha256"
        ],
        "test_split_sha256": manifest["split"]["official_test_image_sha256"],
        "evidence": str((run_dir / "result.json").resolve()),
    }


def deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_variant = {str(row["variant"]): row for row in rows}
    output = []
    for comparison, baseline_name, candidate_name in COMPARISONS:
        baseline = by_variant[baseline_name]
        candidate = by_variant[candidate_name]
        output.append(
            {
                "comparison": comparison,
                "baseline": baseline_name,
                "candidate": candidate_name,
                "validation_hard_acc_delta": float(candidate["discrete_acc"])
                - float(baseline["discrete_acc"]),
                "training_hard_acc_delta": float(candidate["train_discrete_acc"])
                - float(baseline["train_discrete_acc"]),
                "test_hard_acc_delta_report_only": float(
                    candidate["test_discrete_acc"]
                )
                - float(baseline["test_discrete_acc"]),
                "acc_gap_delta": float(candidate["acc_gap"])
                - float(baseline["acc_gap"]),
                "unused_gate_ratio_delta": float(candidate["unused_gate_ratio"])
                - float(baseline["unused_gate_ratio"]),
                "gate_count_ratio": int(candidate["gate_count"])
                / int(baseline["gate_count"]),
                "validation_scaling_success": float(candidate["discrete_acc"])
                > float(baseline["discrete_acc"]),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(
    rows: list[dict[str, object]], comparison_rows: list[dict[str, object]]
) -> str:
    best = max(rows, key=lambda row: float(row["discrete_acc"]))
    lines = [
        "# CIFAR-100 A8 bit-plane LUT capacity screen",
        "",
        "Architecture selection uses the fixed 5,000-row validation split. The",
        "official test split is report-only. Every row is an argmax-hardened,",
        "per-block-frozen Boolean LUT/wiring payload with fixed integer GroupSum.",
        "Training shadows are real-valued; deployment payloads and audited",
        "execution contain no real-valued tensor or learned numeric matrix.",
        "",
        "| variant | soft acc | hard acc | gap | train hard | test hard | hard loss | time | epochs | unused | gates | depth | fanout | vote support | class support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {soft} | {hard} | {gap} | {train} | {test} | "
            "{loss:.4f} | {time:.1f}s | {epochs} | {unused} | {gates} | "
            "{depth} | {fanout} | {vote_support:.1f} | {class_support:.1f} |".format(
                variant=row["variant"],
                soft=percent(float(row["soft_acc"])),
                hard=percent(float(row["discrete_acc"])),
                gap=percent(float(row["acc_gap"])),
                train=percent(float(row["train_discrete_acc"])),
                test=percent(float(row["test_discrete_acc"])),
                loss=float(row["discrete_loss"]),
                time=float(row["train_time_s"]),
                epochs=row["epochs_ran"],
                unused=percent(float(row["unused_gate_ratio"])),
                gates=row["gate_count"],
                depth=row["depth"],
                fanout=row["fanout_max"],
                vote_support=float(row["final_vote_input_support_mean"]),
                class_support=float(row["class_input_support_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Registered deltas",
            "",
            "| comparison | validation hard delta | training hard delta | test delta (report-only) | gap delta | unused delta | gates x | success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| {comparison} | {validation} | {training} | {test} | {gap} | "
            "{unused} | {gates:.2f} | {success} |".format(
                comparison=row["comparison"],
                validation=percent(float(row["validation_hard_acc_delta"])),
                training=percent(float(row["training_hard_acc_delta"])),
                test=percent(float(row["test_hard_acc_delta_report_only"])),
                gap=percent(float(row["acc_gap_delta"])),
                unused=percent(float(row["unused_gate_ratio_delta"])),
                gates=float(row["gate_count_ratio"]),
                success="yes" if row["validation_scaling_success"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Screen conclusion",
            "",
            f"The seed-0 validation winner is `{best['variant']}` at "
            f"{percent(float(best['discrete_acc']))} hard accuracy. Its strict "
            f"training accuracy is {percent(float(best['train_discrete_acc']))}, "
            f"so the remaining fit gap is measured rather than inferred.",
            "This is a capacity screen, not a promotion claim; the selected",
            "configuration requires the pre-registered seed-1/2 repeats.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [flatten_run(args.runs_root, *config) for config in CONFIGS]
    provenance_fields = (
        "protocol_sha256",
        "source_bundle_sha256",
        "train_split_sha256",
        "validation_split_sha256",
        "test_split_sha256",
    )
    for field in provenance_fields:
        if len({row[field] for row in rows}) != 1:
            raise RuntimeError(f"mismatched ladder provenance: {field}")
    comparison_rows = deltas(rows)
    best = max(rows, key=lambda row: float(row["discrete_acc"]))
    summary = {
        "runs": len(rows),
        "selection_split": "validation",
        "best_variant": best["variant"],
        "best_validation_hard_acc": best["discrete_acc"],
        "best_training_hard_acc": best["train_discrete_acc"],
        "best_test_hard_acc_report_only": best["test_discrete_acc"],
        "all_strict_no_real": True,
        "comparisons": comparison_rows,
        "protocol_sha256": best["protocol_sha256"],
        "source_bundle_sha256": best["source_bundle_sha256"],
        "summarizer_sha256": sha256_file(Path(__file__)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "required_results.csv", rows)
    write_csv(args.out_dir / "scale_deltas.csv", comparison_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        render_report(rows, comparison_rows), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
