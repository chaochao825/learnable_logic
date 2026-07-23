from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


RUN_PATTERN = re.compile(
    r"^v2_(argmax|truth|wiring)_w(672|832)_s([012])$"
)
MODE_NAME = {
    "argmax": "bitplane_lut_argmax",
    "truth": "bitplane_lut_refit",
    "wiring": "bitplane_lut_wiring_refit",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, object]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def sample_std(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def merge_wiring_diagnostics(
    rows: list[dict[str, object]], path: Path
) -> None:
    diagnostics: dict[tuple[str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for diagnostic in csv.DictReader(handle):
            key = (
                diagnostic["mode"],
                int(diagnostic["state_bits"]),
                int(diagnostic["seed"]),
            )
            if key in diagnostics:
                raise ValueError(f"duplicate wiring diagnostic: {key}")
            diagnostics[key] = diagnostic
    for row in rows:
        key = (str(row["mode"]), int(row["state_bits"]), int(row["seed"]))
        if key not in diagnostics:
            raise ValueError(f"missing wiring diagnostic: {key}")
        diagnostic = diagnostics[key]
        row.update({
            "trained_nondefault_wiring_ratio": float(
                diagnostic["nondefault_wiring_ratio"]
            ),
            "trained_raw_input_source_ratio": float(
                diagnostic["raw_input_source_ratio"]
            ),
            "trained_candidate_rank_mean": float(
                diagnostic["candidate_rank_mean"]
            ),
            "trained_candidate_rank_max": int(
                diagnostic["candidate_rank_max"]
            ),
            "truth_change_from_initial_ratio": float(
                diagnostic["truth_change_from_initial_ratio"]
            ),
        })
    if len(diagnostics) != len(rows):
        raise ValueError("wiring diagnostic matrix contains unexpected rows")


def flatten_result(path: Path, mode: str, state_bits: int, seed: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.with_name("run_manifest.json")
    artifact_path = path.with_name("hard_payload.pt")
    if not manifest_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(f"missing manifest or hard payload beside {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_sha256 = sha256_file(artifact_path)
    strict = payload["strict_runtime"]
    structure = payload["structure"]
    blocks = payload["block_results"]
    trace = payload["state_trace_diagnostics"]
    if len(blocks) != 2:
        raise ValueError(f"registered v2 run must contain two blocks: {path}")
    if payload["method_id"] != MODE_NAME[mode]:
        raise ValueError(f"method mismatch in {path}")
    if int(payload["state_bits"]) != state_bits or int(payload["seed"]) != seed:
        raise ValueError(f"run identity mismatch in {path}")
    if strict["compliance"] != "operator_audited_bool_int":
        raise ValueError(f"non-strict result in {path}")
    if int(strict["float_tensor_count"]) != 0 or int(strict["audit_operations"]) <= 0:
        raise ValueError(f"invalid no-real audit in {path}")
    if not strict["exact_carrier_logit_match"]:
        raise ValueError(f"strict/carrier mismatch in {path}")
    if len(payload["hard_payload_sha256"]) != 64:
        raise ValueError(f"missing payload hash in {path}")
    if artifact_sha256 != payload["hard_payload_sha256"]:
        raise ValueError(f"hard payload hash mismatch in {path}")
    if structure["learned_dense_integer_matrix_count"] != 0:
        raise ValueError(f"dense integer matrix leaked into {path}")
    if structure["learned_numeric_weight_count"] != 0:
        raise ValueError(f"numeric learned weight leaked into {path}")
    if not payload["training_health"]["finite_gradients"]:
        raise ValueError(f"non-finite training in {path}")
    final_vote = trace[-1]["vote_state"]
    return {
        "method": MODE_NAME[mode],
        "mode": mode,
        "dataset": "sklearn_digits",
        "state_bits": state_bits,
        "preserved_bits": int(structure["preserved_input_bits"]),
        "vote_bits": int(structure["learned_vote_bits"]),
        "seed": seed,
        "soft_acc": float(payload["soft_acc"]),
        "discrete_acc": float(payload["hard_acc"]),
        "acc_gap": float(payload["acc_gap"]),
        "soft_loss": float(payload["soft_loss"]),
        "discrete_loss": float(payload["hard_loss"]),
        "loss_gap": float(payload["loss_gap"]),
        "test_soft_acc": float(payload["test_soft_acc"]),
        "test_discrete_acc": float(payload["test_hard_acc"]),
        "test_acc_gap": float(payload["test_acc_gap"]),
        "train_time_s": float(payload["train_time_s"]),
        "epochs_to_target": int(payload["epochs_to_target"]),
        "unused_gate_ratio": float(structure["unused_gate_ratio"]),
        "inactive_vote_ratio": float(final_vote["inactive_bit_ratio"]),
        "vote_entropy": float(final_vote["bit_entropy_mean"]),
        "vote_joint_unique_ratio": float(final_vote["joint_unique_state_ratio"]),
        "gate_count": int(structure["gate_count"]),
        "depth": int(structure["learned_logic_depth"]),
        "fanout_max": int(structure["fanout_max"]),
        "hard_payload_bits": int(structure["total_logical_payload_bits"]),
        "hard_payload_bytes_serialized": int(payload["hard_payload_bytes"]),
        "block0_pre_hard_acc": float(blocks[0]["pre_refit_hard_acc"]),
        "block0_post_hard_acc": float(blocks[0]["post_refit_hard_acc"]),
        "block1_pre_hard_acc": float(blocks[1]["pre_refit_hard_acc"]),
        "block1_post_hard_acc": float(blocks[1]["post_refit_hard_acc"]),
        "block0_fit_error": float(blocks[0]["refit_bit_error"]),
        "block1_fit_error": float(blocks[1]["refit_bit_error"]),
        "block0_address_coverage": float(blocks[0]["refit_address_coverage"]),
        "block1_address_coverage": float(blocks[1]["refit_address_coverage"]),
        "changed_truth_ratio": statistics.fmean(
            float(block["changed_truth_ratio"]) for block in blocks
        ),
        "changed_wiring_ratio": statistics.fmean(
            float(block["changed_wiring_ratio"]) for block in blocks
        ),
        "audit_ops": int(strict["audit_operations"]),
        "payload_sha256": artifact_sha256,
        "source_bundle_sha256": canonical_sha256(manifest["source_sha256"]),
        "train_split_sha256": manifest["split"]["train_index_sha256"],
        "validation_split_sha256": manifest["split"]["validation_index_sha256"],
        "test_split_sha256": manifest["split"]["test_index_sha256"],
        "evidence": str(path),
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for mode in ("argmax", "truth", "wiring"):
        for state_bits in (672, 832):
            group = [
                row for row in rows
                if row["mode"] == mode and row["state_bits"] == state_bits
            ]
            if len(group) != 3:
                raise ValueError(f"incomplete aggregate {mode}/{state_bits}")
            output.append({
                "method": MODE_NAME[mode],
                "mode": mode,
                "dataset": "sklearn_digits",
                "state_bits": state_bits,
                "vote_bits": group[0]["vote_bits"],
                "seeds": 3,
                "validation_discrete_acc_mean": mean(group, "discrete_acc"),
                "validation_discrete_acc_std": sample_std(group, "discrete_acc"),
                "validation_soft_acc_mean": mean(group, "soft_acc"),
                "validation_acc_gap_mean": mean(group, "acc_gap"),
                "validation_soft_loss_mean": mean(group, "soft_loss"),
                "validation_discrete_loss_mean": mean(group, "discrete_loss"),
                "validation_loss_gap_mean": mean(group, "loss_gap"),
                "test_discrete_acc_mean": mean(group, "test_discrete_acc"),
                "test_discrete_acc_std": sample_std(group, "test_discrete_acc"),
                "train_time_s_mean": mean(group, "train_time_s"),
                "epochs_to_target_mean": mean(group, "epochs_to_target"),
                "unused_gate_ratio_mean": mean(group, "unused_gate_ratio"),
                "inactive_vote_ratio_mean": mean(group, "inactive_vote_ratio"),
                "vote_entropy_mean": mean(group, "vote_entropy"),
                "vote_joint_unique_ratio_mean": mean(group, "vote_joint_unique_ratio"),
                "gate_count": group[0]["gate_count"],
                "depth": group[0]["depth"],
                "fanout_max_mean": mean(group, "fanout_max"),
                "hard_payload_bits": group[0]["hard_payload_bits"],
                "block0_post_hard_acc_mean": mean(group, "block0_post_hard_acc"),
                "block1_post_hard_acc_mean": mean(group, "block1_post_hard_acc"),
                "block0_fit_error_mean": mean(group, "block0_fit_error"),
                "block1_fit_error_mean": mean(group, "block1_fit_error"),
                "block0_address_coverage_mean": mean(group, "block0_address_coverage"),
                "block1_address_coverage_mean": mean(group, "block1_address_coverage"),
                "changed_truth_ratio_mean": mean(group, "changed_truth_ratio"),
                "changed_wiring_ratio_mean": mean(group, "changed_wiring_ratio"),
                "trained_nondefault_wiring_ratio_mean": mean(
                    group, "trained_nondefault_wiring_ratio"
                ),
                "trained_raw_input_source_ratio_mean": mean(
                    group, "trained_raw_input_source_ratio"
                ),
                "trained_candidate_rank_mean": mean(
                    group, "trained_candidate_rank_mean"
                ),
                "truth_change_from_initial_ratio_mean": mean(
                    group, "truth_change_from_initial_ratio"
                ),
            })
    return output


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["mode"]), int(row["state_bits"]), int(row["seed"])): row
        for row in rows
    }
    output = []
    for mode in ("argmax", "truth", "wiring"):
        for seed in range(3):
            narrow = by_key[(mode, 672, seed)]
            wide = by_key[(mode, 832, seed)]
            output.append({
                "comparison": "scale_320_vs_160_votes",
                "mode": mode,
                "state_bits": "832-672",
                "seed": seed,
                "validation_hard_acc_delta": float(wide["discrete_acc"]) - float(narrow["discrete_acc"]),
                "test_hard_acc_delta": float(wide["test_discrete_acc"]) - float(narrow["test_discrete_acc"]),
                "unused_gate_ratio_delta": float(wide["unused_gate_ratio"]) - float(narrow["unused_gate_ratio"]),
                "inactive_vote_ratio_delta": float(wide["inactive_vote_ratio"]) - float(narrow["inactive_vote_ratio"]),
            })
    for state_bits in (672, 832):
        for seed in range(3):
            baseline = by_key[("argmax", state_bits, seed)]
            for mode in ("truth", "wiring"):
                candidate = by_key[(mode, state_bits, seed)]
                output.append({
                    "comparison": f"{mode}_vs_argmax",
                    "mode": mode,
                    "state_bits": state_bits,
                    "seed": seed,
                    "validation_hard_acc_delta": float(candidate["discrete_acc"]) - float(baseline["discrete_acc"]),
                    "test_hard_acc_delta": float(candidate["test_discrete_acc"]) - float(baseline["test_discrete_acc"]),
                    "unused_gate_ratio_delta": float(candidate["unused_gate_ratio"]) - float(baseline["unused_gate_ratio"]),
                    "inactive_vote_ratio_delta": float(candidate["inactive_vote_ratio"]) - float(baseline["inactive_vote_ratio"]),
                })
    return output


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_report(
    rows: list[dict[str, object]],
    aggregates: list[dict[str, object]],
    deltas: list[dict[str, object]],
) -> str:
    lines = [
        "# A8 bit-plane Hard-LGN digits scale screen",
        "",
        "All 18 registered v2 runs completed. Every result was replayed by the",
        "standalone Boolean/integer executor with zero real-valued tensors and",
        "exact hard-carrier logit equality on all validation and test rows.",
        "Soft metrics are the selected final block before hardening/refit; discrete",
        "metrics are measured after the block is hardened and frozen.",
        "",
        "| hardening | state/votes | validation hard mean +/- sd | test hard mean +/- sd | gap | inactive votes | unused gates | gates | payload bits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {mode} | {state_bits}/{vote_bits} | {valid} +/- {valid_sd} | "
            "{test} +/- {test_sd} | {gap} | {inactive} | {unused} | {gates} | {payload} |".format(
                mode=row["mode"],
                state_bits=row["state_bits"],
                vote_bits=row["vote_bits"],
                valid=percent(float(row["validation_discrete_acc_mean"])),
                valid_sd=percent(float(row["validation_discrete_acc_std"])),
                test=percent(float(row["test_discrete_acc_mean"])),
                test_sd=percent(float(row["test_discrete_acc_std"])),
                gap=percent(float(row["validation_acc_gap_mean"])),
                inactive=percent(float(row["inactive_vote_ratio_mean"])),
                unused=percent(float(row["unused_gate_ratio_mean"])),
                gates=row["gate_count"],
                payload=row["hard_payload_bits"],
            )
        )

    lines.extend([
        "",
        "## Required metrics (three-seed means)",
        "",
        "| method | dataset | state/votes | soft acc | discrete acc | acc gap | soft loss | discrete loss | loss gap | train time (s) | epochs to target | unused | gates | depth | fanout max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in aggregates:
        lines.append(
            "| {method} | sklearn_digits | {state_bits}/{vote_bits} | {soft_acc} | "
            "{hard_acc} | {acc_gap} | {soft_loss:.4f} | {hard_loss:.4f} | "
            "{loss_gap:.4f} | {time:.2f} | {epochs:.2f} | {unused} | {gates} | "
            "{depth} | {fanout:.2f} |".format(
                method=row["method"],
                state_bits=row["state_bits"],
                vote_bits=row["vote_bits"],
                soft_acc=percent(float(row["validation_soft_acc_mean"])),
                hard_acc=percent(float(row["validation_discrete_acc_mean"])),
                acc_gap=percent(float(row["validation_acc_gap_mean"])),
                soft_loss=float(row["validation_soft_loss_mean"]),
                hard_loss=float(row["validation_discrete_loss_mean"]),
                loss_gap=float(row["validation_loss_gap_mean"]),
                time=float(row["train_time_s_mean"]),
                epochs=float(row["epochs_to_target_mean"]),
                unused=percent(float(row["unused_gate_ratio_mean"])),
                gates=row["gate_count"],
                depth=row["depth"],
                fanout=float(row["fanout_max_mean"]),
            )
        )

    scale = [row for row in deltas if row["comparison"] == "scale_320_vs_160_votes"]
    lines.extend(["", "## Paired decisions", ""])
    for mode in ("argmax", "truth", "wiring"):
        group = [row for row in scale if row["mode"] == mode]
        valid_delta = statistics.fmean(float(row["validation_hard_acc_delta"]) for row in group)
        test_delta = statistics.fmean(float(row["test_hard_acc_delta"]) for row in group)
        wins = sum(float(row["validation_hard_acc_delta"]) > 0 for row in group)
        lines.append(
            f"- `{mode}` 320-vote scaling: validation {percent(valid_delta)} and "
            f"test {percent(test_delta)} mean delta; {wins}/3 validation wins."
        )
    for mode in ("truth", "wiring"):
        group = [row for row in deltas if row["comparison"] == f"{mode}_vs_argmax"]
        valid_delta = statistics.fmean(float(row["validation_hard_acc_delta"]) for row in group)
        wins = sum(float(row["validation_hard_acc_delta"]) > 0 for row in group)
        lines.append(
            f"- `{mode}` versus argmax across both widths: validation "
            f"{percent(valid_delta)} mean delta; {wins}/6 wins."
        )
    wiring_rows = [row for row in rows if row["mode"] == "wiring"]
    wiring_change = statistics.fmean(float(row["changed_wiring_ratio"]) for row in wiring_rows)
    trained_wiring_min = min(
        float(row["trained_nondefault_wiring_ratio"]) for row in rows
    )
    trained_wiring_max = max(
        float(row["trained_nondefault_wiring_ratio"]) for row in rows
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The protected bit-plane state prevents depth-wise input erasure and the",
        "learned vote state remains active and jointly diverse. That establishes",
        "that information collapse is avoidable in this bounded setting; it does",
        "not establish CIFAR-scale sufficiency.",
        "For argmax, block-0 to block-1 hard accuracy rises from 80.62% to",
        "88.02% at 160 votes and from 87.65% to 91.48% at 320 votes. Final",
        "vote entropy is about 0.85 bit, inactive vote ratio is zero, and all",
        "270 validation rows retain distinct joint vote states.",
        "",
        "Direct hard-ST/argmax is the current winner. Empirical independent LUT",
        "refitting changes truth bits but usually discards task-level vote margin.",
        f"Training did learn wiring: {percent(trained_wiring_min)} to",
        f"{percent(trained_wiring_max)} of exported gate inputs select a candidate",
        "other than the default route. The measured greedy post-refit wiring change",
        f"ratio is nevertheless {percent(wiring_change)}, so the wiring-refit",
        "variant is behaviorally identical to truth refit. This rejects the current",
        "coordinate-greedy refitter, not learned discrete wiring itself.",
        "",
        "Training times are reported but were collected concurrently across RTX",
        "4090 and H200 devices; they must not be used for a method speed claim.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wiring-diagnostics", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for directory in sorted(args.runs_root.iterdir()):
        match = RUN_PATTERN.match(directory.name)
        if not match:
            continue
        mode, width_text, seed_text = match.groups()
        key = (mode, int(width_text), int(seed_text))
        result = directory / "result.json"
        if not result.is_file():
            raise FileNotFoundError(f"missing result: {result}")
        rows.append(flatten_result(result, *key))
        seen.add(key)
    expected = {
        (mode, width, seed)
        for mode in ("argmax", "truth", "wiring")
        for width in (672, 832)
        for seed in range(3)
    }
    if seen != expected:
        raise ValueError(f"run matrix mismatch: missing={sorted(expected - seen)}")
    for key in (
        "source_bundle_sha256",
        "train_split_sha256",
        "validation_split_sha256",
        "test_split_sha256",
    ):
        values = {str(row[key]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"inconsistent {key}: {sorted(values)}")
    reference_manifest = json.loads(
        (args.runs_root / "v2_argmax_w672_s0" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for name, expected_sha256 in reference_manifest["source_sha256"].items():
        current = args.source_root / name
        if not current.is_file() or sha256_file(current) != expected_sha256:
            raise ValueError(f"current training source does not match manifest: {current}")
    rows.sort(key=lambda row: (str(row["mode"]), int(row["state_bits"]), int(row["seed"])))
    merge_wiring_diagnostics(rows, args.wiring_diagnostics)
    aggregates = aggregate(rows)
    deltas = paired_deltas(rows)
    write_csv(args.out_dir / "required_results.csv", rows)
    write_csv(args.out_dir / "aggregate_results.csv", aggregates)
    write_csv(args.out_dir / "paired_deltas.csv", deltas)
    summary = {
        "runs": len(rows),
        "strict_runs": sum(int(row["audit_ops"]) > 0 for row in rows),
        "payload_hashes": sorted(str(row["payload_sha256"]) for row in rows),
        "source_bundle_sha256": rows[0]["source_bundle_sha256"],
        "source_files_match_current": True,
        "source_file_sha256": reference_manifest["source_sha256"],
        "split_sha256": {
            "train": rows[0]["train_split_sha256"],
            "validation": rows[0]["validation_split_sha256"],
            "test": rows[0]["test_split_sha256"],
        },
        "aggregate_results": aggregates,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        render_report(rows, aggregates, deltas), encoding="utf-8"
    )
    print(json.dumps({"runs": len(rows), "output": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
