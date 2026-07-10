#!/usr/bin/env python3
"""Goal 8: b-bit local LUT export, ABC minimization, and cost comparison.

This runner reuses the existing LightLogic K-expansion path, then treats each
thresholded local gate as an explicit b-input LUT.  The current model produces
two-input LUTs; the export/minimization helpers are generic over b so larger
local LUT modules can be plugged in later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from evaluate_abc_blif import evaluate_blif
from hard_lgn_benchmark import fanout_max, load_dataset, set_seed
from lightlogic_experiments import evaluate, unused_gate_ratio
from lightlogic_k_expansion import (
    KExpandedThresholdLayer,
    calibrate_thresholds,
    local_errors,
    make_k_model,
    original_gate_count,
    train_teacher,
)
from lut_minimization import (
    LutMinimizer,
    blif_names_for_lut,
    raw_lut_estimate,
    run_abc_blif,
    run_self_test,
    write_csv,
)


DEFAULT_K_VALUES = [2, 4, 8, 16, 32]


def to_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def safe_stem(dataset: str, seed: int, k: int) -> str:
    return f"{dataset}__seed{seed}__k{k}"


def thresholded_truth_bits(layer: KExpandedThresholdLayer, gate_id: int) -> str:
    qk = layer.qk.detach().cpu()
    thresholds = layer.threshold_value.detach().cpu().reshape(-1)
    threshold = float(thresholds[gate_id].item())
    return "".join("1" if float(qk[gate_id, idx].item()) >= threshold else "0" for idx in range(qk.shape[1]))


def collect_lut_instances(
    expanded,
    dataset_name: str,
    seed: int,
    k: int,
    calibration_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_id, layer in enumerate(expanded.layers):
        if not isinstance(layer, KExpandedThresholdLayer):
            raise TypeError(f"expected KExpandedThresholdLayer, got {type(layer)}")
        idx0 = layer.indices_0.detach().cpu().tolist()
        idx1 = layer.indices_1.detach().cpu().tolist()
        thresholds = layer.threshold_value.detach().cpu().reshape(-1)
        for gate_id in range(layer.out_dim):
            truth_bits = thresholded_truth_bits(layer, gate_id)
            b = int(math.log2(len(truth_bits)))
            estimate = raw_lut_estimate(b, truth_bits)
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "k": k,
                    "calibration_mode": calibration_mode,
                    "layer": layer_id,
                    "gate": gate_id,
                    "b": b,
                    "truth_bits": truth_bits,
                    "threshold": float(thresholds[gate_id].item()),
                    "input_0": int(idx0[gate_id]),
                    "input_1": int(idx1[gate_id]),
                    "raw_lut_bits": estimate.raw_lut_bits,
                    "raw_mux2_count": estimate.raw_mux2_count,
                    "raw_sop_literals": estimate.raw_sop_literals,
                    "raw_sop_not_estimate": estimate.raw_sop_not_estimate,
                    "on_set_size": estimate.on_set_size,
                }
            )
    return rows


def write_expanded_network_blif(expanded, input_dim: int, path: Path, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_names = [f"i{i}" for i in range(input_dim)]
    layers = list(expanded.layers)
    outputs = [f"l{len(layers) - 1}_g{idx}" for idx in range(layers[-1].out_dim)] if layers else prev_names
    lines = [
        f".model {model_name}",
        ".inputs " + " ".join(prev_names),
        ".outputs " + " ".join(outputs),
    ]
    for layer_id, layer in enumerate(layers):
        if not isinstance(layer, KExpandedThresholdLayer):
            raise TypeError(f"expected KExpandedThresholdLayer, got {type(layer)}")
        idx0 = layer.indices_0.detach().cpu().tolist()
        idx1 = layer.indices_1.detach().cpu().tolist()
        next_names = []
        for gate_id in range(layer.out_dim):
            output_name = f"l{layer_id}_g{gate_id}"
            next_names.append(output_name)
            truth_bits = thresholded_truth_bits(layer, gate_id)
            input_names = [prev_names[int(idx0[gate_id])], prev_names[int(idx1[gate_id])]]
            lines.extend(blif_names_for_lut(input_names, output_name, truth_bits))
        prev_names = next_names
    lines.append(".end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_blif_safe(path: Path, dataset, group_tau: float) -> tuple[str, float, float, dict[str, object], str]:
    try:
        acc, loss, stats = evaluate_blif(path, dataset.x_test, dataset.y_test, dataset.num_classes, group_tau)
        return "ok", acc, loss, dict(stats), ""
    except Exception as exc:  # pragma: no cover - operational path.
        return "exception", math.nan, math.nan, {}, repr(exc)


def write_markdown_report(path: Path, rows: list[dict[str, object]], unique_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    ok_unique = sum(
        1
        for row in unique_rows
        if row.get("abc_status") == "ok"
        and str(row.get("source_equivalent")) == "True"
        and str(row.get("equivalent")) == "True"
    )
    lines = [
        "# Goal 8 LUT Logic Minimization Report",
        "",
        "Each thresholded local LightLogic gate is exported as an explicit LUT, minimized with ABC, and checked by exhaustive truth-table equivalence.",
        "",
        f"- backend: {args.backend}",
        f"- ABC path: `{args.abc_path}`",
        f"- datasets: {', '.join(args.datasets)}",
        f"- seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- K values: {', '.join(str(k) for k in args.k_values)}",
        f"- unique LUTs equivalent: {ok_unique}/{len(unique_rows)}",
        "",
        "| dataset | seed | K | teacher_acc | raw_lut_acc | minimized_blif_acc | accuracy_delta | raw_lut_gate_estimate | abc_and_count | abc_level | total_abc_seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("seed", "")),
                    str(row.get("k", "")),
                    f"{to_float(row.get('teacher_acc')):.6g}",
                    f"{to_float(row.get('raw_lut_acc')):.6g}",
                    f"{to_float(row.get('minimized_blif_acc')):.6g}",
                    f"{to_float(row.get('accuracy_delta')):.6g}",
                    str(row.get("raw_lut_gate_estimate", "")),
                    str(row.get("abc_and_count", "")),
                    str(row.get("abc_level", "")),
                    f"{to_float(row.get('total_abc_seconds')):.6g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `raw_lut_gate_estimate` is the sum of per-LUT mux-tree estimates, `2^b - 1` per local LUT.",
            "- `abc_and_count` and `abc_level` are from ABC after `strash; dc2` on the exported full-network BLIF.",
            "- Per-LUT minimization is cached by `(b, truth_bits)`; repeated LUT instances reuse the same minimized result.",
            "- `unique_lut_abc_seconds_total` is the measured cached synthesis time; `per_lut_abc_seconds_uncached_estimate` is the repeated-instance cost estimate.",
            "- `xag_*` counts come from a canonical local ANF/XAG decomposition of each LUT truth table.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(dataset_name: str, seed: int, args: argparse.Namespace, device: torch.device, minimizer: LutMinimizer) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset = load_dataset(dataset_name, seed, args)
    if args.width % dataset.num_classes != 0:
        raise ValueError(f"width={args.width} must be divisible by num_classes={dataset.num_classes}")
    if args.width * 2 < dataset.input_dim:
        raise ValueError(f"width={args.width} too small for input_dim={dataset.input_dim}")

    teacher, train_time = train_teacher(dataset, args, device, seed)
    teacher_acc, teacher_loss = evaluate(
        teacher,
        dataset.x_test,
        dataset.y_test,
        args.eval_batch_size,
        device,
        "continuous",
        args.temp_eval,
    )
    network_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    base_gates = original_gate_count(teacher)

    for k in args.k_values:
        expanded = make_k_model(teacher, dataset.num_classes, args.group_tau, k, args.threshold, args.temp_eval).to(device)
        calibrate_thresholds(
            teacher,
            expanded,
            dataset.x_train,
            args.eval_batch_size,
            device,
            args.calibration_mode,
            args.calibration_grid_size,
            args.temp_eval,
        )
        raw_lut_acc, raw_lut_loss = evaluate(
            expanded,
            dataset.x_test,
            dataset.y_test,
            args.eval_batch_size,
            device,
            "hard",
            1.0,
        )
        err = local_errors(expanded)
        instances = collect_lut_instances(expanded, dataset.name, seed, int(k), args.calibration_mode)
        per_instance_abc_seconds_estimate = 0.0
        unique_abc_seconds = 0.0
        per_instance_abc_and = 0
        per_instance_abc_level_sum = 0
        per_instance_xag_and = 0
        per_instance_xag_xor = 0
        per_instance_xag_not = 0
        per_instance_xag_level_sum = 0
        all_unique_equivalent = True
        for row in instances:
            cache_key = (int(row["b"]), str(row["truth_bits"]))
            was_cached = cache_key in minimizer.cache
            result = minimizer.minimize(int(row["b"]), str(row["truth_bits"]))
            if not was_cached:
                unique_abc_seconds += result.abc_seconds
            row.update(
                {
                    "truth_hash": result.truth_hash,
                    "abc_status": result.abc_status,
                    "abc_and_count": result.abc_and_count,
                    "abc_level": result.abc_level,
                    "abc_seconds": result.abc_seconds,
                    "source_equivalent": result.source_equivalent,
                    "equivalent": result.equivalent,
                    "xag_and_count": result.xag_and_count,
                    "xor_count": result.xor_count,
                    "xag_not_count": result.xag_not_count,
                    "xag_level_estimate": result.xag_level_estimate,
                    "xag_backend": result.xag_backend,
                    "xor_count_status": result.xor_count_status,
                    "unique_raw_blif_path": result.raw_blif_path,
                    "unique_optimized_blif_path": result.optimized_blif_path,
                }
            )
            per_instance_abc_seconds_estimate += result.abc_seconds
            per_instance_abc_and += result.abc_and_count
            per_instance_abc_level_sum += result.abc_level
            per_instance_xag_and += result.xag_and_count
            per_instance_xag_xor += result.xor_count
            per_instance_xag_not += result.xag_not_count
            per_instance_xag_level_sum += result.xag_level_estimate
            all_unique_equivalent = all_unique_equivalent and result.source_equivalent and result.equivalent
        instance_rows.extend(instances)

        stem = safe_stem(dataset.name, seed, int(k))
        raw_network_blif = Path(args.out_dir) / "blif" / "networks" / f"{stem}.raw.blif"
        optimized_network_blif = Path(args.out_dir) / "blif" / "networks" / f"{stem}.abc_optimized.blif"
        network_log = Path(args.out_dir) / "abc_logs" / "networks" / f"{stem}.abc.log"
        write_expanded_network_blif(expanded, dataset.input_dim, raw_network_blif, stem)
        network_abc = run_abc_blif(Path(args.abc_path), raw_network_blif, optimized_network_blif, network_log)
        source_status, source_acc, source_loss, source_stats, source_note = evaluate_blif_safe(raw_network_blif, dataset, args.group_tau)
        if network_abc.abc_status == "ok":
            opt_status, opt_acc, opt_loss, opt_stats, opt_note = evaluate_blif_safe(optimized_network_blif, dataset, args.group_tau)
        else:
            opt_status, opt_acc, opt_loss, opt_stats, opt_note = network_abc.abc_status, math.nan, math.nan, {}, network_abc.abc_error

        raw_lut_gate_estimate = sum(int(row["raw_mux2_count"]) for row in instances)
        raw_lut_bits_total = sum(int(row["raw_lut_bits"]) for row in instances)
        raw_sop_literals_total = sum(int(row["raw_sop_literals"]) for row in instances)
        raw_sop_not_total = sum(int(row["raw_sop_not_estimate"]) for row in instances)
        source_delta = source_acc - raw_lut_acc if math.isfinite(source_acc) else math.nan
        opt_delta_raw = opt_acc - raw_lut_acc if math.isfinite(opt_acc) else math.nan
        opt_delta_source = opt_acc - source_acc if math.isfinite(opt_acc) and math.isfinite(source_acc) else math.nan
        accuracy_preserved = (
            source_status == "ok"
            and opt_status == "ok"
            and math.isfinite(source_delta)
            and math.isfinite(opt_delta_source)
            and abs(source_delta) <= args.accuracy_tolerance
            and abs(opt_delta_source) <= args.accuracy_tolerance
        )
        network_rows.append(
            {
                "dataset": dataset.name,
                "seed": seed,
                "k": int(k),
                "calibration_mode": args.calibration_mode,
                "threshold": float(args.threshold),
                "temp_eval": float(args.temp_eval),
                "estimator": args.estimator,
                "init": args.init,
                "teacher_acc": teacher_acc,
                "teacher_loss": teacher_loss,
                "raw_lut_acc": raw_lut_acc,
                "raw_lut_loss": raw_lut_loss,
                "source_blif_status": source_status,
                "source_blif_acc": source_acc,
                "source_blif_loss": source_loss,
                "source_acc_delta_vs_raw_lut": source_delta,
                "minimized_blif_status": opt_status,
                "minimized_blif_acc": opt_acc,
                "minimized_blif_loss": opt_loss,
                "accuracy_delta": opt_delta_raw,
                "minimized_acc_delta_vs_source_blif": opt_delta_source,
                "accuracy_preserved": bool(accuracy_preserved),
                "local_mae": err["mae"],
                "local_mse": err["mse"],
                "local_max_abs_error": err["max_abs_error"],
                "local_binary_flip_ratio": err["binary_flip_ratio"],
                "original_gate_count": base_gates,
                "expanded_gate_count": base_gates * int(k),
                "lut_instance_count": len(instances),
                "unique_lut_count_seen_so_far": len(minimizer.cache),
                "raw_lut_gate_estimate": raw_lut_gate_estimate,
                "raw_lut_bits_total": raw_lut_bits_total,
                "raw_sop_literals_total": raw_sop_literals_total,
                "raw_sop_not_estimate_total": raw_sop_not_total,
                "raw_lut_depth_estimate": len(expanded.layers) * 2,
                "per_lut_abc_and_count_estimate": per_instance_abc_and,
                "per_lut_abc_level_sum_estimate": per_instance_abc_level_sum,
                "per_lut_xag_and_count_estimate": per_instance_xag_and,
                "per_lut_xag_xor_count_estimate": per_instance_xag_xor,
                "per_lut_xag_not_count_estimate": per_instance_xag_not,
                "per_lut_xag_level_sum_estimate": per_instance_xag_level_sum,
                "per_lut_abc_seconds_uncached_estimate": per_instance_abc_seconds_estimate,
                "unique_lut_abc_seconds_total": unique_abc_seconds,
                "all_unique_luts_equivalent": bool(all_unique_equivalent),
                "abc_status": network_abc.abc_status,
                "abc_returncode": network_abc.abc_returncode,
                "abc_and_count": network_abc.abc_and_count,
                "abc_level": network_abc.abc_level,
                "abc_node_count": network_abc.abc_node_count,
                "abc_edge_count": network_abc.abc_edge_count,
                "abc_cube_count": network_abc.abc_cube_count,
                "abc_stat_lines": network_abc.abc_stat_lines,
                "total_abc_seconds": network_abc.abc_seconds,
                "source_blif_fanout_max": source_stats.get("fanout_max", ""),
                "optimized_blif_fanout_max": opt_stats.get("fanout_max", ""),
                "source_blif_unused_node_ratio": source_stats.get("unused_node_ratio", ""),
                "optimized_blif_unused_node_ratio": opt_stats.get("unused_node_ratio", ""),
                "raw_network_blif": str(raw_network_blif),
                "optimized_network_blif": str(optimized_network_blif),
                "network_abc_log": str(network_log),
                "train_time": train_time,
                "depth": len(expanded.layers),
                "fanout_max": fanout_max(expanded.layers),
                "unused_gate_ratio": unused_gate_ratio(expanded, dataset.x_train, args.eval_batch_size, device),
                "note": "; ".join(item for item in [source_note, opt_note] if item),
            }
        )
    return network_rows, instance_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["abc"], default="abc")
    parser.add_argument("--abc-path", default="/home/spco/boolean_sat/abc/abc")
    parser.add_argument("--datasets", nargs="+", default=["parity8", "majority9", "random_sparse10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calibration-mode", choices=["fixed", "per_layer", "per_neuron"], default="fixed")
    parser.add_argument("--calibration-grid-size", type=int, default=101)
    parser.add_argument("--accuracy-tolerance", type=float, default=1e-9)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--estimator", choices=["sigmoid", "sinusoidal"], default="sinusoidal")
    parser.add_argument("--init", choices=["residual", "and_or", "xor", "random", "uniform"], default="residual")
    parser.add_argument("--init-strength", type=float, default=0.98)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.1)
    parser.add_argument("--temp-eval", type=float, default=1.0)
    parser.add_argument("--anneal-train", action="store_true")
    parser.add_argument("--data-dir", default="/home/spco/data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--threshold-levels", type=int, default=1)
    parser.add_argument("--image-max-train", type=int, default=4000)
    parser.add_argument("--image-max-test", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/lightlogic_lut_min_goal8_bool_seed0_v1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--self-test-first", action="store_true", default=True)
    parser.add_argument("--no-self-test-first", action="store_false", dest="self_test_first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets = ["parity6", "majority7", "random_sparse8"]
        args.seeds = [0]
        if args.k_values == DEFAULT_K_VALUES:
            args.k_values = [2, 4, 8]
        args.width = min(args.width, 32)
        args.layers = min(args.layers, 3)
        args.epochs = min(args.epochs, 12)
    if any(k <= 0 for k in args.k_values):
        raise ValueError("--k-values must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1]")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.self_test_first:
        self_test_rows = run_self_test(Path(args.abc_path), out_dir / "self_test")
        if not all(bool(row["self_test_pass"]) for row in self_test_rows):
            raise RuntimeError("LUT minimization self-test failed")

    minimizer = LutMinimizer(Path(args.abc_path), out_dir)
    network_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    print(f"device={device} out_dir={out_dir}")
    for seed in args.seeds:
        set_seed(seed)
        for dataset_name in args.datasets:
            print(f"dataset={dataset_name} seed={seed}", flush=True)
            dataset_network_rows, dataset_instance_rows = run_dataset(dataset_name, seed, args, device, minimizer)
            network_rows.extend(dataset_network_rows)
            instance_rows.extend(dataset_instance_rows)
            write_csv(out_dir / "network_minimization_results.partial.csv", network_rows)
            write_csv(out_dir / "lut_instances.partial.csv", instance_rows)
            write_csv(out_dir / "unique_lut_minimization.partial.csv", [asdict(item) for item in minimizer.cache.values()])
            for row in dataset_network_rows:
                print(
                    "  K={} teacher_acc={:.4f} raw_lut_acc={:.4f} minimized_blif_acc={:.4f} abc_and={} abc_level={} abc_status={}".format(
                        row["k"],
                        row["teacher_acc"],
                        row["raw_lut_acc"],
                        row["minimized_blif_acc"],
                        row["abc_and_count"],
                        row["abc_level"],
                        row["abc_status"],
                    ),
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    for row in network_rows:
        row["elapsed_total_seconds"] = elapsed
    unique_rows = [asdict(item) for item in minimizer.cache.values()]
    write_csv(out_dir / "network_minimization_results.csv", network_rows)
    write_csv(out_dir / "lut_instances.csv", instance_rows)
    write_csv(out_dir / "unique_lut_minimization.csv", unique_rows)
    write_markdown_report(out_dir / "lut_minimization_summary.md", network_rows, unique_rows, args)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "network_minimization_results.csv")
    print(out_dir / "unique_lut_minimization.csv")
    print(out_dir / "lut_minimization_summary.md")


if __name__ == "__main__":
    main()
