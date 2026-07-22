#!/usr/bin/env python3
"""Evaluate pre/post-ABC BLIF networks on the Boolean test sets.

The training script's ABC path originally measured structure only:
`read_blif; print_stats; strash; dc2; print_stats`.  This helper keeps the
original artifacts untouched, asks ABC to write an optimized BLIF into a report
directory, and evaluates both source and optimized BLIF with the same GroupSum
classification rule used by the PyTorch prototype.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hard_lgn_benchmark import make_boolean_dataset


OUTPUT_FIELDS = [
    "run",
    "dataset",
    "method",
    "seed",
    "abc_eval_status",
    "source_blif",
    "optimized_blif",
    "abc_write_returncode",
    "abc_write_runtime_seconds",
    "abc_write_log",
    "pre_soft_acc",
    "pre_discrete_acc",
    "source_blif_acc",
    "post_abc_discrete_acc",
    "source_acc_delta_vs_results",
    "post_acc_delta_vs_pre_discrete",
    "pre_acc_gap",
    "source_gap_vs_pre_soft",
    "post_abc_gap_vs_pre_soft",
    "post_gap_delta_vs_pre_gap",
    "pre_soft_loss",
    "pre_discrete_loss",
    "source_blif_loss",
    "post_abc_discrete_loss",
    "source_loss_delta_vs_results",
    "post_loss_delta_vs_pre_discrete",
    "post_abc_loss_gap_vs_pre_soft",
    "pre_gate_count",
    "pre_depth",
    "pre_fanout_max",
    "abc_pre_nd",
    "abc_post_and",
    "abc_pre_lev",
    "abc_post_lev",
    "abc_and_reduction_vs_pre_nd",
    "abc_level_delta_vs_pre_lev",
    "source_blif_fanout_max",
    "post_abc_fanout_max",
    "fanout_delta_source_to_post",
    "source_blif_unused_node_count",
    "post_abc_unused_node_count",
    "source_blif_unused_node_ratio",
    "post_abc_unused_node_ratio",
    "unused_node_ratio_delta_source_to_post",
    "source_output_count",
    "post_output_count",
    "source_node_count",
    "post_node_count",
    "note",
]


class BlifNetwork:
    def __init__(self, inputs: list[str], outputs: list[str], nodes: list[tuple[list[str], str, list[str]]]) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def preprocess_blif_lines(path: Path) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        line = current + line
        current = ""
        lines.append(line.strip())
    if current.strip():
        lines.append(current.strip())
    return lines


def parse_blif(path: Path) -> BlifNetwork:
    inputs: list[str] = []
    outputs: list[str] = []
    nodes: list[tuple[list[str], str, list[str]]] = []
    current_names: list[str] | None = None
    current_cover: list[str] = []

    def flush_node() -> None:
        nonlocal current_names, current_cover
        if current_names is None:
            return
        if len(current_names) < 1:
            raise ValueError(f"empty .names in {path}")
        nodes.append((current_names[:-1], current_names[-1], current_cover))
        current_names = None
        current_cover = []

    for line in preprocess_blif_lines(path):
        if line.startswith(".names "):
            flush_node()
            current_names = line.split()[1:]
            current_cover = []
            continue
        if line.startswith("."):
            flush_node()
            parts = line.split()
            directive = parts[0]
            if directive == ".inputs":
                inputs.extend(parts[1:])
            elif directive == ".outputs":
                outputs.extend(parts[1:])
            elif directive in {".model", ".end"}:
                pass
            else:
                raise ValueError(f"unsupported BLIF directive {directive!r} in {path}")
            continue
        if current_names is None:
            raise ValueError(f"cover line outside .names in {path}: {line}")
        current_cover.append(line)
    flush_node()
    if not inputs or not outputs:
        raise ValueError(f"missing .inputs or .outputs in {path}")
    return BlifNetwork(inputs, outputs, nodes)


def cover_matches(pattern: str, values: list[np.ndarray]) -> np.ndarray:
    if len(pattern) != len(values):
        raise ValueError(f"cover pattern length {len(pattern)} does not match inputs {len(values)}")
    if not values:
        return np.ones(1, dtype=bool)
    match = np.ones_like(values[0], dtype=bool)
    for char, value in zip(pattern, values, strict=True):
        if char == "0":
            match &= ~value
        elif char == "1":
            match &= value
        elif char == "-":
            continue
        else:
            raise ValueError(f"unsupported BLIF cover character {char!r}")
    return match


def evaluate_blif_outputs(network: BlifNetwork, x: torch.Tensor) -> np.ndarray:
    x_np = x.detach().cpu().numpy().astype(bool)
    if x_np.shape[1] != len(network.inputs):
        raise ValueError(f"input width mismatch: data={x_np.shape[1]} blif={len(network.inputs)}")
    signals: dict[str, np.ndarray] = {
        name: x_np[:, idx].copy()
        for idx, name in enumerate(network.inputs)
    }
    sample_count = x_np.shape[0]
    for input_names, output_name, cover in network.nodes:
        values = [signals[name] for name in input_names]
        parsed_cover: list[tuple[str, str]] = []
        for line in cover:
            parts = line.split()
            if len(input_names) == 0:
                parsed_cover.append(("", parts[-1] if parts else "0"))
                continue
            if len(parts) != 2:
                raise ValueError(f"unsupported cover line for {output_name}: {line}")
            parsed_cover.append((parts[0], parts[1]))
        output_values = {output_value for _, output_value in parsed_cover}
        if not parsed_cover:
            out = np.zeros(sample_count, dtype=bool)
        elif output_values == {"0"}:
            # ABC often writes complemented single-output covers as an OFF-set:
            # listed cubes drive 0, unspecified minterms default to 1.
            out = np.ones(sample_count, dtype=bool)
            for pattern, _ in parsed_cover:
                if input_names:
                    out &= ~cover_matches(pattern, values)
                else:
                    out[:] = False
        elif output_values == {"1"}:
            out = np.zeros(sample_count, dtype=bool)
            for pattern, _ in parsed_cover:
                if input_names:
                    out |= cover_matches(pattern, values)
                else:
                    out[:] = True
        else:
            out = np.zeros(sample_count, dtype=bool)
            for pattern, output_value in parsed_cover:
                if output_value not in {"0", "1"}:
                    raise ValueError(f"unsupported BLIF output value {output_value!r}")
                if input_names:
                    out[cover_matches(pattern, values)] = output_value == "1"
                else:
                    out[:] = output_value == "1"
        signals[output_name] = out
    missing = [name for name in network.outputs if name not in signals]
    if missing:
        raise ValueError(f"missing output signals in BLIF evaluation: {missing[:5]}")
    return np.stack([signals[name] for name in network.outputs], axis=1).astype(np.float32)


def blif_structure_stats(network: BlifNetwork) -> dict[str, float | int]:
    """Compute simple graph stats directly from BLIF node connectivity."""
    signal_loads: dict[str, int] = {}
    node_by_output: dict[str, tuple[list[str], str, list[str]]] = {}
    for input_names, output_name, cover in network.nodes:
        node_by_output[output_name] = (input_names, output_name, cover)
        signal_loads.setdefault(output_name, 0)
        for input_name in input_names:
            signal_loads[input_name] = signal_loads.get(input_name, 0) + 1
    for output_name in network.outputs:
        signal_loads[output_name] = signal_loads.get(output_name, 0) + 1

    reachable_node_outputs: set[str] = set()
    visited_signals: set[str] = set()
    stack = list(network.outputs)
    while stack:
        signal = stack.pop()
        if signal in visited_signals:
            continue
        visited_signals.add(signal)
        node = node_by_output.get(signal)
        if node is None:
            continue
        reachable_node_outputs.add(signal)
        stack.extend(node[0])

    node_count = len(network.nodes)
    unused_node_count = node_count - len(reachable_node_outputs)
    return {
        "output_count": len(network.outputs),
        "node_count": node_count,
        "reachable_node_count": len(reachable_node_outputs),
        "unused_node_count": unused_node_count,
        "unused_node_ratio": unused_node_count / node_count if node_count else 0.0,
        "fanout_max": max(signal_loads.values()) if signal_loads else 0,
    }


def grouped_logits(output_bits: np.ndarray, num_classes: int, group_tau: float) -> torch.Tensor:
    if output_bits.shape[1] % num_classes != 0:
        raise ValueError(f"output width {output_bits.shape[1]} is not divisible by classes={num_classes}")
    logits = torch.from_numpy(output_bits).float().reshape(output_bits.shape[0], num_classes, -1).sum(dim=-1)
    return logits / group_tau


def evaluate_blif(
    path: Path,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    group_tau: float,
) -> tuple[float, float, dict[str, float | int]]:
    network = parse_blif(path)
    output_bits = evaluate_blif_outputs(network, x)
    logits = grouped_logits(output_bits, num_classes, group_tau)
    loss = F.cross_entropy(logits, y.long(), reduction="mean").item()
    acc = (logits.argmax(dim=1) == y.long()).float().mean().item()
    stats = blif_structure_stats(network)
    stats["output_count"] = output_bits.shape[1]
    return acc, loss, stats


def run_abc_write(
    abc_path: Path,
    run_dir: Path,
    source_blif: Path,
    optimized_blif: Path,
    log_path: Path,
) -> tuple[int, float]:
    optimized_blif.parent.mkdir(parents=True, exist_ok=True)
    relative_source = source_blif.relative_to(run_dir)
    cmd = [
        str(abc_path),
        "-c",
        f"read_blif {relative_source}; strash; dc2; write_blif {optimized_blif}",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=run_dir, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode, elapsed


def safe_stem(dataset: str, method: str, seed: str) -> str:
    raw = f"{dataset}__{method}__seed{seed}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory containing results.csv and synthesis_stats.csv.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for post-ABC evaluation outputs.")
    parser.add_argument("--run-name", default="", help="Optional run label for output rows.")
    parser.add_argument("--abc-path", default="/home/spco/boolean_sat/abc/abc")
    parser.add_argument("--group-tau", type=float, default=1.0)
    parser.add_argument("--skip-abc-write", action="store_true", help="Evaluate existing optimized BLIF files without invoking ABC.")
    args = parser.parse_args()

    args.run_dir = args.run_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    run_name = args.run_name or args.run_dir.name
    results_by_key = {
        (row["dataset"], row["method"], row["seed"]): row
        for row in read_csv(args.run_dir / "results.csv")
    }
    synthesis_rows = read_csv(args.run_dir / "synthesis_stats.csv")
    rows: list[dict[str, object]] = []
    optimized_dir = args.out_dir / "optimized_blif"
    log_dir = args.out_dir / "abc_write_logs"

    for synth_row in synthesis_rows:
        dataset = synth_row.get("dataset", "")
        method = synth_row.get("method", "")
        seed = synth_row.get("seed", "")
        key = (dataset, method, seed)
        result_row = results_by_key.get(key, {})
        stem = safe_stem(dataset, method, seed)
        source_blif = args.run_dir / synth_row.get("blif_path", "")
        optimized_blif = optimized_dir / f"{stem}.abc_optimized.blif"
        write_log = log_dir / f"{stem}.abc_write.log"
        row: dict[str, object] = {
            "run": run_name,
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "abc_eval_status": "not_run",
            "source_blif": str(source_blif),
            "optimized_blif": str(optimized_blif),
            "abc_write_log": str(write_log),
            "pre_soft_acc": result_row.get("soft_acc", ""),
            "pre_discrete_acc": result_row.get("discrete_acc", ""),
            "pre_acc_gap": result_row.get("acc_gap", ""),
            "pre_soft_loss": result_row.get("soft_loss", ""),
            "pre_discrete_loss": result_row.get("discrete_loss", ""),
            "pre_gate_count": synth_row.get("pre_gate_count", ""),
            "pre_depth": synth_row.get("pre_depth", ""),
            "pre_fanout_max": synth_row.get("pre_fanout_max", ""),
            "abc_pre_nd": synth_row.get("abc_pre_nd", ""),
            "abc_post_and": synth_row.get("abc_post_and", ""),
            "abc_pre_lev": synth_row.get("abc_pre_lev", ""),
            "abc_post_lev": synth_row.get("abc_post_lev", ""),
            "abc_and_reduction_vs_pre_nd": synth_row.get("abc_and_reduction_vs_pre_nd", ""),
            "abc_level_delta_vs_pre_lev": synth_row.get("abc_level_delta_vs_pre_lev", ""),
        }
        try:
            if not source_blif.exists():
                raise FileNotFoundError(source_blif)
            if not args.skip_abc_write:
                returncode, runtime = run_abc_write(
                    Path(args.abc_path),
                    args.run_dir,
                    source_blif,
                    optimized_blif,
                    write_log,
                )
                row["abc_write_returncode"] = returncode
                row["abc_write_runtime_seconds"] = runtime
            elif not optimized_blif.exists():
                raise FileNotFoundError(optimized_blif)
            if int(row.get("abc_write_returncode", 0)) != 0:
                row["abc_eval_status"] = "abc_write_failed"
                rows.append(row)
                continue
            seed_int = int(seed)
            bundle = make_boolean_dataset(dataset, seed_int)
            source_acc, source_loss, source_stats = evaluate_blif(
                source_blif,
                bundle.x_test,
                bundle.y_test,
                bundle.num_classes,
                args.group_tau,
            )
            post_acc, post_loss, post_stats = evaluate_blif(
                optimized_blif,
                bundle.x_test,
                bundle.y_test,
                bundle.num_classes,
                args.group_tau,
            )
            pre_soft_acc = to_float(result_row.get("soft_acc"))
            pre_discrete_acc = to_float(result_row.get("discrete_acc"))
            pre_acc_gap = to_float(result_row.get("acc_gap"))
            pre_soft_loss = to_float(result_row.get("soft_loss"))
            pre_discrete_loss = to_float(result_row.get("discrete_loss"))
            row.update(
                {
                    "abc_eval_status": "ok",
                    "source_blif_acc": source_acc,
                    "post_abc_discrete_acc": post_acc,
                    "source_acc_delta_vs_results": source_acc - pre_discrete_acc,
                    "post_acc_delta_vs_pre_discrete": post_acc - pre_discrete_acc,
                    "source_gap_vs_pre_soft": abs(pre_soft_acc - source_acc),
                    "post_abc_gap_vs_pre_soft": abs(pre_soft_acc - post_acc),
                    "post_gap_delta_vs_pre_gap": abs(pre_soft_acc - post_acc) - pre_acc_gap,
                    "source_blif_loss": source_loss,
                    "post_abc_discrete_loss": post_loss,
                    "source_loss_delta_vs_results": source_loss - pre_discrete_loss,
                    "post_loss_delta_vs_pre_discrete": post_loss - pre_discrete_loss,
                    "post_abc_loss_gap_vs_pre_soft": abs(pre_soft_loss - post_loss),
                    "source_blif_fanout_max": source_stats["fanout_max"],
                    "post_abc_fanout_max": post_stats["fanout_max"],
                    "fanout_delta_source_to_post": post_stats["fanout_max"] - source_stats["fanout_max"],
                    "source_blif_unused_node_count": source_stats["unused_node_count"],
                    "post_abc_unused_node_count": post_stats["unused_node_count"],
                    "source_blif_unused_node_ratio": source_stats["unused_node_ratio"],
                    "post_abc_unused_node_ratio": post_stats["unused_node_ratio"],
                    "unused_node_ratio_delta_source_to_post": post_stats["unused_node_ratio"] - source_stats["unused_node_ratio"],
                    "source_output_count": source_stats["output_count"],
                    "post_output_count": post_stats["output_count"],
                    "source_node_count": source_stats["node_count"],
                    "post_node_count": post_stats["node_count"],
                    "note": "post_abc_gap_vs_pre_soft uses the original relaxed soft accuracy and the optimized hard BLIF accuracy; post-ABC fanout/unused are BLIF graph reachability metrics.",
                }
            )
        except Exception as exc:  # pragma: no cover - operational report path.
            row["abc_eval_status"] = "exception"
            row["note"] = repr(exc)
        rows.append(row)

    write_csv(args.out_dir / "post_abc_eval.csv", rows)
    print(args.out_dir / "post_abc_eval.csv")


if __name__ == "__main__":
    main()
