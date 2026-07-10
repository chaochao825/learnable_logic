#!/usr/bin/env python3
"""Generic LUT truth-table export and ABC minimization helpers.

The helpers in this file are intentionally independent from the LightLogic
training code.  They operate on explicit LUT truth strings such as ``0110``
for a two-input XOR with pattern order 00, 01, 10, 11.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lut_xag import xag_from_truth_bits


@dataclass
class RawLutEstimate:
    b: int
    truth_bits: str
    raw_lut_bits: int
    raw_mux2_count: int
    raw_sop_literals: int
    raw_sop_not_estimate: int
    on_set_size: int


@dataclass
class AbcBlifResult:
    abc_status: str
    abc_returncode: int
    abc_seconds: float
    abc_inputs: int
    abc_outputs: int
    abc_node_count: int | str
    abc_edge_count: int | str
    abc_cube_count: int | str
    abc_level: int
    abc_and_count: int
    abc_stat_lines: int
    raw_blif_path: str
    optimized_blif_path: str
    abc_log_path: str
    abc_error: str = ""


@dataclass
class LutMinimizationResult:
    b: int
    truth_bits: str
    truth_hash: str
    raw_lut_bits: int
    raw_mux2_count: int
    raw_sop_literals: int
    raw_sop_not_estimate: int
    on_set_size: int
    abc_status: str
    abc_returncode: int
    abc_seconds: float
    abc_and_count: int
    abc_level: int
    abc_node_count: int | str
    abc_edge_count: int | str
    abc_cube_count: int | str
    abc_stat_lines: int
    xag_and_count: int
    xor_count: int
    xag_not_count: int
    xag_level_estimate: int
    xag_backend: str
    xor_count_status: str
    source_equivalent: bool
    equivalent: bool
    raw_blif_path: str
    optimized_blif_path: str
    abc_log_path: str
    note: str = ""


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def infer_lut_input_count(truth_bits: str) -> int:
    if not truth_bits or any(char not in {"0", "1"} for char in truth_bits):
        raise ValueError(f"truth_bits must be a non-empty binary string, got {truth_bits!r}")
    length = len(truth_bits)
    if length & (length - 1):
        raise ValueError(f"truth table length must be a power of two, got {length}")
    return int(math.log2(length))


def validate_truth_bits(b: int, truth_bits: str) -> None:
    expected = 1 << b
    if len(truth_bits) != expected:
        raise ValueError(f"b={b} expects {expected} truth bits, got {len(truth_bits)}")
    if any(char not in {"0", "1"} for char in truth_bits):
        raise ValueError(f"truth_bits must contain only 0/1, got {truth_bits!r}")


def pattern_for_index(index: int, b: int) -> str:
    return format(index, f"0{b}b")


def raw_lut_estimate(b: int, truth_bits: str) -> RawLutEstimate:
    validate_truth_bits(b, truth_bits)
    on_indices = [idx for idx, bit in enumerate(truth_bits) if bit == "1"]
    complemented_inputs = set()
    for idx in on_indices:
        pattern = pattern_for_index(idx, b)
        for pos, bit in enumerate(pattern):
            if bit == "0":
                complemented_inputs.add(pos)
    return RawLutEstimate(
        b=b,
        truth_bits=truth_bits,
        raw_lut_bits=1 << b,
        raw_mux2_count=max((1 << b) - 1, 0),
        raw_sop_literals=len(on_indices) * b,
        raw_sop_not_estimate=len(complemented_inputs),
        on_set_size=len(on_indices),
    )


def blif_names_for_lut(input_names: list[str], output_name: str, truth_bits: str) -> list[str]:
    b = len(input_names)
    validate_truth_bits(b, truth_bits)
    if "1" not in truth_bits:
        return [f".names {output_name}"]
    if "0" not in truth_bits:
        return [f".names {output_name}", "1"]
    lines = [".names " + " ".join([*input_names, output_name]).strip()]
    for idx, bit in enumerate(truth_bits):
        if bit == "1":
            if b == 0:
                lines.append("1")
            else:
                lines.append(f"{pattern_for_index(idx, b)} 1")
    return lines


def write_single_lut_blif(path: Path, b: int, truth_bits: str, model_name: str = "lut") -> None:
    validate_truth_bits(b, truth_bits)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = [f"x{i}" for i in range(b)]
    lines = [
        f".model {model_name}",
        ".inputs " + " ".join(inputs),
        ".outputs y",
        *blif_names_for_lut(inputs, "y", truth_bits),
        ".end",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_abc_print_stats(output: str) -> dict[str, int]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    stat_lines = [line for line in clean.splitlines() if "i/o =" in line]
    stats: dict[str, int] = {"abc_stat_lines": len(stat_lines)}
    if not stat_lines:
        return stats
    line = stat_lines[-1]
    io_match = re.search(r"i/o\s*=\s*([0-9]+)\s*/\s*([0-9]+)", line)
    if io_match:
        stats["abc_inputs"] = int(io_match.group(1))
        stats["abc_outputs"] = int(io_match.group(2))
    for key, value in re.findall(r"(nd|edge|cube|lev|and)\s*=\s*([0-9]+)", line):
        stats[f"abc_{key}"] = int(value)
    return stats


def run_abc_blif(
    abc_path: Path,
    raw_blif_path: Path,
    optimized_blif_path: Path,
    log_path: Path,
) -> AbcBlifResult:
    optimized_blif_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if not abc_path.exists():
        return AbcBlifResult(
            abc_status="missing_abc",
            abc_returncode=-1,
            abc_seconds=0.0,
            abc_inputs=0,
            abc_outputs=0,
            abc_node_count=0,
            abc_edge_count=0,
            abc_cube_count=0,
            abc_level=0,
            abc_and_count=0,
            abc_stat_lines=0,
            raw_blif_path=str(raw_blif_path),
            optimized_blif_path=str(optimized_blif_path),
            abc_log_path=str(log_path),
            abc_error=f"ABC binary not found: {abc_path}",
        )
    cmd = [
        str(abc_path),
        "-c",
        f"read_blif {raw_blif_path}; strash; dc2; print_stats; write_blif {optimized_blif_path}",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - started
        combined = proc.stdout + proc.stderr
        log_path.write_text(combined, encoding="utf-8")
        parsed = parse_abc_print_stats(combined)
        status = "ok" if proc.returncode == 0 and optimized_blif_path.exists() else "abc_failed"
        return AbcBlifResult(
            abc_status=status,
            abc_returncode=proc.returncode,
            abc_seconds=elapsed,
            abc_inputs=parsed.get("abc_inputs", 0),
            abc_outputs=parsed.get("abc_outputs", 0),
            abc_node_count=parsed.get("abc_nd", parsed.get("abc_and", "")),
            abc_edge_count=parsed.get("abc_edge", ""),
            abc_cube_count=parsed.get("abc_cube", ""),
            abc_level=parsed.get("abc_lev", 0),
            abc_and_count=parsed.get("abc_and", 0),
            abc_stat_lines=parsed.get("abc_stat_lines", 0),
            raw_blif_path=str(raw_blif_path),
            optimized_blif_path=str(optimized_blif_path),
            abc_log_path=str(log_path),
        )
    except Exception as exc:  # pragma: no cover - operational path.
        elapsed = time.perf_counter() - started
        log_path.write_text(repr(exc) + "\n", encoding="utf-8")
        return AbcBlifResult(
            abc_status="exception",
            abc_returncode=-1,
            abc_seconds=elapsed,
            abc_inputs=0,
            abc_outputs=0,
            abc_node_count=0,
            abc_edge_count=0,
            abc_cube_count=0,
            abc_level=0,
            abc_and_count=0,
            abc_stat_lines=0,
            raw_blif_path=str(raw_blif_path),
            optimized_blif_path=str(optimized_blif_path),
            abc_log_path=str(log_path),
            abc_error=repr(exc),
        )


def exact_binary_inputs(b: int) -> torch.Tensor:
    rows = []
    for idx in range(1 << b):
        rows.append([float(bit) for bit in pattern_for_index(idx, b)])
    return torch.tensor(rows, dtype=torch.float32)


def evaluate_blif_truth(path: Path, b: int) -> str:
    from evaluate_abc_blif import evaluate_blif_outputs, parse_blif

    network = parse_blif(path)
    output = evaluate_blif_outputs(network, exact_binary_inputs(b))
    if output.shape[1] != 1:
        raise ValueError(f"expected one BLIF output, got {output.shape[1]}")
    return "".join("1" if value >= 0.5 else "0" for value in output[:, 0].tolist())


class LutMinimizer:
    def __init__(self, abc_path: Path, out_dir: Path) -> None:
        self.abc_path = abc_path
        self.out_dir = out_dir
        self.cache: dict[tuple[int, str], LutMinimizationResult] = {}

    def minimize(self, b: int, truth_bits: str) -> LutMinimizationResult:
        validate_truth_bits(b, truth_bits)
        key = (b, truth_bits)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        estimate = raw_lut_estimate(b, truth_bits)
        truth_hash = hashlib.sha256(f"{b}:{truth_bits}".encode("ascii")).hexdigest()
        stem = f"lut_b{b}_{truth_hash[:16]}"
        raw_path = self.out_dir / "blif" / "unique_luts" / f"{stem}.raw.blif"
        opt_path = self.out_dir / "blif" / "unique_luts" / f"{stem}.abc_optimized.blif"
        log_path = self.out_dir / "abc_logs" / "unique_luts" / f"{stem}.abc.log"
        write_single_lut_blif(raw_path, b, truth_bits, stem)
        source_truth = ""
        optimized_truth = ""
        source_equivalent = False
        equivalent = False
        note = ""
        try:
            source_truth = evaluate_blif_truth(raw_path, b)
            source_equivalent = source_truth == truth_bits
        except Exception as exc:
            note = f"source_equivalence_error={exc!r}"
        abc_result = run_abc_blif(self.abc_path, raw_path, opt_path, log_path)
        xag_result = xag_from_truth_bits(truth_bits)
        if abc_result.abc_status == "ok":
            try:
                optimized_truth = evaluate_blif_truth(opt_path, b)
                equivalent = optimized_truth == truth_bits
            except Exception as exc:
                note = (note + "; " if note else "") + f"optimized_equivalence_error={exc!r}"
        result = LutMinimizationResult(
            b=b,
            truth_bits=truth_bits,
            truth_hash=truth_hash,
            raw_lut_bits=estimate.raw_lut_bits,
            raw_mux2_count=estimate.raw_mux2_count,
            raw_sop_literals=estimate.raw_sop_literals,
            raw_sop_not_estimate=estimate.raw_sop_not_estimate,
            on_set_size=estimate.on_set_size,
            abc_status=abc_result.abc_status,
            abc_returncode=abc_result.abc_returncode,
            abc_seconds=abc_result.abc_seconds,
            abc_and_count=abc_result.abc_and_count,
            abc_level=abc_result.abc_level,
            abc_node_count=abc_result.abc_node_count,
            abc_edge_count=abc_result.abc_edge_count,
            abc_cube_count=abc_result.abc_cube_count,
            abc_stat_lines=abc_result.abc_stat_lines,
            xag_and_count=xag_result.xag_and_count,
            xor_count=xag_result.xag_xor_count,
            xag_not_count=xag_result.xag_not_count,
            xag_level_estimate=xag_result.xag_level_estimate,
            xag_backend=xag_result.xag_backend,
            xor_count_status=xag_result.xag_backend,
            source_equivalent=source_equivalent,
            equivalent=equivalent,
            raw_blif_path=str(raw_path),
            optimized_blif_path=str(opt_path),
            abc_log_path=str(log_path),
            note=note,
        )
        self.cache[key] = result
        return result


def run_self_test(abc_path: Path, out_dir: Path) -> list[dict[str, object]]:
    cases = {
        "const0": "0000",
        "const1": "1111",
        "and": "0001",
        "or": "0111",
        "xor": "0110",
        "xnor": "1001",
    }
    minimizer = LutMinimizer(abc_path, out_dir)
    rows: list[dict[str, object]] = []
    for name, truth_bits in cases.items():
        result = minimizer.minimize(2, truth_bits)
        row = asdict(result)
        row["case"] = name
        row["self_test_pass"] = bool(result.source_equivalent and result.equivalent and result.abc_status == "ok")
        rows.append(row)
    write_csv(out_dir / "self_test_results.csv", rows)
    (out_dir / "self_test_summary.json").write_text(
        json.dumps(
            {
                "passed": all(bool(row["self_test_pass"]) for row in rows),
                "rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--abc-path", type=Path, default=Path("/home/spco/boolean_sat/abc/abc"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/lut_minimization_self_test"))
    args = parser.parse_args()
    if not args.self_test:
        parser.error("pass --self-test to run standalone LUT minimization checks")
    rows = run_self_test(args.abc_path, args.out_dir)
    passed = all(bool(row["self_test_pass"]) for row in rows)
    print(args.out_dir / "self_test_results.csv")
    print("self_test_pass=" + str(passed))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
