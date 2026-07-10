from __future__ import annotations

import glob
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def latest(pattern: str) -> Path | None:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    return paths[-1] if paths else None


def load(pattern: str) -> dict[str, Any] | None:
    path = latest(pattern)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_summary_path"] = str(path)
    return data


def load_group(items: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for label, pattern in items:
        data = load(pattern)
        if data is None:
            rows.append({"label": label, "missing": True, "pattern": pattern})
        else:
            data["label"] = label
            data["missing"] = False
            rows.append(data)
    return rows


def mean_std(values: list[float]) -> dict[str, float | None]:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return {"mean": None, "std": None}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.mean(values), "std": std}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if not row.get("missing")]
    fields = ["teacher_hard_acc", "teacher_soft_acc", "teacher_acc_gap", "teacher_train_time", "peak_memory_mb"]
    out: dict[str, Any] = {"n": len(present)}
    for field in fields:
        stats = mean_std([row.get(field) for row in present])
        out[f"{field}_mean"] = stats["mean"]
        out[f"{field}_std"] = stats["std"]
    return out


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_rows(title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    print(f"\n{title}")
    header = ["label", *fields, "summary"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        if row.get("missing"):
            vals = [row["label"], *(["MISSING"] + [""] * (len(fields) - 1)), row["pattern"]]
        else:
            vals = [row["label"], *[fmt(row.get(field)) for field in fields], row["_summary_path"]]
        print("| " + " | ".join(vals) + " |")


def per_block_table(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    block_ids = sorted(
        {
            int(match.group(1))
            for key in row
            for match in [re.match(r"sctm_b(\d+)_route_entropy", key)]
            if match
        }
    )
    table = []
    for block_id in block_ids:
        table.append(
            {
                "block": block_id,
                "route_entropy": row.get(f"sctm_b{block_id}_route_entropy"),
                "unique_patch_ratio": row.get(f"sctm_b{block_id}_unique_patch_ratio"),
                "patch_norm_before": row.get(f"sctm_b{block_id}_patch_norm_before"),
                "patch_norm_after": row.get(f"sctm_b{block_id}_patch_norm_after"),
                "cls_update_norm": row.get(f"sctm_b{block_id}_cls_update_norm"),
                "skip_acc": row.get(f"skip_sctm_b{block_id}_acc"),
            }
        )
    return table


baseline_multiseed = load_group(
    [
        ("full_seed0_existing", "runs/sctm_primary_full_attention_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
        ("full_seed1", "runs/sctm_validation_full_attention_seed1_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
        ("full_seed2", "runs/sctm_validation_full_attention_seed2_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
    ]
)
sctm_multiseed = load_group(
    [
        ("sctm_seed0_existing", "runs/sctm_primary_lowbit_k8_local3x3_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
        ("sctm_seed1", "runs/sctm_validation_lowbit_k8_b2_local3x3_seed1_noaug_8000_20260628/*/summary.json"),
        ("sctm_seed2", "runs/sctm_validation_lowbit_k8_b2_local3x3_seed2_noaug_8000_20260628/*/summary.json"),
    ]
)
sweep_rows = load_group(
    [
        ("k8_b2_existing", "runs/sctm_primary_lowbit_k8_local3x3_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
        ("k8_b1", "runs/sctm_sweep_lowbit_k8_b1_local3x3_seed0_noaug_8000_20260628/*/summary.json"),
        ("k8_b3", "runs/sctm_sweep_lowbit_k8_b3_local3x3_seed0_noaug_8000_20260628/*/summary.json"),
        ("k12_b2", "runs/sctm_sweep_lowbit_k12_b2_local3x3_seed0_noaug_8000_20260628/*/summary.json"),
        ("k6_b2_optional", "runs/sctm_sweep_lowbit_k6_b2_local3x3_seed0_noaug_8000_20260628/*/summary.json"),
    ]
)
router_rows = load_group(
    [
        ("continuous_existing", "runs/sctm_primary_lowbit_k8_local3x3_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
        ("int4", "runs/sctm_router_lowbit_k8_b2_int4_local3x3_seed0_noaug_8000_20260628/*/summary.json"),
        ("int3", "runs/sctm_router_lowbit_k8_b2_int3_local3x3_seed0_noaug_8000_20260628/*/summary.json"),
        ("binary_xnor", "runs/sctm_router_lowbit_k8_b2_binary_xnor_local3x3_seed0_noaug_8000_20260628/*/summary.json"),
    ]
)

baseline_agg = aggregate(baseline_multiseed)
sctm_agg = aggregate(sctm_multiseed)
full_mean = baseline_agg.get("teacher_hard_acc_mean")
sctm_mean = sctm_agg.get("teacher_hard_acc_mean")
decision_delta = None if full_mean is None or sctm_mean is None else sctm_mean - full_mean

diagnostic_source = next((row for row in sctm_multiseed if not row.get("missing") and "sctm_b0_route_entropy" in row), None)
if diagnostic_source is None:
    diagnostic_source = next((row for row in router_rows if not row.get("missing") and "sctm_b0_route_entropy" in row), None)

payload = {
    "multi_seed": {
        "full_attention": baseline_agg,
        "sctm_lowbit_k8_b2_local3x3": sctm_agg,
        "sctm_minus_full_hard_acc_mean": decision_delta,
    },
    "baseline_rows": baseline_multiseed,
    "sctm_rows": sctm_multiseed,
    "sweep_rows": sweep_rows,
    "router_rows": router_rows,
    "per_block_diagnostics_source": None if diagnostic_source is None else diagnostic_source.get("_summary_path"),
    "per_block_diagnostics": per_block_table(diagnostic_source),
}
print(json.dumps(payload, indent=2, sort_keys=True))

metrics = [
    "teacher_soft_acc",
    "teacher_hard_acc",
    "teacher_acc_gap",
    "teacher_train_time",
    "peak_memory_mb",
]
print_rows("MULTI_SEED_FULL", baseline_multiseed, metrics)
print_rows("MULTI_SEED_SCTM", sctm_multiseed, metrics)
print_rows(
    "SWEEP",
    sweep_rows,
    [
        "teacher_hard_acc",
        "teacher_soft_acc",
        "teacher_acc_gap",
        "sctm_route_entropy",
        "sctm_patch_norm_before",
        "sctm_patch_norm_after",
        "sctm_unique_selected_patch_ratio",
        "sctm_est_logic_depth",
        "sctm_cost_estimated_gate_equivalent_cost",
        "sctm_cost_out_projection_ops",
    ],
)
print_rows(
    "ROUTER",
    router_rows,
    [
        "sctm_score_mode",
        "teacher_hard_acc",
        "teacher_soft_acc",
        "teacher_acc_gap",
        "sctm_route_entropy",
        "sctm_est_logic_depth",
        "sctm_cost_estimated_gate_equivalent_cost",
        "sctm_cost_out_projection_ops",
    ],
)

print("\nPER_BLOCK")
print("| block | entropy | unique | patch_norm | cls_update | skip_acc |")
print("|---|---|---|---|---|---|")
for row in per_block_table(diagnostic_source):
    patch_norm = ""
    if row["patch_norm_before"] is not None:
        patch_norm = f"{float(row['patch_norm_before']):.4g}->{float(row['patch_norm_after']):.4g}"
    vals = [
        row["block"],
        fmt(row["route_entropy"]),
        fmt(row["unique_patch_ratio"]),
        patch_norm,
        fmt(row["cls_update_norm"]),
        fmt(row["skip_acc"]),
    ]
    print("| " + " | ".join(str(v) for v in vals) + " |")
