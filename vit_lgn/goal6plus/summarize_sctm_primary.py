from __future__ import annotations

import glob
import json
from pathlib import Path


RUNS = [
    ("baseline_full", "runs/sctm_primary_full_attention_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
    ("mean_k4_identity", "runs/sctm_primary_mean_k4_identity_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
    ("mean_k8_identity", "runs/sctm_primary_mean_k8_identity_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
    ("lowbit_k4_local3x3", "runs/sctm_primary_lowbit_k4_local3x3_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
    ("lowbit_k8_local3x3", "runs/sctm_primary_lowbit_k8_local3x3_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
    ("static_k8_local3x3", "runs/sctm_primary_static_k8_local3x3_d6_h6_ratio1_noaug_8000_20260628/*/summary.json"),
]


FIELDS = [
    "teacher_soft_acc",
    "teacher_hard_acc",
    "teacher_acc_gap",
    "sctm_score_mode",
    "baseline_cls_cosine",
    "baseline_topk_overlap",
    "sctm_route_entropy",
    "sctm_unique_selected_patch_ratio",
    "sctm_patch_norm_before",
    "sctm_patch_norm_after",
    "skip_mixer_acc",
    "skip_logicffn_acc",
    "teacher_train_time",
    "peak_memory_mb",
    "teacher_gate_count",
    "teacher_depth",
    "teacher_unused_gate_ratio",
    "teacher_dead_gate_ratio",
    "sctm_est_logic_depth",
    "sctm_cost_estimated_gate_equivalent_cost",
    "sctm_cost_score_projection_ops",
    "sctm_cost_out_projection_ops",
    "sctm_cost_score_dot_ops",
    "sctm_cost_topk_comparator_ops",
    "sctm_cost_selected_value_mux_ops",
    "sctm_cost_lowbit_weighted_aggregation_ops",
    "sctm_cost_local3x3_patch_mixer_ops",
    "sctm_selected_patch_histogram_path",
    "checkpoint_path",
]


def latest(pattern: str) -> Path | None:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        return None
    return paths[-1]


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


rows = []
baseline_hard = None
for label, pattern in RUNS:
    path = latest(pattern)
    if path is None:
        row = {"label": label, "summary": pattern, "missing": True}
        for field in FIELDS:
            row[field] = None
        rows.append(row)
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if label == "baseline_full":
        baseline_hard = float(data["teacher_hard_acc"])
    row = {"label": label, "summary": str(path)}
    for field in FIELDS:
        row[field] = data.get(field)
    rows.append(row)

print(json.dumps({"baseline_hard": baseline_hard, "rows": rows}, indent=2, sort_keys=True))

print("\nMARKDOWN_TABLE")
header = [
    "run",
    "score",
    "soft",
    "hard",
    "drop_vs_base",
    "gap",
    "entropy",
    "unique",
    "topk_overlap",
    "cls_cos",
    "patch_norm",
    "skip_mix",
    "skip_ffn",
    "time_s",
    "mem_mb",
    "depth",
    "gate_eq",
]
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")
for row in rows:
    if row.get("missing"):
        vals = [row["label"], "MISSING", *([""] * (len(header) - 2))]
        print("| " + " | ".join(vals) + " |")
        continue
    hard = row["teacher_hard_acc"]
    drop = None if baseline_hard is None or hard is None else float(hard) - baseline_hard
    patch_norm = ""
    if row["sctm_patch_norm_before"] is not None:
        patch_norm = f"{float(row['sctm_patch_norm_before']):.3f}->{float(row['sctm_patch_norm_after']):.3f}"
    vals = [
        row["label"],
        fmt(row["sctm_score_mode"]),
        fmt(row["teacher_soft_acc"]),
        fmt(row["teacher_hard_acc"]),
        fmt(drop),
        fmt(row["teacher_acc_gap"]),
        fmt(row["sctm_route_entropy"]),
        fmt(row["sctm_unique_selected_patch_ratio"]),
        fmt(row["baseline_topk_overlap"]),
        fmt(row["baseline_cls_cosine"]),
        patch_norm,
        fmt(row["skip_mixer_acc"]),
        fmt(row["skip_logicffn_acc"]),
        fmt(row["teacher_train_time"]),
        fmt(row["peak_memory_mb"]),
        fmt(row["sctm_est_logic_depth"]),
        fmt(row["sctm_cost_estimated_gate_equivalent_cost"]),
    ]
    print("| " + " | ".join(vals) + " |")
