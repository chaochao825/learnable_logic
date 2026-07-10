from __future__ import annotations

import glob
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def first_number(data: dict, keys: list[str]):
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def first_value(summary: dict, config: dict, keys: list[str], default=""):
    for key in keys:
        if key in summary:
            return summary[key]
        if key in config:
            return config[key]
    return default


rows = []
for summary_path_str in glob.glob("runs/*/*/summary.json"):
    summary_path = Path(summary_path_str)
    summary = load_json(summary_path)
    config = load_json(summary_path.with_name("config.json"))
    hard = first_number(summary, ["teacher_hard_acc", "hard_acc", "student_hard_acc", "best_test_acc"])
    soft = first_number(summary, ["teacher_soft_acc", "soft_acc", "teacher_relaxed_acc"])
    if hard is None and soft is None:
        continue
    run_root = summary_path.parent.parent.name
    rows.append(
        {
            "hard": hard,
            "soft": soft,
            "run": run_root,
            "time": summary_path.parent.name,
            "mixer": first_value(summary, config, ["mixer", "variant"], ""),
            "depth": first_value(summary, config, ["arch_depth", "depth"], ""),
            "embed": first_value(summary, config, ["arch_embed_dim", "embed_dim"], ""),
            "heads": first_value(summary, config, ["arch_num_heads", "num_heads"], ""),
            "tokens": first_value(summary, config, ["arch_tokens"], ""),
            "logic_ratio": first_value(summary, config, ["logic_mlp_ratio"], ""),
            "augment": first_value(summary, config, ["augment"], ""),
            "head_type": first_value(summary, config, ["head_type"], ""),
            "topk": first_value(summary, config, ["sctm_topk", "topk"], ""),
            "patch_path": first_value(summary, config, ["sctm_patch_path"], ""),
            "param_count": first_value(summary, config, ["param_count"], ""),
            "gate_count": first_value(summary, config, ["teacher_gate_count", "gate_count"], ""),
            "logic_depth": first_value(summary, config, ["teacher_depth", "depth"], ""),
            "memory": first_value(summary, config, ["peak_memory_mb"], ""),
            "path": str(summary_path),
        }
    )

rows.sort(key=lambda row: (-1.0 if row["hard"] is None else -row["hard"], row["run"]))

print("TOP_HARD")
header = [
    "hard",
    "soft",
    "run",
    "mixer",
    "depth",
    "embed",
    "heads",
    "logic_ratio",
    "augment",
    "topk",
    "patch_path",
    "gate_count",
    "memory",
]
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")
for row in rows[:40]:
    print("| " + " | ".join(str(row.get(col, "")) for col in header) + " |")

print("\nSCTM_SCALE_AND_PRIMARY")
for row in rows:
    if "sctm_primary" in row["run"] or "sctm_scale" in row["run"]:
        print(json.dumps(row, sort_keys=True, ensure_ascii=False))
