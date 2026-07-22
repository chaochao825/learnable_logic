"""Collect bit-state experiment summaries into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


RESULT_FIELDS = (
    "campaign",
    "run",
    "method",
    "dataset",
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "train_time",
    "epochs_to_target",
    "epochs_to_20pct",
    "time_to_20pct",
    "unused_gate_ratio",
    "activation_inactive_gate_ratio",
    "unused_gate_definition",
    "gate_count",
    "predicate_count",
    "depth",
    "fanout_max",
    "best_discrete_acc",
    "best_epoch",
    "state_entropy",
    "state_constant_ratio",
    "state_duplicate_ratio",
    "state_flip_rate",
    "layer_gap_max_mae",
    "layer_gap_max_flip_ratio",
    "layer_gap_final_flip_ratio",
    "gate_entropy",
    "gate_confidence",
    "hard_path_acc_gap",
    "hard_path_loss_gap",
    "bit_exact_verified",
    "seed",
    "state_width",
    "encoder_kind",
    "global_token_mode",
    "local_depth",
    "global_depth",
    "heads",
    "qk_bits",
    "topk",
    "votes_per_class",
    "epochs",
    "validation_size",
    "train_limit",
    "eval_limit",
    "group_sum_temperature",
    "gate_init_strength",
    "gate_init_mode",
    "gate_init_normal_std",
    "hardening_logit_scale",
    "soft_warmup_epochs",
    "optimizer",
    "learning_rate",
    "lr_schedule",
)


def summary_paths(roots: Iterable[Path]) -> list[tuple[Path, Path]]:
    paths: list[tuple[Path, Path]] = []
    for root in roots:
        paths.extend((root, path) for path in root.rglob("summary.json"))
    return sorted(paths, key=lambda item: (item[0].name, str(item[1])))


def checkpoint_entropy_unused(path: Path) -> float:
    import torch

    from .regularization import MIND_GAP_ENTROPY_THRESHOLD

    checkpoint_path = path.parent / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = path.parent / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint missing beside {path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    entropy_values = []
    for name, value in state.items():
        if not name.endswith(".logits") or value.ndim != 2 or value.shape[-1] != 16:
            continue
        probabilities = value.to(torch.float32).softmax(dim=-1)
        entropy_values.append(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        )
    if not entropy_values:
        raise ValueError(f"no 16-function gate logits in {checkpoint_path}")
    entropy = torch.cat(entropy_values)
    return float((entropy > MIND_GAP_ENTROPY_THRESHOLD).to(torch.float32).mean())


def summary_row(
    root: Path,
    path: Path,
    *,
    infer_legacy_entropy_unused: bool = False,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    model = summary.get("model_config") or {}
    training = summary.get("training_args") or {}
    row = {
        field: summary.get(field, "")
        for field in RESULT_FIELDS
    }
    row.update(
        campaign=root.name,
        run=str(path.parent.relative_to(root)),
        state_width=model.get("state_width", training.get("state_width", "")),
        encoder_kind=model.get("encoder_kind", training.get("encoder_kind", "")),
        global_token_mode=model.get(
            "global_token_mode", training.get("global_token_mode", "majority")
        ),
        local_depth=model.get("local_depth", training.get("local_depth", "")),
        global_depth=model.get("global_depth", training.get("global_depth", "")),
        heads=model.get("heads", training.get("heads", "")),
        qk_bits=model.get("qk_bits", training.get("qk_bits", "")),
        topk=model.get("topk", training.get("topk", "")),
        votes_per_class=model.get(
            "votes_per_class", training.get("votes_per_class", "")
        ),
        epochs=training.get("epochs", ""),
        validation_size=training.get("validation_size", ""),
        train_limit=training.get("train_limit", ""),
        eval_limit=training.get("eval_limit", ""),
        group_sum_temperature=training.get("group_sum_temperature", ""),
        gate_init_strength=training.get("gate_init_strength", ""),
        gate_init_mode=training.get("gate_init_mode", "targeted_legacy"),
        gate_init_normal_std=training.get("gate_init_normal_std", ""),
        hardening_logit_scale=training.get("hardening_logit_scale", ""),
        soft_warmup_epochs=training.get("soft_warmup_epochs", ""),
        optimizer=training.get("optimizer", ""),
        learning_rate=training.get("learning_rate", ""),
        lr_schedule=training.get("lr_schedule", ""),
    )
    for epoch in summary.get("history") or []:
        if float(epoch.get("discrete_acc", 0.0)) >= 0.2:
            row["epochs_to_20pct"] = epoch.get("epoch", "")
            row["time_to_20pct"] = epoch.get("elapsed", "")
            break
    if "activation_inactive_gate_ratio" in summary:
        row["activation_inactive_gate_ratio"] = summary[
            "activation_inactive_gate_ratio"
        ]
        row["unused_gate_definition"] = summary.get(
            "unused_gate_definition", "mind_gap_entropy"
        )
    else:
        row["activation_inactive_gate_ratio"] = summary.get(
            "unused_gate_ratio", ""
        )
        row["unused_gate_definition"] = "legacy_activation_inactive"
        if infer_legacy_entropy_unused:
            row["unused_gate_ratio"] = checkpoint_entropy_unused(path)
            row["unused_gate_definition"] = "mind_gap_entropy_posthoc"
    posthoc_path = path.parent / "posthoc_diagnostics.json"
    if posthoc_path.exists():
        posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
        for field in (
            "unused_gate_ratio",
            "activation_inactive_gate_ratio",
            "layer_gap_max_mae",
            "layer_gap_max_flip_ratio",
            "layer_gap_final_flip_ratio",
        ):
            if field in posthoc:
                row[field] = posthoc[field]
        row["unused_gate_definition"] = posthoc.get(
            "unused_gate_definition", row["unused_gate_definition"]
        )
    return row


def collect(
    roots: Iterable[Path],
    *,
    infer_legacy_entropy_unused: bool = False,
) -> list[dict[str, Any]]:
    return [
        summary_row(
            root,
            path,
            infer_legacy_entropy_unused=infer_legacy_entropy_unused,
        )
        for root, path in summary_paths(roots)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--infer-legacy-entropy-unused",
        action="store_true",
        help="read adjacent checkpoints to repair pre-schema entropy-unused metrics",
    )
    args = parser.parse_args()

    rows = collect(
        args.roots,
        infer_legacy_entropy_unused=args.infer_legacy_entropy_unused,
    )
    if not rows:
        parser.error("no summary.json files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
