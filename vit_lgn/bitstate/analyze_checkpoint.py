"""Compute deployment and depth-gap diagnostics for a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import BitStateConfig, BitStateViT
from .regularization import entropy_unused_gate_ratio, gate_distribution_metrics
from .train_bitstate import (
    make_loaders,
    measure_inactive,
    measure_layer_gap,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("batches must be positive")

    summary_path = args.run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = args.run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.run_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint missing in {args.run_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    training_args = argparse.Namespace(**summary["training_args"])
    seed_everything(training_args.seed)
    device = torch.device(args.device)
    train_loader, _validation_loader, test_loader, _shape = make_loaders(training_args)
    model = BitStateViT(BitStateConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    first_images, _labels = next(iter(test_loader))
    model.assert_bit_exact(first_images.to(device))
    activation_inactive = measure_inactive(
        model,
        train_loader,
        device,
        args.batches,
    )
    layer_gap = measure_layer_gap(
        model,
        test_loader,
        device,
        tau=training_args.eval_tau,
        max_batches=args.batches,
    )
    gate_entropy, gate_confidence = gate_distribution_metrics(model.gate_layers())
    result = {
        "source_summary": str(summary_path),
        "checkpoint": str(checkpoint_path),
        "bit_exact_verified": True,
        "unused_gate_ratio": entropy_unused_gate_ratio(model.gate_layers()),
        "activation_inactive_gate_ratio": activation_inactive,
        "unused_gate_definition": "mind_gap_entropy_above_initialization_2.5pct",
        "gate_entropy": float(gate_entropy.detach()),
        "gate_confidence": float(gate_confidence.detach()),
        "layer_gap_max_mae": max(item["mae"] for item in layer_gap),
        "layer_gap_max_flip_ratio": max(item["flip_ratio"] for item in layer_gap),
        "layer_gap_final_flip_ratio": layer_gap[-1]["flip_ratio"],
        "layer_gap_diagnostics": layer_gap,
    }
    output = args.output or args.run_dir / "posthoc_diagnostics.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
