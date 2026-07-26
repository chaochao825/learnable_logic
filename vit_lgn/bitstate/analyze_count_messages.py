from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path

import torch

from research_registry.capacity import bitstate_capacity
from vit_lgn.bitstate.gates import GATE_NAMES, TRUTH_TABLE
from vit_lgn.bitstate.model import BitStateConfig, BitStateViT
from vit_lgn.bitstate.train_bitstate import make_loaders


def _entropy(histogram: torch.Tensor) -> float:
    total = int(histogram.sum())
    if not total:
        return 0.0
    entropy = 0.0
    for count in histogram.tolist():
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def merge_gate_usage(model: BitStateViT) -> dict[str, object]:
    """Measure whether selected merge LUTs actually depend on each input."""

    block_rows = []
    aggregate = {
        "gate_count": 0,
        "uses_state": 0,
        "uses_message": 0,
        "state_only": 0,
        "message_only": 0,
        "both": 0,
        "neither": 0,
        "identity_state": 0,
        "identity_message": 0,
    }
    truth = TRUTH_TABLE.cpu()
    for block_id, block in enumerate(model.global_blocks):
        operations = block.merge.hard_ops().cpu()
        selected = truth[operations]
        uses_state = (selected[:, 0] != selected[:, 2]) | (
            selected[:, 1] != selected[:, 3]
        )
        uses_message = (selected[:, 0] != selected[:, 1]) | (
            selected[:, 2] != selected[:, 3]
        )
        counts = {
            "gate_count": operations.numel(),
            "uses_state": int(uses_state.sum()),
            "uses_message": int(uses_message.sum()),
            "state_only": int((uses_state & ~uses_message).sum()),
            "message_only": int((~uses_state & uses_message).sum()),
            "both": int((uses_state & uses_message).sum()),
            "neither": int((~uses_state & ~uses_message).sum()),
            "identity_state": int((operations == GATE_NAMES.index("a")).sum()),
            "identity_message": int((operations == GATE_NAMES.index("b")).sum()),
        }
        for key, value in counts.items():
            aggregate[key] += value
        denominator = max(counts["gate_count"], 1)
        block_rows.append(
            {
                "block": block_id,
                **counts,
                **{
                    f"{key}_ratio": value / denominator
                    for key, value in counts.items()
                    if key != "gate_count"
                },
            }
        )
    denominator = max(aggregate["gate_count"], 1)
    return {
        "aggregate": {
            **aggregate,
            **{
                f"{key}_ratio": value / denominator
                for key, value in aggregate.items()
                if key != "gate_count"
            },
        },
        "blocks": block_rows,
    }


@torch.no_grad()
def analyze(
    model: BitStateViT,
    loader: object,
    device: torch.device,
    max_batches: int,
) -> dict[str, object]:
    block_rows = []
    histograms = [
        torch.zeros(block.topk + 1, dtype=torch.int64)
        for block in model.global_blocks
    ]
    changed = [0 for _ in model.global_blocks]
    elements = [0 for _ in model.global_blocks]
    threshold_active = [
        torch.zeros(block.message_count_thresholds.numel(), dtype=torch.int64)
        for block in model.global_blocks
    ]
    threshold_elements = [0 for _ in model.global_blocks]

    for batch_id, (images, _labels) in enumerate(loader):
        if batch_id >= max_batches:
            break
        state = model.encoder.forward_bits(images.to(device))
        for block in model.local_blocks:
            state = block.forward_bits(state)
        for block_id, block in enumerate(model.global_blocks):
            query = block._reshape_qk(block.query.forward_bits(state))
            key = block._reshape_qk(block.key.forward_bits(state))
            indices = block._topk_indices(query, key)
            selected = block._gather_values(state, indices)
            counts = selected.to(torch.int32).sum(dim=3)
            histograms[block_id] += torch.bincount(
                counts.reshape(-1).cpu(),
                minlength=block.topk + 1,
            )
            majority = counts * 2 >= block.topk
            encoded = block._encode_hard_counts(counts)
            changed[block_id] += int(torch.count_nonzero(encoded != majority))
            elements[block_id] += encoded.numel()
            if block.message_mode != "majority":
                source_counts = counts[..., block.message_count_source_indices]
                active = (
                    source_counts.unsqueeze(-1)
                    >= block.message_count_thresholds.view(1, 1, 1, 1, -1)
                )
                threshold_active[block_id] += active.sum(
                    dim=(0, 1, 2, 3)
                ).cpu()
                threshold_elements[block_id] += active.numel() // active.shape[-1]
            state = block.forward_bits(state)

    for block_id, (block, histogram) in enumerate(
        zip(model.global_blocks, histograms)
    ):
        total = int(histogram.sum())
        non_extreme = int(histogram[1:-1].sum()) if histogram.numel() > 2 else 0
        row: dict[str, object] = {
            "block": block_id,
            "message_mode": block.message_mode,
            "count_histogram": [int(value) for value in histogram.tolist()],
            "observed_count_levels": int(torch.count_nonzero(histogram)),
            "count_entropy_bits": _entropy(histogram),
            "non_extreme_count_ratio": non_extreme / max(total, 1),
            "message_change_vs_majority": changed[block_id]
            / max(elements[block_id], 1),
            "count_output_width_per_head": block.count_output_width,
            "count_source_channels_per_head": int(
                block.message_count_source_indices.numel()
            ),
        }
        if threshold_elements[block_id]:
            row["threshold_active_ratios"] = [
                int(value) / threshold_elements[block_id]
                for value in threshold_active[block_id].tolist()
            ]
        else:
            row["threshold_active_ratios"] = []
        block_rows.append(row)
    return {
        "capacity": bitstate_capacity(model.config, model),
        "batches": min(max_batches, len(loader)),
        "blocks": block_rows,
        "merge_gate_usage": merge_gate_usage(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = BitStateViT(BitStateConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    training_args = Namespace(**checkpoint["result"]["training_args"])
    training_args.device = args.device
    _train, _validation, test_loader, _shape = make_loaders(training_args)
    result = analyze(model, test_loader, torch.device(args.device), args.max_batches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
