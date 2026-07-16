"""Bounded GPU smoke probe for the minimal shared logic-tree branch."""

from __future__ import annotations

import argparse
import json
import time

import torch

from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
from vit_lgn.full_discrete.enhancements_logic_tree import SharedLogicTreeConv3x3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--local-layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(20260714)
    torch.cuda.manual_seed_all(20260714)
    model = EnhancedFullDiscreteViT(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        topk=8,
        mlp_ratio=4.0,
        weight_bits=7,
        activation_bits=8,
        qk_lanes=7,
        local_layers=args.local_layers,
        local_operator="logic_tree3x3",
    ).to(device).train()
    images = torch.rand(args.batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (args.batch_size,), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss_value = 0.0
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    branches = [
        module for module in model.modules()
        if isinstance(module, SharedLogicTreeConv3x3)
    ]
    tree_gradient = sum(
        float(branch.truth_table_logits.grad.abs().sum())
        for branch in branches
        if branch.truth_table_logits.grad is not None
    )
    print(json.dumps({
        "batch_size": args.batch_size,
        "dim": args.dim,
        "depth": args.depth,
        "local_layers": args.local_layers,
        "steps": args.steps,
        "seconds_total": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
        "loss": loss_value,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "logic_tree_parameters": sum(
            branch.truth_table_logits.numel() for branch in branches
        ),
        "unique_shared_lut_gates": sum(
            branch.num_unique_shared_lut_gates for branch in branches
        ),
        "gate_evaluations_per_image": sum(
            branch.gate_evaluations_per_image for branch in branches
        ),
        "logic_tree_gradient_l1": tree_gradient,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
