"""Evaluate normalization counterfactuals on one frozen CIFAR checkpoint.

This diagnostic deliberately does not claim retraining equivalence.  It loads
identical hardened weights into topology-compatible norm variants and reports
the full held-out validation split, making it useful for selecting the small
set of variants that deserve a formal 50k run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from .enhanced_model import EnhancedFullDiscreteViT
from .export_logic_payload import _model_kwargs_from_checkpoint_args


def _parse_variant(specification: str) -> tuple[str, str]:
    block, separator, final = specification.partition(":")
    if not separator:
        final = "same"
    allowed = {"rms_lut", "shift_rms", "requant", "none"}
    if block not in allowed or final not in allowed | {"same"}:
        raise ValueError(f"invalid normalization variant: {specification}")
    return block, final


def _validation_loader(data_root: Path, valid_size: int, batch_size: int) -> DataLoader:
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    dataset = CIFAR10(data_root, train=True, download=False, transform=transform)
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(20260711)
    ).tolist()
    return DataLoader(
        Subset(dataset, order[:valid_size]),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    correct = count = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(torch.nn.functional.cross_entropy(
            logits, labels, reduction="sum"
        ))
        correct += int((logits.argmax(-1) == labels).sum())
        count += labels.numel()
    return correct / count, loss_sum / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--valid-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "rms_lut:same", "shift_rms:same", "requant:same", "none:same",
            "shift_rms:none", "rms_lut:none",
        ],
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args")
    if not isinstance(model_args, dict):
        raise ValueError("checkpoint args are required")
    base_kwargs = _model_kwargs_from_checkpoint_args(model_args)
    loader = _validation_loader(args.data_root, args.valid_size, args.batch_size)
    device = torch.device(args.device)

    for specification in args.variants:
        block_kind, final_kind = _parse_variant(specification)
        kwargs = {
            **base_kwargs,
            "norm_kind": block_kind,
            "final_norm_kind": final_kind,
        }
        model = EnhancedFullDiscreteViT(**kwargs)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        started = time.perf_counter()
        accuracy, cross_entropy = _evaluate(model, loader, device)
        print(json.dumps({
            "variant": specification,
            "accuracy": accuracy,
            "cross_entropy": cross_entropy,
            "examples": args.valid_size,
            "seconds": time.perf_counter() - started,
        }, sort_keys=True), flush=True)
        del model


if __name__ == "__main__":
    main()
