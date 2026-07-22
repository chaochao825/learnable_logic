from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


ATTENTION_CLEAN_PROFILE = "attention_clean_20260413_215439"


def load_attention_clean_teacher(
    *,
    source_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    profile: str = ATTENTION_CLEAN_PROFILE,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the verified 90.4% CIFAR-10 attention-clean checkpoint."""

    if profile != ATTENTION_CLEAN_PROFILE:
        raise ValueError(profile)
    source = Path(source_dir).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not (source / "vit_tiny_attention_logic.py").is_file():
        raise FileNotFoundError(source / "vit_tiny_attention_logic.py")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    from vit_tiny_attention_logic import vit_tiny

    model = vit_tiny(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        attention_only=False,
        attention_k=9,
        topk_impl="torch-topk",
        validate_input=False,
        use_thermometer_encoding=True,
        n_thresholds=3,
        encoding_scale=10.0,
        apply_sigmoid_before_encoding=True,
        decode_output=True,
        boundary_surrogate_temperature=1.0,
        majority_train_temperature=1.0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TypeError("teacher checkpoint must contain state_dict")
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing_suffix = "._fixed_random_topk_indices"
    unexpected_missing = [
        key for key in incompatible.missing_keys if not key.endswith(allowed_missing_suffix)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            {
                "missing": unexpected_missing,
                "unexpected": incompatible.unexpected_keys,
            }
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "profile": profile,
        "source_dir": str(source),
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "allowed_missing_buffers": incompatible.missing_keys,
    }
    return model, metadata
