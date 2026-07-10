from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import load_dataset
from goal6plus_vit_experiment import (
    DEVICE,
    evaluate,
    harden_model,
    make_train_args,
    prepare_forward,
    preprocess_batch,
    train_teacher,
)
from probe_frozen_features import DEFAULTS
from train_logic_vit_tiny import build_model, set_deterministic


EPS = 1e-12


def _copy_linear(dst: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    with torch.no_grad():
        dst.weight.copy_(weight)
        if dst.bias is not None:
            if bias is None:
                dst.bias.zero_()
            else:
                dst.bias.copy_(bias)


class _CLSBaseTopKMixer(nn.Module):
    """CLS-to-patch mixer base: patch-token outputs are always zero residuals."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patches: int,
        topk: int,
        aggregation: str,
        route_temperature: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        if aggregation not in {"mean", "weighted"}:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_patches = int(num_patches)
        self.head_dim = self.embed_dim // self.num_heads
        self.topk = int(topk)
        self.aggregation = aggregation
        self.route_temperature = float(route_temperature)
        self.hard_routing = False

        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    @classmethod
    def _split_qkv(cls, attn: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
        return attn.qkv.weight.detach().chunk(3, dim=0), (
            None if attn.qkv.bias is None else attn.qkv.bias.detach().chunk(3, dim=0)
        )

    def copy_value_and_output_from_attention(self, attn: nn.Module) -> None:
        weight_chunks, bias_chunks = self._split_qkv(attn)
        _q_w, _k_w, v_w = weight_chunks
        if bias_chunks is None:
            v_b = None
        else:
            _q_b, _k_b, v_b = bias_chunks
        _copy_linear(self.v_proj, v_w, v_b)
        _copy_linear(self.out_proj, attn.proj.weight.detach(), None if attn.proj.bias is None else attn.proj.bias.detach())

    def set_hard_routing(self, enabled: bool) -> None:
        self.hard_routing = bool(enabled)

    def route_scores(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        topk = min(max(int(self.topk), 1), self.num_patches)
        return scores.topk(topk, dim=-1).indices

    def route_weights(self, scores: torch.Tensor, hard: bool | None = None) -> torch.Tensor:
        use_hard = self.hard_routing if hard is None else bool(hard)
        topk = min(max(int(self.topk), 1), self.num_patches)
        indices = scores.topk(topk, dim=-1).indices
        top_scores = scores.gather(-1, indices)
        weights = torch.zeros_like(scores)
        if self.aggregation == "mean":
            uniform = torch.full_like(top_scores, 1.0 / topk)
            if use_hard:
                selected_weights = uniform
            else:
                soft_surrogate = F.softmax(top_scores / max(self.route_temperature, 1e-6), dim=-1)
                selected_weights = uniform + soft_surrogate - soft_surrogate.detach()
        else:
            selected_weights = F.softmax(top_scores / max(self.route_temperature, 1e-6), dim=-1)
        weights.scatter_(-1, indices, selected_weights)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if tokens != self.num_patches + 1:
            raise ValueError(f"Expected {self.num_patches + 1} tokens, got {tokens}")
        cls = x[:, 0]
        patches = x[:, 1:]
        scores = self.route_scores(cls, patches)
        weights = self.route_weights(scores)
        values = self.v_proj(patches).reshape(batch, self.num_patches, self.num_heads, self.head_dim)
        values = values.transpose(1, 2)
        mixed = torch.einsum("bhp,bhpd->bhd", weights, values).reshape(batch, channels)
        mixed = self.out_proj(mixed)
        out = x.new_zeros(batch, tokens, channels)
        out[:, 0] = mixed
        return out


class CLSStaticTopKMixer(_CLSBaseTopKMixer):
    """Learned static per-head, per-patch routing scores for CLS update."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patches: int,
        topk: int,
        aggregation: str = "mean",
        route_temperature: float = 1.0,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_patches, topk, aggregation, route_temperature)
        self.route_logits = nn.Parameter(torch.zeros(self.num_heads, self.num_patches))

    @classmethod
    def from_attention(
        cls,
        attn: nn.Module,
        num_patches: int,
        topk: int,
        aggregation: str,
        route_temperature: float,
    ) -> "CLSStaticTopKMixer":
        mixer = cls(
            embed_dim=int(attn.embed_dim),
            num_heads=int(attn.num_heads),
            num_patches=num_patches,
            topk=topk,
            aggregation=aggregation,
            route_temperature=route_temperature,
        )
        mixer.copy_value_and_output_from_attention(attn)
        return mixer

    def route_scores(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        return self.route_logits.unsqueeze(0).expand(patches.shape[0], -1, -1)


class CLSDynamicTopKRouter(_CLSBaseTopKMixer):
    """CLS-to-patch q/k router; never forms full token-to-token attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patches: int,
        topk: int,
        aggregation: str = "mean",
        route_temperature: float = 1.0,
    ) -> None:
        super().__init__(embed_dim, num_heads, num_patches, topk, aggregation, route_temperature)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)

    @classmethod
    def from_attention(
        cls,
        attn: nn.Module,
        num_patches: int,
        topk: int,
        aggregation: str,
        route_temperature: float,
    ) -> "CLSDynamicTopKRouter":
        mixer = cls(
            embed_dim=int(attn.embed_dim),
            num_heads=int(attn.num_heads),
            num_patches=num_patches,
            topk=topk,
            aggregation=aggregation,
            route_temperature=route_temperature,
        )
        weight_chunks, bias_chunks = mixer._split_qkv(attn)
        q_w, k_w, _v_w = weight_chunks
        if bias_chunks is None:
            q_b = k_b = None
        else:
            q_b, k_b, _v_b = bias_chunks
        _copy_linear(mixer.q_proj, q_w, q_b)
        _copy_linear(mixer.k_proj, k_w, k_b)
        mixer.copy_value_and_output_from_attention(attn)
        return mixer

    def route_scores(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        q = self.q_proj(cls).reshape(batch, self.num_heads, self.head_dim)
        k = self.k_proj(patches).reshape(batch, self.num_patches, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)
        return (q.unsqueeze(2) * k).sum(dim=-1) * (self.head_dim ** -0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLS-only top-k token-mixer ablation for ViT-LGN attention blocks.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="runs/cls_topk_token_mixer_ablation")
    parser.add_argument("--teacher-iters", type=int, default=-1)
    parser.add_argument("--eval-split", choices=["valid", "test"], default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--topk-list", type=str, default="4,8,16")
    parser.add_argument("--block-settings", type=str, default="b3;b2,b3")
    parser.add_argument("--include-b1-b2-b3", action="store_true")
    parser.add_argument(
        "--mixer-variants",
        type=str,
        default="static_mean,static_weighted,dynamic_mean,dynamic_weighted",
    )
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--route-stats-batches", type=int, default=20)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SimpleNamespace:
    cfg = dict(DEFAULTS)
    if args.config_json:
        cfg.update(json.loads(Path(args.config_json).read_text(encoding="utf-8")))
    if args.teacher_iters >= 0:
        cfg["teacher_iters"] = args.teacher_iters
    cfg["student_iters"] = 0
    cfg["distill_alpha"] = 0.0
    if args.eval_split is not None:
        cfg["eval_split"] = args.eval_split
    if args.eval_max_batches is not None:
        cfg["eval_max_batches"] = args.eval_max_batches
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    cfg["out_dir"] = args.out_dir
    cfg.setdefault("post_mask_head_calibrate_iters", 0)
    cfg.setdefault("post_mask_head_calibrate_lr", 0.0)
    cfg.setdefault("post_mask_head_calibrate_target", "student")
    cfg.setdefault("head_type", "linear")
    cfg.setdefault("signed_head_topk", 32)
    cfg.setdefault("signed_head_use_topk_mask", False)
    cfg.setdefault("cross_block_logic_history", False)
    cfg.setdefault("cross_block_candidate_frac", 0.0)
    return SimpleNamespace(**cfg)


def parse_int_list(spec: str) -> list[int]:
    values = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _parse_block_token(token: str) -> int:
    token = token.strip().lower()
    if token.startswith("b"):
        token = token[1:]
    return int(token)


def parse_block_setting(spec: str, depth: int) -> tuple[str, set[int]]:
    raw = spec.strip()
    if not raw or raw.lower() in {"none", "baseline"}:
        return "baseline", set()
    blocks: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = _parse_block_token(start_s)
            end = _parse_block_token(end_s)
            if end < start:
                raise ValueError(f"Invalid descending block range: {part}")
            blocks.update(range(start, end + 1))
        else:
            blocks.add(_parse_block_token(part))
    invalid = sorted(block for block in blocks if block < 0 or block >= depth)
    if invalid:
        raise ValueError(f"Block setting {raw!r} contains invalid blocks {invalid} for depth={depth}")
    label = ",".join(f"b{idx}" for idx in sorted(blocks))
    return label, blocks


def parse_block_settings(spec: str, depth: int, include_b1_b2_b3: bool) -> list[tuple[str, set[int]]]:
    parts = [item.strip() for item in spec.split(";") if item.strip()]
    if include_b1_b2_b3 and depth >= 4 and "b1,b2,b3" not in parts:
        parts.append("b1,b2,b3")
    settings = [parse_block_setting(part, depth) for part in parts]
    if not settings:
        raise ValueError("No block settings requested")
    return settings


def variant_spec(name: str) -> tuple[type[_CLSBaseTopKMixer], str]:
    specs: dict[str, tuple[type[_CLSBaseTopKMixer], str]] = {
        "static_mean": (CLSStaticTopKMixer, "mean"),
        "static_weighted": (CLSStaticTopKMixer, "weighted"),
        "dynamic_mean": (CLSDynamicTopKRouter, "mean"),
        "dynamic_weighted": (CLSDynamicTopKRouter, "weighted"),
    }
    if name not in specs:
        raise ValueError(f"Unknown mixer variant {name!r}; expected one of {sorted(specs)}")
    return specs[name]


def set_mixers_hard(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, _CLSBaseTopKMixer):
            module.set_hard_routing(enabled)


def iter_mixers(model: nn.Module) -> Iterable[tuple[int, _CLSBaseTopKMixer]]:
    for block_idx, block in enumerate(model.blocks):
        if isinstance(block.attn, _CLSBaseTopKMixer):
            yield block_idx, block.attn


def replace_attention_blocks(
    model: nn.Module,
    variant: str,
    blocks: set[int],
    topk: int,
    route_temperature: float,
) -> list[tuple[int, _CLSBaseTopKMixer]]:
    mixer_cls, aggregation = variant_spec(variant)
    replacements: list[tuple[int, _CLSBaseTopKMixer]] = []
    num_patches = int(model.patch_embed.n_patches)
    for block_idx in sorted(blocks):
        block = model.blocks[block_idx]
        old_attn = block.attn
        mixer = mixer_cls.from_attention(old_attn, num_patches, topk, aggregation, route_temperature)
        mixer.to(next(old_attn.parameters()).device)
        block.attn = mixer
        replacements.append((block_idx, mixer))
    return replacements


def maybe_apply_head_mask(model: nn.Module, train_loader, transform, cfg: SimpleNamespace, train_args: SimpleNamespace) -> None:
    if cfg.head_type in {"signed_sparse_linear", "signed_counter_linear", "scaled_signed_counter_linear"}:
        raise RuntimeError(
            "CLS top-k mixer ablation is scoped to the current classifier. "
            "Use a linear-head config or add a project-local head-mask adapter."
        )


def sequence_from_images(model: nn.Module, images: torch.Tensor, transform) -> torch.Tensor:
    x = preprocess_batch(images, transform)
    x = model.patch_embed(x)
    cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)
    x = x + model.pos_embed
    return model.pos_drop(x)


def attention_patch_scores(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, tokens, _channels = x.shape
    qkv = attn.qkv(x).reshape(batch, tokens, 3, attn.num_heads, attn.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    q_cls = q[:, :, 0:1, :]
    k_patch = k[:, :, 1:, :]
    return (q_cls @ k_patch.transpose(-2, -1)).squeeze(-2) * (attn.head_dim ** -0.5)


def entropy_norm(weights: torch.Tensor) -> torch.Tensor:
    return -(weights.clamp_min(EPS) * weights.clamp_min(EPS).log()).sum(dim=-1) / math.log(weights.shape[-1])


def overlap_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a, b: [B, H, K]
    intersection = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
    return intersection / max(int(a.shape[-1]), 1)


@torch.no_grad()
def collect_pair_diagnostics(
    full_model: nn.Module,
    variant_model: nn.Module,
    loader,
    transform,
    max_batches: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    prepare_forward(full_model, "hard_forward")
    prepare_forward(variant_model, "hard_forward")
    set_mixers_hard(variant_model, True)

    route_totals: dict[tuple[int, int], dict[str, object]] = {}
    route_counts: dict[tuple[int, int], int] = {}
    cls_cos_total = 0.0
    cls_cos_count = 0
    max_batches = max(1, int(max_batches))
    diagnostic_start = time.time()

    for batch_idx, (images, _targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        full_x = sequence_from_images(full_model, images, transform)
        variant_x = sequence_from_images(variant_model, images, transform)

        for block_idx, (full_block, variant_block) in enumerate(zip(full_model.blocks, variant_model.blocks)):
            full_norm = full_block.norm1(full_x)
            variant_norm = variant_block.norm1(variant_x)
            if isinstance(variant_block.attn, _CLSBaseTopKMixer):
                mixer = variant_block.attn
                scores = mixer.route_scores(variant_norm[:, 0], variant_norm[:, 1:])
                hard_weights = mixer.route_weights(scores, hard=True)
                soft_weights = F.softmax(scores / max(mixer.route_temperature, 1e-6), dim=-1)
                mixer_indices = mixer.topk_indices(scores)
                full_scores = attention_patch_scores(full_block.attn, full_norm)
                full_indices = full_scores.topk(mixer_indices.shape[-1], dim=-1).indices
                overlap = overlap_ratio(mixer_indices, full_indices)
                hard_entropy = entropy_norm(hard_weights)
                soft_entropy = entropy_norm(soft_weights)
                selected = hard_weights > 0
                nonzero = selected.sum(dim=-1).float()
                topk_mass = hard_weights.gather(-1, mixer_indices).sum(dim=-1)

                for head_idx in range(mixer.num_heads):
                    key = (block_idx, head_idx)
                    bucket = route_totals.setdefault(
                        key,
                        {
                            "block": block_idx,
                            "head": head_idx,
                            "router": "static" if isinstance(mixer, CLSStaticTopKMixer) else "dynamic",
                            "aggregation": mixer.aggregation,
                            "topk": int(mixer_indices.shape[-1]),
                            "selected_hist": torch.zeros(mixer.num_patches, dtype=torch.float64),
                            "overlap": 0.0,
                            "hard_entropy": 0.0,
                            "soft_entropy": 0.0,
                            "nonzero": 0.0,
                            "topk_mass": 0.0,
                            "max_weight": 0.0,
                        },
                    )
                    bucket["selected_hist"] += selected[:, head_idx].detach().cpu().double().sum(dim=0)
                    bucket["overlap"] += float(overlap[:, head_idx].sum().item())
                    bucket["hard_entropy"] += float(hard_entropy[:, head_idx].sum().item())
                    bucket["soft_entropy"] += float(soft_entropy[:, head_idx].sum().item())
                    bucket["nonzero"] += float(nonzero[:, head_idx].sum().item())
                    bucket["topk_mass"] += float(topk_mass[:, head_idx].sum().item())
                    bucket["max_weight"] += float(hard_weights[:, head_idx].max(dim=-1).values.sum().item())
                    route_counts[key] = route_counts.get(key, 0) + int(scores.shape[0])

            full_x = full_x + full_block.drop_path(full_block.attn(full_norm))
            full_x = full_x + full_block.drop_path(full_block.ffn(full_block.norm2(full_x)))
            variant_x = variant_x + variant_block.drop_path(variant_block.attn(variant_norm))
            variant_x = variant_x + variant_block.drop_path(variant_block.ffn(variant_block.norm2(variant_x)))

        full_cls = full_model.norm(full_x)[:, 0]
        variant_cls = variant_model.norm(variant_x)[:, 0]
        cls_cos = F.cosine_similarity(full_cls, variant_cls, dim=-1)
        cls_cos_total += float(cls_cos.sum().item())
        cls_cos_count += int(cls_cos.numel())

    route_rows: list[dict[str, object]] = []
    overlap_values: list[float] = []
    hard_entropy_values: list[float] = []
    soft_entropy_values: list[float] = []
    for key, bucket in sorted(route_totals.items()):
        examples = max(route_counts[key], 1)
        hist = bucket["selected_hist"]
        row = {
            "block": bucket["block"],
            "head": bucket["head"],
            "router": bucket["router"],
            "aggregation": bucket["aggregation"],
            "topk": bucket["topk"],
            "examples": examples,
            "mean_overlap_with_full_attn_topk": float(bucket["overlap"]) / examples,
            "mean_hard_routing_entropy_norm": float(bucket["hard_entropy"]) / examples,
            "mean_score_softmax_entropy_norm": float(bucket["soft_entropy"]) / examples,
            "mean_nonzero": float(bucket["nonzero"]) / examples,
            "mean_topk_mass": float(bucket["topk_mass"]) / examples,
            "mean_max_weight": float(bucket["max_weight"]) / examples,
            "unique_selected_patch_ratio": float((hist > 0).double().mean().item()),
            "selected_token_histogram": json.dumps([int(v) for v in hist.tolist()]),
        }
        route_rows.append(row)
        overlap_values.append(float(row["mean_overlap_with_full_attn_topk"]))
        hard_entropy_values.append(float(row["mean_hard_routing_entropy_norm"]))
        soft_entropy_values.append(float(row["mean_score_softmax_entropy_norm"]))

    diag = {
        "cls_feature_cosine_vs_full_attention": cls_cos_total / max(cls_cos_count, 1),
        "mean_overlap_with_full_attn_topk": sum(overlap_values) / max(len(overlap_values), 1),
        "mean_hard_routing_entropy_norm": sum(hard_entropy_values) / max(len(hard_entropy_values), 1),
        "mean_score_softmax_entropy_norm": sum(soft_entropy_values) / max(len(soft_entropy_values), 1),
        "diagnostic_time_sec": time.time() - diagnostic_start,
    }
    return diag, route_rows


@torch.no_grad()
def collect_cls_cosine(
    full_model: nn.Module,
    variant_model: nn.Module,
    loader,
    transform,
    max_batches: int,
) -> float:
    prepare_forward(full_model, "relaxed_eval")
    prepare_forward(variant_model, "relaxed_eval")
    set_mixers_hard(variant_model, False)
    total = 0.0
    count = 0
    for batch_idx, (images, _targets) in enumerate(loader):
        if batch_idx >= max(1, int(max_batches)):
            break
        full_x = sequence_from_images(full_model, images, transform)
        variant_x = sequence_from_images(variant_model, images, transform)
        for full_block, variant_block in zip(full_model.blocks, variant_model.blocks):
            full_x = full_block(full_x)
            variant_x = variant_block(variant_x)
        full_cls = full_model.norm(full_x)[:, 0]
        variant_cls = variant_model.norm(variant_x)[:, 0]
        cos = F.cosine_similarity(full_cls, variant_cls, dim=-1)
        total += float(cos.sum().item())
        count += int(cos.numel())
    return total / max(count, 1)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_and_eval_model(
    cfg: SimpleNamespace,
    train_args: SimpleNamespace,
    train_loader,
    eval_loader,
    transform,
    variant: str,
    block_label: str,
    blocks: set[int],
    topk: int,
    route_temperature: float,
) -> tuple[nn.Module, nn.Module, dict[str, object]]:
    set_deterministic(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = build_model(train_args).to(DEVICE)
    mixer_params = 0
    replaced = 0
    if variant != "full_attention":
        replacements = replace_attention_blocks(model, variant, blocks, topk, route_temperature)
        mixer_params = sum(param.numel() for _idx, mixer in replacements for param in mixer.parameters())
        replaced = len(replacements)

    start = time.time()
    set_mixers_hard(model, False)
    train_time = train_teacher(model, train_loader, transform, cfg, train_args)
    maybe_apply_head_mask(model, train_loader, transform, cfg, train_args)
    train_elapsed = time.time() - start

    eval_max_batches = None if int(cfg.eval_max_batches) < 0 else int(cfg.eval_max_batches)
    set_mixers_hard(model, False)
    eval_start = time.time()
    soft_acc, soft_loss = evaluate(model, eval_loader, transform, "relaxed_eval", eval_max_batches)

    hard_model = copy.deepcopy(model).to(DEVICE)
    harden_model(hard_model, temperature=cfg.temp_end)
    set_mixers_hard(hard_model, True)
    hard_acc, hard_loss = evaluate(hard_model, eval_loader, transform, "hard_forward", eval_max_batches)
    eval_time = time.time() - eval_start

    peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    row = {
        "variant": variant,
        "block_setting": block_label if variant != "full_attention" else "full_attention",
        "replace_blocks": ",".join(str(block) for block in sorted(blocks)) if variant != "full_attention" else "",
        "replaced_block_count": replaced,
        "topk": topk if variant != "full_attention" else -1,
        "aggregation": "" if variant == "full_attention" else variant_spec(variant)[1],
        "router": "full_attention"
        if variant == "full_attention"
        else ("static" if variant.startswith("static") else "dynamic"),
        "candidate_tokens": "full_sequence" if variant == "full_attention" else "patch_only",
        "patch_token_update": "full_attention" if variant == "full_attention" else "unchanged_zero_residual",
        "soft_route_topk_active": variant != "full_attention",
        "seed": int(cfg.seed),
        "teacher_iters": int(cfg.teacher_iters),
        "eval_split": cfg.eval_split,
        "eval_max_batches": int(cfg.eval_max_batches),
        "soft_acc": soft_acc,
        "soft_loss": soft_loss,
        "hard_acc": hard_acc,
        "hard_loss": hard_loss,
        "soft_to_hard_drop": soft_acc - hard_acc,
        "hard_discretization_worsens": bool((soft_acc - hard_acc) > 0.0),
        "train_time_sec": train_time,
        "train_elapsed_sec": train_elapsed,
        "eval_time_sec": eval_time,
        "peak_memory_mb": peak_memory_mb,
        "trainable_params": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "mixer_params": mixer_params,
    }
    return model, hard_model, row


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    set_deterministic(int(cfg.seed))

    topk_values = parse_int_list(args.topk_list)
    mixer_variants = [item.strip() for item in args.mixer_variants.split(",") if item.strip()]
    if not mixer_variants:
        raise ValueError("No mixer variants requested")
    for name in mixer_variants:
        variant_spec(name)

    train_args = make_train_args(cfg)
    train_loader, valid_loader, test_loader, _train_eval_loader, _num_batches, transform = load_dataset(train_args)
    eval_loader = test_loader if cfg.eval_split == "test" else (valid_loader if valid_loader is not None else test_loader)

    block_settings = parse_block_settings(args.block_settings, int(cfg.depth), args.include_b1_b2_b3)
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_pid{os.getpid()}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(cfg),
                "config_json": args.config_json,
                "topk_list": topk_values,
                "block_settings": [{"label": label, "blocks": sorted(blocks)} for label, blocks in block_settings],
                "mixer_variants": mixer_variants,
                "route_temperature": args.route_temperature,
                "route_stats_batches": args.route_stats_batches,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []

    full_model, full_hard_model, baseline_row = train_and_eval_model(
        cfg,
        train_args,
        train_loader,
        eval_loader,
        transform,
        "full_attention",
        "full_attention",
        set(),
        -1,
        args.route_temperature,
    )
    baseline_row.update(
        {
            "soft_drop_vs_full_attention": 0.0,
            "hard_drop_vs_full_attention": 0.0,
            "hard_extra_drop_vs_soft": 0.0,
            "cls_feature_cosine_vs_full_attention": 1.0,
            "cls_feature_cosine_vs_full_attention_soft": 1.0,
            "mean_overlap_with_full_attn_topk": 1.0,
            "mean_hard_routing_entropy_norm": 1.0,
            "mean_score_softmax_entropy_norm": 1.0,
            "diagnostic_time_sec": 0.0,
        }
    )
    rows.append(baseline_row)
    write_csv(out_dir / "results.csv", rows)
    print(json.dumps(baseline_row, sort_keys=True), flush=True)

    baseline_soft_acc = float(baseline_row["soft_acc"])
    baseline_hard_acc = float(baseline_row["hard_acc"])

    for block_label, blocks in block_settings:
        if 0 in blocks:
            raise ValueError("This first-round ablation must not replace b0")
        for topk in topk_values:
            for variant in mixer_variants:
                model, hard_model, row = train_and_eval_model(
                    cfg,
                    train_args,
                    train_loader,
                    eval_loader,
                    transform,
                    variant,
                    block_label,
                    blocks,
                    topk,
                    args.route_temperature,
                )
                diag, current_route_rows = collect_pair_diagnostics(
                    full_hard_model,
                    hard_model,
                    eval_loader,
                    transform,
                    args.route_stats_batches,
                )
                soft_cosine = collect_cls_cosine(
                    full_model,
                    model,
                    eval_loader,
                    transform,
                    args.route_stats_batches,
                )
                row.update(diag)
                row["cls_feature_cosine_vs_full_attention_soft"] = soft_cosine
                row["soft_drop_vs_full_attention"] = baseline_soft_acc - float(row["soft_acc"])
                row["hard_drop_vs_full_attention"] = baseline_hard_acc - float(row["hard_acc"])
                row["hard_extra_drop_vs_soft"] = float(row["hard_drop_vs_full_attention"]) - float(
                    row["soft_drop_vs_full_attention"]
                )
                row["hard_discretization_worsens"] = bool(float(row["hard_extra_drop_vs_soft"]) > 0.0)

                for route_row in current_route_rows:
                    route_row.update(
                        {
                            "variant": variant,
                            "block_setting": block_label,
                            "replace_blocks": ",".join(str(block) for block in sorted(blocks)),
                            "topk_setting": topk,
                            "seed": int(cfg.seed),
                        }
                    )
                route_rows.extend(current_route_rows)
                rows.append(row)
                write_csv(out_dir / "results.csv", rows)
                write_csv(out_dir / "route_diagnostics.csv", route_rows)
                print(json.dumps(row, sort_keys=True), flush=True)

    best_soft = max(rows, key=lambda item: float(item["soft_acc"]))
    best_hard = max(rows, key=lambda item: float(item["hard_acc"]))
    summary = {
        "out_dir": str(out_dir),
        "device": str(DEVICE),
        "result_count": len(rows),
        "route_row_count": len(route_rows),
        "baseline_soft_acc": baseline_soft_acc,
        "baseline_hard_acc": baseline_hard_acc,
        "best_variant_by_soft_acc": best_soft["variant"],
        "best_soft_acc": best_soft["soft_acc"],
        "best_variant_by_hard_acc": best_hard["variant"],
        "best_hard_acc": best_hard["hard_acc"],
        "results_csv": "results.csv",
        "route_diagnostics_csv": "route_diagnostics.csv",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
