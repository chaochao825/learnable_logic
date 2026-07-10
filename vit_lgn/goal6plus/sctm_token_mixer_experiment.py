from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
    apply_signed_sparse_head_mask,
    calibrate_masked_head,
    evaluate,
    evaluate_permanent_hard_with_stats,
    finetune_teacher_fixed_connections,
    make_train_args,
    model_stats,
    prefixed_stats,
    prepare_forward,
    preprocess_batch,
    train_student,
    train_teacher,
    write_metrics,
)
from probe_frozen_features import DEFAULTS
from train_logic_vit_tiny import build_model, harden_model, set_deterministic


EPS = 1e-8


def copy_linear(dst: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    with torch.no_grad():
        dst.weight.copy_(weight)
        if dst.bias is not None:
            if bias is None:
                dst.bias.zero_()
            else:
                dst.bias.copy_(bias)


class SparseCLSPatchTokenMixer(nn.Module):
    """Sparse CLS-to-patch top-k token mixer with optional local patch mixing."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patches: int,
        variant: str,
        topk: int,
        patch_path: str,
        weight_bits: int = 2,
        score_mode: str = "continuous",
        block_id: int = 0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        if variant not in {"mean", "lowbit_weighted", "static_control"}:
            raise ValueError(f"Unknown SCTM variant: {variant}")
        if patch_path not in {"identity", "local3x3"}:
            raise ValueError(f"Unknown patch_path: {patch_path}")
        if score_mode not in {"continuous", "int4", "int3", "binary_xnor"}:
            raise ValueError(f"Unknown SCTM score_mode: {score_mode}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_patches = int(num_patches)
        self.head_dim = self.embed_dim // self.num_heads
        self.variant = variant
        self.topk = min(max(int(topk), 1), self.num_patches)
        self.patch_path = patch_path
        self.weight_bits = int(weight_bits)
        self.score_mode = score_mode
        self.block_id = int(block_id)
        self.grid_size = int(round(math.sqrt(self.num_patches)))
        if self.grid_size * self.grid_size != self.num_patches:
            raise ValueError(f"SCTM local3x3 expects square patch grid, got {self.num_patches}")

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.patch_scale = nn.Parameter(torch.tensor(0.10))
        self.static_weight_logits = nn.Parameter(torch.zeros(self.num_heads, self.topk))
        static_indices = self._make_static_indices()
        self.register_buffer("static_indices", static_indices, persistent=True)
        self.register_buffer("hist_counts", torch.zeros(self.num_heads, self.num_patches), persistent=False)
        self.register_buffer("diag_batches", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("diag_routes", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("diag_entropy_sum", torch.zeros(()), persistent=False)
        self.register_buffer("diag_entropy_count", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("patch_norm_before_sum", torch.zeros(()), persistent=False)
        self.register_buffer("patch_norm_after_sum", torch.zeros(()), persistent=False)
        self.register_buffer("cls_update_norm_sum", torch.zeros(()), persistent=False)
        self.register_buffer("cls_update_norm_count", torch.zeros((), dtype=torch.long), persistent=False)
        self.collect_diagnostics = False
        self.skip_mixer = False

    @classmethod
    def from_attention(
        cls,
        attn: nn.Module,
        num_patches: int,
        variant: str,
        topk: int,
        patch_path: str,
        weight_bits: int,
        score_mode: str,
        block_id: int,
    ) -> "SparseCLSPatchTokenMixer":
        mixer = cls(
            embed_dim=int(attn.embed_dim),
            num_heads=int(attn.num_heads),
            num_patches=num_patches,
            variant=variant,
            topk=topk,
            patch_path=patch_path,
            weight_bits=weight_bits,
            score_mode=score_mode,
            block_id=block_id,
        )
        q_w, k_w, v_w = attn.qkv.weight.detach().chunk(3, dim=0)
        if attn.qkv.bias is None:
            q_b = k_b = v_b = None
        else:
            q_b, k_b, v_b = attn.qkv.bias.detach().chunk(3, dim=0)
        copy_linear(mixer.q_proj, q_w, q_b)
        copy_linear(mixer.k_proj, k_w, k_b)
        copy_linear(mixer.v_proj, v_w, v_b)
        copy_linear(mixer.out_proj, attn.proj.weight.detach(), None if attn.proj.bias is None else attn.proj.bias.detach())
        return mixer

    def _make_static_indices(self) -> torch.Tensor:
        base = torch.linspace(0, self.num_patches - 1, steps=self.topk).round().long()
        rows = []
        for head in range(self.num_heads):
            rows.append((base + head) % self.num_patches)
        return torch.stack(rows, dim=0)

    def reset_diagnostics(self) -> None:
        self.hist_counts.zero_()
        self.diag_batches.zero_()
        self.diag_routes.zero_()
        self.diag_entropy_sum.zero_()
        self.diag_entropy_count.zero_()
        self.patch_norm_before_sum.zero_()
        self.patch_norm_after_sum.zero_()
        self.cls_update_norm_sum.zero_()
        self.cls_update_norm_count.zero_()

    @staticmethod
    def _ste_sign(x: torch.Tensor) -> torch.Tensor:
        signed = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return x + (signed - x).detach()

    @staticmethod
    def _ste_minmax_quantize(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
        levels = max((1 << max(int(bits), 1)) - 1, 1)
        lo = x.amin(dim=dim, keepdim=True)
        hi = x.amax(dim=dim, keepdim=True)
        norm = (x - lo) / (hi - lo).clamp_min(EPS)
        quant = torch.round(norm * levels) / levels
        quantized = lo + quant * (hi - lo)
        return x + (quantized - x).detach()

    def route_scores(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        q = self.q_proj(cls).reshape(batch, self.num_heads, self.head_dim)
        k = self.k_proj(patches).reshape(batch, self.num_patches, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)
        if self.score_mode == "binary_xnor":
            q = self._ste_sign(q)
            k = self._ste_sign(k)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) * (self.head_dim ** -0.5)
        if self.score_mode == "int4":
            return self._ste_minmax_quantize(scores, bits=4, dim=-1)
        if self.score_mode == "int3":
            return self._ste_minmax_quantize(scores, bits=3, dim=-1)
        return scores

    def _topk(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "static_control":
            batch = scores.shape[0]
            indices = self.static_indices.unsqueeze(0).expand(batch, -1, -1)
            selected_scores = scores.gather(-1, indices)
            return indices, selected_scores
        top = scores.topk(self.topk, dim=-1)
        return top.indices, top.values

    def _selected_weights(self, selected_scores: torch.Tensor) -> torch.Tensor:
        if self.variant == "mean":
            return torch.full_like(selected_scores, 1.0 / self.topk)
        if self.variant == "static_control":
            positive = F.softplus(self.static_weight_logits).unsqueeze(0).to(selected_scores.dtype) + EPS
            return positive / positive.sum(dim=-1, keepdim=True).clamp_min(EPS)
        levels = max((1 << max(self.weight_bits, 1)) - 1, 1)
        lo = selected_scores.amin(dim=-1, keepdim=True)
        hi = selected_scores.amax(dim=-1, keepdim=True)
        norm = (selected_scores - lo) / (hi - lo).clamp_min(EPS)
        quant = torch.round(norm * levels) / levels
        quant = norm + (quant - norm).detach()
        weights = quant + (1.0 / (levels + 1.0))
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)

    def _gather_values(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        return values.gather(dim=2, index=gather_idx)

    def _patch_delta(self, patches: torch.Tensor) -> torch.Tensor:
        if self.patch_path == "identity":
            return torch.zeros_like(patches)
        batch = patches.shape[0]
        grid = patches.reshape(batch, self.grid_size, self.grid_size, self.embed_dim).permute(0, 3, 1, 2)
        local = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        local = local.permute(0, 2, 3, 1).reshape(batch, self.num_patches, self.embed_dim)
        scale = torch.tanh(self.patch_scale)
        return scale * (local - patches)

    def _record_diagnostics(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        patches: torch.Tensor,
        patch_delta: torch.Tensor,
        cls_delta: torch.Tensor,
    ) -> None:
        if not self.collect_diagnostics:
            return
        with torch.no_grad():
            self.diag_batches += int(indices.shape[0])
            self.diag_routes += int(indices.numel())
            for head in range(self.num_heads):
                flat = indices[:, head, :].reshape(-1)
                counts = torch.bincount(flat, minlength=self.num_patches).to(self.hist_counts.dtype)
                self.hist_counts[head].add_(counts.to(self.hist_counts.device))
            entropy = -(weights * weights.clamp_min(EPS).log()).sum(dim=-1)
            self.diag_entropy_sum += entropy.sum()
            self.diag_entropy_count += int(entropy.numel())
            self.patch_norm_before_sum += patches.norm(dim=-1).mean(dim=1).sum()
            self.patch_norm_after_sum += (patches + patch_delta).norm(dim=-1).mean(dim=1).sum()
            self.cls_update_norm_sum += cls_delta.norm(dim=-1).sum()
            self.cls_update_norm_count += int(cls_delta.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_mixer:
            return torch.zeros_like(x)
        batch, tokens, channels = x.shape
        if tokens != self.num_patches + 1 or channels != self.embed_dim:
            raise ValueError(f"SCTM expected {(self.num_patches + 1, self.embed_dim)}, got {(tokens, channels)}")
        cls = x[:, 0]
        patches = x[:, 1:]
        scores = self.route_scores(cls, patches)
        indices, selected_scores = self._topk(scores)
        weights = self._selected_weights(selected_scores)
        values = self.v_proj(patches).reshape(batch, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        selected_values = self._gather_values(values, indices)
        mixed = (weights.unsqueeze(-1) * selected_values).sum(dim=2).reshape(batch, channels)
        cls_delta = self.out_proj(mixed)
        patch_delta = self._patch_delta(patches)
        self._record_diagnostics(
            indices.detach(),
            weights.detach(),
            patches.detach(),
            patch_delta.detach(),
            cls_delta.detach(),
        )
        out = x.new_zeros(batch, tokens, channels)
        out[:, 0] = cls_delta
        out[:, 1:] = patch_delta
        return out


def sctm_modules(model: nn.Module) -> list[SparseCLSPatchTokenMixer]:
    return [m for m in model.modules() if isinstance(m, SparseCLSPatchTokenMixer)]


def replace_attention_with_sctm(
    model: nn.Module,
    variant: str,
    topk: int,
    patch_path: str,
    weight_bits: int,
    score_mode: str,
) -> None:
    num_patches = int(model.patch_embed.n_patches)
    for block_id, block in enumerate(model.blocks):
        block.attn = SparseCLSPatchTokenMixer.from_attention(
            block.attn,
            num_patches=num_patches,
            variant=variant,
            topk=topk,
            patch_path=patch_path,
            weight_bits=weight_bits,
            score_mode=score_mode,
            block_id=block_id,
        )


def set_sctm_collect(model: nn.Module, enabled: bool) -> None:
    for module in sctm_modules(model):
        module.collect_diagnostics = bool(enabled)
        if enabled:
            module.reset_diagnostics()


def sctm_stats(model: nn.Module) -> dict[str, float]:
    modules = sctm_modules(model)
    if not modules:
        return {}
    stats: dict[str, float] = {}
    entropy_sum = 0.0
    entropy_count = 0
    unique_ratios = []
    patch_before = 0.0
    patch_after = 0.0
    patch_count = 0
    for module in modules:
        hist = module.hist_counts.detach().float().cpu()
        total = float(hist.sum().item())
        used = float((hist > 0).sum().item())
        denom = max(float(module.num_heads * module.num_patches), 1.0)
        stats[f"sctm_b{module.block_id}_unique_patch_ratio"] = used / denom
        unique_ratios.append(used / denom)
        for head in range(module.num_heads):
            h = hist[head]
            stats[f"sctm_b{module.block_id}_h{head}_hist_max"] = float(h.max().item()) if h.numel() else 0.0
            stats[f"sctm_b{module.block_id}_h{head}_hist_used"] = float((h > 0).sum().item())
        entropy_sum += float(module.diag_entropy_sum.item())
        module_entropy_count = int(module.diag_entropy_count.item())
        entropy_count += module_entropy_count
        stats[f"sctm_b{module.block_id}_route_entropy"] = (
            float(module.diag_entropy_sum.item()) / max(module_entropy_count, 1)
        )
        batches = int(module.diag_batches.item())
        if batches > 0:
            block_patch_before = float(module.patch_norm_before_sum.item()) / batches
            block_patch_after = float(module.patch_norm_after_sum.item()) / batches
            stats[f"sctm_b{module.block_id}_patch_norm_before"] = block_patch_before
            stats[f"sctm_b{module.block_id}_patch_norm_after"] = block_patch_after
            patch_before += float(module.patch_norm_before_sum.item())
            patch_after += float(module.patch_norm_after_sum.item())
            patch_count += batches
        cls_count = int(module.cls_update_norm_count.item())
        stats[f"sctm_b{module.block_id}_cls_update_norm"] = (
            float(module.cls_update_norm_sum.item()) / max(cls_count, 1)
        )
    stats["sctm_route_entropy"] = entropy_sum / max(entropy_count, 1)
    stats["sctm_unique_selected_patch_ratio"] = sum(unique_ratios) / max(len(unique_ratios), 1)
    stats["sctm_patch_norm_before"] = patch_before / max(patch_count, 1)
    stats["sctm_patch_norm_after"] = patch_after / max(patch_count, 1)
    return stats


def _ceil_log2(x: int) -> int:
    return int(math.ceil(math.log2(max(int(x), 2))))


def _score_bits(score_mode: str) -> int:
    if score_mode == "int4":
        return 4
    if score_mode == "int3":
        return 3
    if score_mode == "binary_xnor":
        return 1
    return 16


def sctm_cost_proxy(model: nn.Module) -> dict[str, float]:
    modules = sctm_modules(model)
    if not modules:
        return {}
    total_names = [
        "score_projection_ops",
        "value_projection_ops",
        "out_projection_ops",
        "score_dot_ops",
        "topk_comparator_ops",
        "selected_value_mux_ops",
        "lowbit_weighted_aggregation_ops",
        "local3x3_patch_mixer_ops",
        "estimated_gate_equivalent_cost",
    ]
    totals = {name: 0.0 for name in total_names}
    max_depth = 0
    stats: dict[str, float] = {}
    for module in modules:
        e = int(module.embed_dim)
        h = int(module.num_heads)
        n = int(module.num_patches)
        d = int(module.head_dim)
        k = int(module.topk)
        bits = _score_bits(module.score_mode)
        weight_bits = max(int(module.weight_bits), 1) if module.variant == "lowbit_weighted" else 1

        score_projection_ops = float((1 + n) * e * e)
        value_projection_ops = float(n * e * e)
        out_projection_ops = float(e * e)
        score_dot_ops = float(h * n * d)
        topk_comparator_ops = float(h * n * _ceil_log2(k + 1))
        selected_value_mux_ops = float(h * k * d)
        lowbit_weighted_aggregation_ops = float(h * k * d)
        local3x3_patch_mixer_ops = float(n * e * 9) if module.patch_path == "local3x3" else 0.0

        projection_depth = _ceil_log2(e)
        score_dot_depth = _ceil_log2(d) + (0 if module.score_mode == "binary_xnor" else 1)
        topk_depth = _ceil_log2(n) * _ceil_log2(k + 1)
        aggregation_depth = _ceil_log2(k) + weight_bits
        local_depth = _ceil_log2(9) if module.patch_path == "local3x3" else 0
        estimated_logic_depth = float(
            projection_depth + score_dot_depth + topk_depth + aggregation_depth + local_depth
        )

        gate_equiv = (
            score_projection_ops * 16.0 * bits
            + value_projection_ops * 16.0 * max(weight_bits, 1)
            + out_projection_ops * 16.0 * max(weight_bits, 1)
            + score_dot_ops * float(bits * bits)
            + topk_comparator_ops * float(bits)
            + selected_value_mux_ops
            + lowbit_weighted_aggregation_ops * float(weight_bits)
            + local3x3_patch_mixer_ops
        )

        prefix = f"sctm_b{module.block_id}_cost_"
        block_values = {
            "score_projection_ops": score_projection_ops,
            "value_projection_ops": value_projection_ops,
            "out_projection_ops": out_projection_ops,
            "score_dot_ops": score_dot_ops,
            "topk_comparator_ops": topk_comparator_ops,
            "selected_value_mux_ops": selected_value_mux_ops,
            "lowbit_weighted_aggregation_ops": lowbit_weighted_aggregation_ops,
            "local3x3_patch_mixer_ops": local3x3_patch_mixer_ops,
            "estimated_logic_depth": estimated_logic_depth,
            "estimated_gate_equivalent_cost": gate_equiv,
            "score_bits": float(bits),
            "aggregation_weight_bits": float(weight_bits),
        }
        for name, value in block_values.items():
            stats[prefix + name] = float(value)
        for name in total_names:
            totals[name] += float(block_values[name])
        max_depth = max(max_depth, int(estimated_logic_depth))

    for name, value in totals.items():
        stats[f"sctm_cost_{name}"] = float(value)
    stats["sctm_est_logic_depth"] = float(max_depth)
    stats["sctm_cost_blocks"] = float(len(modules))
    return stats


def write_sctm_histograms(model: nn.Module, path: Path) -> None:
    payload = {}
    for module in sctm_modules(model):
        block_payload = {}
        hist = module.hist_counts.detach().long().cpu()
        for head in range(module.num_heads):
            block_payload[f"head_{head}"] = hist[head].tolist()
        payload[f"block_{module.block_id}"] = {
            "num_patches": module.num_patches,
            "topk": module.topk,
            "counts": block_payload,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def install_zero_hooks(modules: Iterable[nn.Module]):
    handles = []
    for module in modules:
        handles.append(module.register_forward_hook(lambda _m, _inp, out: torch.zeros_like(out)))
    return handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def full_attention_cls_topk(attn: nn.Module, x: torch.Tensor, topk: int) -> torch.Tensor:
    batch, tokens, channels = x.shape
    qkv = attn.qkv(x).reshape(batch, tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    cls_q = q[:, :, 0:1, :]
    patch_k = k[:, :, 1:, :]
    scores = (cls_q * patch_k).sum(dim=-1) * (attn.head_dim ** -0.5)
    return scores.topk(min(topk, scores.shape[-1]), dim=-1).indices


@torch.no_grad()
def baseline_pair_diagnostics(
    model: nn.Module,
    baseline: nn.Module,
    loader,
    transform,
    max_batches: int,
    topk: int,
) -> dict[str, float]:
    if not sctm_modules(model):
        return {}
    prepare_forward(model, "relaxed_eval")
    prepare_forward(baseline, "relaxed_eval")
    overlap_sum = 0.0
    overlap_count = 0
    cos_sum = 0.0
    cos_count = 0
    for batch_idx, (images, _targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = preprocess_batch(images, transform)

        xb = baseline.patch_embed(images)
        xv = model.patch_embed(images)
        batch = xb.shape[0]
        cls_b = baseline.cls_token.expand(batch, -1, -1)
        cls_v = model.cls_token.expand(batch, -1, -1)
        xb = torch.cat([cls_b, xb], dim=1) + baseline.pos_embed
        xv = torch.cat([cls_v, xv], dim=1) + model.pos_embed
        xb = baseline.pos_drop(xb)
        xv = model.pos_drop(xv)

        for block_b, block_v in zip(baseline.blocks, model.blocks):
            xb_norm = block_b.norm1(xb)
            xv_norm = block_v.norm1(xv)
            base_topk = full_attention_cls_topk(block_b.attn, xb_norm, topk)
            scores = block_v.attn.route_scores(xv_norm[:, 0], xv_norm[:, 1:])
            sctm_topk, _ = block_v.attn._topk(scores)
            for head in range(base_topk.shape[1]):
                a = base_topk[:, head, :]
                b = sctm_topk[:, head, :]
                matches = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).float().mean()
                overlap_sum += float(matches.item())
                overlap_count += 1
            xb = block_b(xb)
            xv = block_v(xv)
            cos = F.cosine_similarity(xb[:, 0], xv[:, 0], dim=-1).mean()
            cos_sum += float(cos.item())
            cos_count += 1
    return {
        "baseline_cls_cosine": cos_sum / max(cos_count, 1),
        "baseline_topk_overlap": overlap_sum / max(overlap_count, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse CLS Patch Token Mixer experiment for ViT-LGN.")
    for key, value in DEFAULTS.items():
        if key == "out_dir":
            continue
        arg = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, int):
            parser.add_argument(arg, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(arg, type=float, default=value)
        else:
            parser.add_argument(arg, type=str, default=value)
    parser.set_defaults(
        augment=False,
        depth=6,
        embed_dim=192,
        num_heads=6,
        patch_size=4,
        logic_ffn_layers=2,
        logic_n_thresholds=7,
        logic_connectivity="fixed",
        logic_connections="random",
        student_iters=0,
        distill_alpha=0.0,
        eval_split="test",
    )
    parser.add_argument("--logic-mlp-ratio", type=float, default=1.0)
    parser.add_argument("--mixer", choices=["full_attention", "sctm_mean", "sctm_lowbit_weighted", "sctm_static_control"], default="full_attention")
    parser.add_argument("--sctm-topk", type=int, default=8)
    parser.add_argument("--sctm-weight-bits", type=int, default=2)
    parser.add_argument("--sctm-patch-path", choices=["identity", "local3x3"], default="identity")
    parser.add_argument("--sctm-score-mode", choices=["continuous", "int4", "int3", "binary_xnor"], default="continuous")
    parser.add_argument("--baseline-checkpoint", type=str, default="")
    parser.add_argument("--diagnostic-batches", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="runs/sctm_token_mixer")
    return parser.parse_args()


def build_experiment_model(args: argparse.Namespace, train_args: SimpleNamespace) -> nn.Module:
    model = build_model(train_args).to(DEVICE)
    if args.mixer != "full_attention":
        variant = args.mixer.replace("sctm_", "")
        replace_attention_with_sctm(
            model,
            variant=variant,
            topk=args.sctm_topk,
            patch_path=args.sctm_patch_path,
            weight_bits=args.sctm_weight_bits,
            score_mode=args.sctm_score_mode,
        )
    return model.to(DEVICE)


def load_checkpoint_model(path: str, fallback_args: argparse.Namespace) -> nn.Module:
    payload = torch.load(path, map_location=DEVICE)
    cfg = dict(vars(fallback_args))
    payload_args = payload.get("args", {})
    cfg.update(payload_args)
    if "mixer" not in payload_args:
        cfg["mixer"] = "full_attention"
    args = SimpleNamespace(**cfg)
    train_args = make_train_args(args)
    model = build_experiment_model(args, train_args)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(DEVICE)


def main() -> None:
    args = parse_args()
    cfg = vars(args).copy()
    args = argparse.Namespace(**cfg)
    set_deterministic(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out_root = Path(args.out_dir)
    out_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_args = make_train_args(args)
    train_loader, valid_loader, test_loader, _train_eval_loader, _num_batches, transform = load_dataset(train_args)
    eval_loader = test_loader if args.eval_split == "test" else (valid_loader if valid_loader is not None else test_loader)
    eval_max_batches = None if args.eval_max_batches < 0 else args.eval_max_batches

    teacher = build_experiment_model(args, train_args)
    token_count = int(getattr(teacher.patch_embed, "n_patches", 0)) + 1
    param_count = sum(p.numel() for p in teacher.parameters())
    teacher_time = train_teacher(teacher, train_loader, transform, args, train_args)
    post_disc_ft_time, post_disc_ft_trainable = finetune_teacher_fixed_connections(
        teacher, train_loader, transform, args, train_args
    )

    student = copy.deepcopy(teacher).to(DEVICE)
    student_time, trainable_params = train_student(teacher, student, train_loader, transform, args, train_args)
    teacher_head_calib_time = 0.0
    teacher_head_calib_trainable = 0
    student_head_calib_time = 0.0
    student_head_calib_trainable = 0
    if args.head_type in {"signed_sparse_linear", "signed_counter_linear", "scaled_signed_counter_linear"}:
        apply_signed_sparse_head_mask(teacher)
        apply_signed_sparse_head_mask(student)
        if args.post_mask_head_calibrate_iters > 0:
            if args.post_mask_head_calibrate_target in {"teacher", "both"}:
                teacher_head_calib_time, teacher_head_calib_trainable = calibrate_masked_head(
                    teacher, train_loader, transform, args, train_args
                )
            if args.post_mask_head_calibrate_target in {"student", "both"}:
                student_head_calib_time, student_head_calib_trainable = calibrate_masked_head(
                    student, train_loader, transform, args, train_args
                )

    peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
    checkpoint_path = out_dir / "final_model.pt"
    torch.save({"args": vars(args), "model_state": teacher.state_dict()}, checkpoint_path)

    rows: list[dict[str, object]] = []
    eval_specs = [
        ("teacher_relaxed_trainpath", teacher, "relaxed_trainpath", False),
        ("teacher_relaxed_eval", teacher, "relaxed_eval", False),
        ("teacher_hard_forward", teacher, "hard_forward", False),
        ("teacher_permanent_hard", teacher, "hard_forward", True),
        ("student_relaxed_trainpath", student, "relaxed_trainpath", False),
        ("student_hard_forward", student, "hard_forward", False),
        ("student_permanent_hard", student, "hard_forward", True),
    ]
    teacher_stats = model_stats(teacher, args.signed_head_topk)
    student_stats = model_stats(student, args.signed_head_topk)
    sctm_diag_stats: dict[str, float] = {}
    for method, model, mode, permanent in eval_specs:
        if permanent:
            acc, loss, current_stats = evaluate_permanent_hard_with_stats(
                model, eval_loader, transform, args.temp_end, eval_max_batches, args.signed_head_topk
            )
        else:
            current_stats = model_stats(model, args.signed_head_topk)
            collect = method == "teacher_relaxed_eval" and args.mixer != "full_attention"
            if collect:
                set_sctm_collect(model, True)
            acc, loss = evaluate(model, eval_loader, transform, mode, eval_max_batches)
            if collect:
                set_sctm_collect(model, False)
                sctm_diag_stats = sctm_stats(model)
        row = {
            "method": method,
            "mixer": args.mixer,
            "sctm_topk": args.sctm_topk,
            "sctm_patch_path": args.sctm_patch_path,
            "acc": acc,
            "loss": loss,
            "teacher_train_time": teacher_time,
            "student_train_time": student_time,
            "post_disc_ft_time": post_disc_ft_time,
            "post_disc_ft_trainable_params": post_disc_ft_trainable,
            "peak_memory_mb": peak_memory_mb,
            "teacher_iters": args.teacher_iters,
            "student_iters": args.student_iters,
            "student_trainable_params": trainable_params,
            **current_stats,
        }
        rows.append(row)

    skip_stats: dict[str, float] = {}
    if args.mixer != "full_attention":
        handles = install_zero_hooks([block.attn for block in teacher.blocks])
        skip_acc, _skip_loss = evaluate(teacher, eval_loader, transform, "relaxed_eval", eval_max_batches)
        remove_hooks(handles)
        skip_stats["skip_mixer_acc"] = skip_acc
        for block_id, block in enumerate(teacher.blocks):
            handles = install_zero_hooks([block.attn])
            block_skip_acc, _block_skip_loss = evaluate(
                teacher, eval_loader, transform, "relaxed_eval", eval_max_batches
            )
            remove_hooks(handles)
            skip_stats[f"skip_sctm_b{block_id}_acc"] = block_skip_acc
        handles = install_zero_hooks([block.ffn for block in teacher.blocks])
        skip_ffn_acc, _skip_ffn_loss = evaluate(teacher, eval_loader, transform, "relaxed_eval", eval_max_batches)
        remove_hooks(handles)
        skip_stats["skip_logicffn_acc"] = skip_ffn_acc

    baseline_diag: dict[str, float] = {}
    if args.baseline_checkpoint and args.mixer != "full_attention":
        baseline = load_checkpoint_model(args.baseline_checkpoint, args)
        baseline_diag = baseline_pair_diagnostics(
            teacher,
            baseline,
            eval_loader,
            transform,
            max_batches=max(int(args.diagnostic_batches), 1),
            topk=args.sctm_topk,
        )

    by_method = {str(r["method"]): r for r in rows}
    teacher_soft_trainpath = float(by_method["teacher_relaxed_trainpath"]["acc"])
    teacher_soft = float(by_method["teacher_relaxed_eval"]["acc"])
    teacher_hard = float(by_method["teacher_permanent_hard"]["acc"])
    student_hard = float(by_method["student_permanent_hard"]["acc"])
    sctm_histogram_path = ""
    if args.mixer != "full_attention":
        sctm_histogram_file = out_dir / "sctm_selected_patch_histograms.json"
        write_sctm_histograms(teacher, sctm_histogram_file)
        sctm_histogram_path = str(sctm_histogram_file)
    summary = {
        "out_dir": str(out_dir),
        "checkpoint_path": str(checkpoint_path),
        "mixer": args.mixer,
        "sctm_topk": args.sctm_topk,
        "sctm_patch_path": args.sctm_patch_path,
        "sctm_weight_bits": args.sctm_weight_bits,
        "sctm_score_mode": args.sctm_score_mode,
        "arch_depth": args.depth,
        "arch_embed_dim": args.embed_dim,
        "arch_num_heads": args.num_heads,
        "arch_patch_size": args.patch_size,
        "arch_tokens": token_count,
        "logic_mlp_ratio": args.logic_mlp_ratio,
        "logic_ffn_layers": args.logic_ffn_layers,
        "logic_n_thresholds": args.logic_n_thresholds,
        "logic_connectivity": args.logic_connectivity,
        "logic_connections": args.logic_connections,
        "augment": args.augment,
        "head_type": args.head_type,
        "signed_head_topk": args.signed_head_topk,
        "param_count": param_count,
        "eval_split": args.eval_split,
        "eval_max_batches": args.eval_max_batches,
        "teacher_soft_acc": teacher_soft,
        "teacher_relaxed_trainpath_acc": teacher_soft_trainpath,
        "teacher_hard_acc": teacher_hard,
        "teacher_acc_gap": abs(teacher_soft - teacher_hard),
        "teacher_soft_to_discrete_drop": teacher_soft - teacher_hard,
        "student_hard_acc": student_hard,
        "student_delta_vs_teacher_hard": student_hard - teacher_hard,
        "student_gap_vs_teacher_soft": abs(teacher_soft - student_hard),
        "teacher_train_time": teacher_time,
        "student_train_time": student_time,
        "post_disc_ft_time": post_disc_ft_time,
        "post_disc_ft_trainable_params": post_disc_ft_trainable,
        "teacher_head_calib_time": teacher_head_calib_time,
        "teacher_head_calib_trainable_params": teacher_head_calib_trainable,
        "student_head_calib_time": student_head_calib_time,
        "student_head_calib_trainable_params": student_head_calib_trainable,
        "peak_memory_mb": peak_memory_mb,
        "student_trainable_params": trainable_params,
        "sctm_selected_patch_histogram_path": sctm_histogram_path,
        **prefixed_stats("teacher_", teacher_stats),
        **prefixed_stats("student_", student_stats),
        **sctm_cost_proxy(teacher),
        **sctm_diag_stats,
        **skip_stats,
        **baseline_diag,
    }
    write_metrics(out_dir, rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
