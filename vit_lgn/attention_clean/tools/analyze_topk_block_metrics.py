from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_logic_vit_tiny import (  # noqa: E402
    build_model,
    initialize_lazy_modules,
    parse_args as parse_train_args,
    set_attention_soft_train_temperatures,
    set_deterministic,
)
from data_pipeline import load_dataset  # noqa: E402


@dataclass
class BlockMetricSummary:
    block_index: int
    rows_seen: int
    soft_mask_total_sum_over_k_mean: float
    selected_soft_mask_fraction_mean: float
    kth_gap_mean: float | None
    kth_gap_valid_rows: int
    grad_compare_rows: int
    grad_cosine_mean: float | None
    grad_cosine_std: float | None
    grad_sign_flip_ratio_mean: float | None
    grad_norm_ratio_mean: float | None
    grad_raw_abs_mean: float | None
    grad_norm_abs_mean: float | None
    actual_grad_rows: int
    actual_grad_abs_mean: float | None
    actual_vs_raw_cosine_mean: float | None
    actual_vs_raw_cosine_std: float | None
    actual_vs_raw_sign_flip_ratio_mean: float | None
    actual_vs_norm_cosine_mean: float | None
    actual_vs_norm_cosine_std: float | None
    actual_vs_norm_sign_flip_ratio_mean: float | None


class BlockMetricCollector:
    def __init__(self, block_index: int) -> None:
        self.block_index = int(block_index)
        self.soft_mask_total_sum_over_k_sum = 0.0
        self.selected_soft_fraction_sum = 0.0
        self.rows_seen = 0
        self.kth_gap_sum = 0.0
        self.kth_gap_rows = 0
        self.grad_compare_rows = 0
        self.grad_cosine_sum = 0.0
        self.grad_cosine_sq_sum = 0.0
        self.grad_sign_flip_sum = 0.0
        self.grad_norm_ratio_sum = 0.0
        self.grad_raw_abs_sum = 0.0
        self.grad_norm_abs_sum = 0.0
        self.grad_elem_count = 0
        self.actual_grad_rows = 0
        self.actual_grad_abs_sum = 0.0
        self.actual_grad_elem_count = 0
        self.actual_vs_raw_cosine_sum = 0.0
        self.actual_vs_raw_cosine_sq_sum = 0.0
        self.actual_vs_raw_sign_flip_sum = 0.0
        self.actual_vs_norm_cosine_sum = 0.0
        self.actual_vs_norm_cosine_sq_sum = 0.0
        self.actual_vs_norm_sign_flip_sum = 0.0
        self._pending_grad_payload: dict[str, torch.Tensor | int | float] | None = None

    def observe(
        self,
        module,
        q: torch.Tensor,
        k: torch.Tensor,
        topk: int,
        proxy_scores: torch.Tensor | None,
        output: dict[str, torch.Tensor],
        collect_grad_compare: bool,
    ) -> None:
        raw_scores = output["xnor_popcount"].detach().to(torch.float32)
        topk_indices = output["topk_indices"].detach().to(torch.long)
        row_count = raw_scores.shape[-1]
        limit = min(int(topk), row_count)
        if limit <= 0:
            return

        hard_mask = module._build_selector_mask(topk_indices, row_count).to(torch.float32)
        if proxy_scores is None:
            proxy_scores = module._xnor_similarity_proxy(q, k)
        proxy_scores = proxy_scores.detach().to(torch.float32)

        soft_mask = self._compute_soft_mask(
            module=module,
            hard_mask=hard_mask,
            proxy_scores=proxy_scores,
            limit=limit,
            topk_indices=topk_indices,
        )

        selected_soft_sum = (soft_mask * hard_mask).sum(dim=-1)
        total_soft_sum = soft_mask.sum(dim=-1)
        selected_soft_fraction = selected_soft_sum / total_soft_sum.clamp_min(1e-12)
        total_soft_sum_over_k = total_soft_sum / float(limit)

        self.soft_mask_total_sum_over_k_sum += float(total_soft_sum_over_k.sum().item())
        self.selected_soft_fraction_sum += float(selected_soft_fraction.sum().item())
        self.rows_seen += int(selected_soft_fraction.numel())

        if limit < row_count:
            selected_scores = torch.gather(raw_scores, -1, topk_indices)
            kth_selected = selected_scores.min(dim=-1).values

            unselected_scores = raw_scores.masked_fill(hard_mask.to(torch.bool), float("-inf"))
            next_best_unselected = unselected_scores.max(dim=-1).values

            kth_gap = kth_selected - next_best_unselected
            self.kth_gap_sum += float(kth_gap.sum().item())
            self.kth_gap_rows += int(kth_gap.numel())

        if collect_grad_compare and output["selector_mask"] is not None and output["selector_mask"].requires_grad:
            output["selector_mask"].retain_grad()
            actual_proxy_scores = None
            if proxy_scores is not None and proxy_scores.requires_grad:
                proxy_scores.retain_grad()
                actual_proxy_scores = proxy_scores
            self._pending_grad_payload = {
                "selector_mask": output["selector_mask"],
                "proxy_scores": proxy_scores.detach().to(torch.float32),
                "limit": limit,
                "temperature": float(module.boundary_surrogate_temperature),
                "actual_proxy_scores": actual_proxy_scores,
            }

    def finalize_backward(self) -> None:
        if self._pending_grad_payload is None:
            return

        selector_mask = self._pending_grad_payload["selector_mask"]
        proxy_scores = self._pending_grad_payload["proxy_scores"]
        limit = int(self._pending_grad_payload["limit"])
        temperature = float(self._pending_grad_payload["temperature"])
        actual_proxy_scores = self._pending_grad_payload["actual_proxy_scores"]
        self._pending_grad_payload = None

        if selector_mask.grad is None:
            return

        metrics = self._compare_norm_raw_score_grad(
            proxy_scores=proxy_scores,
            mask_grad=selector_mask.grad.detach().to(torch.float32),
            limit=limit,
            temperature=temperature,
        )
        if metrics is None:
            return

        cosine = metrics["cosine"]
        sign_flip = metrics["sign_flip"]
        norm_ratio = metrics["norm_ratio"]
        grad_raw_abs = metrics["grad_raw_abs"]
        grad_norm_abs = metrics["grad_norm_abs"]

        self.grad_compare_rows += int(cosine.numel())
        self.grad_cosine_sum += float(cosine.sum().item())
        self.grad_cosine_sq_sum += float((cosine * cosine).sum().item())
        self.grad_sign_flip_sum += float(sign_flip.sum().item())
        self.grad_norm_ratio_sum += float(norm_ratio.sum().item())
        self.grad_raw_abs_sum += float(grad_raw_abs.sum().item())
        self.grad_norm_abs_sum += float(grad_norm_abs.sum().item())
        self.grad_elem_count += int(grad_raw_abs.numel())

        if actual_proxy_scores is not None and actual_proxy_scores.grad is not None:
            actual_grad = actual_proxy_scores.grad.detach().to(torch.float32)
            actual_metrics = self._compare_actual_with_refs(
                actual_grad=actual_grad,
                grad_raw=metrics["grad_raw"],
                grad_norm=metrics["grad_norm"],
            )
            self.actual_grad_rows += int(actual_metrics["actual_vs_raw_cosine"].numel())
            self.actual_grad_abs_sum += float(actual_grad.abs().sum().item())
            self.actual_grad_elem_count += int(actual_grad.numel())
            self.actual_vs_raw_cosine_sum += float(actual_metrics["actual_vs_raw_cosine"].sum().item())
            self.actual_vs_raw_cosine_sq_sum += float((actual_metrics["actual_vs_raw_cosine"] ** 2).sum().item())
            self.actual_vs_raw_sign_flip_sum += float(actual_metrics["actual_vs_raw_sign_flip"].sum().item())
            self.actual_vs_norm_cosine_sum += float(actual_metrics["actual_vs_norm_cosine"].sum().item())
            self.actual_vs_norm_cosine_sq_sum += float((actual_metrics["actual_vs_norm_cosine"] ** 2).sum().item())
            self.actual_vs_norm_sign_flip_sum += float(actual_metrics["actual_vs_norm_sign_flip"].sum().item())

    def summary(self) -> BlockMetricSummary:
        total_sum_over_k_mean = 0.0
        selected_fraction_mean = 0.0
        if self.rows_seen > 0:
            total_sum_over_k_mean = self.soft_mask_total_sum_over_k_sum / float(self.rows_seen)
            selected_fraction_mean = self.selected_soft_fraction_sum / float(self.rows_seen)

        kth_gap_mean = None
        if self.kth_gap_rows > 0:
            kth_gap_mean = self.kth_gap_sum / float(self.kth_gap_rows)

        grad_cosine_mean = None
        grad_cosine_std = None
        grad_sign_flip_ratio_mean = None
        grad_norm_ratio_mean = None
        grad_raw_abs_mean = None
        grad_norm_abs_mean = None
        actual_grad_abs_mean = None
        actual_vs_raw_cosine_mean = None
        actual_vs_raw_cosine_std = None
        actual_vs_raw_sign_flip_ratio_mean = None
        actual_vs_norm_cosine_mean = None
        actual_vs_norm_cosine_std = None
        actual_vs_norm_sign_flip_ratio_mean = None
        if self.grad_compare_rows > 0:
            grad_cosine_mean = self.grad_cosine_sum / float(self.grad_compare_rows)
            cosine_sq_mean = self.grad_cosine_sq_sum / float(self.grad_compare_rows)
            grad_cosine_std = max(cosine_sq_mean - grad_cosine_mean * grad_cosine_mean, 0.0) ** 0.5
            grad_sign_flip_ratio_mean = self.grad_sign_flip_sum / float(self.grad_compare_rows)
            grad_norm_ratio_mean = self.grad_norm_ratio_sum / float(self.grad_compare_rows)
        if self.grad_elem_count > 0:
            grad_raw_abs_mean = self.grad_raw_abs_sum / float(self.grad_elem_count)
            grad_norm_abs_mean = self.grad_norm_abs_sum / float(self.grad_elem_count)
        if self.actual_grad_elem_count > 0:
            actual_grad_abs_mean = self.actual_grad_abs_sum / float(self.actual_grad_elem_count)
        if self.actual_grad_rows > 0:
            actual_vs_raw_cosine_mean = self.actual_vs_raw_cosine_sum / float(self.actual_grad_rows)
            actual_vs_raw_cosine_sq_mean = self.actual_vs_raw_cosine_sq_sum / float(self.actual_grad_rows)
            actual_vs_raw_cosine_std = max(actual_vs_raw_cosine_sq_mean - actual_vs_raw_cosine_mean * actual_vs_raw_cosine_mean, 0.0) ** 0.5
            actual_vs_raw_sign_flip_ratio_mean = self.actual_vs_raw_sign_flip_sum / float(self.actual_grad_rows)
            actual_vs_norm_cosine_mean = self.actual_vs_norm_cosine_sum / float(self.actual_grad_rows)
            actual_vs_norm_cosine_sq_mean = self.actual_vs_norm_cosine_sq_sum / float(self.actual_grad_rows)
            actual_vs_norm_cosine_std = max(actual_vs_norm_cosine_sq_mean - actual_vs_norm_cosine_mean * actual_vs_norm_cosine_mean, 0.0) ** 0.5
            actual_vs_norm_sign_flip_ratio_mean = self.actual_vs_norm_sign_flip_sum / float(self.actual_grad_rows)

        return BlockMetricSummary(
            block_index=self.block_index,
            rows_seen=self.rows_seen,
            soft_mask_total_sum_over_k_mean=total_sum_over_k_mean,
            selected_soft_mask_fraction_mean=selected_fraction_mean,
            kth_gap_mean=kth_gap_mean,
            kth_gap_valid_rows=self.kth_gap_rows,
            grad_compare_rows=self.grad_compare_rows,
            grad_cosine_mean=grad_cosine_mean,
            grad_cosine_std=grad_cosine_std,
            grad_sign_flip_ratio_mean=grad_sign_flip_ratio_mean,
            grad_norm_ratio_mean=grad_norm_ratio_mean,
            grad_raw_abs_mean=grad_raw_abs_mean,
            grad_norm_abs_mean=grad_norm_abs_mean,
            actual_grad_rows=self.actual_grad_rows,
            actual_grad_abs_mean=actual_grad_abs_mean,
            actual_vs_raw_cosine_mean=actual_vs_raw_cosine_mean,
            actual_vs_raw_cosine_std=actual_vs_raw_cosine_std,
            actual_vs_raw_sign_flip_ratio_mean=actual_vs_raw_sign_flip_ratio_mean,
            actual_vs_norm_cosine_mean=actual_vs_norm_cosine_mean,
            actual_vs_norm_cosine_std=actual_vs_norm_cosine_std,
            actual_vs_norm_sign_flip_ratio_mean=actual_vs_norm_sign_flip_ratio_mean,
        )

    @staticmethod
    def _compute_soft_mask(
        module,
        hard_mask: torch.Tensor,
        proxy_scores: torch.Tensor,
        limit: int,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        mode = module.topk_surrogate_mode
        temperature = module.boundary_surrogate_temperature

        if mode == "random-kth":
            _, soft_mask, _ = module.hard_topk_mask_random_kth_boundary_surrogate(
                hard_mask,
                proxy_scores,
                limit,
                temperature=temperature,
                detach_kth_value=module.topk_kth_detach_value,
                candidate_indices=topk_indices if module.topk_forward_mode != "topk" else None,
                use_midpoint_theta=module.topk_kth_use_midpoint_theta,
                normalize_soft_mask=module.topk_kth_normalize_soft_mask,
            )
            return soft_mask.detach().to(torch.float32)

        if mode == "soft-rank":
            _, soft_mask, _ = module.hard_topk_mask_soft_rank_surrogate(
                hard_mask,
                proxy_scores,
                limit,
                temperature=temperature,
            )
            return soft_mask.detach().to(torch.float32)

        if mode == "sigmoid-topk":
            _, soft_mask, _ = module.hard_topk_mask_sigmoid_topk_surrogate(
                hard_mask,
                proxy_scores,
                limit,
                temperature=temperature,
            )
            return soft_mask.detach().to(torch.float32)

        if mode == "subset-gibbs":
            _, soft_mask, _ = module.hard_topk_mask_subset_gibbs_surrogate(
                hard_mask,
                proxy_scores,
                limit,
                temperature=temperature,
            )
            return soft_mask.detach().to(torch.float32)

        _, soft_mask, _ = module.hard_topk_mask_kth_boundary_surrogate(
            hard_mask,
            proxy_scores,
            limit,
            temperature=temperature,
            detach_kth_value=module.topk_kth_detach_value,
            use_midpoint_theta=module.topk_kth_use_midpoint_theta,
            normalize_soft_mask=module.topk_kth_normalize_soft_mask,
        )
        return soft_mask.detach().to(torch.float32)

    @staticmethod
    def _compare_norm_raw_score_grad(
        proxy_scores: torch.Tensor,
        mask_grad: torch.Tensor,
        limit: int,
        temperature: float,
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor] | None:
        width = proxy_scores.shape[-1]
        if limit <= 0 or limit >= width:
            return None

        tau = max(float(temperature), eps)
        topk_vals = torch.topk(proxy_scores, k=limit, dim=-1, largest=True, sorted=True).values
        theta = topk_vals[..., -1:].detach()

        u = torch.sigmoid((proxy_scores - theta) / tau)
        du_ds = u * (1.0 - u) / tau

        grad_raw = mask_grad * du_ds

        z = u.sum(dim=-1, keepdim=True) + eps
        weight = u / z
        g_bar = (mask_grad * weight).sum(dim=-1, keepdim=True)
        grad_norm = (float(limit) / z) * (mask_grad - g_bar) * du_ds

        cosine = F.cosine_similarity(grad_raw, grad_norm, dim=-1)
        sign_flip = ((grad_raw * grad_norm) < 0).to(torch.float32).mean(dim=-1)
        norm_ratio = grad_norm.norm(dim=-1) / (grad_raw.norm(dim=-1) + eps)

        return {
            "cosine": cosine.detach(),
            "sign_flip": sign_flip.detach(),
            "norm_ratio": norm_ratio.detach(),
            "grad_raw_abs": grad_raw.abs().detach(),
            "grad_norm_abs": grad_norm.abs().detach(),
            "grad_raw": grad_raw.detach(),
            "grad_norm": grad_norm.detach(),
        }

    @staticmethod
    def _compare_actual_with_refs(
        actual_grad: torch.Tensor,
        grad_raw: torch.Tensor,
        grad_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        actual_vs_raw_cosine = F.cosine_similarity(actual_grad, grad_raw, dim=-1)
        actual_vs_raw_sign_flip = ((actual_grad * grad_raw) < 0).to(torch.float32).mean(dim=-1)
        actual_vs_norm_cosine = F.cosine_similarity(actual_grad, grad_norm, dim=-1)
        actual_vs_norm_sign_flip = ((actual_grad * grad_norm) < 0).to(torch.float32).mean(dim=-1)
        return {
            "actual_vs_raw_cosine": actual_vs_raw_cosine.detach(),
            "actual_vs_raw_sign_flip": actual_vs_raw_sign_flip.detach(),
            "actual_vs_norm_cosine": actual_vs_norm_cosine.detach(),
            "actual_vs_norm_sign_flip": actual_vs_norm_sign_flip.detach(),
        }


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Analyze block-wise top-k surrogate statistics from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--split",
        choices=["train-eval", "valid", "test"],
        default="test",
        help="Dataset split used for forward statistics",
    )
    parser.add_argument("--num-batches", type=int, default=4, help="How many batches to run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON results")
    parser.add_argument(
        "--init-split",
        choices=["train-eval", "valid", "test"],
        default="test",
        help="Split used once for lazy module initialization",
    )
    parser.add_argument(
        "--print-config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print parsed train/model args before running",
    )
    parser.add_argument(
        "--collect-grad-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a real CE backward pass and compare score-space gradients for raw-vs-normalized kth surrogate",
    )
    script_args, remaining = parser.parse_known_args(argv)

    saved_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        train_args = parse_train_args()
    finally:
        sys.argv = saved_argv

    return script_args, train_args


def load_checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint, checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return {"state_dict": checkpoint}, checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")


def select_loader(
    split: str,
    train_eval_loader,
    valid_loader,
    test_loader,
):
    if split == "train-eval":
        return train_eval_loader
    if split == "valid":
        if valid_loader is None:
            raise ValueError("valid split is empty; increase --valid-set-size or use another split")
        return valid_loader
    return test_loader


def infer_and_report_mismatches(load_result) -> None:
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing:
        print(f"missing_keys={len(missing)}")
        for key in missing[:10]:
            print(f"  missing: {key}")
        if len(missing) > 10:
            print(f"  ... {len(missing) - 10} more")
    if unexpected:
        print(f"unexpected_keys={len(unexpected)}")
        for key in unexpected[:10]:
            print(f"  unexpected: {key}")
        if len(unexpected) > 10:
            print(f"  ... {len(unexpected) - 10} more")


def attach_collectors(model, collect_grad_compare: bool) -> list[BlockMetricCollector]:
    collectors: list[BlockMetricCollector] = []

    for block_index, block in enumerate(model.blocks):
        collector = BlockMetricCollector(block_index)
        packed_topk = block.attn.logic_attention.packed_topk
        original_forward = packed_topk.forward

        def wrapped_forward(self, q, k, topk, return_selector_mask=True, return_topk_packed_scores=True, proxy_scores=None, _orig=original_forward, _collector=collector):
            output = _orig(
                q,
                k,
                topk=topk,
                return_selector_mask=return_selector_mask,
                return_topk_packed_scores=return_topk_packed_scores,
                proxy_scores=proxy_scores,
            )
            _collector.observe(self, q, k, topk, proxy_scores, output, collect_grad_compare=collect_grad_compare)
            return output

        packed_topk.forward = MethodType(wrapped_forward, packed_topk)
        collectors.append(collector)

    return collectors


def enable_attention_surrogate_backward(model) -> None:
    for block in model.blocks:
        block.attn.logic_attention.packed_topk.train(True)
        block.attn.logic_attention.selector_majority.train(True)


def main() -> None:
    script_args, train_args = parse_combined_args()
    checkpoint_path = script_args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    set_deterministic(int(train_args.seed))
    device = torch.device(script_args.device)

    train_loader, valid_loader, test_loader, train_eval_loader, _, transform = load_dataset(train_args)
    init_loader = select_loader(script_args.init_split, train_eval_loader, valid_loader, test_loader)
    run_loader = select_loader(script_args.split, train_eval_loader, valid_loader, test_loader)

    if script_args.print_config:
        print("=== parsed train/model args ===")
        for key, value in sorted(vars(train_args).items()):
            print(f"{key}={value}")
        print()

    model = build_model(train_args).to(device)
    initialize_lazy_modules(model, init_loader, device, transform)

    checkpoint, state_dict = load_checkpoint_state(checkpoint_path)
    load_result = model.load_state_dict(state_dict, strict=False)
    infer_and_report_mismatches(load_result)

    set_attention_soft_train_temperatures(
        model,
        boundary_surrogate_temperature=float(train_args.boundary_surrogate_temp_max),
        majority_temperature=float(train_args.majority_train_temp_max),
    )
    model.eval()
    if script_args.collect_grad_compare:
        enable_attention_surrogate_backward(model)

    collectors = attach_collectors(model, collect_grad_compare=script_args.collect_grad_compare)

    if script_args.collect_grad_compare:
        for batch_index, (images, labels) in enumerate(run_loader):
            if batch_index >= script_args.num_batches:
                break
            model.zero_grad(set_to_none=True)
            images = transform(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            for collector in collectors:
                collector.finalize_backward()
    else:
        with torch.no_grad():
            for batch_index, (images, _) in enumerate(run_loader):
                if batch_index >= script_args.num_batches:
                    break
                images = transform(images.to(device, non_blocking=True))
                _ = model(images)

    summaries = [collector.summary() for collector in collectors]

    print(f"checkpoint={checkpoint_path}")
    print(f"step={checkpoint.get('step', 'unknown')}")
    print(f"split={script_args.split}")
    print(f"num_batches={script_args.num_batches}")
    print(f"topk_surrogate_mode={train_args.topk_surrogate_mode}")
    print(f"topk_forward_mode={train_args.topk_forward_mode}")
    print(f"collect_grad_compare={script_args.collect_grad_compare}")
    print()
    print(
        "block rows_seen soft_mask_total_sum_over_k_mean selected_soft_mask_fraction_mean kth_gap_mean kth_gap_valid_rows grad_compare_rows grad_cosine_mean grad_cosine_std grad_sign_flip_ratio_mean grad_norm_ratio_mean grad_raw_abs_mean grad_norm_abs_mean actual_grad_rows actual_grad_abs_mean actual_vs_raw_cosine_mean actual_vs_raw_cosine_std actual_vs_raw_sign_flip_ratio_mean actual_vs_norm_cosine_mean actual_vs_norm_cosine_std actual_vs_norm_sign_flip_ratio_mean"
    )
    for item in summaries:
        kth_gap_text = "nan" if item.kth_gap_mean is None else f"{item.kth_gap_mean:.6f}"
        grad_cosine_mean_text = "nan" if item.grad_cosine_mean is None else f"{item.grad_cosine_mean:.6f}"
        grad_cosine_std_text = "nan" if item.grad_cosine_std is None else f"{item.grad_cosine_std:.6f}"
        grad_sign_flip_text = "nan" if item.grad_sign_flip_ratio_mean is None else f"{item.grad_sign_flip_ratio_mean:.6f}"
        grad_norm_ratio_text = "nan" if item.grad_norm_ratio_mean is None else f"{item.grad_norm_ratio_mean:.6f}"
        grad_raw_abs_text = "nan" if item.grad_raw_abs_mean is None else f"{item.grad_raw_abs_mean:.6f}"
        grad_norm_abs_text = "nan" if item.grad_norm_abs_mean is None else f"{item.grad_norm_abs_mean:.6f}"
        actual_grad_abs_text = "nan" if item.actual_grad_abs_mean is None else f"{item.actual_grad_abs_mean:.6f}"
        actual_vs_raw_cosine_text = "nan" if item.actual_vs_raw_cosine_mean is None else f"{item.actual_vs_raw_cosine_mean:.6f}"
        actual_vs_raw_cosine_std_text = "nan" if item.actual_vs_raw_cosine_std is None else f"{item.actual_vs_raw_cosine_std:.6f}"
        actual_vs_raw_sign_flip_text = "nan" if item.actual_vs_raw_sign_flip_ratio_mean is None else f"{item.actual_vs_raw_sign_flip_ratio_mean:.6f}"
        actual_vs_norm_cosine_text = "nan" if item.actual_vs_norm_cosine_mean is None else f"{item.actual_vs_norm_cosine_mean:.6f}"
        actual_vs_norm_cosine_std_text = "nan" if item.actual_vs_norm_cosine_std is None else f"{item.actual_vs_norm_cosine_std:.6f}"
        actual_vs_norm_sign_flip_text = "nan" if item.actual_vs_norm_sign_flip_ratio_mean is None else f"{item.actual_vs_norm_sign_flip_ratio_mean:.6f}"
        print(
            f"{item.block_index:>5d} "
            f"{item.rows_seen:>8d} "
            f"{item.soft_mask_total_sum_over_k_mean:>33.6f} "
            f"{item.selected_soft_mask_fraction_mean:>33.6f} "
            f"{kth_gap_text:>12s} "
            f"{item.kth_gap_valid_rows:>18d} "
            f"{item.grad_compare_rows:>17d} "
            f"{grad_cosine_mean_text:>16s} "
            f"{grad_cosine_std_text:>15s} "
            f"{grad_sign_flip_text:>25s} "
            f"{grad_norm_ratio_text:>20s} "
            f"{grad_raw_abs_text:>17s} "
            f"{grad_norm_abs_text:>18s} "
            f"{item.actual_grad_rows:>16d} "
            f"{actual_grad_abs_text:>20s} "
            f"{actual_vs_raw_cosine_text:>25s} "
            f"{actual_vs_raw_cosine_std_text:>24s} "
            f"{actual_vs_raw_sign_flip_text:>35s} "
            f"{actual_vs_norm_cosine_text:>26s} "
            f"{actual_vs_norm_cosine_std_text:>25s} "
            f"{actual_vs_norm_sign_flip_text:>36s}"
        )

    if script_args.output_json is not None:
        payload = {
            "checkpoint": str(checkpoint_path),
            "step": checkpoint.get("step", None),
            "split": script_args.split,
            "num_batches": int(script_args.num_batches),
            "device": str(device),
            "model_args": vars(train_args),
            "blocks": [asdict(item) for item in summaries],
        }
        script_args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with script_args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print()
        print(f"json_saved_to={script_args.output_json}")


if __name__ == "__main__":
    main()
