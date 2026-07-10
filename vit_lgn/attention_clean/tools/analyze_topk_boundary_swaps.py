from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import load_dataset  # noqa: E402
from train_logic_vit_tiny import (  # noqa: E402
    build_model,
    initialize_lazy_modules,
    parse_args as parse_train_args,
    set_attention_soft_train_temperatures,
    set_deterministic,
)


@dataclass
class SwapRecord:
    block_index: int
    row_flat_index: int
    batch_index: int
    head_index: int
    query_index: int
    inside_rank: int
    outside_rank: int
    swap_out_index: int
    swap_in_index: int
    score_out: float
    score_in: float
    loss_orig: float
    loss_swap: float
    benefit: float
    correct: bool
    raw_pred_score: float | None
    raw_predict_good: bool | None
    norm_pred_score: float | None
    norm_predict_good: bool | None


@dataclass
class BlockSwapSummary:
    block_index: int
    rows_analyzed: int
    swaps_analyzed: int
    correct_swap_ratio: float
    mean_benefit: float
    raw_prediction_accuracy: float | None
    norm_prediction_accuracy: float | None
    raw_positive_ratio: float | None
    norm_positive_ratio: float | None
    raw_precision: float | None
    norm_precision: float | None
    raw_recall: float | None
    norm_recall: float | None
    raw_top1_hit: float | None
    norm_top1_hit: float | None
    raw_top3_hit: float | None
    norm_top3_hit: float | None
    raw_top5_hit: float | None
    norm_top5_hit: float | None
    raw_benefit_pearson: float | None
    norm_benefit_pearson: float | None
    raw_benefit_spearman: float | None
    norm_benefit_spearman: float | None


class BlockCapture:
    def __init__(self, block_index: int) -> None:
        self.block_index = int(block_index)
        self.hard_mask: torch.Tensor | None = None
        self.proxy_scores: torch.Tensor | None = None
        self.selector_mask: torch.Tensor | None = None
        self.limit: int | None = None
        self.raw_score_grad: torch.Tensor | None = None
        self.norm_score_grad: torch.Tensor | None = None

    def capture(
        self,
        module,
        q: torch.Tensor,
        k: torch.Tensor,
        topk: int,
        proxy_scores: torch.Tensor | None,
        output: dict[str, torch.Tensor],
    ) -> None:
        topk_indices = output["topk_indices"].detach().to(torch.long)
        row_count = topk_indices.shape[-2]
        limit = min(int(topk), row_count)
        hard_mask = module._build_selector_mask(topk_indices, row_count).detach().to(torch.bool)

        if proxy_scores is None:
            proxy_scores = module._xnor_similarity_proxy(q, k)
        proxy_scores = proxy_scores.detach().to(torch.float32)

        self.hard_mask = hard_mask
        self.proxy_scores = proxy_scores
        self.limit = limit
        self.selector_mask = output["selector_mask"]
        if self.selector_mask is not None and self.selector_mask.requires_grad:
            self.selector_mask.retain_grad()

    def finalize_backward(self, temperature: float, eps: float = 1e-6) -> None:
        if self.selector_mask is None or self.selector_mask.grad is None:
            return
        if self.proxy_scores is None or self.limit is None:
            return

        proxy_scores = self.proxy_scores
        mask_grad = self.selector_mask.grad.detach().to(torch.float32)
        limit = int(self.limit)
        width = proxy_scores.shape[-1]
        if limit <= 0 or limit >= width:
            return

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

        self.raw_score_grad = grad_raw.detach()
        self.norm_score_grad = grad_norm.detach()


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate boundary swaps by true loss and compare raw-vs-normalized kth score-gradient predictions. "
            "For each chosen block/query-row, it swaps low-score selected entries with high-score unselected entries "
            "near the decision boundary and checks whether the forced swap reduces CE loss."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--init-split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-index", type=int, default=0, help="Start batch index on the chosen split")
    parser.add_argument("--num-batches", type=int, default=1, help="Analyze this many consecutive batches")
    parser.add_argument("--block-index", type=int, default=None, help="Analyze one block only; default analyzes all blocks")
    parser.add_argument("--max-rows-per-block", type=int, default=2, help="Analyze at most this many flattened rows per block")
    parser.add_argument("--boundary-width", type=int, default=4, help="Swap candidates from lowest selected / highest unselected")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--print-record-limit", type=int, default=0, help="How many detailed swap records to print")
    parser.add_argument("--print-config", action=argparse.BooleanOptionalAction, default=False)
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


def select_loader(split: str, train_eval_loader, valid_loader, test_loader):
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


def attach_captures(model) -> tuple[list[BlockCapture], list[tuple[Any, Any]]]:
    captures: list[BlockCapture] = []
    originals: list[tuple[Any, Any]] = []

    for block_index, block in enumerate(model.blocks):
        capture = BlockCapture(block_index)
        packed_topk = block.attn.logic_attention.packed_topk
        original_forward = packed_topk.forward
        originals.append((packed_topk, original_forward))

        def wrapped_forward(
            self,
            q,
            k,
            topk,
            return_selector_mask=True,
            return_topk_packed_scores=True,
            proxy_scores=None,
            _orig=original_forward,
            _capture=capture,
        ):
            output = _orig(
                q,
                k,
                topk=topk,
                return_selector_mask=return_selector_mask,
                return_topk_packed_scores=return_topk_packed_scores,
                proxy_scores=proxy_scores,
            )
            _capture.capture(self, q, k, topk, proxy_scores, output)
            return output

        packed_topk.forward = MethodType(wrapped_forward, packed_topk)
        captures.append(capture)

    return captures, originals


def restore_forwards(originals: list[tuple[Any, Any]]) -> None:
    for module, original_forward in originals:
        module.forward = original_forward


def enable_attention_surrogate_backward(model) -> None:
    for block in model.blocks:
        block.attn.logic_attention.packed_topk.train(True)
        block.attn.logic_attention.selector_majority.train(True)


def get_batches(loader, start_batch_index: int, num_batches: int):
    if start_batch_index < 0:
        raise ValueError("batch_index must be >= 0")
    if num_batches <= 0:
        raise ValueError("num_batches must be > 0")

    outputs = []
    end_batch_index = start_batch_index + num_batches
    for idx, batch in enumerate(loader):
        if idx < start_batch_index:
            continue
        if idx >= end_batch_index:
            break
        outputs.append((idx, batch))
    if len(outputs) != num_batches:
        raise IndexError(
            f"requested batches [{start_batch_index}, {end_batch_index}) but only found {len(outputs)} batches"
        )
    return outputs


@contextmanager
def forced_block_mask(model, block_index: int, forced_mask: torch.Tensor):
    packed_topk = model.blocks[block_index].attn.logic_attention.packed_topk
    original_forward = packed_topk.forward
    forced_mask = forced_mask.detach()

    def wrapped_forward(
        self,
        q,
        k,
        topk,
        return_selector_mask=True,
        return_topk_packed_scores=True,
        proxy_scores=None,
    ):
        output = original_forward(
            q,
            k,
            topk=topk,
            return_selector_mask=return_selector_mask,
            return_topk_packed_scores=return_topk_packed_scores,
            proxy_scores=proxy_scores,
        )
        if return_selector_mask:
            output["selector_mask"] = forced_mask.to(device=q.device)
        return output

    packed_topk.forward = MethodType(wrapped_forward, packed_topk)
    try:
        yield
    finally:
        packed_topk.forward = original_forward


def unflatten_row_index(row_flat_index: int, prefix_shape: torch.Size) -> tuple[int, ...]:
    coords = []
    remain = int(row_flat_index)
    for size in reversed(prefix_shape):
        coords.append(remain % int(size))
        remain //= int(size)
    return tuple(reversed(coords))


def force_swap(mask: torch.Tensor, row_flat_index: int, idx_out: int, idx_in: int) -> torch.Tensor:
    swapped = mask.clone()
    flat = swapped.reshape(-1, swapped.shape[-1])
    flat[row_flat_index, idx_out] = False
    flat[row_flat_index, idx_in] = True
    return swapped


def define_correct_swaps_by_loss(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    block_index: int,
    original_mask: torch.Tensor,
    proxy_scores: torch.Tensor,
    raw_score_grad: torch.Tensor | None,
    norm_score_grad: torch.Tensor | None,
    boundary_width: int,
    max_rows_to_scan: int,
    show_row_progress: bool = False,
    row_progress_desc: str | None = None,
) -> tuple[list[SwapRecord], float]:
    with torch.no_grad():
        logits_orig = model(images)
        loss_orig = F.cross_entropy(logits_orig, labels).item()

    prefix_shape = original_mask.shape[:-1]
    flat_mask = original_mask.reshape(-1, original_mask.shape[-1])
    flat_scores = proxy_scores.reshape(-1, proxy_scores.shape[-1])
    flat_raw_grad = None if raw_score_grad is None else raw_score_grad.reshape(-1, raw_score_grad.shape[-1])
    flat_norm_grad = None if norm_score_grad is None else norm_score_grad.reshape(-1, norm_score_grad.shape[-1])

    records: list[SwapRecord] = []
    rows_to_scan = min(int(max_rows_to_scan), flat_mask.shape[0])

    row_iter = range(rows_to_scan)
    if show_row_progress:
        row_iter = tqdm(
            row_iter,
            total=rows_to_scan,
            desc=row_progress_desc or "analyze_rows",
            leave=False,
        )

    for row_flat_index in row_iter:
        row_mask = flat_mask[row_flat_index]
        row_scores = flat_scores[row_flat_index]
        k = int(row_mask.to(torch.int32).sum().item())
        width = row_mask.numel()
        if k <= 0 or k >= width:
            continue

        b = min(int(boundary_width), k, width - k)
        if b <= 0:
            continue

        selected_scores = row_scores.masked_fill(~row_mask, float("inf"))
        inside_low = torch.topk(selected_scores, k=b, largest=False, sorted=True).indices

        outside_scores = row_scores.masked_fill(row_mask, float("-inf"))
        outside_high = torch.topk(outside_scores, k=b, largest=True, sorted=True).indices

        coords = unflatten_row_index(row_flat_index, prefix_shape)
        batch_index, head_index, query_index = coords

        for i_idx in range(b):
            for j_idx in range(b):
                idx_out = int(inside_low[i_idx].item())
                idx_in = int(outside_high[j_idx].item())

                swapped_mask = force_swap(original_mask, row_flat_index, idx_out, idx_in)
                with torch.no_grad():
                    with forced_block_mask(model, block_index, swapped_mask):
                        logits_swap = model(images)
                    loss_swap = F.cross_entropy(logits_swap, labels).item()

                benefit = float(loss_orig - loss_swap)
                correct = bool(benefit > 0.0)

                raw_pred_score = None
                raw_predict_good = None
                if flat_raw_grad is not None:
                    grad_row = flat_raw_grad[row_flat_index]
                    raw_pred_score = float((grad_row[idx_out] - grad_row[idx_in]).item())
                    raw_predict_good = bool(raw_pred_score > 0.0)

                norm_pred_score = None
                norm_predict_good = None
                if flat_norm_grad is not None:
                    grad_row = flat_norm_grad[row_flat_index]
                    norm_pred_score = float((grad_row[idx_out] - grad_row[idx_in]).item())
                    norm_predict_good = bool(norm_pred_score > 0.0)

                records.append(
                    SwapRecord(
                        block_index=block_index,
                        row_flat_index=row_flat_index,
                        batch_index=int(batch_index),
                        head_index=int(head_index),
                        query_index=int(query_index),
                        inside_rank=i_idx,
                        outside_rank=j_idx,
                        swap_out_index=idx_out,
                        swap_in_index=idx_in,
                        score_out=float(row_scores[idx_out].item()),
                        score_in=float(row_scores[idx_in].item()),
                        loss_orig=loss_orig,
                        loss_swap=float(loss_swap),
                        benefit=benefit,
                        correct=correct,
                        raw_pred_score=raw_pred_score,
                        raw_predict_good=raw_predict_good,
                        norm_pred_score=norm_pred_score,
                        norm_predict_good=norm_predict_good,
                    )
                )

    return records, loss_orig


def summarize_block(block_index: int, records: list[SwapRecord]) -> BlockSwapSummary:
    swaps_analyzed = len(records)
    rows_analyzed = len({(r.batch_index, r.head_index, r.query_index) for r in records})
    correct_swap_ratio = 0.0
    mean_benefit = 0.0
    raw_prediction_accuracy = None
    norm_prediction_accuracy = None
    raw_positive_ratio = None
    norm_positive_ratio = None
    raw_precision = None
    norm_precision = None
    raw_recall = None
    norm_recall = None
    raw_top1_hit = None
    norm_top1_hit = None
    raw_top3_hit = None
    norm_top3_hit = None
    raw_top5_hit = None
    norm_top5_hit = None
    raw_benefit_pearson = None
    norm_benefit_pearson = None
    raw_benefit_spearman = None
    norm_benefit_spearman = None

    if swaps_analyzed > 0:
        correct_swap_ratio = sum(1 for r in records if r.correct) / float(swaps_analyzed)
        mean_benefit = sum(r.benefit for r in records) / float(swaps_analyzed)

    raw_valid = [r for r in records if r.raw_predict_good is not None]
    if raw_valid:
        raw_prediction_accuracy = sum(int(r.raw_predict_good == r.correct) for r in raw_valid) / float(len(raw_valid))
        raw_positive_ratio = sum(int(bool(r.raw_predict_good)) for r in raw_valid) / float(len(raw_valid))
        raw_precision = _precision(raw_valid, use_norm=False)
        raw_recall = _recall(raw_valid, use_norm=False)
        raw_top1_hit = _topk_hit(raw_valid, k=1, use_norm=False)
        raw_top3_hit = _topk_hit(raw_valid, k=3, use_norm=False)
        raw_top5_hit = _topk_hit(raw_valid, k=5, use_norm=False)
        raw_benefit_pearson = _benefit_corr(raw_valid, use_norm=False, kind="pearson")
        raw_benefit_spearman = _benefit_corr(raw_valid, use_norm=False, kind="spearman")

    norm_valid = [r for r in records if r.norm_predict_good is not None]
    if norm_valid:
        norm_prediction_accuracy = sum(int(r.norm_predict_good == r.correct) for r in norm_valid) / float(len(norm_valid))
        norm_positive_ratio = sum(int(bool(r.norm_predict_good)) for r in norm_valid) / float(len(norm_valid))
        norm_precision = _precision(norm_valid, use_norm=True)
        norm_recall = _recall(norm_valid, use_norm=True)
        norm_top1_hit = _topk_hit(norm_valid, k=1, use_norm=True)
        norm_top3_hit = _topk_hit(norm_valid, k=3, use_norm=True)
        norm_top5_hit = _topk_hit(norm_valid, k=5, use_norm=True)
        norm_benefit_pearson = _benefit_corr(norm_valid, use_norm=True, kind="pearson")
        norm_benefit_spearman = _benefit_corr(norm_valid, use_norm=True, kind="spearman")

    return BlockSwapSummary(
        block_index=block_index,
        rows_analyzed=rows_analyzed,
        swaps_analyzed=swaps_analyzed,
        correct_swap_ratio=correct_swap_ratio,
        mean_benefit=mean_benefit,
        raw_prediction_accuracy=raw_prediction_accuracy,
        norm_prediction_accuracy=norm_prediction_accuracy,
        raw_positive_ratio=raw_positive_ratio,
        norm_positive_ratio=norm_positive_ratio,
        raw_precision=raw_precision,
        norm_precision=norm_precision,
        raw_recall=raw_recall,
        norm_recall=norm_recall,
        raw_top1_hit=raw_top1_hit,
        norm_top1_hit=norm_top1_hit,
        raw_top3_hit=raw_top3_hit,
        norm_top3_hit=norm_top3_hit,
        raw_top5_hit=raw_top5_hit,
        norm_top5_hit=norm_top5_hit,
        raw_benefit_pearson=raw_benefit_pearson,
        norm_benefit_pearson=norm_benefit_pearson,
        raw_benefit_spearman=raw_benefit_spearman,
        norm_benefit_spearman=norm_benefit_spearman,
    )


def _fmt_metric(value: float | None) -> str:
    return "nan" if value is None else f"{value:.6f}"


def _metric_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _print_high_signal_table(label: str, summaries: list[BlockSwapSummary]) -> None:
    print(label)
    print(
        "scope rows swaps pos_ratio "
        "raw_precision norm_precision d_precision "
        "raw_recall norm_recall d_recall "
        "raw_spearman norm_spearman d_spearman "
        "raw_pearson norm_pearson d_pearson"
    )
    for item in summaries:
        scope = "all" if item.block_index < 0 else str(item.block_index)
        print(
            f"{scope:>5s} "
            f"{item.rows_analyzed:>4d} "
            f"{item.swaps_analyzed:>5d} "
            f"{item.correct_swap_ratio:>9.6f} "
            f"{_fmt_metric(item.raw_precision):>13s} "
            f"{_fmt_metric(item.norm_precision):>14s} "
            f"{_fmt_metric(_metric_delta(item.raw_precision, item.norm_precision)):>11s} "
            f"{_fmt_metric(item.raw_recall):>10s} "
            f"{_fmt_metric(item.norm_recall):>11s} "
            f"{_fmt_metric(_metric_delta(item.raw_recall, item.norm_recall)):>8s} "
            f"{_fmt_metric(item.raw_benefit_spearman):>12s} "
            f"{_fmt_metric(item.norm_benefit_spearman):>13s} "
            f"{_fmt_metric(_metric_delta(item.raw_benefit_spearman, item.norm_benefit_spearman)):>10s} "
            f"{_fmt_metric(item.raw_benefit_pearson):>11s} "
            f"{_fmt_metric(item.norm_benefit_pearson):>12s} "
            f"{_fmt_metric(_metric_delta(item.raw_benefit_pearson, item.norm_benefit_pearson)):>9s}"
        )


def _precision(records: list[SwapRecord], use_norm: bool) -> float | None:
    predicted_positive = [
        r for r in records if (r.norm_predict_good if use_norm else r.raw_predict_good)
    ]
    if not predicted_positive:
        return None
    true_positive = sum(int(r.correct) for r in predicted_positive)
    return true_positive / float(len(predicted_positive))


def _recall(records: list[SwapRecord], use_norm: bool) -> float | None:
    actual_positive = [r for r in records if r.correct]
    if not actual_positive:
        return None
    hit = 0
    for r in actual_positive:
        predicted = r.norm_predict_good if use_norm else r.raw_predict_good
        hit += int(bool(predicted))
    return hit / float(len(actual_positive))


def _topk_hit(records: list[SwapRecord], k: int, use_norm: bool) -> float | None:
    if not records:
        return None
    key = (lambda r: r.norm_pred_score) if use_norm else (lambda r: r.raw_pred_score)
    valid = [r for r in records if key(r) is not None]
    if not valid:
        return None
    topk = sorted(valid, key=lambda r: float(key(r)), reverse=True)[: min(k, len(valid))]
    return float(any(r.correct for r in topk))


def _benefit_corr(records: list[SwapRecord], use_norm: bool, kind: str) -> float | None:
    score_key = (lambda r: r.norm_pred_score) if use_norm else (lambda r: r.raw_pred_score)
    valid = [r for r in records if score_key(r) is not None]
    if len(valid) < 2:
        return None

    scores = torch.tensor([float(score_key(r)) for r in valid], dtype=torch.float32)
    benefits = torch.tensor([float(r.benefit) for r in valid], dtype=torch.float32)
    if kind == "pearson":
        return _pearson_corr(scores, benefits)
    if kind == "spearman":
        return _spearman_corr(scores, benefits)
    raise ValueError(f"Unknown correlation kind: {kind}")


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= eps:
        return None
    return float((x_centered * y_centered).sum().item() / denom.item())


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2:
        return None
    x_rank = torch.argsort(torch.argsort(x)).to(torch.float32)
    y_rank = torch.argsort(torch.argsort(y)).to(torch.float32)
    return _pearson_corr(x_rank, y_rank)


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

    selected_batches = get_batches(run_loader, script_args.batch_index, script_args.num_batches)
    target_block_indices = range(len(model.blocks)) if script_args.block_index is None else [int(script_args.block_index)]
    all_records: list[SwapRecord] = []
    block_to_records: dict[int, list[SwapRecord]] = {int(block_index): [] for block_index in target_block_indices}
    baseline_losses: list[float] = []

    for _, (images, labels) in tqdm(
        selected_batches,
        desc="analyze_batches",
        total=len(selected_batches),
    ):
        images = transform(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        model.eval()
        enable_attention_surrogate_backward(model)
        captures, originals = attach_captures(model)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        baseline_loss = F.cross_entropy(logits, labels)
        baseline_loss.backward()
        baseline_losses.append(float(baseline_loss.item()))

        for block_index, capture in enumerate(captures):
            capture.finalize_backward(
                temperature=float(model.blocks[block_index].attn.logic_attention.packed_topk.boundary_surrogate_temperature)
            )
        restore_forwards(originals)

        model.zero_grad(set_to_none=True)
        model.eval()

        for block_index in target_block_indices:
            capture = captures[block_index]
            if capture.hard_mask is None or capture.proxy_scores is None:
                continue

            records, _ = define_correct_swaps_by_loss(
                model=model,
                images=images,
                labels=labels,
                block_index=block_index,
                original_mask=capture.hard_mask,
                proxy_scores=capture.proxy_scores,
                raw_score_grad=capture.raw_score_grad,
                norm_score_grad=capture.norm_score_grad,
                boundary_width=int(script_args.boundary_width),
                max_rows_to_scan=int(script_args.max_rows_per_block),
                show_row_progress=True,
                row_progress_desc=f"rows[b{block_index}]",
            )
            all_records.extend(records)
            block_to_records[int(block_index)].extend(records)

    block_summaries = [
        summarize_block(block_index, block_to_records[block_index])
        for block_index in sorted(block_to_records.keys())
        if block_to_records[block_index]
    ]
    overall_summary = summarize_block(-1, all_records) if all_records else None

    print(f"checkpoint={checkpoint_path}")
    print(f"step={checkpoint.get('step', 'unknown')}")
    print(f"split={script_args.split}")
    print(f"batch_index={script_args.batch_index}")
    print(f"num_batches={script_args.num_batches}")
    if baseline_losses:
        print(f"baseline_loss_mean={sum(baseline_losses) / len(baseline_losses):.6f}")
    print(f"boundary_width={script_args.boundary_width}")
    print(f"max_rows_per_block={script_args.max_rows_per_block}")
    print()
    print(
        "This script defines a swap as correct iff forcing that boundary swap lowers CE loss. "
        "raw/norm columns are first-order score-gradient predictions from unnormalized vs normalized kth surrogate."
    )
    print()
    if overall_summary is not None:
        _print_high_signal_table("overall", [overall_summary])
        print()
    _print_high_signal_table("per_block", block_summaries)

    if script_args.print_record_limit > 0:
        print()
        print("sample swap records")
        print(
            "block row batch head query out_idx in_idx score_out score_in benefit correct raw_pred_score raw_good norm_pred_score norm_good"
        )
        for record in all_records[: script_args.print_record_limit]:
            raw_pred = "nan" if record.raw_pred_score is None else f"{record.raw_pred_score:.6f}"
            norm_pred = "nan" if record.norm_pred_score is None else f"{record.norm_pred_score:.6f}"
            raw_good = "nan" if record.raw_predict_good is None else str(int(record.raw_predict_good))
            norm_good = "nan" if record.norm_predict_good is None else str(int(record.norm_predict_good))
            print(
                f"{record.block_index:>5d} "
                f"{record.row_flat_index:>3d} "
                f"{record.batch_index:>5d} "
                f"{record.head_index:>4d} "
                f"{record.query_index:>5d} "
                f"{record.swap_out_index:>7d} "
                f"{record.swap_in_index:>6d} "
                f"{record.score_out:>9.4f} "
                f"{record.score_in:>8.4f} "
                f"{record.benefit:>8.6f} "
                f"{int(record.correct):>7d} "
                f"{raw_pred:>13s} "
                f"{raw_good:>8s} "
                f"{norm_pred:>14s} "
                f"{norm_good:>9s}"
            )

    if script_args.output_json is not None:
        payload = {
            "checkpoint": str(checkpoint_path),
            "step": checkpoint.get("step", None),
            "split": script_args.split,
            "batch_index": int(script_args.batch_index),
            "num_batches": int(script_args.num_batches),
            "baseline_loss_mean": (sum(baseline_losses) / len(baseline_losses)) if baseline_losses else None,
            "boundary_width": int(script_args.boundary_width),
            "max_rows_per_block": int(script_args.max_rows_per_block),
            "model_args": vars(train_args),
            "overall_summary": None if overall_summary is None else asdict(overall_summary),
            "summaries": [asdict(item) for item in block_summaries],
            "records": [asdict(item) for item in all_records],
        }
        script_args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with script_args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print()
        print(f"json_saved_to={script_args.output_json}")


if __name__ == "__main__":
    main()
