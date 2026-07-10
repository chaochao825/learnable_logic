from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
class SurrogateSpec:
    label: str
    checkpoint: Path
    topk_surrogate_mode: str
    topk_kth_normalize_soft_mask: bool


@dataclass
class SwapRecord:
    surrogate_label: str
    block_index: int
    row_flat_index: int
    batch_loader_index: int
    batch_index: int
    head_index: int
    query_index: int
    swap_out_index: int
    swap_in_index: int
    score_out: float
    score_in: float
    pred_score: float
    pred_good: bool
    benefit: float
    correct: bool


@dataclass
class SurrogateSummary:
    label: str
    rows_analyzed: int
    swaps_analyzed: int
    pos_ratio: float
    precision: float | None
    recall: float | None
    average_precision: float | None
    top_truek_precision: float | None
    top_truek_benefit_capture: float | None
    benefit_spearman: float | None
    benefit_pearson: float | None
    pred_score_mean: float
    benefit_mean: float


@dataclass
class SurrogateBlockSummary:
    label: str
    block_index: int
    rows_analyzed: int
    swaps_analyzed: int
    pos_ratio: float
    precision: float | None
    recall: float | None
    average_precision: float | None
    top_truek_precision: float | None
    top_truek_benefit_capture: float | None
    benefit_spearman: float | None
    benefit_pearson: float | None


class BlockCapture:
    def __init__(self, block_index: int) -> None:
        self.block_index = int(block_index)
        self.hard_mask: torch.Tensor | None = None
        self.proxy_scores: torch.Tensor | None = None
        self.proxy_scores_grad: torch.Tensor | None = None

    def capture_proxy(self, proxy_scores: torch.Tensor) -> torch.Tensor:
        self.proxy_scores = proxy_scores
        if proxy_scores.requires_grad:
            proxy_scores.retain_grad()
        return proxy_scores

    def capture_forward(self, module, output: dict[str, torch.Tensor]) -> None:
        topk_indices = output["topk_indices"].detach().to(torch.long)
        row_count = topk_indices.shape[-2]
        self.hard_mask = module._build_selector_mask(topk_indices, row_count).detach().to(torch.bool)

    def finalize_backward(self) -> None:
        if self.proxy_scores is None or self.proxy_scores.grad is None:
            return
        self.proxy_scores_grad = self.proxy_scores.grad.detach().to(torch.float32)


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Compare boundary-swap alignment across kth-raw, kth-norm, sigmoid-topk and subset-gibbs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fallback checkpoint used for all surrogates if per-surrogate checkpoint is omitted")
    parser.add_argument("--checkpoint-kth-raw", type=Path, default=None)
    parser.add_argument("--checkpoint-kth-norm", type=Path, default=None)
    parser.add_argument("--checkpoint-sigmoid-topk", type=Path, default=None)
    parser.add_argument("--checkpoint-subset-gibbs", type=Path, default=None)
    parser.add_argument("--split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--init-split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-index", type=int, default=0, help="Start batch index")
    parser.add_argument("--num-batches", type=int, default=1, help="Analyze this many consecutive batches")
    parser.add_argument("--block-index", type=int, default=None, help="Analyze one block only; default analyzes all blocks")
    parser.add_argument("--max-rows-per-block", type=int, default=10, help="Analyze at most this many flattened rows per block per batch")
    parser.add_argument("--boundary-width", type=int, default=4, help="Swap candidates from lowest selected / highest unselected")
    parser.add_argument("--print-record-limit", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--output-table-prefix",
        type=Path,
        default=None,
        help="Write detailed CSV tables to <prefix>_overall.csv, <prefix>_per_block.csv and <prefix>_records.csv",
    )
    parser.add_argument("--print-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--print-detailed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the full metrics table in addition to the compact ranking-focused summary",
    )
    script_args, remaining = parser.parse_known_args(argv)

    saved_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        train_args = parse_train_args()
    finally:
        sys.argv = saved_argv

    return script_args, train_args


def select_loader(split: str, train_eval_loader, valid_loader, test_loader):
    if split == "train-eval":
        return train_eval_loader
    if split == "valid":
        if valid_loader is None:
            raise ValueError("valid split is empty; increase --valid-set-size or use another split")
        return valid_loader
    return test_loader


def get_batches(loader, start_batch_index: int, num_batches: int):
    if start_batch_index < 0:
        raise ValueError("batch-index must be >= 0")
    if num_batches <= 0:
        raise ValueError("num-batches must be > 0")
    end = start_batch_index + num_batches
    batches = []
    for idx, batch in enumerate(loader):
        if idx < start_batch_index:
            continue
        if idx >= end:
            break
        batches.append((idx, batch))
    if len(batches) != num_batches:
        raise IndexError(f"requested {num_batches} batches starting at {start_batch_index}, got {len(batches)}")
    return batches


def load_checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint, checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return {"state_dict": checkpoint}, checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")


def infer_and_report_mismatches(load_result, label: str) -> None:
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing:
        print(f"[{label}] missing_keys={len(missing)}")
        for key in missing[:5]:
            print(f"  missing: {key}")
    if unexpected:
        print(f"[{label}] unexpected_keys={len(unexpected)}")
        for key in unexpected[:5]:
            print(f"  unexpected: {key}")


def build_specs(script_args: argparse.Namespace) -> list[SurrogateSpec]:
    def resolve(path: Path | None) -> Path:
        if path is not None:
            return path.expanduser().resolve()
        if script_args.checkpoint is None:
            raise ValueError("Need either --checkpoint or per-surrogate checkpoint args")
        return script_args.checkpoint.expanduser().resolve()

    specs = [
        SurrogateSpec(
            label="kth-raw",
            checkpoint=resolve(script_args.checkpoint_kth_raw),
            topk_surrogate_mode="kth",
            topk_kth_normalize_soft_mask=False,
        ),
        SurrogateSpec(
            label="kth-norm",
            checkpoint=resolve(script_args.checkpoint_kth_norm),
            topk_surrogate_mode="kth",
            topk_kth_normalize_soft_mask=True,
        ),
        SurrogateSpec(
            label="sigmoid-topk",
            checkpoint=resolve(script_args.checkpoint_sigmoid_topk),
            topk_surrogate_mode="sigmoid-topk",
            topk_kth_normalize_soft_mask=True,
        ),
        SurrogateSpec(
            label="subset-gibbs",
            checkpoint=resolve(script_args.checkpoint_subset_gibbs),
            topk_surrogate_mode="subset-gibbs",
            topk_kth_normalize_soft_mask=True,
        ),
    ]
    for spec in specs:
        if not spec.checkpoint.exists():
            raise FileNotFoundError(f"{spec.label} checkpoint not found: {spec.checkpoint}")
    return specs


def make_model_args(train_args: argparse.Namespace, spec: SurrogateSpec) -> argparse.Namespace:
    args = copy.deepcopy(train_args)
    args.topk_surrogate_mode = spec.topk_surrogate_mode
    args.topk_kth_normalize_soft_mask = spec.topk_kth_normalize_soft_mask
    return args


def enable_surrogate_backward(model) -> None:
    for block in model.blocks:
        block.attn.logic_attention.packed_topk.train(True)
        block.attn.logic_attention.selector_majority.train(True)


def attach_captures(model) -> tuple[list[BlockCapture], list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    captures: list[BlockCapture] = []
    forward_originals: list[tuple[Any, Any]] = []
    proxy_originals: list[tuple[Any, Any]] = []

    for block_index, block in enumerate(model.blocks):
        capture = BlockCapture(block_index)
        packed_topk = block.attn.logic_attention.packed_topk
        original_forward = packed_topk.forward
        original_proxy = packed_topk._xnor_similarity_proxy
        forward_originals.append((packed_topk, original_forward))
        proxy_originals.append((packed_topk, original_proxy))

        def wrapped_proxy(self, q, k, _orig=original_proxy, _capture=capture):
            proxy = _orig(q, k)
            return _capture.capture_proxy(proxy)

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
            _capture.capture_forward(self, output)
            return output

        packed_topk._xnor_similarity_proxy = MethodType(wrapped_proxy, packed_topk)
        packed_topk.forward = MethodType(wrapped_forward, packed_topk)
        captures.append(capture)

    return captures, forward_originals, proxy_originals


def restore_forwards(originals: list[tuple[Any, Any]]) -> None:
    for module, original_forward in originals:
        module.forward = original_forward


def restore_proxies(originals: list[tuple[Any, Any]]) -> None:
    for module, original_proxy in originals:
        module._xnor_similarity_proxy = original_proxy


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


def force_swap(mask: torch.Tensor, row_flat_index: int, idx_out: int, idx_in: int) -> torch.Tensor:
    swapped = mask.clone()
    flat = swapped.reshape(-1, swapped.shape[-1])
    flat[row_flat_index, idx_out] = False
    flat[row_flat_index, idx_in] = True
    return swapped


def unflatten_row_index(row_flat_index: int, prefix_shape: torch.Size) -> tuple[int, ...]:
    coords = []
    remain = int(row_flat_index)
    for size in reversed(prefix_shape):
        coords.append(remain % int(size))
        remain //= int(size)
    return tuple(reversed(coords))


def corr_pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float | None:
    if x.numel() < 2:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.norm() * y_centered.norm()
    if float(denom.item()) <= eps:
        return None
    return float((x_centered * y_centered).sum().item() / denom.item())


def corr_spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    x_rank = torch.argsort(torch.argsort(x)).to(torch.float32)
    y_rank = torch.argsort(torch.argsort(y)).to(torch.float32)
    return corr_pearson(x_rank, y_rank)


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    if scores.numel() == 0:
        return None
    positives = int(labels.sum().item())
    if positives == 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    cumsum_tp = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32, device=sorted_labels.device)
    precision_at_i = cumsum_tp / ranks
    ap = (precision_at_i * sorted_labels).sum() / float(positives)
    return float(ap.item())


def top_truek_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    positives = int(labels.sum().item())
    if positives <= 0:
        return None
    order = torch.argsort(scores, descending=True)[:positives]
    return float(labels[order].mean().item())


def top_truek_benefit_capture(scores: torch.Tensor, benefits: torch.Tensor) -> float | None:
    positive_benefits = benefits.clamp_min(0.0)
    positives = int((positive_benefits > 0).sum().item())
    total_positive_benefit = float(positive_benefits.sum().item())
    if positives <= 0 or total_positive_benefit <= 0.0:
        return None
    order = torch.argsort(scores, descending=True)[:positives]
    captured = float(positive_benefits[order].sum().item())
    return captured / total_positive_benefit


def summarize_surrogate(label: str, records: list[SwapRecord]) -> SurrogateSummary:
    swaps = len(records)
    rows = len({(r.batch_loader_index, r.block_index, r.batch_index, r.head_index, r.query_index) for r in records})
    if swaps == 0:
        return SurrogateSummary(label, 0, 0, 0.0, None, None, None, None, None, None, None, 0.0, 0.0)

    scores = torch.tensor([r.pred_score for r in records], dtype=torch.float32)
    labels = torch.tensor([1.0 if r.correct else 0.0 for r in records], dtype=torch.float32)
    benefits = torch.tensor([r.benefit for r in records], dtype=torch.float32)
    pred_positive = torch.tensor([1.0 if r.pred_good else 0.0 for r in records], dtype=torch.float32)

    positives = int(labels.sum().item())
    pred_pos = int(pred_positive.sum().item())
    tp = int((labels * pred_positive).sum().item())

    precision = None if pred_pos == 0 else tp / float(pred_pos)
    recall = None if positives == 0 else tp / float(positives)

    return SurrogateSummary(
        label=label,
        rows_analyzed=rows,
        swaps_analyzed=swaps,
        pos_ratio=float(labels.mean().item()),
        precision=precision,
        recall=recall,
        average_precision=average_precision(scores, labels),
        top_truek_precision=top_truek_precision(scores, labels),
        top_truek_benefit_capture=top_truek_benefit_capture(scores, benefits),
        benefit_spearman=corr_spearman(scores, benefits),
        benefit_pearson=corr_pearson(scores, benefits),
        pred_score_mean=float(scores.mean().item()),
        benefit_mean=float(benefits.mean().item()),
    )


def summarize_surrogate_block(label: str, block_index: int, records: list[SwapRecord]) -> SurrogateBlockSummary:
    summary = summarize_surrogate(label, records)
    return SurrogateBlockSummary(
        label=label,
        block_index=block_index,
        rows_analyzed=summary.rows_analyzed,
        swaps_analyzed=summary.swaps_analyzed,
        pos_ratio=summary.pos_ratio,
        precision=summary.precision,
        recall=summary.recall,
        average_precision=summary.average_precision,
        top_truek_precision=summary.top_truek_precision,
        top_truek_benefit_capture=summary.top_truek_benefit_capture,
        benefit_spearman=summary.benefit_spearman,
        benefit_pearson=summary.benefit_pearson,
    )


def analyze_spec(
    spec: SurrogateSpec,
    train_args: argparse.Namespace,
    init_loader,
    batches,
    transform,
    device: torch.device,
    block_index: int | None,
    max_rows_per_block: int,
    boundary_width: int,
) -> tuple[SurrogateSummary, list[SwapRecord], float]:
    args = make_model_args(train_args, spec)
    model = build_model(args).to(device)
    initialize_lazy_modules(model, init_loader, device, transform)

    checkpoint, state_dict = load_checkpoint_state(spec.checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    infer_and_report_mismatches(load_result, spec.label)

    set_attention_soft_train_temperatures(
        model,
        boundary_surrogate_temperature=float(args.boundary_surrogate_temp_max),
        majority_temperature=float(args.majority_train_temp_max),
    )

    target_blocks = list(range(len(model.blocks))) if block_index is None else [int(block_index)]
    records: list[SwapRecord] = []
    baseline_losses: list[float] = []

    for batch_loader_index, (images, labels) in tqdm(batches, desc=f"{spec.label}:batches", total=len(batches), leave=False):
        images = transform(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        model.eval()
        enable_surrogate_backward(model)
        captures, fwd_originals, proxy_originals = attach_captures(model)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        baseline_loss = F.cross_entropy(logits, labels)
        baseline_loss.backward()
        baseline_loss_value = float(baseline_loss.item())
        baseline_losses.append(baseline_loss_value)

        restore_forwards(fwd_originals)
        restore_proxies(proxy_originals)
        for capture in captures:
            capture.finalize_backward()

        model.zero_grad(set_to_none=True)
        model.eval()

        for bidx in target_blocks:
            capture = captures[bidx]
            if capture.hard_mask is None or capture.proxy_scores is None or capture.proxy_scores_grad is None:
                continue

            flat_mask = capture.hard_mask.reshape(-1, capture.hard_mask.shape[-1])
            flat_scores = capture.proxy_scores.reshape(-1, capture.proxy_scores.shape[-1])
            flat_grad = capture.proxy_scores_grad.reshape(-1, capture.proxy_scores_grad.shape[-1])
            prefix_shape = capture.hard_mask.shape[:-1]
            rows_to_scan = min(int(max_rows_per_block), flat_mask.shape[0])

            for row_flat_index in tqdm(
                range(rows_to_scan),
                desc=f"{spec.label}:rows[b{bidx}|batch{batch_loader_index}]",
                total=rows_to_scan,
                leave=False,
            ):
                row_mask = flat_mask[row_flat_index]
                row_scores = flat_scores[row_flat_index]
                row_grad = flat_grad[row_flat_index]
                k = int(row_mask.to(torch.int32).sum().item())
                width = row_mask.numel()
                if k <= 0 or k >= width:
                    continue

                bw = min(int(boundary_width), k, width - k)
                if bw <= 0:
                    continue

                selected_scores = row_scores.masked_fill(~row_mask, float("inf"))
                inside_low = torch.topk(selected_scores, k=bw, largest=False, sorted=True).indices
                outside_scores = row_scores.masked_fill(row_mask, float("-inf"))
                outside_high = torch.topk(outside_scores, k=bw, largest=True, sorted=True).indices

                batch_coord, head_coord, query_coord = unflatten_row_index(row_flat_index, prefix_shape)

                for i_idx in range(bw):
                    for j_idx in range(bw):
                        idx_out = int(inside_low[i_idx].item())
                        idx_in = int(outside_high[j_idx].item())
                        swapped_mask = force_swap(capture.hard_mask, row_flat_index, idx_out, idx_in)

                        with torch.no_grad():
                            with forced_block_mask(model, bidx, swapped_mask):
                                swap_logits = model(images)
                            swap_loss = F.cross_entropy(swap_logits, labels).item()

                        benefit = baseline_loss_value - float(swap_loss)
                        pred_score = float((row_grad[idx_out] - row_grad[idx_in]).item())
                        pred_good = bool(pred_score > 0.0)
                        correct = bool(benefit > 0.0)

                        records.append(
                            SwapRecord(
                                surrogate_label=spec.label,
                                block_index=bidx,
                                row_flat_index=row_flat_index,
                                batch_loader_index=batch_loader_index,
                                batch_index=int(batch_coord),
                                head_index=int(head_coord),
                                query_index=int(query_coord),
                                swap_out_index=idx_out,
                                swap_in_index=idx_in,
                                score_out=float(row_scores[idx_out].item()),
                                score_in=float(row_scores[idx_in].item()),
                                pred_score=pred_score,
                                pred_good=pred_good,
                                benefit=benefit,
                                correct=correct,
                            )
                        )

    summary = summarize_surrogate(spec.label, records)
    baseline_loss_mean = float(sum(baseline_losses) / len(baseline_losses)) if baseline_losses else math.nan
    return summary, records, baseline_loss_mean


def fmt(v: float | None) -> str:
    return "nan" if v is None else f"{v:.6f}"


def delta_fmt(v: float | None, ref: float | None) -> str:
    if v is None or ref is None:
        return "nan"
    return f"{(v - ref):+.6f}"


def build_summary_map(items: list[SurrogateSummary]) -> dict[str, SurrogateSummary]:
    return {item.label: item for item in items}


def ordered_labels() -> list[str]:
    return ["kth-raw", "kth-norm", "sigmoid-topk", "subset-gibbs"]


def dense_rank_desc(metric_by_label: dict[str, float | None]) -> dict[str, int | None]:
    valid = [(label, value) for label, value in metric_by_label.items() if value is not None]
    if not valid:
        return {label: None for label in metric_by_label}
    unique_values = sorted({float(value) for _, value in valid}, reverse=True)
    value_to_rank = {value: rank + 1 for rank, value in enumerate(unique_values)}
    ranked = {label: value_to_rank[float(value)] for label, value in valid}
    for label in metric_by_label:
        ranked.setdefault(label, None)
    return ranked


def metric_worst(metric_by_label: dict[str, float | None]) -> float | None:
    valid = [float(value) for value in metric_by_label.values() if value is not None]
    if not valid:
        return None
    return min(valid)


def sort_labels_for_compact_table(
    label_to_item: dict[str, Any],
    ap_rank: dict[str, int | None],
    capture_rank: dict[str, int | None],
    spearman_rank: dict[str, int | None],
) -> list[str]:
    def mean_rank(label: str) -> float:
        ranks = [r for r in [ap_rank.get(label), capture_rank.get(label), spearman_rank.get(label)] if r is not None]
        if not ranks:
            return float("inf")
        return sum(ranks) / len(ranks)

    preferred_order = {label: idx for idx, label in enumerate(ordered_labels())}
    labels = list(label_to_item.keys())
    labels.sort(key=lambda label: (mean_rank(label), preferred_order.get(label, 999), label))
    return labels


def print_compact_overall(summaries: list[SurrogateSummary]) -> None:
    label_to_item = build_summary_map(summaries)
    ap_by_label = {item.label: item.average_precision for item in summaries}
    capture_by_label = {item.label: item.top_truek_benefit_capture for item in summaries}
    spearman_by_label = {item.label: item.benefit_spearman for item in summaries}

    ap_rank = dense_rank_desc(ap_by_label)
    capture_rank = dense_rank_desc(capture_by_label)
    spearman_rank = dense_rank_desc(spearman_by_label)

    ap_worst = metric_worst(ap_by_label)
    capture_worst = metric_worst(capture_by_label)
    spearman_worst = metric_worst(spearman_by_label)

    labels = sort_labels_for_compact_table(label_to_item, ap_rank, capture_rank, spearman_rank)

    print("overall_compact")
    print(
        "label ap ap_vs_worst ap_rank capture capture_vs_worst capture_rank spearman spearman_vs_worst spearman_rank mean_rank"
    )
    for label in labels:
        item = label_to_item[label]
        mean_rank_values = [r for r in [ap_rank[label], capture_rank[label], spearman_rank[label]] if r is not None]
        mean_rank = None if not mean_rank_values else sum(mean_rank_values) / len(mean_rank_values)
        print(
            f"{item.label:>13s} "
            f"{fmt(item.average_precision):>9s} "
            f"{delta_fmt(item.average_precision, ap_worst):>11s} "
            f"{str(ap_rank[label]) if ap_rank[label] is not None else 'nan':>7s} "
            f"{fmt(item.top_truek_benefit_capture):>9s} "
            f"{delta_fmt(item.top_truek_benefit_capture, capture_worst):>16s} "
            f"{str(capture_rank[label]) if capture_rank[label] is not None else 'nan':>12s} "
            f"{fmt(item.benefit_spearman):>9s} "
            f"{delta_fmt(item.benefit_spearman, spearman_worst):>17s} "
            f"{str(spearman_rank[label]) if spearman_rank[label] is not None else 'nan':>13s} "
            f"{fmt(mean_rank):>9s}"
        )


def print_compact_per_block(block_summaries: list[SurrogateBlockSummary]) -> None:
    if not block_summaries:
        return
    per_block: dict[int, dict[str, SurrogateBlockSummary]] = {}
    for item in block_summaries:
        per_block.setdefault(int(item.block_index), {})[item.label] = item

    print()
    print("per_block_compact")
    print(
        "block label ap ap_vs_worst ap_rank capture capture_vs_worst capture_rank spearman spearman_vs_worst spearman_rank mean_rank"
    )
    for block_index in sorted(per_block.keys()):
        label_to_item = per_block[block_index]
        ap_by_label = {label: item.average_precision for label, item in label_to_item.items()}
        capture_by_label = {label: item.top_truek_benefit_capture for label, item in label_to_item.items()}
        spearman_by_label = {label: item.benefit_spearman for label, item in label_to_item.items()}

        ap_rank = dense_rank_desc(ap_by_label)
        capture_rank = dense_rank_desc(capture_by_label)
        spearman_rank = dense_rank_desc(spearman_by_label)

        ap_worst = metric_worst(ap_by_label)
        capture_worst = metric_worst(capture_by_label)
        spearman_worst = metric_worst(spearman_by_label)

        labels = sort_labels_for_compact_table(label_to_item, ap_rank, capture_rank, spearman_rank)
        for label in labels:
            item = label_to_item[label]
            mean_rank_values = [r for r in [ap_rank[label], capture_rank[label], spearman_rank[label]] if r is not None]
            mean_rank = None if not mean_rank_values else sum(mean_rank_values) / len(mean_rank_values)
            print(
                f"{item.block_index:>5d} "
                f"{item.label:>13s} "
                f"{fmt(item.average_precision):>9s} "
                f"{delta_fmt(item.average_precision, ap_worst):>11s} "
                f"{str(ap_rank[label]) if ap_rank[label] is not None else 'nan':>7s} "
                f"{fmt(item.top_truek_benefit_capture):>9s} "
                f"{delta_fmt(item.top_truek_benefit_capture, capture_worst):>16s} "
                f"{str(capture_rank[label]) if capture_rank[label] is not None else 'nan':>12s} "
                f"{fmt(item.benefit_spearman):>9s} "
                f"{delta_fmt(item.benefit_spearman, spearman_worst):>17s} "
                f"{str(spearman_rank[label]) if spearman_rank[label] is not None else 'nan':>13s} "
                f"{fmt(mean_rank):>9s}"
            )


def print_detailed_overall(
    summaries: list[SurrogateSummary],
    baseline_loss_means: dict[str, float],
) -> None:
    print()
    print("overall_detailed")
    print(
        "label baseline_loss_mean rows swaps pos_ratio precision recall average_precision "
        "top_truek_precision top_truek_benefit_capture benefit_spearman benefit_pearson"
    )
    for item in summaries:
        print(
            f"{item.label:>13s} "
            f"{baseline_loss_means[item.label]:>18.6f} "
            f"{item.rows_analyzed:>4d} "
            f"{item.swaps_analyzed:>5d} "
            f"{item.pos_ratio:>9.6f} "
            f"{fmt(item.precision):>9s} "
            f"{fmt(item.recall):>8s} "
            f"{fmt(item.average_precision):>17s} "
            f"{fmt(item.top_truek_precision):>19s} "
            f"{fmt(item.top_truek_benefit_capture):>24s} "
            f"{fmt(item.benefit_spearman):>16s} "
            f"{fmt(item.benefit_pearson):>15s}"
        )


def print_detailed_per_block(block_summaries: list[SurrogateBlockSummary]) -> None:
    if not block_summaries:
        return
    print()
    print("per_block_detailed")
    print(
        "label block rows swaps pos_ratio precision recall average_precision "
        "top_truek_precision top_truek_benefit_capture benefit_spearman benefit_pearson"
    )
    for item in block_summaries:
        print(
            f"{item.label:>13s} "
            f"{item.block_index:>5d} "
            f"{item.rows_analyzed:>4d} "
            f"{item.swaps_analyzed:>5d} "
            f"{item.pos_ratio:>9.6f} "
            f"{fmt(item.precision):>9s} "
            f"{fmt(item.recall):>8s} "
            f"{fmt(item.average_precision):>17s} "
            f"{fmt(item.top_truek_precision):>19s} "
            f"{fmt(item.top_truek_benefit_capture):>24s} "
            f"{fmt(item.benefit_spearman):>16s} "
            f"{fmt(item.benefit_pearson):>15s}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    script_args, train_args = parse_combined_args()
    specs = build_specs(script_args)

    set_deterministic(int(train_args.seed))
    device = torch.device(script_args.device)

    train_loader, valid_loader, test_loader, train_eval_loader, _, transform = load_dataset(train_args)
    init_loader = select_loader(script_args.init_split, train_eval_loader, valid_loader, test_loader)
    run_loader = select_loader(script_args.split, train_eval_loader, valid_loader, test_loader)
    batches = get_batches(run_loader, script_args.batch_index, script_args.num_batches)

    if script_args.print_config:
        print("=== parsed train/model args ===")
        for key, value in sorted(vars(train_args).items()):
            print(f"{key}={value}")
        print()

    summaries: list[SurrogateSummary] = []
    block_summaries: list[SurrogateBlockSummary] = []
    all_records: list[SwapRecord] = []
    baseline_loss_means: dict[str, float] = {}

    for spec in tqdm(specs, desc="surrogates", total=len(specs)):
        summary, records, baseline_loss_mean = analyze_spec(
            spec=spec,
            train_args=train_args,
            init_loader=init_loader,
            batches=batches,
            transform=transform,
            device=device,
            block_index=script_args.block_index,
            max_rows_per_block=int(script_args.max_rows_per_block),
            boundary_width=int(script_args.boundary_width),
        )
        summaries.append(summary)
        all_records.extend(records)
        baseline_loss_means[spec.label] = baseline_loss_mean
        if script_args.block_index is None:
            block_to_records: dict[int, list[SwapRecord]] = {}
            for record in records:
                block_to_records.setdefault(int(record.block_index), []).append(record)
            for bidx in sorted(block_to_records.keys()):
                block_summaries.append(summarize_surrogate_block(spec.label, bidx, block_to_records[bidx]))

    print(f"split={script_args.split}")
    print(f"batch_index={script_args.batch_index}")
    print(f"num_batches={script_args.num_batches}")
    print(f"block_index={'all' if script_args.block_index is None else script_args.block_index}")
    print(f"max_rows_per_block={script_args.max_rows_per_block}")
    print(f"boundary_width={script_args.boundary_width}")
    print()
    print("Primary comparison metrics: average_precision, top_truek_benefit_capture, benefit_spearman.")
    print("Compact tables report each surrogate's absolute value, gain vs the worst method, per-metric rank, and mean rank.")
    print()
    print_compact_overall(summaries)

    if script_args.block_index is None and block_summaries:
        print_compact_per_block(block_summaries)

    if script_args.print_detailed:
        print_detailed_overall(summaries, baseline_loss_means)
        if script_args.block_index is None and block_summaries:
            print_detailed_per_block(block_summaries)

    if script_args.print_record_limit > 0:
        print()
        print("sample records")
        print("label block loader_batch row out_idx in_idx pred_score benefit correct")
        for record in all_records[: script_args.print_record_limit]:
            print(
                f"{record.surrogate_label:>13s} "
                f"{record.block_index:>5d} "
                f"{record.batch_loader_index:>12d} "
                f"{record.row_flat_index:>3d} "
                f"{record.swap_out_index:>7d} "
                f"{record.swap_in_index:>6d} "
                f"{record.pred_score:>10.6f} "
                f"{record.benefit:>10.6f} "
                f"{int(record.correct):>7d}"
            )

    if script_args.output_json is not None:
        payload = {
            "split": script_args.split,
            "batch_index": int(script_args.batch_index),
            "num_batches": int(script_args.num_batches),
            "block_index": script_args.block_index,
            "max_rows_per_block": int(script_args.max_rows_per_block),
            "boundary_width": int(script_args.boundary_width),
            "model_args": vars(train_args),
            "surrogate_specs": [
                {
                    "label": spec.label,
                    "checkpoint": str(spec.checkpoint),
                    "topk_surrogate_mode": spec.topk_surrogate_mode,
                    "topk_kth_normalize_soft_mask": bool(spec.topk_kth_normalize_soft_mask),
                }
                for spec in specs
            ],
            "baseline_loss_means": baseline_loss_means,
            "summaries": [asdict(x) for x in summaries],
            "block_summaries": [asdict(x) for x in block_summaries],
            "records": [asdict(x) for x in all_records],
        }
        script_args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with script_args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if script_args.output_table_prefix is not None:
        prefix = script_args.output_table_prefix.expanduser()
        overall_rows = []
        for item in summaries:
            row = asdict(item)
            row["baseline_loss_mean"] = baseline_loss_means[item.label]
            overall_rows.append(row)

        block_rows = [asdict(item) for item in block_summaries]
        record_rows = [asdict(item) for item in all_records]

        write_csv(
            Path(f"{prefix}_overall.csv"),
            overall_rows,
            [
                "label",
                "baseline_loss_mean",
                "rows_analyzed",
                "swaps_analyzed",
                "pos_ratio",
                "precision",
                "recall",
                "average_precision",
                "top_truek_precision",
                "top_truek_benefit_capture",
                "benefit_spearman",
                "benefit_pearson",
                "pred_score_mean",
                "benefit_mean",
            ],
        )
        write_csv(
            Path(f"{prefix}_per_block.csv"),
            block_rows,
            [
                "label",
                "block_index",
                "rows_analyzed",
                "swaps_analyzed",
                "pos_ratio",
                "precision",
                "recall",
                "average_precision",
                "top_truek_precision",
                "top_truek_benefit_capture",
                "benefit_spearman",
                "benefit_pearson",
            ],
        )
        write_csv(
            Path(f"{prefix}_records.csv"),
            record_rows,
            [
                "surrogate_label",
                "block_index",
                "row_flat_index",
                "batch_loader_index",
                "batch_index",
                "head_index",
                "query_index",
                "swap_out_index",
                "swap_in_index",
                "score_out",
                "score_in",
                "pred_score",
                "pred_good",
                "benefit",
                "correct",
            ],
        )


if __name__ == "__main__":
    main()
