from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_topk_boundary_swap_surrogates import (  # noqa: E402
    BlockCapture,
    SwapRecord,
    attach_captures,
    enable_surrogate_backward,
    forced_block_mask,
    get_batches,
    infer_and_report_mismatches,
    load_checkpoint_state,
    ordered_labels,
    restore_forwards,
    restore_proxies,
    select_loader,
    unflatten_row_index,
    write_csv,
)
from data_pipeline import load_dataset  # noqa: E402
from train_logic_vit_tiny import (  # noqa: E402
    build_model,
    initialize_lazy_modules,
    parse_args as parse_train_args,
    set_attention_soft_train_temperatures,
    set_deterministic,
)


@dataclass
class BackwardSpec:
    label: str
    topk_surrogate_mode: str
    topk_kth_normalize_soft_mask: bool


@dataclass
class BackwardSummary:
    label: str
    rows_analyzed: int
    swaps_analyzed: int
    beneficial_swaps: int
    harmful_swaps: int
    neutral_swaps: int
    beneficial_swap_rate: float
    top_m_budget: int
    row_top1_hit_rate: float | None
    row_top1_benefit_mean: float | None
    row_top1_regret_mean: float | None
    macro_row_auc: float | None
    macro_row_auc_rows: int
    actionable_row_spearman_mean: float | None
    actionable_row_spearman_rows: int
    row_soft_expected_benefit_mean: float | None
    row_soft_positive_mass_mean: float | None
    actionable_row_soft_positive_mass_mean: float | None
    actionable_row_soft_expected_benefit_mean: float | None
    actionable_row_soft_gap_closed_mean: float | None
    actionable_row_soft_benefit_recall_mean: float | None
    actionable_row_oracle_overlap_mean: float | None
    actionable_row_oracle_js_similarity_mean: float | None
    actionable_row_count: int
    swap_sign_acc: float | None
    swap_auc: float | None
    top_m_precision: float | None
    benefit_capture: float | None
    positive_negative_margin: float | None
    score_benefit_spearman: float | None
    score_benefit_pearson: float | None
    pred_score_mean: float
    benefit_mean: float


@dataclass
class BackwardBlockSummary:
    label: str
    block_index: int
    rows_analyzed: int
    swaps_analyzed: int
    beneficial_swaps: int
    harmful_swaps: int
    neutral_swaps: int
    beneficial_swap_rate: float
    top_m_budget: int
    row_top1_hit_rate: float | None
    row_top1_benefit_mean: float | None
    row_top1_regret_mean: float | None
    macro_row_auc: float | None
    macro_row_auc_rows: int
    actionable_row_spearman_mean: float | None
    actionable_row_spearman_rows: int
    row_soft_expected_benefit_mean: float | None
    row_soft_positive_mass_mean: float | None
    actionable_row_soft_positive_mass_mean: float | None
    actionable_row_soft_expected_benefit_mean: float | None
    actionable_row_soft_gap_closed_mean: float | None
    actionable_row_soft_benefit_recall_mean: float | None
    actionable_row_oracle_overlap_mean: float | None
    actionable_row_oracle_js_similarity_mean: float | None
    actionable_row_count: int
    swap_sign_acc: float | None
    swap_auc: float | None
    top_m_precision: float | None
    benefit_capture: float | None
    positive_negative_margin: float | None
    score_benefit_spearman: float | None
    score_benefit_pearson: float | None


PRIMARY_METRICS = (
    ("actionable_row_oracle_overlap_mean", "oracle_overlap", True),
    ("actionable_row_oracle_js_similarity_mean", "oracle_js_sim", True),
    ("macro_row_auc", "row_auc", True),
    ("actionable_row_spearman_mean", "row_spearman", True),
)


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Compare top-k backward estimators on the same checkpoint and same hard top-k forward states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Single checkpoint used for all surrogate estimators")
    parser.add_argument("--split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--init-split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--block-index", type=int, default=None)
    parser.add_argument("--max-rows-per-block", type=int, default=10)
    parser.add_argument("--boundary-width", type=int, default=4)
    parser.add_argument("--benefit-eps", type=float, default=1e-6, help="Ignore |benefit| <= eps when defining beneficial vs harmful swaps")
    parser.add_argument("--score-eps", type=float, default=0.0, help="pred_score > eps counts as the surrogate favoring that swap")
    parser.add_argument("--top-m-count", type=int, default=32, help="Evaluate precision among the top-m scored swaps")
    parser.add_argument("--row-softmax-temp", type=float, default=1.0, help="Temperature for row-wise softmax distribution metrics")
    parser.add_argument("--row-score-norm-eps", type=float, default=1e-6, help="Stability epsilon for per-row score standardization")
    parser.add_argument("--swap-forward-batch-size", type=int, default=32, help="How many swap masks to evaluate together in one forward pass")
    parser.add_argument("--print-record-limit", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-table-prefix", type=Path, default=None)
    parser.add_argument("--print-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-detailed", action=argparse.BooleanOptionalAction, default=False)
    script_args, remaining = parser.parse_known_args(argv)

    saved_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        train_args = parse_train_args()
    finally:
        sys.argv = saved_argv

    return script_args, train_args


def build_specs() -> list[BackwardSpec]:
    return [
        BackwardSpec("kth-raw", "kth", False),
        BackwardSpec("kth-norm", "kth", True),
        BackwardSpec("sigmoid-topk", "sigmoid-topk", True),
        BackwardSpec("subset-gibbs", "subset-gibbs", True),
    ]


def set_model_backward_spec(model, spec: BackwardSpec) -> None:
    for block in model.blocks:
        packed_topk = block.attn.logic_attention.packed_topk
        packed_topk.topk_surrogate_mode = spec.topk_surrogate_mode
        packed_topk.topk_kth_normalize_soft_mask = bool(spec.topk_kth_normalize_soft_mask)


def select_boundary_candidates(
    row_mask: torch.Tensor,
    row_scores: torch.Tensor,
    boundary_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_mask = row_mask.to(torch.bool)
    width = int(row_mask.numel())
    limit = int(row_mask.to(torch.int32).sum().item())
    band = min(int(boundary_width), limit, width - limit)
    if band <= 0:
        empty = torch.empty(0, dtype=torch.long, device=row_mask.device)
        return empty, empty

    selected_scores = row_scores.masked_fill(~row_mask, float("inf"))
    inside_low = torch.topk(selected_scores, k=band, largest=False, sorted=True).indices.to(torch.long)

    outside_scores = row_scores.masked_fill(row_mask, float("-inf"))
    outside_high = torch.topk(outside_scores, k=band, largest=True, sorted=True).indices.to(torch.long)
    return inside_low, outside_high


def flatten_row_coords(coords: tuple[int, ...], shape: torch.Size) -> int:
    if len(coords) != len(shape):
        raise ValueError("coords and shape must have the same rank")
    flat_index = 0
    for coord, size in zip(coords, shape):
        flat_index = flat_index * int(size) + int(coord)
    return flat_index


def build_forced_masks_for_swaps(
    base_mask: torch.Tensor,
    row_flat_index: int,
    swap_out_indices: torch.Tensor,
    swap_in_indices: torch.Tensor,
) -> torch.Tensor:
    forced_masks = base_mask.unsqueeze(0).expand(int(swap_out_indices.shape[0]), *base_mask.shape).clone()
    flat = forced_masks.reshape(int(swap_out_indices.shape[0]), -1, forced_masks.shape[-1])
    row_masks = flat[:, row_flat_index, :]
    row_ids = torch.arange(int(swap_out_indices.shape[0]), device=row_masks.device)
    row_masks[row_ids, swap_out_indices] = False
    row_masks[row_ids, swap_in_indices] = True
    flat[:, row_flat_index, :] = row_masks
    return forced_masks


def evaluate_forced_masks_chunk(
    model,
    block_index: int,
    forced_masks: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    chunk_size = int(forced_masks.shape[0])
    repeated_images = images.repeat(chunk_size, 1, 1, 1)
    repeated_labels = labels.repeat(chunk_size)
    forced_mask = forced_masks.reshape(-1, *forced_masks.shape[2:])
    with torch.inference_mode():
        with forced_block_mask(model, block_index, forced_mask):
            logits = model(repeated_images)
            loss_values = F.cross_entropy(logits, repeated_labels, reduction="none")
    return loss_values.reshape(chunk_size, int(labels.shape[0]))


def evaluate_swap_benefits_for_row(
    model,
    block_index: int,
    sample_mask: torch.Tensor,
    sample_row_flat_index: int,
    swap_out_indices: torch.Tensor,
    swap_in_indices: torch.Tensor,
    sample_images: torch.Tensor,
    sample_labels: torch.Tensor,
    baseline_loss_value: float,
    swap_forward_batch_size: int,
) -> torch.Tensor:
    benefits: list[torch.Tensor] = []
    total_swaps = int(swap_out_indices.shape[0])
    for start in range(0, total_swaps, int(swap_forward_batch_size)):
        end = min(start + int(swap_forward_batch_size), total_swaps)
        forced_masks = build_forced_masks_for_swaps(
            base_mask=sample_mask,
            row_flat_index=sample_row_flat_index,
            swap_out_indices=swap_out_indices[start:end],
            swap_in_indices=swap_in_indices[start:end],
        )
        swap_loss_values = evaluate_forced_masks_chunk(
            model=model,
            block_index=block_index,
            forced_masks=forced_masks,
            images=sample_images,
            labels=sample_labels,
        ).reshape(-1)
        benefits.append(torch.full_like(swap_loss_values, float(baseline_loss_value)) - swap_loss_values)
    return torch.cat(benefits, dim=0) if benefits else torch.empty(0, dtype=torch.float32, device=sample_images.device)


def collect_capture_grads(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    spec: BackwardSpec,
) -> tuple[list[BlockCapture], float, torch.Tensor]:
    set_model_backward_spec(model, spec)
    model.eval()
    enable_surrogate_backward(model)
    captures, fwd_originals, proxy_originals = attach_captures(model)
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss_values = F.cross_entropy(logits, labels, reduction="none")
    loss = loss_values.mean()
    loss.backward()
    baseline_loss_value = float(loss.item())
    restore_forwards(fwd_originals)
    restore_proxies(proxy_originals)
    for capture in captures:
        capture.finalize_backward()
    model.zero_grad(set_to_none=True)
    model.eval()
    return captures, baseline_loss_value, loss_values.detach()


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
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    return corr_pearson(x_rank, y_rank)


def average_ranks(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError("average_ranks expects a 1D tensor")
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks = torch.empty_like(x, dtype=torch.float32)

    start = 0
    total = int(sorted_x.numel())
    while start < total:
        end = start + 1
        while end < total and bool(sorted_x[end] == sorted_x[start]):
            end += 1
        avg_rank = 0.5 * float(start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def compute_swap_auc(scores: torch.Tensor, beneficial_mask: torch.Tensor, harmful_mask: torch.Tensor) -> float | None:
    valid = torch.logical_or(beneficial_mask, harmful_mask)
    if int(valid.sum().item()) <= 0:
        return None

    valid_scores = scores[valid]
    valid_labels = beneficial_mask[valid].to(torch.bool)
    positives = int(valid_labels.sum().item())
    negatives = int((~valid_labels).sum().item())
    if positives <= 0 or negatives <= 0:
        return None

    ranks = average_ranks(valid_scores)
    rank_sum = float(ranks[valid_labels].sum().item())
    auc = (rank_sum - (positives * (positives + 1) / 2.0)) / float(positives * negatives)
    return float(auc)


def compute_swap_sign_acc(
    scores: torch.Tensor,
    beneficial_mask: torch.Tensor,
    harmful_mask: torch.Tensor,
    score_eps: float,
) -> float | None:
    valid = torch.logical_or(beneficial_mask, harmful_mask)
    if int(valid.sum().item()) <= 0:
        return None
    predicted_positive = scores[valid] > float(score_eps)
    target_positive = beneficial_mask[valid]
    return float((predicted_positive == target_positive).to(torch.float32).mean().item())


def compute_top_m_precision(scores: torch.Tensor, beneficial_mask: torch.Tensor, top_m_count: int) -> tuple[int, float | None]:
    if scores.numel() == 0:
        return 0, None
    budget = min(int(top_m_count), int(scores.numel()))
    if budget <= 0:
        return 0, None
    order = torch.argsort(scores, descending=True)[:budget]
    precision = float(beneficial_mask[order].to(torch.float32).mean().item())
    return budget, precision


def compute_benefit_capture(scores: torch.Tensor, benefits: torch.Tensor, beneficial_mask: torch.Tensor) -> float | None:
    positive_budget = int(beneficial_mask.sum().item())
    if positive_budget <= 0:
        return None
    positive_benefits = benefits.clamp_min(0.0) * beneficial_mask.to(torch.float32)
    total_positive_benefit = float(positive_benefits.sum().item())
    if total_positive_benefit <= 0.0:
        return None
    order = torch.argsort(scores, descending=True)[:positive_budget]
    captured = float(positive_benefits[order].sum().item())
    return captured / total_positive_benefit


def compute_positive_negative_margin(
    scores: torch.Tensor,
    beneficial_mask: torch.Tensor,
    harmful_mask: torch.Tensor,
) -> float | None:
    positives = scores[beneficial_mask]
    negatives = scores[harmful_mask]
    if positives.numel() == 0 or negatives.numel() == 0:
        return None
    return float(positives.mean().item() - negatives.mean().item())


def build_row_soft_distribution(scores: torch.Tensor, temperature: float, score_norm_eps: float) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("build_row_soft_distribution expects a 1D score tensor")
    centered = scores - scores.mean()
    scale = centered.std(unbiased=False)
    if float(scale.item()) > float(score_norm_eps):
        normalized = centered / scale
    else:
        normalized = centered
    return torch.softmax(normalized / float(temperature), dim=0)


def compute_js_similarity(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape:
        raise ValueError("compute_js_similarity expects matching 1D tensors")
    p_safe = p.clamp_min(float(eps))
    q_safe = q.clamp_min(float(eps))
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    m_safe = (0.5 * (p_safe + q_safe)).clamp_min(float(eps))
    js_div = 0.5 * torch.sum(p_safe * (torch.log(p_safe) - torch.log(m_safe)))
    js_div = js_div + 0.5 * torch.sum(q_safe * (torch.log(q_safe) - torch.log(m_safe)))
    js_sim = 1.0 - float(js_div.item()) / float(math.log(2.0))
    return max(0.0, min(1.0, js_sim))


def build_row_record_groups(records: list[SwapRecord]) -> dict[tuple[int, int, int, int, int], list[SwapRecord]]:
    grouped: dict[tuple[int, int, int, int, int], list[SwapRecord]] = {}
    for record in records:
        key = (
            int(record.batch_loader_index),
            int(record.block_index),
            int(record.batch_index),
            int(record.head_index),
            int(record.query_index),
        )
        grouped.setdefault(key, []).append(record)
    return grouped


def compute_row_level_metrics(
    records: list[SwapRecord],
    benefit_eps: float,
    row_softmax_temp: float,
    row_score_norm_eps: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    int,
    float | None,
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    int,
]:
    row_groups = build_row_record_groups(records)
    if not row_groups:
        return None, None, None, None, 0, None, 0, None, None, None, None, None, None, None, None, 0

    top1_hits: list[float] = []
    top1_benefits: list[float] = []
    top1_regrets: list[float] = []
    row_aucs: list[float] = []
    actionable_row_spearmans: list[float] = []
    soft_expected_benefits: list[float] = []
    soft_positive_masses: list[float] = []
    actionable_soft_positive_masses: list[float] = []
    actionable_soft_expected_benefits: list[float] = []
    actionable_soft_gap_closed: list[float] = []
    actionable_soft_benefit_recalls: list[float] = []
    actionable_oracle_overlaps: list[float] = []
    actionable_oracle_js_similarities: list[float] = []

    for row_records in row_groups.values():
        scores = torch.tensor([record.pred_score for record in row_records], dtype=torch.float32)
        benefits = torch.tensor([record.benefit for record in row_records], dtype=torch.float32)

        chosen_index = int(torch.argmax(scores).item())
        chosen_benefit = float(benefits[chosen_index].item())
        best_true_benefit = float(benefits.max().item())

        top1_hits.append(1.0 if chosen_benefit > float(benefit_eps) else 0.0)
        top1_benefits.append(chosen_benefit)
        top1_regrets.append(max(0.0, best_true_benefit - chosen_benefit))

        beneficial_mask = benefits > float(benefit_eps)
        harmful_mask = benefits < -float(benefit_eps)
        row_auc = compute_swap_auc(scores, beneficial_mask, harmful_mask)
        if row_auc is not None:
            row_aucs.append(row_auc)
        row_spearman = corr_spearman(scores, benefits)

        probs = build_row_soft_distribution(
            scores,
            temperature=float(row_softmax_temp),
            score_norm_eps=float(row_score_norm_eps),
        )
        positive_benefits = benefits.clamp_min(0.0)
        soft_expected_benefits.append(float((probs * benefits).sum().item()))
        soft_positive_masses.append(float(probs[beneficial_mask].sum().item()))
        best_positive_benefit = float(positive_benefits.max().item())
        total_positive_benefit = float(positive_benefits.sum().item())
        if total_positive_benefit > float(benefit_eps):
            oracle_distribution = positive_benefits / float(total_positive_benefit)
            if row_spearman is not None:
                actionable_row_spearmans.append(row_spearman)
            actionable_soft_positive_masses.append(float(probs[beneficial_mask].sum().item()))
            actionable_soft_expected_benefits.append(float((probs * benefits).sum().item()))
            actionable_soft_gap_closed.append(float((probs * positive_benefits).sum().item() / best_positive_benefit))
            actionable_soft_benefit_recalls.append(float((probs * positive_benefits).sum().item() / total_positive_benefit))
            actionable_oracle_overlaps.append(float(torch.minimum(probs, oracle_distribution).sum().item()))
            actionable_oracle_js_similarities.append(compute_js_similarity(probs, oracle_distribution))

    hit_rate = float(torch.tensor(top1_hits, dtype=torch.float32).mean().item()) if top1_hits else None
    benefit_mean = float(torch.tensor(top1_benefits, dtype=torch.float32).mean().item()) if top1_benefits else None
    regret_mean = float(torch.tensor(top1_regrets, dtype=torch.float32).mean().item()) if top1_regrets else None
    macro_row_auc = float(torch.tensor(row_aucs, dtype=torch.float32).mean().item()) if row_aucs else None
    actionable_row_spearman_mean = (
        float(torch.tensor(actionable_row_spearmans, dtype=torch.float32).mean().item())
        if actionable_row_spearmans
        else None
    )
    soft_expected_benefit_mean = (
        float(torch.tensor(soft_expected_benefits, dtype=torch.float32).mean().item()) if soft_expected_benefits else None
    )
    soft_positive_mass_mean = (
        float(torch.tensor(soft_positive_masses, dtype=torch.float32).mean().item()) if soft_positive_masses else None
    )
    actionable_soft_positive_mass_mean = (
        float(torch.tensor(actionable_soft_positive_masses, dtype=torch.float32).mean().item())
        if actionable_soft_positive_masses
        else None
    )
    actionable_soft_expected_benefit_mean = (
        float(torch.tensor(actionable_soft_expected_benefits, dtype=torch.float32).mean().item())
        if actionable_soft_expected_benefits
        else None
    )
    actionable_soft_gap_closed_mean = (
        float(torch.tensor(actionable_soft_gap_closed, dtype=torch.float32).mean().item())
        if actionable_soft_gap_closed
        else None
    )
    actionable_soft_benefit_recall_mean = (
        float(torch.tensor(actionable_soft_benefit_recalls, dtype=torch.float32).mean().item())
        if actionable_soft_benefit_recalls
        else None
    )
    actionable_oracle_overlap_mean = (
        float(torch.tensor(actionable_oracle_overlaps, dtype=torch.float32).mean().item())
        if actionable_oracle_overlaps
        else None
    )
    actionable_oracle_js_similarity_mean = (
        float(torch.tensor(actionable_oracle_js_similarities, dtype=torch.float32).mean().item())
        if actionable_oracle_js_similarities
        else None
    )
    return (
        hit_rate,
        benefit_mean,
        regret_mean,
        macro_row_auc,
        len(row_aucs),
        actionable_row_spearman_mean,
        len(actionable_row_spearmans),
        soft_expected_benefit_mean,
        soft_positive_mass_mean,
        actionable_soft_positive_mass_mean,
        actionable_soft_expected_benefit_mean,
        actionable_soft_gap_closed_mean,
        actionable_soft_benefit_recall_mean,
        actionable_oracle_overlap_mean,
        actionable_oracle_js_similarity_mean,
        len(actionable_soft_gap_closed),
    )


def summarize_backward_records(
    label: str,
    records: list[SwapRecord],
    top_m_count: int,
    benefit_eps: float,
    score_eps: float,
    row_softmax_temp: float,
    row_score_norm_eps: float,
) -> BackwardSummary:
    rows = len({(r.batch_loader_index, r.block_index, r.batch_index, r.head_index, r.query_index) for r in records})
    swaps = len(records)
    if swaps == 0:
        return BackwardSummary(
            label=label,
            rows_analyzed=0,
            swaps_analyzed=0,
            beneficial_swaps=0,
            harmful_swaps=0,
            neutral_swaps=0,
            beneficial_swap_rate=0.0,
            top_m_budget=0,
            row_top1_hit_rate=None,
            row_top1_benefit_mean=None,
            row_top1_regret_mean=None,
            macro_row_auc=None,
            macro_row_auc_rows=0,
            actionable_row_spearman_mean=None,
            actionable_row_spearman_rows=0,
            row_soft_expected_benefit_mean=None,
            row_soft_positive_mass_mean=None,
            actionable_row_soft_positive_mass_mean=None,
            actionable_row_soft_expected_benefit_mean=None,
            actionable_row_soft_gap_closed_mean=None,
            actionable_row_soft_benefit_recall_mean=None,
            actionable_row_oracle_overlap_mean=None,
            actionable_row_oracle_js_similarity_mean=None,
            actionable_row_count=0,
            swap_sign_acc=None,
            swap_auc=None,
            top_m_precision=None,
            benefit_capture=None,
            positive_negative_margin=None,
            score_benefit_spearman=None,
            score_benefit_pearson=None,
            pred_score_mean=0.0,
            benefit_mean=0.0,
        )

    scores = torch.tensor([r.pred_score for r in records], dtype=torch.float32)
    benefits = torch.tensor([r.benefit for r in records], dtype=torch.float32)

    beneficial_mask = benefits > float(benefit_eps)
    harmful_mask = benefits < -float(benefit_eps)
    neutral_mask = ~(beneficial_mask | harmful_mask)

    top_m_budget, top_m_precision = compute_top_m_precision(scores, beneficial_mask, top_m_count)
    (
        row_top1_hit_rate,
        row_top1_benefit_mean,
        row_top1_regret_mean,
        macro_row_auc,
        macro_row_auc_rows,
        actionable_row_spearman_mean,
        actionable_row_spearman_rows,
        row_soft_expected_benefit_mean,
        row_soft_positive_mass_mean,
        actionable_row_soft_positive_mass_mean,
        actionable_row_soft_expected_benefit_mean,
        actionable_row_soft_gap_closed_mean,
        actionable_row_soft_benefit_recall_mean,
        actionable_row_oracle_overlap_mean,
        actionable_row_oracle_js_similarity_mean,
        actionable_row_count,
    ) = compute_row_level_metrics(
        records,
        benefit_eps,
        row_softmax_temp,
        row_score_norm_eps,
    )

    return BackwardSummary(
        label=label,
        rows_analyzed=rows,
        swaps_analyzed=swaps,
        beneficial_swaps=int(beneficial_mask.sum().item()),
        harmful_swaps=int(harmful_mask.sum().item()),
        neutral_swaps=int(neutral_mask.sum().item()),
        beneficial_swap_rate=float(beneficial_mask.to(torch.float32).mean().item()),
        top_m_budget=top_m_budget,
        row_top1_hit_rate=row_top1_hit_rate,
        row_top1_benefit_mean=row_top1_benefit_mean,
        row_top1_regret_mean=row_top1_regret_mean,
        macro_row_auc=macro_row_auc,
        macro_row_auc_rows=macro_row_auc_rows,
        actionable_row_spearman_mean=actionable_row_spearman_mean,
        actionable_row_spearman_rows=actionable_row_spearman_rows,
        row_soft_expected_benefit_mean=row_soft_expected_benefit_mean,
        row_soft_positive_mass_mean=row_soft_positive_mass_mean,
        actionable_row_soft_positive_mass_mean=actionable_row_soft_positive_mass_mean,
        actionable_row_soft_expected_benefit_mean=actionable_row_soft_expected_benefit_mean,
        actionable_row_soft_gap_closed_mean=actionable_row_soft_gap_closed_mean,
        actionable_row_soft_benefit_recall_mean=actionable_row_soft_benefit_recall_mean,
        actionable_row_oracle_overlap_mean=actionable_row_oracle_overlap_mean,
        actionable_row_oracle_js_similarity_mean=actionable_row_oracle_js_similarity_mean,
        actionable_row_count=actionable_row_count,
        swap_sign_acc=compute_swap_sign_acc(scores, beneficial_mask, harmful_mask, score_eps),
        swap_auc=compute_swap_auc(scores, beneficial_mask, harmful_mask),
        top_m_precision=top_m_precision,
        benefit_capture=compute_benefit_capture(scores, benefits, beneficial_mask),
        positive_negative_margin=compute_positive_negative_margin(scores, beneficial_mask, harmful_mask),
        score_benefit_spearman=corr_spearman(scores, benefits),
        score_benefit_pearson=corr_pearson(scores, benefits),
        pred_score_mean=float(scores.mean().item()),
        benefit_mean=float(benefits.mean().item()),
    )


def summarize_backward_block(
    label: str,
    block_index: int,
    records: list[SwapRecord],
    top_m_count: int,
    benefit_eps: float,
    score_eps: float,
    row_softmax_temp: float,
    row_score_norm_eps: float,
) -> BackwardBlockSummary:
    summary = summarize_backward_records(
        label,
        records,
        top_m_count,
        benefit_eps,
        score_eps,
        row_softmax_temp,
        row_score_norm_eps,
    )
    return BackwardBlockSummary(
        label=label,
        block_index=block_index,
        rows_analyzed=summary.rows_analyzed,
        swaps_analyzed=summary.swaps_analyzed,
        beneficial_swaps=summary.beneficial_swaps,
        harmful_swaps=summary.harmful_swaps,
        neutral_swaps=summary.neutral_swaps,
        beneficial_swap_rate=summary.beneficial_swap_rate,
        top_m_budget=summary.top_m_budget,
        row_top1_hit_rate=summary.row_top1_hit_rate,
        row_top1_benefit_mean=summary.row_top1_benefit_mean,
        row_top1_regret_mean=summary.row_top1_regret_mean,
        macro_row_auc=summary.macro_row_auc,
        macro_row_auc_rows=summary.macro_row_auc_rows,
        actionable_row_spearman_mean=summary.actionable_row_spearman_mean,
        actionable_row_spearman_rows=summary.actionable_row_spearman_rows,
        row_soft_expected_benefit_mean=summary.row_soft_expected_benefit_mean,
        row_soft_positive_mass_mean=summary.row_soft_positive_mass_mean,
        actionable_row_soft_positive_mass_mean=summary.actionable_row_soft_positive_mass_mean,
        actionable_row_soft_expected_benefit_mean=summary.actionable_row_soft_expected_benefit_mean,
        actionable_row_soft_gap_closed_mean=summary.actionable_row_soft_gap_closed_mean,
        actionable_row_soft_benefit_recall_mean=summary.actionable_row_soft_benefit_recall_mean,
        actionable_row_oracle_overlap_mean=summary.actionable_row_oracle_overlap_mean,
        actionable_row_oracle_js_similarity_mean=summary.actionable_row_oracle_js_similarity_mean,
        actionable_row_count=summary.actionable_row_count,
        swap_sign_acc=summary.swap_sign_acc,
        swap_auc=summary.swap_auc,
        top_m_precision=summary.top_m_precision,
        benefit_capture=summary.benefit_capture,
        positive_negative_margin=summary.positive_negative_margin,
        score_benefit_spearman=summary.score_benefit_spearman,
        score_benefit_pearson=summary.score_benefit_pearson,
    )


def fmt(v: float | int | None) -> str:
    if v is None:
        return "nan"
    if isinstance(v, int):
        return str(v)
    return f"{v:.6f}"


def delta_fmt(v: float | None, worst: float | None, higher_is_better: bool = True) -> str:
    if v is None or worst is None:
        return "nan"
    delta = float(v - worst) if higher_is_better else float(worst - v)
    return f"{delta:+.6f}"


def dense_rank(metric_by_label: dict[str, float | None], higher_is_better: bool = True) -> dict[str, int | None]:
    valid = [(label, value) for label, value in metric_by_label.items() if value is not None]
    if not valid:
        return {label: None for label in metric_by_label}
    unique_values = sorted({float(value) for _, value in valid}, reverse=higher_is_better)
    value_to_rank = {value: rank + 1 for rank, value in enumerate(unique_values)}
    ranked = {label: value_to_rank[float(value)] for label, value in valid}
    for label in metric_by_label:
        ranked.setdefault(label, None)
    return ranked


def metric_worst(metric_by_label: dict[str, float | None], higher_is_better: bool = True) -> float | None:
    valid = [float(value) for value in metric_by_label.values() if value is not None]
    if not valid:
        return None
    return min(valid) if higher_is_better else max(valid)


def sort_labels_for_compact_table(
    label_to_item: dict[str, BackwardSummary | BackwardBlockSummary],
    rank_maps: list[dict[str, int | None]],
) -> list[str]:
    def mean_rank(label: str) -> float:
        ranks = [rank_map.get(label) for rank_map in rank_maps]
        valid = [rank for rank in ranks if rank is not None]
        if not valid:
            return float("inf")
        return sum(valid) / len(valid)

    preferred_order = {label: idx for idx, label in enumerate(ordered_labels())}
    labels = list(label_to_item.keys())
    labels.sort(key=lambda label: (mean_rank(label), preferred_order.get(label, 999), label))
    return labels


def print_compact_overall(summaries: list[BackwardSummary]) -> None:
    label_to_item = {item.label: item for item in summaries}
    metric_maps = [{item.label: getattr(item, field) for item in summaries} for field, _, _ in PRIMARY_METRICS]
    rank_maps = [
        dense_rank(metric_map, higher_is_better=higher_is_better)
        for metric_map, (_, _, higher_is_better) in zip(metric_maps, PRIMARY_METRICS)
    ]
    worst_values = [
        metric_worst(metric_map, higher_is_better=higher_is_better)
        for metric_map, (_, _, higher_is_better) in zip(metric_maps, PRIMARY_METRICS)
    ]
    labels = sort_labels_for_compact_table(label_to_item, rank_maps)

    print("overall_compact")
    print(
        "label oracle_overlap overlap_vs_worst overlap_rank oracle_js_sim js_vs_worst js_rank "
        "row_auc auc_vs_worst auc_rank row_spearman spearman_vs_worst spearman_rank mean_rank"
    )
    for label in labels:
        item = label_to_item[label]
        mean_rank_values = [rank_map[label] for rank_map in rank_maps if rank_map[label] is not None]
        mean_rank = None if not mean_rank_values else sum(mean_rank_values) / len(mean_rank_values)
        print(
            f"{item.label:>13s} "
            f"{fmt(item.actionable_row_oracle_overlap_mean):>14s} "
            f"{delta_fmt(item.actionable_row_oracle_overlap_mean, worst_values[0], higher_is_better=True):>16s} "
            f"{str(rank_maps[0][label]) if rank_maps[0][label] is not None else 'nan':>12s} "
            f"{fmt(item.actionable_row_oracle_js_similarity_mean):>9s} "
            f"{delta_fmt(item.actionable_row_oracle_js_similarity_mean, worst_values[1], higher_is_better=True):>13s} "
            f"{str(rank_maps[1][label]) if rank_maps[1][label] is not None else 'nan':>7s} "
            f"{fmt(item.macro_row_auc):>8s} "
            f"{delta_fmt(item.macro_row_auc, worst_values[2], higher_is_better=True):>12s} "
            f"{str(rank_maps[2][label]) if rank_maps[2][label] is not None else 'nan':>9s} "
            f"{fmt(item.actionable_row_spearman_mean):>12s} "
            f"{delta_fmt(item.actionable_row_spearman_mean, worst_values[3], higher_is_better=True):>18s} "
            f"{str(rank_maps[3][label]) if rank_maps[3][label] is not None else 'nan':>13s} "
            f"{fmt(mean_rank):>9s}"
        )


def print_compact_per_block(block_summaries: list[BackwardBlockSummary]) -> None:
    if not block_summaries:
        return

    per_block: dict[int, dict[str, BackwardBlockSummary]] = {}
    for item in block_summaries:
        per_block.setdefault(int(item.block_index), {})[item.label] = item

    print()
    print("per_block_compact")
    print(
        "block label oracle_overlap overlap_vs_worst overlap_rank oracle_js_sim js_vs_worst js_rank "
        "row_auc auc_vs_worst auc_rank row_spearman spearman_vs_worst spearman_rank mean_rank"
    )
    for block_index in sorted(per_block.keys()):
        label_to_item = per_block[block_index]
        metric_maps = [
            {label: getattr(item, field) for label, item in label_to_item.items()} for field, _, _ in PRIMARY_METRICS
        ]
        rank_maps = [
            dense_rank(metric_map, higher_is_better=higher_is_better)
            for metric_map, (_, _, higher_is_better) in zip(metric_maps, PRIMARY_METRICS)
        ]
        worst_values = [
            metric_worst(metric_map, higher_is_better=higher_is_better)
            for metric_map, (_, _, higher_is_better) in zip(metric_maps, PRIMARY_METRICS)
        ]
        labels = sort_labels_for_compact_table(label_to_item, rank_maps)

        for label in labels:
            item = label_to_item[label]
            mean_rank_values = [rank_map[label] for rank_map in rank_maps if rank_map[label] is not None]
            mean_rank = None if not mean_rank_values else sum(mean_rank_values) / len(mean_rank_values)
            print(
                f"{item.block_index:>5d} "
                f"{item.label:>13s} "
                f"{fmt(item.actionable_row_oracle_overlap_mean):>14s} "
                f"{delta_fmt(item.actionable_row_oracle_overlap_mean, worst_values[0], higher_is_better=True):>16s} "
                f"{str(rank_maps[0][label]) if rank_maps[0][label] is not None else 'nan':>12s} "
                f"{fmt(item.actionable_row_oracle_js_similarity_mean):>9s} "
                f"{delta_fmt(item.actionable_row_oracle_js_similarity_mean, worst_values[1], higher_is_better=True):>13s} "
                f"{str(rank_maps[1][label]) if rank_maps[1][label] is not None else 'nan':>7s} "
                f"{fmt(item.macro_row_auc):>8s} "
                f"{delta_fmt(item.macro_row_auc, worst_values[2], higher_is_better=True):>12s} "
                f"{str(rank_maps[2][label]) if rank_maps[2][label] is not None else 'nan':>9s} "
                f"{fmt(item.actionable_row_spearman_mean):>12s} "
                f"{delta_fmt(item.actionable_row_spearman_mean, worst_values[3], higher_is_better=True):>18s} "
                f"{str(rank_maps[3][label]) if rank_maps[3][label] is not None else 'nan':>13s} "
                f"{fmt(mean_rank):>9s}"
            )


def print_detailed_overall(summaries: list[BackwardSummary]) -> None:
    print()
    print("overall_detailed")
    print(
        "label rows swaps beneficial harmful neutral top_m_budget beneficial_rate "
        "row_top1_hit row_top1_benefit row_top1_regret macro_row_auc macro_row_auc_rows "
        "actionable_row_spearman actionable_row_spearman_rows "
        "row_soft_benefit row_soft_pos_mass actionable_pos_mass actionable_soft_benefit actionable_gap_closed "
        "actionable_soft_recall actionable_oracle_overlap actionable_oracle_js_sim actionable_rows "
        "swap_auc top_m_precision benefit_capture swap_sign_acc pos_neg_margin "
        "score_benefit_spearman score_benefit_pearson pred_score_mean benefit_mean"
    )
    for item in summaries:
        print(
            f"{item.label:>13s} "
            f"{item.rows_analyzed:>4d} "
            f"{item.swaps_analyzed:>5d} "
            f"{item.beneficial_swaps:>10d} "
            f"{item.harmful_swaps:>7d} "
            f"{item.neutral_swaps:>7d} "
            f"{item.top_m_budget:>12d} "
            f"{fmt(item.beneficial_swap_rate):>15s} "
            f"{fmt(item.row_top1_hit_rate):>12s} "
            f"{fmt(item.row_top1_benefit_mean):>16s} "
            f"{fmt(item.row_top1_regret_mean):>15s} "
            f"{fmt(item.macro_row_auc):>13s} "
            f"{item.macro_row_auc_rows:>18d} "
            f"{fmt(item.actionable_row_spearman_mean):>24s} "
            f"{item.actionable_row_spearman_rows:>28d} "
            f"{fmt(item.row_soft_expected_benefit_mean):>16s} "
            f"{fmt(item.row_soft_positive_mass_mean):>17s} "
            f"{fmt(item.actionable_row_soft_positive_mass_mean):>19s} "
            f"{fmt(item.actionable_row_soft_expected_benefit_mean):>22s} "
            f"{fmt(item.actionable_row_soft_gap_closed_mean):>21s} "
            f"{fmt(item.actionable_row_soft_benefit_recall_mean):>22s} "
            f"{fmt(item.actionable_row_oracle_overlap_mean):>24s} "
            f"{fmt(item.actionable_row_oracle_js_similarity_mean):>19s} "
            f"{item.actionable_row_count:>15d} "
            f"{fmt(item.swap_auc):>8s} "
            f"{fmt(item.top_m_precision):>15s} "
            f"{fmt(item.benefit_capture):>15s} "
            f"{fmt(item.swap_sign_acc):>13s} "
            f"{fmt(item.positive_negative_margin):>14s} "
            f"{fmt(item.score_benefit_spearman):>22s} "
            f"{fmt(item.score_benefit_pearson):>21s} "
            f"{fmt(item.pred_score_mean):>15s} "
            f"{fmt(item.benefit_mean):>12s}"
        )


def print_detailed_per_block(block_summaries: list[BackwardBlockSummary]) -> None:
    if not block_summaries:
        return
    print()
    print("per_block_detailed")
    print(
        "block label rows swaps beneficial harmful neutral top_m_budget beneficial_rate "
        "row_top1_hit row_top1_benefit row_top1_regret macro_row_auc macro_row_auc_rows "
        "actionable_row_spearman actionable_row_spearman_rows "
        "row_soft_benefit row_soft_pos_mass actionable_pos_mass actionable_soft_benefit actionable_gap_closed "
        "actionable_soft_recall actionable_oracle_overlap actionable_oracle_js_sim actionable_rows "
        "swap_auc top_m_precision benefit_capture swap_sign_acc pos_neg_margin "
        "score_benefit_spearman score_benefit_pearson"
    )
    for item in block_summaries:
        print(
            f"{item.block_index:>5d} "
            f"{item.label:>13s} "
            f"{item.rows_analyzed:>4d} "
            f"{item.swaps_analyzed:>5d} "
            f"{item.beneficial_swaps:>10d} "
            f"{item.harmful_swaps:>7d} "
            f"{item.neutral_swaps:>7d} "
            f"{item.top_m_budget:>12d} "
            f"{fmt(item.beneficial_swap_rate):>15s} "
            f"{fmt(item.row_top1_hit_rate):>12s} "
            f"{fmt(item.row_top1_benefit_mean):>16s} "
            f"{fmt(item.row_top1_regret_mean):>15s} "
            f"{fmt(item.macro_row_auc):>13s} "
            f"{item.macro_row_auc_rows:>18d} "
            f"{fmt(item.actionable_row_spearman_mean):>24s} "
            f"{item.actionable_row_spearman_rows:>28d} "
            f"{fmt(item.row_soft_expected_benefit_mean):>16s} "
            f"{fmt(item.row_soft_positive_mass_mean):>17s} "
            f"{fmt(item.actionable_row_soft_positive_mass_mean):>19s} "
            f"{fmt(item.actionable_row_soft_expected_benefit_mean):>22s} "
            f"{fmt(item.actionable_row_soft_gap_closed_mean):>21s} "
            f"{fmt(item.actionable_row_soft_benefit_recall_mean):>22s} "
            f"{fmt(item.actionable_row_oracle_overlap_mean):>24s} "
            f"{fmt(item.actionable_row_oracle_js_similarity_mean):>19s} "
            f"{item.actionable_row_count:>15d} "
            f"{fmt(item.swap_auc):>8s} "
            f"{fmt(item.top_m_precision):>15s} "
            f"{fmt(item.benefit_capture):>15s} "
            f"{fmt(item.swap_sign_acc):>13s} "
            f"{fmt(item.positive_negative_margin):>14s} "
            f"{fmt(item.score_benefit_spearman):>22s} "
            f"{fmt(item.score_benefit_pearson):>21s}"
        )


def main() -> None:
    script_args, train_args = parse_combined_args()
    specs = build_specs()

    if script_args.top_m_count <= 0:
        raise ValueError("top-m-count must be > 0")
    if script_args.row_softmax_temp <= 0.0:
        raise ValueError("row-softmax-temp must be > 0")
    if script_args.row_score_norm_eps <= 0.0:
        raise ValueError("row-score-norm-eps must be > 0")
    if script_args.swap_forward_batch_size <= 0:
        raise ValueError("swap-forward-batch-size must be > 0")
    if script_args.benefit_eps < 0.0:
        raise ValueError("benefit-eps must be >= 0")

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

    checkpoint_path = script_args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model = build_model(train_args).to(device)
    initialize_lazy_modules(model, init_loader, device, transform)
    _, state_dict = load_checkpoint_state(checkpoint_path)
    load_result = model.load_state_dict(state_dict, strict=False)
    infer_and_report_mismatches(load_result, "shared-checkpoint")

    set_attention_soft_train_temperatures(
        model,
        boundary_surrogate_temperature=float(train_args.boundary_surrogate_temp_max),
        majority_temperature=float(train_args.majority_train_temp_max),
    )
    model.eval()

    target_blocks = list(range(len(model.blocks))) if script_args.block_index is None else [int(script_args.block_index)]
    all_records: list[SwapRecord] = []
    summaries: list[BackwardSummary] = []
    block_summaries: list[BackwardBlockSummary] = []
    baseline_loss_means: dict[str, float] = {}

    for batch_loader_index, (images, labels) in tqdm(batches, desc="batches", total=len(batches), leave=False):
        images = transform(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        capture_by_label: dict[str, list[BlockCapture]] = {}
        baseline_loss_values_this_batch: dict[str, torch.Tensor] = {}
        for spec in tqdm(specs, desc=f"backward[batch{batch_loader_index}]", total=len(specs), leave=False):
            captures, baseline_loss_value, baseline_loss_values = collect_capture_grads(model, images, labels, spec)
            capture_by_label[spec.label] = captures
            baseline_loss_values_this_batch[spec.label] = baseline_loss_values
            baseline_loss_means.setdefault(spec.label, 0.0)
            baseline_loss_means[spec.label] += baseline_loss_value

        reference_captures = capture_by_label["kth-raw"]

        for bidx in target_blocks:
            ref_capture = reference_captures[bidx]
            if ref_capture.hard_mask is None or ref_capture.proxy_scores is None:
                continue

            flat_mask = ref_capture.hard_mask.reshape(-1, ref_capture.hard_mask.shape[-1])
            flat_scores = ref_capture.proxy_scores.reshape(-1, ref_capture.proxy_scores.shape[-1])
            prefix_shape = ref_capture.hard_mask.shape[:-1]
            rows_to_scan = min(int(script_args.max_rows_per_block), flat_mask.shape[0])

            grad_by_label = {}
            for spec in specs:
                capture = capture_by_label[spec.label][bidx]
                if capture.proxy_scores_grad is None:
                    continue
                grad_by_label[spec.label] = capture.proxy_scores_grad.reshape(-1, capture.proxy_scores_grad.shape[-1])

            for row_flat_index in tqdm(
                range(rows_to_scan),
                desc=f"rows[b{bidx}|batch{batch_loader_index}]",
                total=rows_to_scan,
                leave=False,
            ):
                row_mask = flat_mask[row_flat_index]
                row_scores = flat_scores[row_flat_index]
                inside_low, outside_high = select_boundary_candidates(row_mask, row_scores, int(script_args.boundary_width))
                band = int(inside_low.numel())
                if band <= 0:
                    continue

                row_coords = unflatten_row_index(row_flat_index, prefix_shape)
                batch_coord = int(row_coords[0])
                head_coord = int(row_coords[1]) if len(row_coords) > 1 else 0
                query_coord = int(row_coords[2]) if len(row_coords) > 2 else 0

                sample_mask = ref_capture.hard_mask[batch_coord : batch_coord + 1].clone()
                sample_images = images[batch_coord : batch_coord + 1]
                sample_labels = labels[batch_coord : batch_coord + 1]
                baseline_loss_value = float(baseline_loss_values_this_batch["kth-raw"][batch_coord].item())
                sample_row_coords = (0, *row_coords[1:])
                sample_row_flat_index = flatten_row_coords(sample_row_coords, sample_mask.shape[:-1])

                swap_out_indices = inside_low.repeat_interleave(band)
                swap_in_indices = outside_high.repeat(band)
                swap_benefits = evaluate_swap_benefits_for_row(
                    model=model,
                    block_index=bidx,
                    sample_mask=sample_mask,
                    sample_row_flat_index=sample_row_flat_index,
                    swap_out_indices=swap_out_indices,
                    swap_in_indices=swap_in_indices,
                    sample_images=sample_images,
                    sample_labels=sample_labels,
                    baseline_loss_value=baseline_loss_value,
                    swap_forward_batch_size=int(script_args.swap_forward_batch_size),
                )

                for swap_index in range(int(swap_out_indices.shape[0])):
                    idx_out = int(swap_out_indices[swap_index].item())
                    idx_in = int(swap_in_indices[swap_index].item())
                    benefit = float(swap_benefits[swap_index].item())
                    correct = bool(benefit > 0.0)

                    for spec in specs:
                        row_grad = grad_by_label[spec.label][row_flat_index]
                        pred_score = float((row_grad[idx_out] - row_grad[idx_in]).item())
                        pred_good = bool(pred_score > 0.0)
                        all_records.append(
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

    batch_count = max(len(batches), 1)
    for label in list(baseline_loss_means.keys()):
        baseline_loss_means[label] /= float(batch_count)

    for spec in specs:
        spec_records = [record for record in all_records if record.surrogate_label == spec.label]
        summaries.append(
            summarize_backward_records(
                label=spec.label,
                records=spec_records,
                top_m_count=int(script_args.top_m_count),
                benefit_eps=float(script_args.benefit_eps),
                score_eps=float(script_args.score_eps),
                row_softmax_temp=float(script_args.row_softmax_temp),
                row_score_norm_eps=float(script_args.row_score_norm_eps),
            )
        )
        if script_args.block_index is None:
            block_to_records: dict[int, list[SwapRecord]] = {}
            for record in spec_records:
                block_to_records.setdefault(int(record.block_index), []).append(record)
            for bidx in sorted(block_to_records.keys()):
                block_summaries.append(
                    summarize_backward_block(
                        label=spec.label,
                        block_index=bidx,
                        records=block_to_records[bidx],
                        top_m_count=int(script_args.top_m_count),
                        benefit_eps=float(script_args.benefit_eps),
                        score_eps=float(script_args.score_eps),
                        row_softmax_temp=float(script_args.row_softmax_temp),
                        row_score_norm_eps=float(script_args.row_score_norm_eps),
                    )
                )

    print(f"checkpoint={checkpoint_path}")
    print(f"split={script_args.split}")
    print(f"batch_index={script_args.batch_index}")
    print(f"num_batches={script_args.num_batches}")
    print(f"block_index={'all' if script_args.block_index is None else script_args.block_index}")
    print(f"max_rows_per_block={script_args.max_rows_per_block}")
    print(f"boundary_width={script_args.boundary_width}")
    print(f"benefit_eps={script_args.benefit_eps}")
    print(f"score_eps={script_args.score_eps}")
    print(f"top_m_count={script_args.top_m_count}")
    print(f"row_softmax_temp={script_args.row_softmax_temp}")
    print(f"row_score_norm_eps={script_args.row_score_norm_eps}")
    print(f"swap_forward_batch_size={script_args.swap_forward_batch_size}")
    print()
    print("This script isolates top-k backward quality on the same checkpoint and the same hard top-k forward states.")
    print("Each candidate is a boundary swap: remove one selected item and add one near-boundary unselected item.")
    print("benefit = per-sample baseline_loss - per-sample swap_loss for the affected sample.")
    print("So benefit > 0 means that discrete swap would have helped that sample directly.")
    print("pred_score = grad[out] - grad[in], so larger pred_score means the surrogate favors that swap more.")
    print("Primary metrics are row-level backward-alignment metrics: oracle_overlap, oracle_js_sim, macro_row_auc, row_spearman.")
    print("Each row turns surrogate scores into a z-normalized softmax distribution before comparing it to the oracle positive-benefit distribution.")
    print("oracle_overlap and oracle_js_sim measure full distribution alignment to the oracle positive-benefit distribution.")
    print("row_spearman measures whether the surrogate ranks swap candidates in the same order as their true discrete benefits.")
    print("This is more aligned with sigmoid/subset-style distributional surrogates than greedy top1 alone.")
    print()
    print_compact_overall(summaries)
    if script_args.block_index is None and block_summaries:
        print_compact_per_block(block_summaries)

    if script_args.print_detailed:
        print_detailed_overall(summaries)
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
            "checkpoint": str(checkpoint_path),
            "split": script_args.split,
            "batch_index": int(script_args.batch_index),
            "num_batches": int(script_args.num_batches),
            "block_index": script_args.block_index,
            "max_rows_per_block": int(script_args.max_rows_per_block),
            "boundary_width": int(script_args.boundary_width),
            "benefit_eps": float(script_args.benefit_eps),
            "score_eps": float(script_args.score_eps),
            "top_m_count": int(script_args.top_m_count),
            "row_softmax_temp": float(script_args.row_softmax_temp),
            "row_score_norm_eps": float(script_args.row_score_norm_eps),
            "swap_forward_batch_size": int(script_args.swap_forward_batch_size),
            "model_args": vars(train_args),
            "backward_specs": [asdict(spec) for spec in specs],
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
            row["baseline_loss_mean"] = baseline_loss_means.get(item.label)
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
                "beneficial_swaps",
                "harmful_swaps",
                "neutral_swaps",
                "beneficial_swap_rate",
                "top_m_budget",
                "row_top1_hit_rate",
                "row_top1_benefit_mean",
                "row_top1_regret_mean",
                "macro_row_auc",
                "macro_row_auc_rows",
                "actionable_row_spearman_mean",
                "actionable_row_spearman_rows",
                "row_soft_expected_benefit_mean",
                "row_soft_positive_mass_mean",
                "actionable_row_soft_positive_mass_mean",
                "actionable_row_soft_expected_benefit_mean",
                "actionable_row_soft_gap_closed_mean",
                "actionable_row_soft_benefit_recall_mean",
                "actionable_row_oracle_overlap_mean",
                "actionable_row_oracle_js_similarity_mean",
                "actionable_row_count",
                "swap_sign_acc",
                "swap_auc",
                "top_m_precision",
                "benefit_capture",
                "positive_negative_margin",
                "score_benefit_spearman",
                "score_benefit_pearson",
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
                "beneficial_swaps",
                "harmful_swaps",
                "neutral_swaps",
                "beneficial_swap_rate",
                "top_m_budget",
                "row_top1_hit_rate",
                "row_top1_benefit_mean",
                "row_top1_regret_mean",
                "macro_row_auc",
                "macro_row_auc_rows",
                "actionable_row_spearman_mean",
                "actionable_row_spearman_rows",
                "row_soft_expected_benefit_mean",
                "row_soft_positive_mass_mean",
                "actionable_row_soft_positive_mass_mean",
                "actionable_row_soft_expected_benefit_mean",
                "actionable_row_soft_gap_closed_mean",
                "actionable_row_soft_benefit_recall_mean",
                "actionable_row_oracle_overlap_mean",
                "actionable_row_oracle_js_similarity_mean",
                "actionable_row_count",
                "swap_sign_acc",
                "swap_auc",
                "top_m_precision",
                "benefit_capture",
                "positive_negative_margin",
                "score_benefit_spearman",
                "score_benefit_pearson",
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
