from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_topk_boundary_swap_surrogates import (  # noqa: E402
    SurrogateSpec,
    build_specs,
    forced_block_mask,
    get_batches,
    infer_and_report_mismatches,
    load_checkpoint_state,
    ordered_labels,
    select_loader,
    unflatten_row_index,
)
from data_pipeline import load_dataset  # noqa: E402
from topk_surrogate_eval_utils import (  # noqa: E402
    run_forward_with_captures,
    select_boundary_candidates,
)
from train_logic_vit_tiny import (  # noqa: E402
    build_model,
    initialize_lazy_modules,
    parse_args as parse_train_args,
    set_attention_soft_train_temperatures,
    set_deterministic,
)


_COMBINATION_INDEX_CACHE: dict[tuple[int, int], torch.Tensor] = {}


@dataclass
class LocalOracleRecord:
    surrogate_label: str
    block_index: int
    row_flat_index: int
    batch_loader_index: int
    batch_index: int
    head_index: int
    query_index: int
    boundary_width: int
    candidate_pool_size: int
    exact_match: bool
    overlap_fraction: float
    jaccard: float
    baseline_loss: float
    oracle_loss: float
    oracle_regret: float


@dataclass
class LocalOracleSummary:
    label: str
    clean_accuracy: float
    rows_analyzed: int
    exact_match_rate: float
    overlap_fraction_mean: float
    jaccard_mean: float
    weighted_overlap_fraction_mean: float
    weighted_exact_match_rate: float
    actionable_rows: int
    actionable_overlap_fraction_mean: float
    actionable_exact_match_rate: float
    actionable_weighted_overlap_fraction_mean: float
    actionable_weighted_exact_match_rate: float
    oracle_regret_mean: float
    oracle_regret_median: float
    oracle_regret_actionable_mean: float
    normalized_oracle_regret_mean: float
    oracle_positive_rate: float


@dataclass
class LocalOracleBlockSummary:
    label: str
    block_index: int
    rows_analyzed: int
    exact_match_rate: float
    overlap_fraction_mean: float
    jaccard_mean: float
    weighted_overlap_fraction_mean: float
    weighted_exact_match_rate: float
    actionable_rows: int
    actionable_overlap_fraction_mean: float
    actionable_exact_match_rate: float
    actionable_weighted_overlap_fraction_mean: float
    actionable_weighted_exact_match_rate: float
    oracle_regret_mean: float
    oracle_regret_actionable_mean: float
    normalized_oracle_regret_mean: float
    oracle_positive_rate: float


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) <= 0:
        return math.nan
    return float(values[mask].mean().item())


def weighted_local_oracle_stats(
    overlap: torch.Tensor,
    exact: torch.Tensor,
    regret: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[float, float]:
    if mask is not None:
        overlap = overlap[mask]
        exact = exact[mask]
        regret = regret[mask]
        if overlap.numel() == 0:
            return math.nan, math.nan
    weights = regret.clamp_min(0.0)
    weight_sum = float(weights.sum().item())
    if weight_sum <= 0.0:
        return math.nan, math.nan
    weighted_overlap = float((overlap * weights).sum().item() / weight_sum)
    weighted_exact = float((exact * weights).sum().item() / weight_sum)
    return weighted_overlap, weighted_exact


class LocalOracleAccumulator:
    def __init__(self) -> None:
        self.records: list[LocalOracleRecord] = []

    @classmethod
    def from_records(cls, records: list[LocalOracleRecord]) -> "LocalOracleAccumulator":
        acc = cls()
        for record in records:
            acc.add(record)
        return acc

    def add(self, record: LocalOracleRecord) -> None:
        self.records.append(record)

    def summary(
        self,
        label: str,
        clean_accuracy: float,
        actionable_oracle_eps: float,
        normalized_regret_eps: float,
    ) -> LocalOracleSummary:
        if not self.records:
            return LocalOracleSummary(
                label=label,
                clean_accuracy=clean_accuracy,
                rows_analyzed=0,
                exact_match_rate=0.0,
                overlap_fraction_mean=0.0,
                jaccard_mean=0.0,
                weighted_overlap_fraction_mean=math.nan,
                weighted_exact_match_rate=math.nan,
                actionable_rows=0,
                actionable_overlap_fraction_mean=math.nan,
                actionable_exact_match_rate=math.nan,
                actionable_weighted_overlap_fraction_mean=math.nan,
                actionable_weighted_exact_match_rate=math.nan,
                oracle_regret_mean=math.nan,
                oracle_regret_median=math.nan,
                oracle_regret_actionable_mean=math.nan,
                normalized_oracle_regret_mean=math.nan,
                oracle_positive_rate=0.0,
            )

        exact = torch.tensor([1.0 if row.exact_match else 0.0 for row in self.records], dtype=torch.float32)
        overlap = torch.tensor([row.overlap_fraction for row in self.records], dtype=torch.float32)
        jaccard = torch.tensor([row.jaccard for row in self.records], dtype=torch.float32)
        regret = torch.tensor([row.oracle_regret for row in self.records], dtype=torch.float32)
        baseline = torch.tensor([row.baseline_loss for row in self.records], dtype=torch.float32)
        actionable = regret > float(actionable_oracle_eps)

        weighted_overlap, weighted_exact = weighted_local_oracle_stats(overlap, exact, regret)
        actionable_overlap = masked_mean(overlap, actionable)
        actionable_exact = masked_mean(exact, actionable)
        actionable_weighted_overlap, actionable_weighted_exact = weighted_local_oracle_stats(
            overlap, exact, regret, mask=actionable
        )
        actionable_regret = masked_mean(regret, actionable)
        normalized_regret = regret / (baseline.abs() + float(normalized_regret_eps))

        return LocalOracleSummary(
            label=label,
            clean_accuracy=clean_accuracy,
            rows_analyzed=len(self.records),
            exact_match_rate=float(exact.mean().item()),
            overlap_fraction_mean=float(overlap.mean().item()),
            jaccard_mean=float(jaccard.mean().item()),
            weighted_overlap_fraction_mean=weighted_overlap,
            weighted_exact_match_rate=weighted_exact,
            actionable_rows=int(actionable.sum().item()),
            actionable_overlap_fraction_mean=actionable_overlap,
            actionable_exact_match_rate=actionable_exact,
            actionable_weighted_overlap_fraction_mean=actionable_weighted_overlap,
            actionable_weighted_exact_match_rate=actionable_weighted_exact,
            oracle_regret_mean=float(regret.mean().item()),
            oracle_regret_median=float(regret.median().item()),
            oracle_regret_actionable_mean=actionable_regret,
            normalized_oracle_regret_mean=float(normalized_regret.mean().item()),
            oracle_positive_rate=float((regret > 1e-12).to(torch.float32).mean().item()),
        )

    def block_summary(
        self,
        label: str,
        block_index: int,
        actionable_oracle_eps: float,
        normalized_regret_eps: float,
    ) -> LocalOracleBlockSummary:
        block_records = [row for row in self.records if int(row.block_index) == int(block_index)]
        if not block_records:
            return LocalOracleBlockSummary(
                label=label,
                block_index=int(block_index),
                rows_analyzed=0,
                exact_match_rate=0.0,
                overlap_fraction_mean=0.0,
                jaccard_mean=0.0,
                weighted_overlap_fraction_mean=math.nan,
                weighted_exact_match_rate=math.nan,
                actionable_rows=0,
                actionable_overlap_fraction_mean=math.nan,
                actionable_exact_match_rate=math.nan,
                actionable_weighted_overlap_fraction_mean=math.nan,
                actionable_weighted_exact_match_rate=math.nan,
                oracle_regret_mean=math.nan,
                oracle_regret_actionable_mean=math.nan,
                normalized_oracle_regret_mean=math.nan,
                oracle_positive_rate=0.0,
            )

        exact = torch.tensor([1.0 if row.exact_match else 0.0 for row in block_records], dtype=torch.float32)
        overlap = torch.tensor([row.overlap_fraction for row in block_records], dtype=torch.float32)
        jaccard = torch.tensor([row.jaccard for row in block_records], dtype=torch.float32)
        regret = torch.tensor([row.oracle_regret for row in block_records], dtype=torch.float32)
        baseline = torch.tensor([row.baseline_loss for row in block_records], dtype=torch.float32)
        actionable = regret > float(actionable_oracle_eps)

        weighted_overlap, weighted_exact = weighted_local_oracle_stats(overlap, exact, regret)
        actionable_overlap = masked_mean(overlap, actionable)
        actionable_exact = masked_mean(exact, actionable)
        actionable_weighted_overlap, actionable_weighted_exact = weighted_local_oracle_stats(
            overlap, exact, regret, mask=actionable
        )
        actionable_regret = masked_mean(regret, actionable)
        normalized_regret = regret / (baseline.abs() + float(normalized_regret_eps))

        return LocalOracleBlockSummary(
            label=label,
            block_index=int(block_index),
            rows_analyzed=len(block_records),
            exact_match_rate=float(exact.mean().item()),
            overlap_fraction_mean=float(overlap.mean().item()),
            jaccard_mean=float(jaccard.mean().item()),
            weighted_overlap_fraction_mean=weighted_overlap,
            weighted_exact_match_rate=weighted_exact,
            actionable_rows=int(actionable.sum().item()),
            actionable_overlap_fraction_mean=actionable_overlap,
            actionable_exact_match_rate=actionable_exact,
            actionable_weighted_overlap_fraction_mean=actionable_weighted_overlap,
            actionable_weighted_exact_match_rate=actionable_weighted_exact,
            oracle_regret_mean=float(regret.mean().item()),
            oracle_regret_actionable_mean=actionable_regret,
            normalized_oracle_regret_mean=float(normalized_regret.mean().item()),
            oracle_positive_rate=float((regret > 1e-12).to(torch.float32).mean().item()),
        )


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Compare hard top-k local support recovery and local oracle regret across surrogate checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fallback checkpoint used for all surrogates")
    parser.add_argument("--checkpoint-kth-raw", type=Path, default=None)
    parser.add_argument("--checkpoint-kth-norm", type=Path, default=None)
    parser.add_argument("--checkpoint-sigmoid-topk", type=Path, default=None)
    parser.add_argument("--checkpoint-subset-gibbs", type=Path, default=None)
    parser.add_argument("--split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--init-split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--block-index", type=int, default=None)
    parser.add_argument("--max-rows-per-block", type=int, default=10)
    parser.add_argument("--boundary-width", type=int, default=4)
    parser.add_argument("--oracle-max-combinations", type=int, default=256)
    parser.add_argument("--oracle-forward-batch-size", type=int, default=32)
    parser.add_argument("--oracle-tie-tol", type=float, default=1e-8)
    parser.add_argument(
        "--actionable-oracle-eps",
        type=float,
        default=1e-4,
        help="Rows with oracle_regret > eps are treated as actionable for functional local-oracle metrics",
    )
    parser.add_argument(
        "--normalized-regret-eps",
        type=float,
        default=1e-8,
        help="Stability constant in normalized oracle regret = regret / (baseline_loss + eps)",
    )
    parser.add_argument(
        "--compact-regret-min-spread",
        type=float,
        default=1e-4,
        help="If oracle_regret spread is smaller than this, compact ranking treats regret as diagnostic-only",
    )
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


def make_model_args(train_args: argparse.Namespace, spec: SurrogateSpec) -> argparse.Namespace:
    args = copy.deepcopy(train_args)
    args.topk_surrogate_mode = spec.topk_surrogate_mode
    args.topk_kth_normalize_soft_mask = spec.topk_kth_normalize_soft_mask
    return args


def evaluate_oracle_row(
    model,
    block_index: int,
    base_mask: torch.Tensor,
    row_flat_index: int,
    candidate_pool: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    sample_index: int,
    baseline_loss: float,
    forward_batch_size: int,
    tie_tol: float,
) -> tuple[torch.Tensor, float]:
    candidate_size = int(candidate_pool.numel() // 2)
    combo_positions = get_combination_positions(int(candidate_pool.numel()), candidate_size, candidate_pool.device)
    chosen_sets = candidate_pool.index_select(0, combo_positions.reshape(-1)).reshape(combo_positions.shape[0], candidate_size)

    best_loss = float(baseline_loss)
    best_candidates = chosen_sets[0].clone()

    for start in range(0, int(chosen_sets.shape[0]), int(forward_batch_size)):
        end = min(start + int(forward_batch_size), int(chosen_sets.shape[0]))
        chosen_chunk = chosen_sets[start:end]
        forced_masks = build_forced_masks_for_row(
            base_mask=base_mask,
            row_flat_index=row_flat_index,
            candidate_pool=candidate_pool,
            chosen_sets=chosen_chunk,
        )
        loss_values = evaluate_forced_masks_chunk(
            model=model,
            block_index=block_index,
            forced_masks=forced_masks,
            images=images,
            labels=labels,
        )
        sample_loss_values = loss_values[:, int(sample_index)]
        chunk_best_idx = int(torch.argmin(sample_loss_values).item())
        chunk_best_loss = float(sample_loss_values[chunk_best_idx].item())
        if chunk_best_loss < best_loss - tie_tol:
            best_loss = chunk_best_loss
            best_candidates = chosen_chunk[chunk_best_idx].clone()

    return best_candidates, best_loss


def get_combination_positions(pool_size: int, choose_size: int, device: torch.device) -> torch.Tensor:
    cache_key = (int(pool_size), int(choose_size))
    cached = _COMBINATION_INDEX_CACHE.get(cache_key)
    if cached is None:
        combos = list(itertools.combinations(range(pool_size), choose_size))
        cached = torch.tensor(combos, dtype=torch.long)
        _COMBINATION_INDEX_CACHE[cache_key] = cached
    return cached.to(device=device)


def build_forced_masks_for_row(
    base_mask: torch.Tensor,
    row_flat_index: int,
    candidate_pool: torch.Tensor,
    chosen_sets: torch.Tensor,
) -> torch.Tensor:
    forced_masks = base_mask.unsqueeze(0).expand(chosen_sets.shape[0], *base_mask.shape).clone()
    flat = forced_masks.reshape(chosen_sets.shape[0], -1, forced_masks.shape[-1])
    row_masks = flat[:, row_flat_index, :]
    row_masks[:, candidate_pool] = False
    row_masks.scatter_(1, chosen_sets, True)
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
    with torch.no_grad():
        with forced_block_mask(model, block_index, forced_mask):
            logits = model(repeated_images)
            loss_values = torch.nn.functional.cross_entropy(logits, repeated_labels, reduction="none")
    return loss_values.reshape(chunk_size, images.shape[0])


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
    oracle_forward_batch_size: int,
    oracle_tie_tol: float,
    actionable_oracle_eps: float,
    normalized_regret_eps: float,
) -> tuple[LocalOracleSummary, list[LocalOracleRecord], list[LocalOracleBlockSummary]]:
    args = make_model_args(train_args, spec)
    model = build_model(args).to(device)
    initialize_lazy_modules(model, init_loader, device, transform)

    _, state_dict = load_checkpoint_state(spec.checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    infer_and_report_mismatches(load_result, spec.label)

    set_attention_soft_train_temperatures(
        model,
        boundary_surrogate_temperature=float(args.boundary_surrogate_temp_max),
        majority_temperature=float(args.majority_train_temp_max),
    )
    model.eval()

    target_blocks = list(range(len(model.blocks))) if block_index is None else [int(block_index)]
    records: list[LocalOracleRecord] = []
    clean_accuracy_sum = 0.0
    clean_accuracy_count = 0

    for batch_loader_index, (images, labels) in tqdm(batches, desc=f"{spec.label}:batches", total=len(batches), leave=False):
        images = transform(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        clean_run = run_forward_with_captures(model, images, labels)
        clean_accuracy_sum += clean_run.accuracy
        clean_accuracy_count += 1

        snapshots = {snapshot.block_index: snapshot for snapshot in clean_run.block_snapshots}
        baseline_loss_values = clean_run.loss_values

        for bidx in target_blocks:
            snapshot = snapshots[bidx]
            flat_mask = snapshot.hard_mask.reshape(-1, snapshot.hard_mask.shape[-1])
            flat_scores = snapshot.scores.reshape(-1, snapshot.scores.shape[-1])
            prefix_shape = snapshot.hard_mask.shape[:-1]
            rows_to_scan = min(int(max_rows_per_block), flat_mask.shape[0])

            for row_flat_index in tqdm(
                range(rows_to_scan),
                desc=f"{spec.label}:rows[b{bidx}|batch{batch_loader_index}]",
                total=rows_to_scan,
                leave=False,
            ):
                row_mask = flat_mask[row_flat_index]
                row_scores = flat_scores[row_flat_index]
                inside_low, outside_high = select_boundary_candidates(row_mask, row_scores, boundary_width)
                band = int(inside_low.numel())
                if band <= 0:
                    continue

                candidate_pool = torch.cat([inside_low, outside_high], dim=0)
                batch_coord, head_coord, query_coord = unflatten_row_index(row_flat_index, prefix_shape)
                sample_baseline_loss = float(baseline_loss_values[int(batch_coord)].item())
                oracle_candidates, oracle_loss = evaluate_oracle_row(
                    model=model,
                    block_index=bidx,
                    base_mask=snapshot.hard_mask,
                    row_flat_index=row_flat_index,
                    candidate_pool=candidate_pool,
                    images=images,
                    labels=labels,
                    sample_index=int(batch_coord),
                    baseline_loss=sample_baseline_loss,
                    forward_batch_size=oracle_forward_batch_size,
                    tie_tol=oracle_tie_tol,
                )

                current_set = set(int(x) for x in inside_low.tolist())
                oracle_set = set(int(x) for x in oracle_candidates.tolist())
                overlap = len(current_set & oracle_set)
                union = len(current_set | oracle_set)
                overlap_fraction = float(overlap) / float(band)
                jaccard = float(overlap) / float(union) if union > 0 else 1.0

                records.append(
                    LocalOracleRecord(
                        surrogate_label=spec.label,
                        block_index=bidx,
                        row_flat_index=row_flat_index,
                        batch_loader_index=batch_loader_index,
                        batch_index=int(batch_coord),
                        head_index=int(head_coord),
                        query_index=int(query_coord),
                        boundary_width=band,
                        candidate_pool_size=int(candidate_pool.numel()),
                        exact_match=bool(current_set == oracle_set),
                        overlap_fraction=overlap_fraction,
                        jaccard=jaccard,
                        baseline_loss=sample_baseline_loss,
                        oracle_loss=float(oracle_loss),
                        oracle_regret=max(0.0, sample_baseline_loss - float(oracle_loss)),
                    )
                )

    clean_accuracy = clean_accuracy_sum / float(clean_accuracy_count) if clean_accuracy_count > 0 else math.nan
    accumulator = LocalOracleAccumulator.from_records(records)
    summary = accumulator.summary(spec.label, clean_accuracy, actionable_oracle_eps, normalized_regret_eps)
    block_summaries = [
        accumulator.block_summary(spec.label, bidx, actionable_oracle_eps, normalized_regret_eps)
        for bidx in sorted(target_blocks)
    ]
    return summary, records, block_summaries


def fmt(value: float | None) -> str:
    return "nan" if value is None or math.isnan(float(value)) else f"{float(value):.6f}"


def dense_rank(metric_by_label: dict[str, float | None], higher_is_better: bool) -> dict[str, int | None]:
    valid = [(label, value) for label, value in metric_by_label.items() if value is not None and not math.isnan(float(value))]
    if not valid:
        return {label: None for label in metric_by_label}
    unique_values = sorted({float(value) for _, value in valid}, reverse=higher_is_better)
    value_to_rank = {value: rank + 1 for rank, value in enumerate(unique_values)}
    ranked = {label: value_to_rank[float(value)] for label, value in valid}
    for label in metric_by_label:
        ranked.setdefault(label, None)
    return ranked


def metric_spread(metric_by_label: dict[str, float | None]) -> float | None:
    valid = [float(value) for value in metric_by_label.values() if value is not None and not math.isnan(float(value))]
    if not valid:
        return None
    return max(valid) - min(valid)


def print_compact_overall(summaries: list[LocalOracleSummary], compact_regret_min_spread: float) -> None:
    weighted_overlap_by_label = {item.label: item.weighted_overlap_fraction_mean for item in summaries}
    actionable_weighted_overlap_by_label = {
        item.label: item.actionable_weighted_overlap_fraction_mean for item in summaries
    }
    actionable_overlap_by_label = {item.label: item.actionable_overlap_fraction_mean for item in summaries}
    normalized_regret_by_label = {item.label: item.normalized_oracle_regret_mean for item in summaries}
    regret_spread = metric_spread(normalized_regret_by_label)
    use_regret_for_rank = regret_spread is not None and regret_spread >= float(compact_regret_min_spread)

    weighted_overlap_rank = dense_rank(weighted_overlap_by_label, higher_is_better=True)
    actionable_weighted_overlap_rank = dense_rank(actionable_weighted_overlap_by_label, higher_is_better=True)
    actionable_overlap_rank = dense_rank(actionable_overlap_by_label, higher_is_better=True)
    regret_rank = dense_rank(normalized_regret_by_label, higher_is_better=False)

    print("overall_compact")
    print(
        "label weighted_overlap weighted_rank actionable_weighted_overlap actionable_weighted_rank "
        "actionable_overlap clean_acc norm_regret regret_rank mean_rank"
    )
    preferred = {label: idx for idx, label in enumerate(ordered_labels())}
    for item in sorted(
        summaries,
        key=lambda x: (
            sum(
                rank
                for rank in (
                    [
                        weighted_overlap_rank[x.label],
                        actionable_weighted_overlap_rank[x.label],
                    ]
                )
                if rank is not None
            ),
            preferred.get(x.label, 999),
        ),
    ):
        ranks = [
            weighted_overlap_rank[item.label],
            actionable_weighted_overlap_rank[item.label],
        ]
        valid_ranks = [rank for rank in ranks if rank is not None]
        mean_rank = None if not valid_ranks else sum(valid_ranks) / len(valid_ranks)
        print(
            f"{item.label:>13s} "
            f"{fmt(item.weighted_overlap_fraction_mean):>16s} "
            f"{str(weighted_overlap_rank[item.label]) if weighted_overlap_rank[item.label] is not None else 'nan':>13s} "
            f"{fmt(item.actionable_weighted_overlap_fraction_mean):>27s} "
            f"{str(actionable_weighted_overlap_rank[item.label]) if actionable_weighted_overlap_rank[item.label] is not None else 'nan':>24s} "
            f"{fmt(item.actionable_overlap_fraction_mean):>18s} "
            f"{fmt(item.clean_accuracy):>9s} "
            f"{fmt(item.normalized_oracle_regret_mean):>11s} "
            f"{(str(regret_rank[item.label]) if use_regret_for_rank and regret_rank[item.label] is not None else 'diag'):>11s} "
            f"{fmt(mean_rank):>9s}"
        )


def print_compact_per_block(block_summaries: list[LocalOracleBlockSummary], compact_regret_min_spread: float) -> None:
    if not block_summaries:
        return
    per_block: dict[int, list[LocalOracleBlockSummary]] = {}
    for item in block_summaries:
        per_block.setdefault(int(item.block_index), []).append(item)

    print()
    print("per_block_compact")
    print(
        "block label weighted_overlap weighted_rank actionable_weighted_overlap actionable_weighted_rank "
        "actionable_overlap norm_regret regret_rank mean_rank"
    )
    for block_index in sorted(per_block.keys()):
        items = per_block[block_index]
        weighted_overlap_by_label = {item.label: item.weighted_overlap_fraction_mean for item in items}
        actionable_weighted_overlap_by_label = {
            item.label: item.actionable_weighted_overlap_fraction_mean for item in items
        }
        actionable_overlap_by_label = {item.label: item.actionable_overlap_fraction_mean for item in items}
        normalized_regret_by_label = {item.label: item.normalized_oracle_regret_mean for item in items}
        regret_spread = metric_spread(normalized_regret_by_label)
        use_regret_for_rank = regret_spread is not None and regret_spread >= float(compact_regret_min_spread)
        weighted_overlap_rank = dense_rank(weighted_overlap_by_label, higher_is_better=True)
        actionable_weighted_overlap_rank = dense_rank(actionable_weighted_overlap_by_label, higher_is_better=True)
        actionable_overlap_rank = dense_rank(actionable_overlap_by_label, higher_is_better=True)
        regret_rank = dense_rank(normalized_regret_by_label, higher_is_better=False)

        preferred = {label: idx for idx, label in enumerate(ordered_labels())}
        for item in sorted(
            items,
            key=lambda x: (
                sum(
                    rank
                    for rank in (
                        [
                            weighted_overlap_rank[x.label],
                            actionable_weighted_overlap_rank[x.label],
                        ]
                    )
                    if rank is not None
                ),
                preferred.get(x.label, 999),
            ),
        ):
            ranks = [
                weighted_overlap_rank[item.label],
                actionable_weighted_overlap_rank[item.label],
            ]
            valid_ranks = [rank for rank in ranks if rank is not None]
            mean_rank = None if not valid_ranks else sum(valid_ranks) / len(valid_ranks)
            print(
                f"{item.block_index:>5d} "
                f"{item.label:>13s} "
                f"{fmt(item.weighted_overlap_fraction_mean):>16s} "
                f"{str(weighted_overlap_rank[item.label]) if weighted_overlap_rank[item.label] is not None else 'nan':>13s} "
                f"{fmt(item.actionable_weighted_overlap_fraction_mean):>27s} "
                f"{str(actionable_weighted_overlap_rank[item.label]) if actionable_weighted_overlap_rank[item.label] is not None else 'nan':>24s} "
                f"{fmt(item.actionable_overlap_fraction_mean):>18s} "
                f"{fmt(item.normalized_oracle_regret_mean):>11s} "
                f"{(str(regret_rank[item.label]) if use_regret_for_rank and regret_rank[item.label] is not None else 'diag'):>11s} "
                f"{fmt(mean_rank):>9s}"
            )


def print_detailed_overall(summaries: list[LocalOracleSummary]) -> None:
    print()
    print("overall_detailed")
    print(
        "label clean_acc rows exact overlap weighted_overlap actionable_rows actionable_overlap "
        "actionable_weighted_overlap weighted_exact actionable_weighted_exact "
        "jaccard oracle_regret oracle_regret_actionable norm_regret oracle_regret_median oracle_positive_rate"
    )
    for item in summaries:
        print(
            f"{item.label:>13s} "
            f"{fmt(item.clean_accuracy):>9s} "
            f"{item.rows_analyzed:>4d} "
            f"{fmt(item.exact_match_rate):>9s} "
            f"{fmt(item.overlap_fraction_mean):>9s} "
            f"{fmt(item.weighted_overlap_fraction_mean):>16s} "
            f"{item.actionable_rows:>15d} "
            f"{fmt(item.actionable_overlap_fraction_mean):>18s} "
            f"{fmt(item.actionable_weighted_overlap_fraction_mean):>27s} "
            f"{fmt(item.weighted_exact_match_rate):>14s} "
            f"{fmt(item.actionable_weighted_exact_match_rate):>25s} "
            f"{fmt(item.jaccard_mean):>9s} "
            f"{fmt(item.oracle_regret_mean):>13s} "
            f"{fmt(item.oracle_regret_actionable_mean):>24s} "
            f"{fmt(item.normalized_oracle_regret_mean):>11s} "
            f"{fmt(item.oracle_regret_median):>20s} "
            f"{fmt(item.oracle_positive_rate):>20s}"
        )


def print_detailed_per_block(block_summaries: list[LocalOracleBlockSummary]) -> None:
    if not block_summaries:
        return
    print()
    print("per_block_detailed")
    print(
        "label block rows exact overlap weighted_overlap actionable_rows actionable_overlap "
        "actionable_weighted_overlap weighted_exact actionable_weighted_exact "
        "jaccard oracle_regret oracle_regret_actionable norm_regret oracle_positive_rate"
    )
    for item in block_summaries:
        print(
            f"{item.label:>13s} "
            f"{item.block_index:>5d} "
            f"{item.rows_analyzed:>4d} "
            f"{fmt(item.exact_match_rate):>9s} "
            f"{fmt(item.overlap_fraction_mean):>9s} "
            f"{fmt(item.weighted_overlap_fraction_mean):>16s} "
            f"{item.actionable_rows:>15d} "
            f"{fmt(item.actionable_overlap_fraction_mean):>18s} "
            f"{fmt(item.actionable_weighted_overlap_fraction_mean):>27s} "
            f"{fmt(item.weighted_exact_match_rate):>14s} "
            f"{fmt(item.actionable_weighted_exact_match_rate):>25s} "
            f"{fmt(item.jaccard_mean):>9s} "
            f"{fmt(item.oracle_regret_mean):>13s} "
            f"{fmt(item.oracle_regret_actionable_mean):>24s} "
            f"{fmt(item.normalized_oracle_regret_mean):>11s} "
            f"{fmt(item.oracle_positive_rate):>20s}"
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
    if int(script_args.oracle_forward_batch_size) <= 0:
        raise ValueError("oracle-forward-batch-size must be > 0")
    if float(script_args.actionable_oracle_eps) < 0.0:
        raise ValueError("actionable-oracle-eps must be >= 0")
    if float(script_args.normalized_regret_eps) <= 0.0:
        raise ValueError("normalized-regret-eps must be > 0")
    if math.comb(2 * int(script_args.boundary_width), int(script_args.boundary_width)) > int(script_args.oracle_max_combinations):
        raise ValueError(
            "boundary-width is too large for oracle-max-combinations; reduce --boundary-width or increase --oracle-max-combinations"
        )

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

    summaries: list[LocalOracleSummary] = []
    block_summaries: list[LocalOracleBlockSummary] = []
    all_records: list[LocalOracleRecord] = []

    for spec in tqdm(specs, desc="surrogates", total=len(specs)):
        summary, records, spec_block_summaries = analyze_spec(
            spec=spec,
            train_args=train_args,
            init_loader=init_loader,
            batches=batches,
            transform=transform,
            device=device,
            block_index=script_args.block_index,
            max_rows_per_block=int(script_args.max_rows_per_block),
            boundary_width=int(script_args.boundary_width),
            oracle_forward_batch_size=int(script_args.oracle_forward_batch_size),
            oracle_tie_tol=float(script_args.oracle_tie_tol),
            actionable_oracle_eps=float(script_args.actionable_oracle_eps),
            normalized_regret_eps=float(script_args.normalized_regret_eps),
        )
        summaries.append(summary)
        all_records.extend(records)
        block_summaries.extend(spec_block_summaries)

    print(f"split={script_args.split}")
    print(f"batch_index={script_args.batch_index}")
    print(f"num_batches={script_args.num_batches}")
    print(f"block_index={'all' if script_args.block_index is None else script_args.block_index}")
    print(f"max_rows_per_block={script_args.max_rows_per_block}")
    print(f"boundary_width={script_args.boundary_width}")
    print(f"oracle_combinations={math.comb(2 * int(script_args.boundary_width), int(script_args.boundary_width))}")
    print(f"oracle_forward_batch_size={script_args.oracle_forward_batch_size}")
    print(f"actionable_oracle_eps={script_args.actionable_oracle_eps}")
    print(f"normalized_regret_eps={script_args.normalized_regret_eps}")
    print(f"compact_regret_min_spread={script_args.compact_regret_min_spread}")
    print()
    print("Primary ranking metrics: weighted_overlap_f1, actionable_weighted_overlap_f1, actionable_overlap_f1.")
    print("These are top-k local-support metrics only: they do not rank by clean_acc.")
    print("weighted_overlap_f1 weights each row by its oracle improvement gap, so rows with bigger local top-k effect count more.")
    print("actionable_weighted_overlap_f1 further restricts to rows with oracle_regret > actionable_oracle_eps, removing low-gain noise.")
    print("clean_acc and normalized_oracle_regret are printed as diagnostics only.")
    print()
    print_compact_overall(summaries, compact_regret_min_spread=float(script_args.compact_regret_min_spread))
    if script_args.block_index is None:
        print_compact_per_block(block_summaries, compact_regret_min_spread=float(script_args.compact_regret_min_spread))

    if script_args.print_detailed:
        print_detailed_overall(summaries)
        if script_args.block_index is None:
            print_detailed_per_block(block_summaries)

    if script_args.print_record_limit > 0:
        print()
        print("sample records")
        print("label block loader_batch row exact overlap jaccard baseline_loss oracle_loss oracle_regret")
        for row in all_records[: script_args.print_record_limit]:
            print(
                f"{row.surrogate_label:>13s} "
                f"{row.block_index:>5d} "
                f"{row.batch_loader_index:>12d} "
                f"{row.row_flat_index:>3d} "
                f"{int(row.exact_match):>5d} "
                f"{row.overlap_fraction:>7.4f} "
                f"{row.jaccard:>7.4f} "
                f"{row.baseline_loss:>13.6f} "
                f"{row.oracle_loss:>11.6f} "
                f"{row.oracle_regret:>13.6f}"
            )

    if script_args.output_json is not None:
        payload = {
            "split": script_args.split,
            "batch_index": int(script_args.batch_index),
            "num_batches": int(script_args.num_batches),
            "block_index": script_args.block_index,
            "max_rows_per_block": int(script_args.max_rows_per_block),
            "boundary_width": int(script_args.boundary_width),
            "oracle_max_combinations": int(script_args.oracle_max_combinations),
            "oracle_forward_batch_size": int(script_args.oracle_forward_batch_size),
            "oracle_tie_tol": float(script_args.oracle_tie_tol),
            "actionable_oracle_eps": float(script_args.actionable_oracle_eps),
            "normalized_regret_eps": float(script_args.normalized_regret_eps),
            "compact_regret_min_spread": float(script_args.compact_regret_min_spread),
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
            "summaries": [asdict(x) for x in summaries],
            "block_summaries": [asdict(x) for x in block_summaries],
            "records": [asdict(x) for x in all_records],
        }
        script_args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with script_args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if script_args.output_table_prefix is not None:
        prefix = script_args.output_table_prefix.expanduser()
        write_csv(
            Path(f"{prefix}_overall.csv"),
            [asdict(row) for row in summaries],
            [
                "label",
                "clean_accuracy",
                "rows_analyzed",
                "exact_match_rate",
                "overlap_fraction_mean",
                "jaccard_mean",
                "weighted_overlap_fraction_mean",
                "weighted_exact_match_rate",
                "actionable_rows",
                "actionable_overlap_fraction_mean",
                "actionable_exact_match_rate",
                "actionable_weighted_overlap_fraction_mean",
                "actionable_weighted_exact_match_rate",
                "oracle_regret_mean",
                "oracle_regret_median",
                "oracle_regret_actionable_mean",
                "normalized_oracle_regret_mean",
                "oracle_positive_rate",
            ],
        )
        write_csv(
            Path(f"{prefix}_per_block.csv"),
            [asdict(row) for row in block_summaries],
            [
                "label",
                "block_index",
                "rows_analyzed",
                "exact_match_rate",
                "overlap_fraction_mean",
                "jaccard_mean",
                "weighted_overlap_fraction_mean",
                "weighted_exact_match_rate",
                "actionable_rows",
                "actionable_overlap_fraction_mean",
                "actionable_exact_match_rate",
                "actionable_weighted_overlap_fraction_mean",
                "actionable_weighted_exact_match_rate",
                "oracle_regret_mean",
                "oracle_regret_actionable_mean",
                "normalized_oracle_regret_mean",
                "oracle_positive_rate",
            ],
        )
        write_csv(
            Path(f"{prefix}_records.csv"),
            [asdict(row) for row in all_records],
            [
                "surrogate_label",
                "block_index",
                "row_flat_index",
                "batch_loader_index",
                "batch_index",
                "head_index",
                "query_index",
                "boundary_width",
                "candidate_pool_size",
                "exact_match",
                "overlap_fraction",
                "jaccard",
                "baseline_loss",
                "oracle_loss",
                "oracle_regret",
            ],
        )


if __name__ == "__main__":
    main()
