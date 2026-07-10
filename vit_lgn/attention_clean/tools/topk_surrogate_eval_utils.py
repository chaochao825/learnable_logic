from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import MethodType

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BlockSnapshot:
    block_index: int
    hard_mask: torch.Tensor
    scores: torch.Tensor


@dataclass
class ForwardRun:
    logits: torch.Tensor
    loss_value: float
    accuracy: float
    block_snapshots: list[BlockSnapshot]
    loss_values: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor


class BlockCapture:
    def __init__(self, block_index: int) -> None:
        self.block_index = int(block_index)
        self.hard_mask: torch.Tensor | None = None
        self.scores: torch.Tensor | None = None

    def capture_forward(self, module, output: dict[str, torch.Tensor]) -> None:
        topk_indices = output["topk_indices"].detach().to(torch.long)
        row_count = topk_indices.shape[-2]
        self.hard_mask = module._build_selector_mask(topk_indices, row_count).detach().to(torch.bool)
        self.scores = output["xnor_popcount"].detach().to(torch.float32)

    def snapshot(self) -> BlockSnapshot:
        if self.hard_mask is None or self.scores is None:
            raise RuntimeError("block capture is incomplete")
        return BlockSnapshot(
            block_index=self.block_index,
            hard_mask=self.hard_mask.clone(),
            scores=self.scores.clone(),
        )


def attach_eval_captures(model) -> tuple[list[BlockCapture], list[tuple[object, object]]]:
    captures: list[BlockCapture] = []
    originals: list[tuple[object, object]] = []

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
            _capture.capture_forward(self, output)
            return output

        packed_topk.forward = MethodType(wrapped_forward, packed_topk)
        captures.append(capture)

    return captures, originals


def restore_forwards(originals: list[tuple[object, object]]) -> None:
    for module, original_forward in originals:
        module.forward = original_forward


def run_forward_with_captures(model, images: torch.Tensor, labels: torch.Tensor) -> ForwardRun:
    captures, originals = attach_eval_captures(model)
    try:
        with torch.no_grad():
            logits = model(images)
            loss_values = F.cross_entropy(logits, labels, reduction="none")
            loss = loss_values.mean()
            pred = logits.argmax(dim=-1)
            correct = pred == labels
            accuracy = float(correct.to(torch.float32).mean().item())
        snapshots = [capture.snapshot() for capture in captures]
    finally:
        restore_forwards(originals)

    return ForwardRun(
        logits=logits.detach(),
        loss_value=float(loss.item()),
        accuracy=accuracy,
        block_snapshots=snapshots,
        loss_values=loss_values.detach(),
        predictions=pred.detach(),
        correct=correct.detach(),
    )


def split_forward_run(run: ForwardRun, batch_size: int) -> list[ForwardRun]:
    total = int(run.logits.shape[0])
    if batch_size <= 0 or total % batch_size != 0:
        raise ValueError("batch_size must divide total run batch size")

    group_count = total // batch_size
    groups: list[ForwardRun] = []
    for group_index in range(group_count):
        start = group_index * batch_size
        end = start + batch_size
        snapshots = [
            BlockSnapshot(
                block_index=snapshot.block_index,
                hard_mask=snapshot.hard_mask[start:end].clone(),
                scores=snapshot.scores[start:end].clone(),
            )
            for snapshot in run.block_snapshots
        ]
        group_logits = run.logits[start:end].clone()
        group_loss_values = run.loss_values[start:end].clone()
        group_predictions = run.predictions[start:end].clone()
        group_correct = run.correct[start:end].clone()
        groups.append(
            ForwardRun(
                logits=group_logits,
                loss_value=float(group_loss_values.mean().item()),
                accuracy=float(group_correct.to(torch.float32).mean().item()),
                block_snapshots=snapshots,
                loss_values=group_loss_values,
                predictions=group_predictions,
                correct=group_correct,
            )
        )
    return groups


def build_rank_keys(row_scores: torch.Tensor) -> torch.Tensor:
    width = int(row_scores.numel())
    row_ids = torch.arange(width, device=row_scores.device, dtype=torch.int64)
    return row_scores.round().to(torch.int64) * int(width + 1) + row_ids


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

    keys = build_rank_keys(row_scores)
    max_key = torch.iinfo(keys.dtype).max
    min_key = torch.iinfo(keys.dtype).min

    selected_keys = keys.masked_fill(~row_mask, max_key)
    inside_low = torch.topk(selected_keys, k=band, dim=-1, largest=False, sorted=True).indices.to(torch.long)

    unselected_keys = keys.masked_fill(row_mask, min_key)
    outside_high = torch.topk(unselected_keys, k=band, dim=-1, largest=True, sorted=True).indices.to(torch.long)
    return inside_low, outside_high


def force_row_candidate_subset(
    base_mask: torch.Tensor,
    row_flat_index: int,
    candidate_pool: torch.Tensor,
    chosen_candidates: torch.Tensor,
) -> torch.Tensor:
    forced = base_mask.clone()
    flat = forced.reshape(-1, forced.shape[-1])
    row_mask = flat[row_flat_index].clone()
    row_mask[candidate_pool] = False
    row_mask[chosen_candidates] = True
    flat[row_flat_index] = row_mask
    return forced


def jaccard_and_flip_rate(mask_a: torch.Tensor, mask_b: torch.Tensor) -> tuple[float, float, bool]:
    mask_a = mask_a.to(torch.bool)
    mask_b = mask_b.to(torch.bool)
    intersection = int(torch.logical_and(mask_a, mask_b).to(torch.int32).sum().item())
    union = int(torch.logical_or(mask_a, mask_b).to(torch.int32).sum().item())
    exact_match = bool(torch.equal(mask_a, mask_b))
    jaccard = 1.0 if union == 0 else float(intersection) / float(union)

    selected = int(mask_a.to(torch.int32).sum().item())
    if selected <= 0:
        flip_rate = 0.0
    else:
        hamming = int(torch.logical_xor(mask_a, mask_b).to(torch.int32).sum().item())
        flip_rate = float(hamming) / float(2 * selected)
    return jaccard, flip_rate, exact_match


def jaccard_and_flip_rate_batch(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks_a = masks_a.to(torch.bool)
    masks_b = masks_b.to(torch.bool)
    intersection = torch.logical_and(masks_a, masks_b).to(torch.int32).sum(dim=-1)
    union = torch.logical_or(masks_a, masks_b).to(torch.int32).sum(dim=-1)
    exact_match = torch.all(masks_a == masks_b, dim=-1)

    jaccard = torch.ones_like(intersection, dtype=torch.float32)
    valid_union = union > 0
    jaccard[valid_union] = intersection[valid_union].to(torch.float32) / union[valid_union].to(torch.float32)

    selected = masks_a.to(torch.int32).sum(dim=-1)
    hamming = torch.logical_xor(masks_a, masks_b).to(torch.int32).sum(dim=-1)
    flip_rate = torch.zeros_like(intersection, dtype=torch.float32)
    valid_selected = selected > 0
    flip_rate[valid_selected] = hamming[valid_selected].to(torch.float32) / (
        2.0 * selected[valid_selected].to(torch.float32)
    )
    return jaccard, flip_rate, exact_match
