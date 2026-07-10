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
    benefit: float
    correct: bool
    pred_score: float
    pred_good: bool


@dataclass
class BlockSwapSummary:
    block_index: int
    rows_analyzed: int
    swaps_analyzed: int
    correct_swap_ratio: float
    mean_benefit: float
    precision: float | None
    recall: float | None
    top1_hit: float | None
    top3_hit: float | None
    top5_hit: float | None
    benefit_pearson: float | None
    benefit_spearman: float | None


class BlockCapture:
    def __init__(self, block_index: int) -> None:
        self.block_index = int(block_index)
        self.hard_mask: torch.Tensor | None = None
        self.proxy_scores: torch.Tensor | None = None
        self.proxy_scores_grad: torch.Tensor | None = None
        self.selector_mask: torch.Tensor | None = None

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
        self.hard_mask = module._build_selector_mask(topk_indices, row_count).detach().to(torch.bool)
        self.selector_mask = output["selector_mask"]
        if self.selector_mask is not None and self.selector_mask.requires_grad:
            self.selector_mask.retain_grad()

    def capture_proxy(self, proxy_scores: torch.Tensor) -> torch.Tensor:
        self.proxy_scores = proxy_scores
        if proxy_scores.requires_grad:
            proxy_scores.retain_grad()
        return proxy_scores

    def finalize_backward(self) -> None:
        if self.proxy_scores is None or self.proxy_scores.grad is None:
            return
        self.proxy_scores_grad = self.proxy_scores.grad.detach().to(torch.float32)


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Evaluate boundary swaps using the actual surrogate backward of sigmoid-topk / subset-gibbs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--init-split", choices=["train-eval", "valid", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-index", type=int, default=0, help="Start batch index")
    parser.add_argument("--num-batches", type=int, default=1, help="Analyze this many consecutive batches")
    parser.add_argument("--block-index", type=int, default=None, help="Analyze one block only")
    parser.add_argument("--max-rows-per-block", type=int, default=10, help="Analyze at most this many rows per block")
    parser.add_argument("--boundary-width", type=int, default=4, help="Swap candidates from low-selected/high-unselected boundary")
    parser.add_argument("--print-record-limit", type=int, default=0, help="How many detailed swap records to print")
    parser.add_argument("--output-json", type=Path, default=None)
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
    if unexpected:
        print(f"unexpected_keys={len(unexpected)}")
        for key in unexpected[:10]:
            print(f"  unexpected: {key}")


def get_batches(loader, start_batch_index: int, num_batches: int):
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
            _capture.capture(self, q, k, topk, proxy_scores, output)
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


def enable_surrogate_backward(model) -> None:
    for block in model.blocks:
        block.attn.logic_attention.packed_topk.train(True)
        block.attn.logic_attention.selector_majority.train(True)


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
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= eps:
        return None
    return float((x * y).sum().item() / denom.item())


def corr_spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    x_rank = torch.argsort(torch.argsort(x)).to(torch.float32)
    y_rank = torch.argsort(torch.argsort(y)).to(torch.float32)
    return corr_pearson(x_rank, y_rank)


def summarize_block(block_index: int, records: list[SwapRecord]) -> BlockSwapSummary:
    swaps = len(records)
    rows = len({(r.batch_index, r.head_index, r.query_index) for r in records})
    if swaps == 0:
        return BlockSwapSummary(block_index, 0, 0, 0.0, 0.0, None, None, None, None, None, None, None)

    correct_ratio = sum(int(r.correct) for r in records) / float(swaps)
    mean_benefit = sum(r.benefit for r in records) / float(swaps)

    preds = torch.tensor([r.pred_score for r in records], dtype=torch.float32)
    benefits = torch.tensor([r.benefit for r in records], dtype=torch.float32)
    pred_positive = [r for r in records if r.pred_good]
    actual_positive = [r for r in records if r.correct]

    precision = None if not pred_positive else sum(int(r.correct) for r in pred_positive) / float(len(pred_positive))
    recall = None if not actual_positive else sum(int(r.pred_good) for r in actual_positive) / float(len(actual_positive))
    top1_hit = _topk_hit(records, 1)
    top3_hit = _topk_hit(records, 3)
    top5_hit = _topk_hit(records, 5)

    return BlockSwapSummary(
        block_index=block_index,
        rows_analyzed=rows,
        swaps_analyzed=swaps,
        correct_swap_ratio=correct_ratio,
        mean_benefit=mean_benefit,
        precision=precision,
        recall=recall,
        top1_hit=top1_hit,
        top3_hit=top3_hit,
        top5_hit=top5_hit,
        benefit_pearson=corr_pearson(preds, benefits),
        benefit_spearman=corr_spearman(preds, benefits),
    )


def _topk_hit(records: list[SwapRecord], k: int) -> float | None:
    if not records:
        return None
    ordered = sorted(records, key=lambda r: r.pred_score, reverse=True)
    topk = ordered[: min(k, len(ordered))]
    return float(any(r.correct for r in topk))


def analyze_batch(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_index: int,
    captures: list[BlockCapture],
    block_indices: list[int],
    boundary_width: int,
    max_rows_per_block: int,
):
    model.zero_grad(set_to_none=True)
    logits = model(images)
    baseline_loss = F.cross_entropy(logits, labels)
    baseline_loss.backward()

    for capture in captures:
        capture.finalize_backward()

    restore_forwards([])

    return float(baseline_loss.item())


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
    batches = get_batches(run_loader, script_args.batch_index, script_args.num_batches)

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

    target_block_indices = list(range(len(model.blocks))) if script_args.block_index is None else [int(script_args.block_index)]
    all_records: list[SwapRecord] = []
    block_to_records: dict[int, list[SwapRecord]] = {i: [] for i in target_block_indices}
    baseline_losses: list[float] = []

    for batch_idx, (images, labels) in tqdm(batches, desc="batches", total=len(batches)):
        images = transform(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        model.eval()
        enable_surrogate_backward(model)
        captures, fwd_originals, proxy_originals = attach_captures(model)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        baseline_loss = F.cross_entropy(logits, labels)
        baseline_loss.backward()
        baseline_losses.append(float(baseline_loss.item()))

        restore_forwards(fwd_originals)
        restore_proxies(proxy_originals)

        for capture in captures:
            capture.finalize_backward()

        for block_index in target_block_indices:
            capture = captures[block_index]
            if capture.hard_mask is None or capture.proxy_scores is None or capture.proxy_scores_grad is None:
                continue

            flat_mask = capture.hard_mask.reshape(-1, capture.hard_mask.shape[-1])
            flat_scores = capture.proxy_scores.reshape(-1, capture.proxy_scores.shape[-1])
            flat_grad = capture.proxy_scores_grad.reshape(-1, capture.proxy_scores_grad.shape[-1])
            row_iter = range(min(int(script_args.max_rows_per_block), flat_mask.shape[0]))
            row_iter = tqdm(row_iter, desc=f"rows[b{block_index}|batch{batch_idx}]", leave=False)

            for row_flat_index in row_iter:
                row_mask = flat_mask[row_flat_index]
                row_scores = flat_scores[row_flat_index]
                row_grad = flat_grad[row_flat_index]
                k = int(row_mask.to(torch.int32).sum().item())
                width = row_mask.numel()
                if k <= 0 or k >= width:
                    continue

                b = min(int(script_args.boundary_width), k, width - k)
                if b <= 0:
                    continue

                selected_scores = row_scores.masked_fill(~row_mask, float("inf"))
                inside_low = torch.topk(selected_scores, k=b, largest=False, sorted=True).indices
                outside_scores = row_scores.masked_fill(row_mask, float("-inf"))
                outside_high = torch.topk(outside_scores, k=b, largest=True, sorted=True).indices
                coords = unflatten_row_index(row_flat_index, capture.hard_mask.shape[:-1])
                batch_coord, head_coord, query_coord = coords

                for i_idx in range(b):
                    for j_idx in range(b):
                        idx_out = int(inside_low[i_idx].item())
                        idx_in = int(outside_high[j_idx].item())
                        swapped_mask = force_swap(capture.hard_mask, row_flat_index, idx_out, idx_in)
                        with torch.no_grad():
                            with forced_block_mask(model, block_index, swapped_mask):
                                swap_logits = model(images)
                            swap_loss = F.cross_entropy(swap_logits, labels).item()

                        benefit = float(baseline_loss.item() - swap_loss)
                        pred_score = float((row_grad[idx_out] - row_grad[idx_in]).item())
                        pred_good = bool(pred_score > 0.0)
                        correct = bool(benefit > 0.0)

                        record = SwapRecord(
                            block_index=block_index,
                            row_flat_index=row_flat_index,
                            batch_index=int(batch_coord),
                            head_index=int(head_coord),
                            query_index=int(query_coord),
                            inside_rank=i_idx,
                            outside_rank=j_idx,
                            swap_out_index=idx_out,
                            swap_in_index=idx_in,
                            score_out=float(row_scores[idx_out].item()),
                            score_in=float(row_scores[idx_in].item()),
                            benefit=benefit,
                            correct=correct,
                            pred_score=pred_score,
                            pred_good=pred_good,
                        )
                        all_records.append(record)
                        block_to_records[block_index].append(record)

    block_summaries = [summarize_block(i, block_to_records[i]) for i in target_block_indices if block_to_records[i]]
    overall_summary = summarize_block(-1, all_records) if all_records else None

    print(f"checkpoint={checkpoint_path}")
    print(f"step={checkpoint.get('step', 'unknown')}")
    print(f"split={script_args.split}")
    print(f"batch_index={script_args.batch_index}")
    print(f"num_batches={script_args.num_batches}")
    print(f"baseline_loss_mean={sum(baseline_losses)/len(baseline_losses):.6f}" if baseline_losses else "baseline_loss_mean=nan")
    print(f"surrogate_mode={train_args.topk_surrogate_mode}")
    print()
    print("overall")
    print("scope rows swaps pos_ratio precision recall top1_hit top3_hit top5_hit spearman pearson")
    if overall_summary is not None:
        print(
            f"all {overall_summary.rows_analyzed} {overall_summary.swaps_analyzed} {overall_summary.correct_swap_ratio:.6f} "
            f"{overall_summary.precision if overall_summary.precision is not None else float('nan'):.6f} "
            f"{overall_summary.recall if overall_summary.recall is not None else float('nan'):.6f} "
            f"{overall_summary.top1_hit if overall_summary.top1_hit is not None else float('nan'):.6f} "
            f"{overall_summary.top3_hit if overall_summary.top3_hit is not None else float('nan'):.6f} "
            f"{overall_summary.top5_hit if overall_summary.top5_hit is not None else float('nan'):.6f} "
            f"{overall_summary.benefit_spearman if overall_summary.benefit_spearman is not None else float('nan'):.6f} "
            f"{overall_summary.benefit_pearson if overall_summary.benefit_pearson is not None else float('nan'):.6f}"
        )
    print()
    print("per_block")
    print("scope rows swaps pos_ratio precision recall top1_hit top3_hit top5_hit spearman pearson")
    for item in block_summaries:
        print(
            f"{item.block_index} {item.rows_analyzed} {item.swaps_analyzed} {item.correct_swap_ratio:.6f} "
            f"{item.precision if item.precision is not None else float('nan'):.6f} "
            f"{item.recall if item.recall is not None else float('nan'):.6f} "
            f"{item.top1_hit if item.top1_hit is not None else float('nan'):.6f} "
            f"{item.top3_hit if item.top3_hit is not None else float('nan'):.6f} "
            f"{item.top5_hit if item.top5_hit is not None else float('nan'):.6f} "
            f"{item.benefit_spearman if item.benefit_spearman is not None else float('nan'):.6f} "
            f"{item.benefit_pearson if item.benefit_pearson is not None else float('nan'):.6f}"
        )

    if script_args.output_json is not None:
        payload = {
            "checkpoint": str(checkpoint_path),
            "step": checkpoint.get("step", None),
            "split": script_args.split,
            "batch_index": int(script_args.batch_index),
            "num_batches": int(script_args.num_batches),
            "baseline_loss_mean": (sum(baseline_losses) / len(baseline_losses)) if baseline_losses else None,
            "surrogate_mode": train_args.topk_surrogate_mode,
            "model_args": vars(train_args),
            "overall_summary": None if overall_summary is None else asdict(overall_summary),
            "block_summaries": [asdict(x) for x in block_summaries],
            "records": [asdict(x) for x in all_records],
        }
        script_args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with script_args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
