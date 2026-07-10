from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import sys
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
class GradSummary:
    label: str
    rows_analyzed: int
    valid_grad_rows: int
    zero_grad_row_ratio: float
    grad_nonzero_ratio_mean: float
    selected_grad_mass_fraction_mean: float | None
    selected_abs_mean: float
    unselected_abs_mean: float
    selected_to_unselected_abs_ratio_mean: float | None
    inside_boundary_mass_fraction_mean: float | None
    outside_boundary_mass_fraction_mean: float | None
    kth_pair_mass_fraction_mean: float | None
    kth_pair_density_ratio_mean: float | None
    boundary_band_mass_fraction_mean: float | None
    boundary_band_density_ratio_mean: float | None
    grad_entropy_mean: float | None
    grad_gini_mean: float | None


@dataclass
class GradBlockSummary:
    label: str
    block_index: int
    rows_analyzed: int
    valid_grad_rows: int
    zero_grad_row_ratio: float
    grad_nonzero_ratio_mean: float
    selected_grad_mass_fraction_mean: float | None
    selected_abs_mean: float
    unselected_abs_mean: float
    selected_to_unselected_abs_ratio_mean: float | None
    inside_boundary_mass_fraction_mean: float | None
    outside_boundary_mass_fraction_mean: float | None
    kth_pair_mass_fraction_mean: float | None
    kth_pair_density_ratio_mean: float | None
    boundary_band_mass_fraction_mean: float | None
    boundary_band_density_ratio_mean: float | None
    grad_entropy_mean: float | None
    grad_gini_mean: float | None


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


class GradDistributionAccumulator:
    def __init__(self, boundary_width: int, grad_eps: float) -> None:
        self.boundary_width = int(boundary_width)
        self.grad_eps = float(grad_eps)
        self.rows_analyzed = 0
        self.valid_grad_rows = 0
        self.grad_nonzero_ratio_sum = 0.0
        self.selected_abs_sum = 0.0
        self.selected_abs_count = 0
        self.unselected_abs_sum = 0.0
        self.unselected_abs_count = 0
        self.selected_to_unselected_abs_ratio_sum = 0.0
        self.selected_to_unselected_abs_ratio_rows = 0
        self.selected_grad_mass_fraction_sum = 0.0
        self.inside_boundary_mass_fraction_sum = 0.0
        self.outside_boundary_mass_fraction_sum = 0.0
        self.kth_pair_mass_fraction_sum = 0.0
        self.kth_pair_density_ratio_sum = 0.0
        self.boundary_band_mass_fraction_sum = 0.0
        self.boundary_band_density_ratio_sum = 0.0
        self.grad_entropy_sum = 0.0
        self.grad_gini_sum = 0.0
        self.profile_sum: torch.Tensor | None = None
        self.profile_rows = 0

    def add_row(
        self,
        hard_mask: torch.Tensor,
        proxy_scores: torch.Tensor,
        proxy_grad: torch.Tensor,
    ) -> None:
        width = int(hard_mask.numel())
        limit = int(hard_mask.to(torch.int32).sum().item())
        if limit <= 0 or limit >= width:
            return

        abs_grad = proxy_grad.abs().to(torch.float32)
        selected_mask = hard_mask.to(torch.bool)
        unselected_mask = ~selected_mask

        self.rows_analyzed += 1
        self.grad_nonzero_ratio_sum += float((abs_grad > self.grad_eps).to(torch.float32).mean().item())

        selected_count = int(selected_mask.sum().item())
        unselected_count = int(unselected_mask.sum().item())
        if selected_count > 0:
            self.selected_abs_sum += float(abs_grad[selected_mask].sum().item())
            self.selected_abs_count += selected_count
        if unselected_count > 0:
            self.unselected_abs_sum += float(abs_grad[unselected_mask].sum().item())
            self.unselected_abs_count += unselected_count

        selected_abs_mean_row = float(abs_grad[selected_mask].mean().item()) if selected_count > 0 else 0.0
        unselected_abs_mean_row = float(abs_grad[unselected_mask].mean().item()) if unselected_count > 0 else 0.0
        if unselected_abs_mean_row > self.grad_eps:
            self.selected_to_unselected_abs_ratio_sum += selected_abs_mean_row / unselected_abs_mean_row
            self.selected_to_unselected_abs_ratio_rows += 1

        total_abs = float(abs_grad.sum().item())
        if total_abs <= self.grad_eps:
            return

        self.valid_grad_rows += 1
        norm_abs_grad = abs_grad / total_abs

        selected_grad_mass_fraction = float(norm_abs_grad[selected_mask].sum().item())
        self.selected_grad_mass_fraction_sum += selected_grad_mass_fraction

        order = torch.argsort(proxy_scores, descending=True)
        sorted_norm_abs_grad = norm_abs_grad[order]
        self._add_profile(sorted_norm_abs_grad)

        bw = min(self.boundary_width, limit, width - limit)
        if bw > 0:
            inside_boundary_mass = float(sorted_norm_abs_grad[limit - bw : limit].sum().item())
            outside_boundary_mass = float(sorted_norm_abs_grad[limit : limit + bw].sum().item())
            boundary_band_mass = inside_boundary_mass + outside_boundary_mass
            expected_boundary_mass = float(2 * bw) / float(width)

            self.inside_boundary_mass_fraction_sum += inside_boundary_mass
            self.outside_boundary_mass_fraction_sum += outside_boundary_mass
            self.boundary_band_mass_fraction_sum += boundary_band_mass
            self.boundary_band_density_ratio_sum += boundary_band_mass / max(expected_boundary_mass, self.grad_eps)

        kth_pair_mass = float(sorted_norm_abs_grad[limit - 1 : limit + 1].sum().item())
        expected_kth_pair_mass = float(2) / float(width)
        self.kth_pair_mass_fraction_sum += kth_pair_mass
        self.kth_pair_density_ratio_sum += kth_pair_mass / max(expected_kth_pair_mass, self.grad_eps)

        entropy = self._normalized_entropy(sorted_norm_abs_grad)
        gini = self._gini(sorted_norm_abs_grad)
        self.grad_entropy_sum += entropy
        self.grad_gini_sum += gini

    def _add_profile(self, sorted_norm_abs_grad: torch.Tensor) -> None:
        if self.profile_sum is None:
            self.profile_sum = torch.zeros_like(sorted_norm_abs_grad, dtype=torch.float64)
        self.profile_sum += sorted_norm_abs_grad.to(torch.float64)
        self.profile_rows += 1

    @staticmethod
    def _normalized_entropy(prob: torch.Tensor, eps: float = 1e-12) -> float:
        width = int(prob.numel())
        if width <= 1:
            return 0.0
        entropy = -(prob * (prob.clamp_min(eps).log())).sum()
        return float((entropy / math.log(width)).item())

    @staticmethod
    def _gini(prob: torch.Tensor, eps: float = 1e-12) -> float:
        x = torch.sort(prob.to(torch.float64).flatten())[0]
        n = int(x.numel())
        total = float(x.sum().item())
        if n == 0 or total <= eps:
            return 0.0
        index = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        gini = (2.0 * (index * x).sum() / (n * x.sum())) - (n + 1.0) / n
        return float(gini.item())

    def summary(self, label: str) -> GradSummary:
        zero_grad_row_ratio = 0.0
        grad_nonzero_ratio_mean = 0.0
        if self.rows_analyzed > 0:
            zero_grad_row_ratio = 1.0 - float(self.valid_grad_rows) / float(self.rows_analyzed)
            grad_nonzero_ratio_mean = self.grad_nonzero_ratio_sum / float(self.rows_analyzed)

        selected_abs_mean = self.selected_abs_sum / float(self.selected_abs_count) if self.selected_abs_count > 0 else 0.0
        unselected_abs_mean = self.unselected_abs_sum / float(self.unselected_abs_count) if self.unselected_abs_count > 0 else 0.0

        selected_to_unselected_abs_ratio_mean = None
        if self.selected_to_unselected_abs_ratio_rows > 0:
            selected_to_unselected_abs_ratio_mean = (
                self.selected_to_unselected_abs_ratio_sum / float(self.selected_to_unselected_abs_ratio_rows)
            )

        selected_grad_mass_fraction_mean = None
        inside_boundary_mass_fraction_mean = None
        outside_boundary_mass_fraction_mean = None
        kth_pair_mass_fraction_mean = None
        kth_pair_density_ratio_mean = None
        boundary_band_mass_fraction_mean = None
        boundary_band_density_ratio_mean = None
        grad_entropy_mean = None
        grad_gini_mean = None

        if self.valid_grad_rows > 0:
            denom = float(self.valid_grad_rows)
            selected_grad_mass_fraction_mean = self.selected_grad_mass_fraction_sum / denom
            inside_boundary_mass_fraction_mean = self.inside_boundary_mass_fraction_sum / denom
            outside_boundary_mass_fraction_mean = self.outside_boundary_mass_fraction_sum / denom
            kth_pair_mass_fraction_mean = self.kth_pair_mass_fraction_sum / denom
            kth_pair_density_ratio_mean = self.kth_pair_density_ratio_sum / denom
            boundary_band_mass_fraction_mean = self.boundary_band_mass_fraction_sum / denom
            boundary_band_density_ratio_mean = self.boundary_band_density_ratio_sum / denom
            grad_entropy_mean = self.grad_entropy_sum / denom
            grad_gini_mean = self.grad_gini_sum / denom

        return GradSummary(
            label=label,
            rows_analyzed=self.rows_analyzed,
            valid_grad_rows=self.valid_grad_rows,
            zero_grad_row_ratio=zero_grad_row_ratio,
            grad_nonzero_ratio_mean=grad_nonzero_ratio_mean,
            selected_grad_mass_fraction_mean=selected_grad_mass_fraction_mean,
            selected_abs_mean=selected_abs_mean,
            unselected_abs_mean=unselected_abs_mean,
            selected_to_unselected_abs_ratio_mean=selected_to_unselected_abs_ratio_mean,
            inside_boundary_mass_fraction_mean=inside_boundary_mass_fraction_mean,
            outside_boundary_mass_fraction_mean=outside_boundary_mass_fraction_mean,
            kth_pair_mass_fraction_mean=kth_pair_mass_fraction_mean,
            kth_pair_density_ratio_mean=kth_pair_density_ratio_mean,
            boundary_band_mass_fraction_mean=boundary_band_mass_fraction_mean,
            boundary_band_density_ratio_mean=boundary_band_density_ratio_mean,
            grad_entropy_mean=grad_entropy_mean,
            grad_gini_mean=grad_gini_mean,
        )

    def block_summary(self, label: str, block_index: int) -> GradBlockSummary:
        summary = self.summary(label)
        return GradBlockSummary(
            label=label,
            block_index=int(block_index),
            rows_analyzed=summary.rows_analyzed,
            valid_grad_rows=summary.valid_grad_rows,
            zero_grad_row_ratio=summary.zero_grad_row_ratio,
            grad_nonzero_ratio_mean=summary.grad_nonzero_ratio_mean,
            selected_grad_mass_fraction_mean=summary.selected_grad_mass_fraction_mean,
            selected_abs_mean=summary.selected_abs_mean,
            unselected_abs_mean=summary.unselected_abs_mean,
            selected_to_unselected_abs_ratio_mean=summary.selected_to_unselected_abs_ratio_mean,
            inside_boundary_mass_fraction_mean=summary.inside_boundary_mass_fraction_mean,
            outside_boundary_mass_fraction_mean=summary.outside_boundary_mass_fraction_mean,
            kth_pair_mass_fraction_mean=summary.kth_pair_mass_fraction_mean,
            kth_pair_density_ratio_mean=summary.kth_pair_density_ratio_mean,
            boundary_band_mass_fraction_mean=summary.boundary_band_mass_fraction_mean,
            boundary_band_density_ratio_mean=summary.boundary_band_density_ratio_mean,
            grad_entropy_mean=summary.grad_entropy_mean,
            grad_gini_mean=summary.grad_gini_mean,
        )

    def rank_profile_rows(self, label: str, block_index: int | None) -> list[dict[str, Any]]:
        if self.profile_sum is None or self.profile_rows <= 0:
            return []
        profile_mean = (self.profile_sum / float(self.profile_rows)).to(torch.float32)
        rows: list[dict[str, Any]] = []
        for rank_index, mean_mass in enumerate(profile_mean.tolist(), start=1):
            rows.append(
                {
                    "label": label,
                    "block_index": block_index,
                    "rank": rank_index,
                    "mean_abs_grad_mass": float(mean_mass),
                }
            )
        return rows


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Compare proxy-score gradient distribution across kth-raw, kth-norm, sigmoid-topk and subset-gibbs.",
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
    parser.add_argument("--block-index", type=int, default=None, help="Analyze one block only; default analyzes all blocks")
    parser.add_argument("--max-rows-per-block", type=int, default=10)
    parser.add_argument("--boundary-width", type=int, default=2, help="Use this many positions on each side of the top-k boundary")
    parser.add_argument("--grad-eps", type=float, default=1e-12, help="Threshold for treating gradient magnitude as zero")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--output-table-prefix",
        type=Path,
        default=None,
        help="Write CSV tables to <prefix>_overall.csv, <prefix>_per_block.csv and <prefix>_rank_profile.csv",
    )
    parser.add_argument("--print-config", action=argparse.BooleanOptionalAction, default=False)
    script_args, remaining = parser.parse_known_args(argv)

    saved_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        train_args = parse_train_args()
    finally:
        sys.argv = saved_argv

    return script_args, train_args


def fmt(v: float | None) -> str:
    return "nan" if v is None else f"{v:.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def build_specs(script_args: argparse.Namespace) -> list[SurrogateSpec]:
    def resolve(path: Path | None) -> Path:
        if path is not None:
            return path.expanduser().resolve()
        if script_args.checkpoint is None:
            raise ValueError("Need either --checkpoint or per-surrogate checkpoint args")
        return script_args.checkpoint.expanduser().resolve()

    specs = [
        SurrogateSpec("kth-raw", resolve(script_args.checkpoint_kth_raw), "kth", False),
        SurrogateSpec("kth-norm", resolve(script_args.checkpoint_kth_norm), "kth", True),
        SurrogateSpec("sigmoid-topk", resolve(script_args.checkpoint_sigmoid_topk), "sigmoid-topk", True),
        SurrogateSpec("subset-gibbs", resolve(script_args.checkpoint_subset_gibbs), "subset-gibbs", True),
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


def summarize_block(
    accumulator: GradDistributionAccumulator,
    label: str,
    block_index: int,
) -> GradBlockSummary:
    return accumulator.block_summary(label, block_index)


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
    grad_eps: float,
) -> tuple[GradSummary, list[GradBlockSummary], list[dict[str, Any]]]:
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

    overall = GradDistributionAccumulator(boundary_width=boundary_width, grad_eps=grad_eps)
    target_blocks = list(range(len(model.blocks))) if block_index is None else [int(block_index)]
    per_block = {
        bidx: GradDistributionAccumulator(boundary_width=boundary_width, grad_eps=grad_eps)
        for bidx in target_blocks
    }

    for batch_loader_index, (images, labels) in tqdm(
        batches,
        desc=f"{spec.label}:batches",
        total=len(batches),
        leave=False,
    ):
        del batch_loader_index
        images = transform(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        model.eval()
        enable_surrogate_backward(model)
        captures, fwd_originals, proxy_originals = attach_captures(model)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        restore_forwards(fwd_originals)
        restore_proxies(proxy_originals)
        for capture in captures:
            capture.finalize_backward()

        model.zero_grad(set_to_none=True)

        for bidx in target_blocks:
            capture = captures[bidx]
            if capture.hard_mask is None or capture.proxy_scores is None or capture.proxy_scores_grad is None:
                continue

            flat_mask = capture.hard_mask.reshape(-1, capture.hard_mask.shape[-1])
            flat_scores = capture.proxy_scores.detach().reshape(-1, capture.proxy_scores.shape[-1]).to(torch.float32)
            flat_grad = capture.proxy_scores_grad.reshape(-1, capture.proxy_scores_grad.shape[-1]).to(torch.float32)
            rows_to_scan = min(int(max_rows_per_block), flat_mask.shape[0])

            for row_flat_index in range(rows_to_scan):
                row_mask = flat_mask[row_flat_index]
                row_scores = flat_scores[row_flat_index]
                row_grad = flat_grad[row_flat_index]
                overall.add_row(row_mask, row_scores, row_grad)
                per_block[bidx].add_row(row_mask, row_scores, row_grad)

    overall_summary = overall.summary(spec.label)
    block_summaries = [summarize_block(per_block[bidx], spec.label, bidx) for bidx in sorted(per_block.keys())]

    rank_profile_rows = overall.rank_profile_rows(spec.label, None)
    for bidx in sorted(per_block.keys()):
        rank_profile_rows.extend(per_block[bidx].rank_profile_rows(spec.label, bidx))

    return overall_summary, block_summaries, rank_profile_rows


def print_overall_table(summaries: list[GradSummary]) -> None:
    print("overall")
    print(
        "label rows valid zero_grad nonzero_ratio sel_mass sel_abs unsel_abs sel_over_unsel "
        "entropy gini inside_boundary_mass outside_boundary_mass kthpair_mass kthpair_density boundary_mass boundary_density"
    )
    for item in summaries:
        print(
            f"{item.label:>13s} "
            f"{item.rows_analyzed:>5d} "
            f"{item.valid_grad_rows:>5d} "
            f"{item.zero_grad_row_ratio:>9.6f} "
            f"{item.grad_nonzero_ratio_mean:>12.6f} "
            f"{fmt(item.selected_grad_mass_fraction_mean):>8s} "
            f"{item.selected_abs_mean:>8.6f} "
            f"{item.unselected_abs_mean:>9.6f} "
            f"{fmt(item.selected_to_unselected_abs_ratio_mean):>14s} "
            f"{fmt(item.grad_entropy_mean):>8s} "
            f"{fmt(item.grad_gini_mean):>8s} "
            f"{fmt(item.inside_boundary_mass_fraction_mean):>20s} "
            f"{fmt(item.outside_boundary_mass_fraction_mean):>21s} "
            f"{fmt(item.kth_pair_mass_fraction_mean):>12s} "
            f"{fmt(item.kth_pair_density_ratio_mean):>15s} "
            f"{fmt(item.boundary_band_mass_fraction_mean):>13s} "
            f"{fmt(item.boundary_band_density_ratio_mean):>16s}"
        )


def print_per_block_table(block_summaries: list[GradBlockSummary]) -> None:
    if not block_summaries:
        return
    print()
    print("per_block")
    print(
        "label block rows valid zero_grad nonzero_ratio sel_mass sel_abs unsel_abs sel_over_unsel "
        "entropy gini inside_boundary_mass outside_boundary_mass kthpair_mass kthpair_density boundary_mass boundary_density"
    )
    for item in block_summaries:
        print(
            f"{item.label:>13s} "
            f"{item.block_index:>5d} "
            f"{item.rows_analyzed:>5d} "
            f"{item.valid_grad_rows:>5d} "
            f"{item.zero_grad_row_ratio:>9.6f} "
            f"{item.grad_nonzero_ratio_mean:>12.6f} "
            f"{fmt(item.selected_grad_mass_fraction_mean):>8s} "
            f"{item.selected_abs_mean:>8.6f} "
            f"{item.unselected_abs_mean:>9.6f} "
            f"{fmt(item.selected_to_unselected_abs_ratio_mean):>14s} "
            f"{fmt(item.grad_entropy_mean):>8s} "
            f"{fmt(item.grad_gini_mean):>8s} "
            f"{fmt(item.inside_boundary_mass_fraction_mean):>20s} "
            f"{fmt(item.outside_boundary_mass_fraction_mean):>21s} "
            f"{fmt(item.kth_pair_mass_fraction_mean):>12s} "
            f"{fmt(item.kth_pair_density_ratio_mean):>15s} "
            f"{fmt(item.boundary_band_mass_fraction_mean):>13s} "
            f"{fmt(item.boundary_band_density_ratio_mean):>16s}"
        )


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

    summaries: list[GradSummary] = []
    block_summaries: list[GradBlockSummary] = []
    rank_profile_rows: list[dict[str, Any]] = []

    for spec in tqdm(specs, desc="surrogates", total=len(specs)):
        summary, spec_block_summaries, spec_rank_profile_rows = analyze_spec(
            spec=spec,
            train_args=train_args,
            init_loader=init_loader,
            batches=batches,
            transform=transform,
            device=device,
            block_index=script_args.block_index,
            max_rows_per_block=int(script_args.max_rows_per_block),
            boundary_width=int(script_args.boundary_width),
            grad_eps=float(script_args.grad_eps),
        )
        summaries.append(summary)
        block_summaries.extend(spec_block_summaries)
        rank_profile_rows.extend(spec_rank_profile_rows)

    print(f"split={script_args.split}")
    print(f"batch_index={script_args.batch_index}")
    print(f"num_batches={script_args.num_batches}")
    print(f"block_index={'all' if script_args.block_index is None else script_args.block_index}")
    print(f"max_rows_per_block={script_args.max_rows_per_block}")
    print(f"boundary_width={script_args.boundary_width}")
    print(f"grad_eps={script_args.grad_eps}")
    print()
    print("Interpretation: lower entropy + higher gini/density means gradients are more concentrated.")
    print("All `*_mass` columns are normalized absolute gradient mass from `proxy_scores.grad`, not raw scores.")
    print("`inside_boundary_mass` is top-k inner-side boundary mass; `outside_boundary_mass` is top-k outer-side boundary mass.")
    print("`kthpair_density` measures concentration on rank-k and rank-(k+1); `boundary_density` measures a wider boundary band.")
    print()
    print_overall_table(summaries)
    if script_args.block_index is None:
        print_per_block_table(block_summaries)

    if script_args.output_json is not None:
        payload = {
            "split": script_args.split,
            "batch_index": int(script_args.batch_index),
            "num_batches": int(script_args.num_batches),
            "block_index": script_args.block_index,
            "max_rows_per_block": int(script_args.max_rows_per_block),
            "boundary_width": int(script_args.boundary_width),
            "grad_eps": float(script_args.grad_eps),
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
            "rank_profile_rows": rank_profile_rows,
        }
        script_args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with script_args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if script_args.output_table_prefix is not None:
        prefix = script_args.output_table_prefix.expanduser()
        write_csv(
            Path(f"{prefix}_overall.csv"),
            [asdict(item) for item in summaries],
            [
                "label",
                "rows_analyzed",
                "valid_grad_rows",
                "zero_grad_row_ratio",
                "grad_nonzero_ratio_mean",
                "selected_grad_mass_fraction_mean",
                "selected_abs_mean",
                "unselected_abs_mean",
                "selected_to_unselected_abs_ratio_mean",
                "inside_boundary_mass_fraction_mean",
                "outside_boundary_mass_fraction_mean",
                "kth_pair_mass_fraction_mean",
                "kth_pair_density_ratio_mean",
                "boundary_band_mass_fraction_mean",
                "boundary_band_density_ratio_mean",
                "grad_entropy_mean",
                "grad_gini_mean",
            ],
        )
        write_csv(
            Path(f"{prefix}_per_block.csv"),
            [asdict(item) for item in block_summaries],
            [
                "label",
                "block_index",
                "rows_analyzed",
                "valid_grad_rows",
                "zero_grad_row_ratio",
                "grad_nonzero_ratio_mean",
                "selected_grad_mass_fraction_mean",
                "selected_abs_mean",
                "unselected_abs_mean",
                "selected_to_unselected_abs_ratio_mean",
                "inside_boundary_mass_fraction_mean",
                "outside_boundary_mass_fraction_mean",
                "kth_pair_mass_fraction_mean",
                "kth_pair_density_ratio_mean",
                "boundary_band_mass_fraction_mean",
                "boundary_band_density_ratio_mean",
                "grad_entropy_mean",
                "grad_gini_mean",
            ],
        )
        write_csv(
            Path(f"{prefix}_rank_profile.csv"),
            rank_profile_rows,
            ["label", "block_index", "rank", "mean_abs_grad_mass"],
        )


if __name__ == "__main__":
    main()
