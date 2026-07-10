from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_topk_boundary_swap_surrogates import (  # noqa: E402
    SurrogateSpec,
    build_specs,
    get_batches,
    infer_and_report_mismatches,
    load_checkpoint_state,
    ordered_labels,
    select_loader,
)
from data_pipeline import load_dataset  # noqa: E402
from topk_surrogate_eval_utils import (  # noqa: E402
    run_forward_with_captures,
    split_forward_run,
)
from train_logic_vit_tiny import (  # noqa: E402
    build_model,
    initialize_lazy_modules,
    parse_args as parse_train_args,
    set_attention_soft_train_temperatures,
    set_deterministic,
)


@dataclass
class RobustnessSummary:
    label: str
    clean_accuracy: float
    clean_nll_mean: float
    clean_brier_mean: float
    perturbed_accuracy_mean: float
    accuracy_drop_mean: float
    perturbed_nll_mean: float
    nll_increase_mean: float
    perturbed_brier_mean: float
    brier_increase_mean: float
    perturbed_true_class_prob_mean: float
    perturbed_true_class_margin_mean: float
    clean_correct_retention_mean: float
    prediction_consistency_mean: float
    correct_class_logit_drop_mean: float
    target_margin_drop_mean: float
    js_divergence_mean: float
    perturbation_evals: int


class RobustnessAccumulator:
    def __init__(self) -> None:
        self.clean_accuracies: list[float] = []
        self.clean_nlls: list[float] = []
        self.clean_briers: list[float] = []
        self.perturbed_accuracies: list[float] = []
        self.perturbed_nlls: list[float] = []
        self.nll_increases: list[float] = []
        self.perturbed_briers: list[float] = []
        self.brier_increases: list[float] = []
        self.perturbed_true_class_probs: list[float] = []
        self.perturbed_true_class_margins: list[float] = []
        self.clean_correct_retentions: list[float] = []
        self.prediction_consistencies: list[float] = []
        self.correct_class_logit_drops: list[float] = []
        self.target_margin_drops: list[float] = []
        self.js_divergences: list[float] = []

    def add_clean_eval(self, clean_run, labels: torch.Tensor) -> None:
        self.clean_accuracies.append(float(clean_run.accuracy))
        clean_nll = float(clean_run.loss_values.mean().item())
        clean_brier = float(compute_brier_score(clean_run.logits, labels).mean().item())
        self.clean_nlls.append(clean_nll)
        self.clean_briers.append(clean_brier)

    def add_noisy_eval(self, clean_run, noisy_run, labels: torch.Tensor) -> None:
        self.perturbed_accuracies.append(float(noisy_run.accuracy))
        noisy_nll = float(noisy_run.loss_values.mean().item())
        clean_nll = float(clean_run.loss_values.mean().item())
        self.perturbed_nlls.append(noisy_nll)
        self.nll_increases.append(noisy_nll - clean_nll)

        noisy_brier = float(compute_brier_score(noisy_run.logits, labels).mean().item())
        clean_brier = float(compute_brier_score(clean_run.logits, labels).mean().item())
        self.perturbed_briers.append(noisy_brier)
        self.brier_increases.append(noisy_brier - clean_brier)

        noisy_true_prob = float(gather_true_class_prob(noisy_run.logits, labels).mean().item())
        noisy_margin = float(compute_true_class_margin(noisy_run.logits, labels).mean().item())
        self.perturbed_true_class_probs.append(noisy_true_prob)
        self.perturbed_true_class_margins.append(noisy_margin)

        clean_pred = clean_run.predictions
        noisy_pred = noisy_run.predictions
        pred_consistency = (clean_pred == noisy_pred).to(torch.float32).mean().item()
        self.prediction_consistencies.append(float(pred_consistency))

        clean_correct = clean_run.correct
        if bool(clean_correct.any()):
            clean_correct_retention = noisy_run.correct[clean_correct].to(torch.float32).mean().item()
        else:
            clean_correct_retention = math.nan
        self.clean_correct_retentions.append(float(clean_correct_retention))

        clean_true_logits = gather_true_class_logits(clean_run.logits, labels)
        noisy_true_logits = gather_true_class_logits(noisy_run.logits, labels)
        correct_class_logit_drop = (clean_true_logits - noisy_true_logits).mean().item()
        self.correct_class_logit_drops.append(float(correct_class_logit_drop))

        clean_margin = compute_true_class_margin(clean_run.logits, labels)
        noisy_margin = compute_true_class_margin(noisy_run.logits, labels)
        target_margin_drop = (clean_margin - noisy_margin).mean().item()
        self.target_margin_drops.append(float(target_margin_drop))

        js_divergence = compute_js_divergence(clean_run.logits, noisy_run.logits).mean().item()
        self.js_divergences.append(float(js_divergence))

    def summary(self, label: str) -> RobustnessSummary:
        clean_accuracy = mean_or_nan(self.clean_accuracies)
        perturbed_accuracy = mean_or_nan(self.perturbed_accuracies)
        accuracy_drop = clean_accuracy - perturbed_accuracy if not math.isnan(clean_accuracy) and not math.isnan(perturbed_accuracy) else math.nan
        return RobustnessSummary(
            label=label,
            clean_accuracy=clean_accuracy,
            clean_nll_mean=mean_or_nan(self.clean_nlls),
            clean_brier_mean=mean_or_nan(self.clean_briers),
            perturbed_accuracy_mean=perturbed_accuracy,
            accuracy_drop_mean=accuracy_drop,
            perturbed_nll_mean=mean_or_nan(self.perturbed_nlls),
            nll_increase_mean=mean_or_nan(self.nll_increases),
            perturbed_brier_mean=mean_or_nan(self.perturbed_briers),
            brier_increase_mean=mean_or_nan(self.brier_increases),
            perturbed_true_class_prob_mean=mean_or_nan(self.perturbed_true_class_probs),
            perturbed_true_class_margin_mean=mean_or_nan(self.perturbed_true_class_margins),
            clean_correct_retention_mean=mean_or_nan(self.clean_correct_retentions),
            prediction_consistency_mean=mean_or_nan(self.prediction_consistencies),
            correct_class_logit_drop_mean=mean_or_nan(self.correct_class_logit_drops),
            target_margin_drop_mean=mean_or_nan(self.target_margin_drops),
            js_divergence_mean=mean_or_nan(self.js_divergences),
            perturbation_evals=len(self.perturbed_accuracies),
        )


def mean_or_nan(values: list[float]) -> float:
    valid = [float(value) for value in values if not math.isnan(float(value))]
    if not valid:
        return math.nan
    return float(sum(valid) / len(valid))


def gather_true_class_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 1, labels.unsqueeze(1)).squeeze(1)


def gather_true_class_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    return torch.gather(prob, 1, labels.unsqueeze(1)).squeeze(1)


def compute_brier_score(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=-1)
    target = F.one_hot(labels, num_classes=logits.shape[-1]).to(prob.dtype)
    return ((prob - target) ** 2).sum(dim=-1)


def compute_true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = gather_true_class_logits(logits, labels)
    masked_logits = logits.clone()
    masked_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))
    best_other = masked_logits.max(dim=1).values
    return true_logits - best_other


def compute_js_divergence(clean_logits: torch.Tensor, noisy_logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    clean_log_prob = torch.log_softmax(clean_logits, dim=-1)
    noisy_log_prob = torch.log_softmax(noisy_logits, dim=-1)
    clean_prob = clean_log_prob.exp()
    noisy_prob = noisy_log_prob.exp()
    mean_prob = (0.5 * (clean_prob + noisy_prob)).clamp_min(eps)
    mean_log_prob = mean_prob.log()
    clean_kl = (clean_prob * (clean_log_prob - mean_log_prob)).sum(dim=-1)
    noisy_kl = (noisy_prob * (noisy_log_prob - mean_log_prob)).sum(dim=-1)
    return 0.5 * (clean_kl + noisy_kl)


def parse_combined_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Compare perturbation robustness of surrogate checkpoints using prediction and logit stability metrics.",
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
    parser.add_argument("--num-perturbations", type=int, default=4)
    parser.add_argument("--perturb-forward-batch-size", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-seed-base", type=int, default=1234)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-table-prefix", type=Path, default=None)
    parser.add_argument("--print-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-detailed", action=argparse.BooleanOptionalAction, default=False)
    # Deprecated compatibility args: retained so older command lines still parse,
    # but they are no longer used because this script no longer compares row-wise support flips.
    parser.add_argument("--block-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-rows-per-block", type=int, default=None, help=argparse.SUPPRESS)
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


def sample_noise_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device="cpu")
    return noise.to(device=x.device)


def build_noisy_batch(
    images: torch.Tensor,
    num_perturbations: int,
    noise_std: float,
    seed_start: int,
) -> torch.Tensor:
    noisy_images = []
    for offset in range(int(num_perturbations)):
        noisy_images.append(images + float(noise_std) * sample_noise_like(images, seed_start + offset))
    return torch.cat(noisy_images, dim=0)


def analyze_spec(
    spec: SurrogateSpec,
    train_args: argparse.Namespace,
    init_loader,
    batches,
    transform,
    device: torch.device,
    num_perturbations: int,
    perturb_forward_batch_size: int,
    noise_std: float,
    noise_seed_base: int,
) -> RobustnessSummary:
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

    accumulator = RobustnessAccumulator()

    for batch_loader_index, (images, labels) in tqdm(batches, desc=f"{spec.label}:batches", total=len(batches), leave=False):
        images = transform(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        clean_run = run_forward_with_captures(model, images, labels)
        accumulator.add_clean_eval(clean_run, labels)

        perturb_batch_size = max(1, int(perturb_forward_batch_size))
        seed_base = int(noise_seed_base) + batch_loader_index * int(num_perturbations)
        for start in tqdm(
            range(0, int(num_perturbations), perturb_batch_size),
            desc=f"{spec.label}:noise[batch{batch_loader_index}]",
            total=math.ceil(int(num_perturbations) / perturb_batch_size),
            leave=False,
        ):
            chunk_size = min(perturb_batch_size, int(num_perturbations) - start)
            noisy_batch = build_noisy_batch(
                images=images,
                num_perturbations=chunk_size,
                noise_std=noise_std,
                seed_start=seed_base + start,
            )
            noisy_run_full = run_forward_with_captures(model, noisy_batch, labels.repeat(chunk_size))
            noisy_runs = split_forward_run(noisy_run_full, batch_size=images.shape[0])
            for noisy_run in noisy_runs:
                accumulator.add_noisy_eval(clean_run, noisy_run, labels)

    return accumulator.summary(spec.label)


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


def print_compact_overall(summaries: list[RobustnessSummary]) -> None:
    pert_acc_by_label = {item.label: item.perturbed_accuracy_mean for item in summaries}
    nll_by_label = {item.label: item.perturbed_nll_mean for item in summaries}
    brier_by_label = {item.label: item.perturbed_brier_mean for item in summaries}
    true_prob_by_label = {item.label: item.perturbed_true_class_prob_mean for item in summaries}
    margin_by_label = {item.label: item.perturbed_true_class_margin_mean for item in summaries}

    pert_rank = dense_rank(pert_acc_by_label, higher_is_better=True)
    nll_rank = dense_rank(nll_by_label, higher_is_better=False)
    brier_rank = dense_rank(brier_by_label, higher_is_better=False)
    true_prob_rank = dense_rank(true_prob_by_label, higher_is_better=True)
    margin_rank = dense_rank(margin_by_label, higher_is_better=True)

    preferred = {label: idx for idx, label in enumerate(ordered_labels())}
    print("overall_compact")
    print("label clean_acc pert_acc pert_rank nll nll_rank brier brier_rank true_prob true_prob_rank margin margin_rank mean_rank")
    for item in sorted(
        summaries,
        key=lambda x: (
            sum(
                rank
                for rank in [pert_rank[x.label], nll_rank[x.label], brier_rank[x.label], true_prob_rank[x.label], margin_rank[x.label]]
                if rank is not None
            ),
            preferred.get(x.label, 999),
        ),
    ):
        ranks = [pert_rank[item.label], nll_rank[item.label], brier_rank[item.label], true_prob_rank[item.label], margin_rank[item.label]]
        valid_ranks = [rank for rank in ranks if rank is not None]
        mean_rank = None if not valid_ranks else sum(valid_ranks) / len(valid_ranks)
        print(
            f"{item.label:>13s} "
            f"{fmt(item.clean_accuracy):>9s} "
            f"{fmt(item.perturbed_accuracy_mean):>9s} "
            f"{str(pert_rank[item.label]) if pert_rank[item.label] is not None else 'nan':>9s} "
            f"{fmt(item.perturbed_nll_mean):>9s} "
            f"{str(nll_rank[item.label]) if nll_rank[item.label] is not None else 'nan':>8s} "
            f"{fmt(item.perturbed_brier_mean):>9s} "
            f"{str(brier_rank[item.label]) if brier_rank[item.label] is not None else 'nan':>10s} "
            f"{fmt(item.perturbed_true_class_prob_mean):>9s} "
            f"{str(true_prob_rank[item.label]) if true_prob_rank[item.label] is not None else 'nan':>15s} "
            f"{fmt(item.perturbed_true_class_margin_mean):>9s} "
            f"{str(margin_rank[item.label]) if margin_rank[item.label] is not None else 'nan':>11s} "
            f"{fmt(mean_rank):>9s}"
        )


def print_detailed_overall(summaries: list[RobustnessSummary]) -> None:
    print()
    print("overall_detailed")
    print("label clean_acc clean_nll clean_brier pert_acc pert_nll nll_inc pert_brier brier_inc true_prob margin retention pred_consistency true_logit_drop margin_drop js_div perturb_evals")
    for item in summaries:
        print(
            f"{item.label:>13s} "
            f"{fmt(item.clean_accuracy):>9s} "
            f"{fmt(item.clean_nll_mean):>9s} "
            f"{fmt(item.clean_brier_mean):>11s} "
            f"{fmt(item.perturbed_accuracy_mean):>9s} "
            f"{fmt(item.perturbed_nll_mean):>9s} "
            f"{fmt(item.nll_increase_mean):>8s} "
            f"{fmt(item.perturbed_brier_mean):>10s} "
            f"{fmt(item.brier_increase_mean):>9s} "
            f"{fmt(item.perturbed_true_class_prob_mean):>9s} "
            f"{fmt(item.perturbed_true_class_margin_mean):>9s} "
            f"{fmt(item.clean_correct_retention_mean):>9s} "
            f"{fmt(item.prediction_consistency_mean):>16s} "
            f"{fmt(item.correct_class_logit_drop_mean):>15s} "
            f"{fmt(item.target_margin_drop_mean):>11s} "
            f"{fmt(item.js_divergence_mean):>9s} "
            f"{item.perturbation_evals:>13d}"
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
    if int(script_args.num_perturbations) <= 0:
        raise ValueError("num-perturbations must be > 0")
    if int(script_args.perturb_forward_batch_size) <= 0:
        raise ValueError("perturb-forward-batch-size must be > 0")

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

    summaries: list[RobustnessSummary] = []
    for spec in tqdm(specs, desc="surrogates", total=len(specs)):
        summaries.append(
            analyze_spec(
                spec=spec,
                train_args=train_args,
                init_loader=init_loader,
                batches=batches,
                transform=transform,
                device=device,
                num_perturbations=int(script_args.num_perturbations),
                perturb_forward_batch_size=int(script_args.perturb_forward_batch_size),
                noise_std=float(script_args.noise_std),
                noise_seed_base=int(script_args.noise_seed_base),
            )
        )

    print(f"split={script_args.split}")
    print(f"batch_index={script_args.batch_index}")
    print(f"num_batches={script_args.num_batches}")
    print(f"num_perturbations={script_args.num_perturbations}")
    print(f"perturb_forward_batch_size={script_args.perturb_forward_batch_size}")
    print(f"noise_std={script_args.noise_std}")
    print()
    print("Primary ranking metrics: perturbed_acc, perturbed_nll, perturbed_brier, true_class_prob, true_class_margin.")
    print("Higher is better for perturbed_acc / true_class_prob / true_class_margin; lower is better for perturbed_nll / perturbed_brier.")
    print("These are supervised logit-distribution metrics; NLL and Brier are proper scoring rules, unlike unsupervised clean-vs-noisy consistency alone.")
    print("Deprecated args --block-index and --max-rows-per-block are now ignored.")
    print()
    print_compact_overall(summaries)

    if script_args.print_detailed:
        print_detailed_overall(summaries)

    if script_args.output_json is not None:
        payload = {
            "split": script_args.split,
            "batch_index": int(script_args.batch_index),
            "num_batches": int(script_args.num_batches),
            "num_perturbations": int(script_args.num_perturbations),
            "perturb_forward_batch_size": int(script_args.perturb_forward_batch_size),
            "noise_std": float(script_args.noise_std),
            "noise_seed_base": int(script_args.noise_seed_base),
            "ignored_args": {
                "block_index": script_args.block_index,
                "max_rows_per_block": script_args.max_rows_per_block,
            },
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
                "clean_nll_mean",
                "clean_brier_mean",
                "perturbed_accuracy_mean",
                "accuracy_drop_mean",
                "perturbed_nll_mean",
                "nll_increase_mean",
                "perturbed_brier_mean",
                "brier_increase_mean",
                "perturbed_true_class_prob_mean",
                "perturbed_true_class_margin_mean",
                "clean_correct_retention_mean",
                "prediction_consistency_mean",
                "correct_class_logit_drop_mean",
                "target_margin_drop_mean",
                "js_divergence_mean",
                "perturbation_evals",
            ],
        )


if __name__ == "__main__":
    main()
