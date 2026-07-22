from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
from vit_lgn.full_discrete.enhancements_global_lut import A8GlobalLUTTreeMixer
from vit_lgn.full_discrete.enhancements_logic_tree import SharedLogicTreeConv3x3


PROTOCOL_SOURCE_FILES = (
    "model.py", "enhanced_model.py", "enhancements_lut.py",
    "enhancements_spatial.py", "enhancements_logic_tree.py", "enhancements_hadamard.py",
    "enhancements_global_lut.py", "enhancements_expert.py",
    "__init__.py", "logic_backend.py", "shiftadd.py", "train_cifar.py",
)
CIFAR10_PAYLOAD_FILES = (
    "batches.meta", "data_batch_1", "data_batch_2", "data_batch_3",
    "data_batch_4", "data_batch_5", "test_batch",
)


class StatefulBatchSampler(Sampler):
    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size, self.batch_size = int(size), int(batch_size)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.position = 0

    def __iter__(self):
        while True:
            if self.position + self.batch_size > self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.position = 0
            result = self.order[self.position:self.position + self.batch_size].tolist()
            self.position += self.batch_size
            yield result

    def __len__(self) -> int:
        return self.size // self.batch_size

    def state_dict(self) -> dict:
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order,
                "position": self.position, "generator": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("sampler checkpoint is incompatible")
        self.order, self.position = state["order"], int(state["position"])
        self.generator.set_state(state["generator"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fully discrete scalable ViT")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--valid-size", type=int, default=5_000)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--weight-magnitude-bits", type=int, default=7)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--qk-lanes", type=int, default=7)
    parser.add_argument(
        "--norm-kind",
        choices=["rms_lut", "shift_rms", "requant", "none"],
        default="rms_lut",
    )
    parser.add_argument(
        "--final-norm-kind",
        choices=["same", "rms_lut", "shift_rms", "requant", "none"],
        default="same",
    )
    parser.add_argument("--learned-gap", action="store_true")
    parser.add_argument("--group-lut-groups", type=int, default=0)
    parser.add_argument("--local-layers", type=int, default=0)
    parser.add_argument(
        "--local-operator",
        choices=["depthwise_shiftadd", "logic_tree3x3"],
        default="depthwise_shiftadd",
    )
    parser.add_argument(
        "--global-mixer",
        choices=["attention", "hadamard", "hybrid", "parallel", "parallel_lut_tree"],
        default="attention",
    )
    parser.add_argument("--hadamard-group-size", type=int, default=32)
    parser.add_argument("--hadamard-branch-shift", type=int, default=2)
    parser.add_argument("--hybrid-attention-period", type=int, default=3)
    parser.add_argument("--global-lut-group-size", type=int, default=32)
    parser.add_argument("--global-lut-branch-shift", type=int, default=2)
    parser.add_argument("--logic-expert-width", type=int, default=0)
    parser.add_argument("--logic-expert-count", type=int, default=1)
    parser.add_argument(
        "--state-control", choices=["none", "dynamic", "static", "random", "script"],
        default="none",
    )
    parser.add_argument("--state-expert-width", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(source_root: Path) -> dict[str, str]:
    return {name: sha256(source_root / name) for name in PROTOCOL_SOURCE_FILES}


def dataset_hashes(data_root: Path) -> dict[str, object]:
    archive = data_root / "cifar-10-python.tar.gz"
    extracted = data_root / "cifar-10-batches-py"
    missing = [name for name in CIFAR10_PAYLOAD_FILES if not (extracted / name).is_file()]
    if missing:
        raise FileNotFoundError(f"CIFAR-10 extracted payload is incomplete: {missing}")
    return {
        "archive_sha256": sha256(archive) if archive.is_file() else None,
        "extracted_files_sha256": {
            name: sha256(extracted / name) for name in CIFAR10_PAYLOAD_FILES
        },
    }


def protocol(args: argparse.Namespace) -> dict:
    source_root = Path(__file__).resolve().parent
    canonical_args = {
        **vars(args),
        "data_root": str(args.data_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
    }
    canonical_args.pop("resume")
    sources = source_hashes(source_root)
    payload = {
        "args": canonical_args,
        "sources": sources,
        "dataset": dataset_hashes(args.data_root),
        "split": {"seed": 20260711, "valid_size": args.valid_size},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def make_data(args: argparse.Namespace):
    train_transform = v2.Compose([
        v2.ToImage(), v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    eval_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_full = CIFAR10(args.data_root, train=True, download=False, transform=train_transform)
    eval_full = CIFAR10(args.data_root, train=True, download=False, transform=eval_transform)
    order = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(20260711)).tolist()
    return Subset(train_full, order[args.valid_size:]), Subset(eval_full, order[:args.valid_size])


def lr_factor(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    floor = args.min_learning_rate / args.learning_rate
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    if hasattr(model, "reset_evaluation_statistics"):
        model.reset_evaluation_statistics()
    if hasattr(model, "reset_state_statistics"):
        model.reset_state_statistics()
    correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += int((model(images).argmax(-1) == labels).sum())
        count += labels.numel()
    model.train()
    result = {"validation_accuracy": correct / count}
    if hasattr(model, "evaluation_statistics"):
        statistics = model.evaluation_statistics()
        if statistics:
            result["normalization_statistics"] = statistics
    if hasattr(model, "state_statistics"):
        statistics = model.state_statistics()
        if statistics:
            result["state_histogram_per_block"] = statistics
    return result


@contextmanager
def forced_projection_a(model: torch.nn.Module):
    """Temporarily force every logic-tree LUT to hard projection A (0xC)."""

    branches = [
        module for module in model.modules()
        if isinstance(module, SharedLogicTreeConv3x3)
    ]
    snapshots = [branch.truth_table_logits.detach().clone() for branch in branches]
    try:
        with torch.no_grad():
            for branch in branches:
                pattern = branch.truth_table_logits.new_tensor([-1.0, -1.0, 1.0, 1.0])
                branch.truth_table_logits.copy_(
                    pattern.expand_as(branch.truth_table_logits)
                )
        yield
    finally:
        with torch.no_grad():
            for branch, snapshot in zip(branches, snapshots):
                branch.truth_table_logits.copy_(snapshot)


def logic_tree_gradient_l2(model: torch.nn.Module) -> float:
    squared = None
    for module in model.modules():
        if not isinstance(module, SharedLogicTreeConv3x3):
            continue
        gradient = module.truth_table_logits.grad
        if gradient is None:
            continue
        value = gradient.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return float(torch.sqrt(squared)) if squared is not None else 0.0


def optimizer_parameter_groups(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    """Keep ROM truth-table codes free of unobserved AdamW drift."""

    lut_parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, A8GlobalLUTTreeMixer)
        for parameter in module.parameters(recurse=False)
        if parameter.requires_grad
    ]
    lut_ids = {id(parameter) for parameter in lut_parameters}
    base_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in lut_ids
    ]
    groups: list[dict[str, object]] = [{
        "params": base_parameters,
        "weight_decay": float(weight_decay),
        "parameter_role": "base_model",
    }]
    if lut_parameters:
        groups.append({
            "params": lut_parameters,
            "weight_decay": 0.0,
            "parameter_role": "global_lut_payload",
        })
    return groups


def optimizer_role_gradient_l2(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Measure each optimizer role before the shared global clip is applied."""

    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        role = str(group.get("parameter_role", f"group_{index}"))
        if role in result:
            raise ValueError(f"duplicate optimizer parameter role: {role}")
        squared = None
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            value = gradient.detach().float().square().sum()
            squared = value if squared is None else squared + value
        result[role] = float(torch.sqrt(squared)) if squared is not None else 0.0
    return result


def atomic_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def freeze_protocol_manifest(path: Path, run_protocol: dict) -> None:
    if path.exists():
        existing_protocol = json.loads(path.read_text(encoding="utf-8"))
        if existing_protocol.get("sha256") != run_protocol["sha256"]:
            raise RuntimeError(
                "existing protocol manifest does not match; it was left untouched"
            )
        return
    atomic_write_json(path, run_protocol)


def verify_current_sources(run_protocol: dict) -> dict[str, str]:
    current = source_hashes(Path(__file__).resolve().parent)
    if current != run_protocol["sources"]:
        raise RuntimeError("training source changed after the protocol was frozen")
    return current


def main() -> None:
    args = parse_args()
    if not 0 < args.valid_size < 50_000:
        raise ValueError("valid-size must be in (0,50000)")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup-steps must be in [0,steps)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    run_protocol = protocol(args)
    protocol_path = args.out_dir / "protocol.json"
    checkpoint_path = args.out_dir / "checkpoint.pt"
    saved = None
    if checkpoint_path.exists():
        if not args.resume:
            raise RuntimeError("checkpoint exists; pass --resume instead of overwriting it")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("protocol_sha256") != run_protocol["sha256"]:
            raise RuntimeError("checkpoint protocol/source/data hash does not match this run")
    freeze_protocol_manifest(protocol_path, run_protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedFullDiscreteViT(
        dim=args.dim, depth=args.depth, heads=args.heads, topk=args.topk,
        mlp_ratio=args.mlp_ratio, weight_bits=args.weight_magnitude_bits,
        activation_bits=args.activation_bits, qk_lanes=args.qk_lanes,
        norm_kind=args.norm_kind,
        final_norm_kind=args.final_norm_kind,
        learned_gap=args.learned_gap,
        group_lut_groups=args.group_lut_groups,
        local_layers=args.local_layers,
        local_operator=args.local_operator,
        global_mixer=args.global_mixer,
        hadamard_group_size=args.hadamard_group_size,
        hadamard_branch_shift=args.hadamard_branch_shift,
        hybrid_attention_period=args.hybrid_attention_period,
        global_lut_group_size=args.global_lut_group_size,
        global_lut_branch_shift=args.global_lut_branch_shift,
        logic_expert_width=args.logic_expert_width,
        logic_expert_count=args.logic_expert_count,
        state_control=args.state_control,
        state_expert_width=args.state_expert_width,
    ).to(device)
    # Model variants consume different numbers of initialization draws.  Reset
    # the data/runtime RNG after construction so paired variants see identical
    # crop/flip streams and sampler order.  Current queued controls are
    # deterministic in forward; random-state controls require recorded replay.
    seed_all(args.seed + 30_000)
    train_set, valid_set = make_data(args)
    sampler = StatefulBatchSampler(len(train_set), args.batch_size, args.seed + 10_000)
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed + 20_000),
                              pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_set, batch_size=args.eval_batch_size, num_workers=0,
                              pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, args))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    history, start_step = [], 0
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        sampler.load_state_dict(saved["sampler"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_step, history = int(saved["step"]), saved["history"]

    iterator = iter(train_loader)
    model.train()
    has_logic_tree = any(
        isinstance(module, SharedLogicTreeConv3x3) for module in model.modules()
    )
    started = time.perf_counter()
    interval_started = started
    interval_start_step = start_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step, args.steps):
        images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        completed = step + 1
        evaluation_due = completed % args.eval_every == 0 or completed == args.steps
        role_gradient = (
            optimizer_role_gradient_l2(optimizer) if evaluation_due else {}
        )
        unclipped_gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        tree_gradient = logic_tree_gradient_l2(model) if evaluation_due else 0.0
        optimizer.step()
        scheduler.step()
        if evaluation_due:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_interval_seconds = time.perf_counter() - interval_started
            interval_steps = completed - interval_start_step
            peak_allocated_gib = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            )
            evaluation = evaluate(model, valid_loader, device)
            if has_logic_tree:
                with forced_projection_a(model):
                    forced = evaluate(model, valid_loader, device)
                evaluation["forced_a_validation_accuracy"] = forced[
                    "validation_accuracy"
                ]
                evaluation["forced_a_logic_tree_statistics"] = {
                    name: statistics
                    for name, statistics in forced.get(
                        "normalization_statistics", {}
                    ).items()
                    if name.startswith("local_branches.")
                }
            unclipped_value = float(unclipped_gradient.detach())
            row = {"step": completed, **evaluation,
                   "train_loss": float(loss.detach()),
                   "next_learning_rate": optimizer.param_groups[0]["lr"],
                   "unclipped_global_gradient_l2": unclipped_value,
                   "gradient_clip_coefficient": min(
                       1.0, 1.0 / (unclipped_value + 1e-6)
                   ),
                   "logic_tree_gradient_l2_after_clip": tree_gradient,
                   "training_interval_seconds": training_interval_seconds,
                   "training_seconds_per_step": (
                       training_interval_seconds / max(interval_steps, 1)
                   ),
                   "peak_allocated_gib": peak_allocated_gib}
            row.update({
                f"preclip_{role}_gradient_l2": value
                for role, value in role_gradient.items()
            })
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_every == 0 or completed == args.steps:
            verify_current_sources(run_protocol)
            atomic_save(checkpoint_path, {
                "step": completed, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "history": history,
                "protocol_sha256": run_protocol["sha256"],
                "source_sha256": run_protocol["sources"],
                "args": {**vars(args), "data_root": str(args.data_root), "out_dir": str(args.out_dir)},
            })
            atomic_save(args.out_dir / f"model_step_{completed:05d}.pt", {
                "step": completed,
                "model": model.state_dict(),
                "history_row": history[-1] if history else None,
                "protocol_sha256": run_protocol["sha256"],
                "source_sha256": run_protocol["sources"],
                "args": {
                    **vars(args),
                    "data_root": str(args.data_root),
                    "out_dir": str(args.out_dir),
                },
            })
        if evaluation_due:
            interval_started = time.perf_counter()
            interval_start_step = completed
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

    final_source_hashes = verify_current_sources(run_protocol)
    result = {
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "history": history,
        "deployment_contract": model.deployment_contract(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": final_source_hashes,
        "protocol_sha256": run_protocol["sha256"],
    }
    atomic_write_json(args.out_dir / "result.json", result)


if __name__ == "__main__":
    main()
