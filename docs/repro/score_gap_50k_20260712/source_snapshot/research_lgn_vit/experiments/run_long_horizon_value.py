from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


REPO_ROOT = Path(__file__).resolve().parents[2]
ATTENTION_CLEAN = REPO_ROOT / "vit_lgn" / "attention_clean"
if str(ATTENTION_CLEAN) not in sys.path:
    sys.path.insert(0, str(ATTENTION_CLEAN))

from vit_tiny_attention_logic import vit_tiny


VARIANTS = {
    "uniform-final1": "uniform-mean",
    "score-gap-final1": "score-gap-lut",
}


class StatefulBatchSampler(Sampler[list[int]]):
    """Infinite shuffled batches whose exact next batch is checkpointable."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size = size
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.position = 0

    def __iter__(self):
        while True:
            if self.position + self.batch_size > self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.position = 0
            batch = self.order[self.position : self.position + self.batch_size].tolist()
            self.position += self.batch_size
            yield batch

    def __len__(self) -> int:
        return self.size // self.batch_size

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state(),
            "order": self.order,
            "position": self.position,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["size"] != self.size or state["batch_size"] != self.batch_size:
            raise ValueError("checkpoint sampler shape does not match current arguments")
        self.generator.set_state(state["generator_state"])
        self.order = state["order"]
        self.position = int(state["position"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="50k-step paired CIFAR-10 selected-V experiment")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--valid-size", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--attention-k", type=int, default=8)
    parser.add_argument("--dataset-provenance", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(args: argparse.Namespace) -> dict:
    files = [
        Path(__file__),
        ATTENTION_CLEAN / "vit_tiny_attention_logic.py",
        ATTENTION_CLEAN / "vit_tiny_baseline.py",
        ATTENTION_CLEAN / "src" / "attention_logic.py",
        ATTENTION_CLEAN / "src" / "packed_xnor_topk.py",
        ATTENTION_CLEAN / "src" / "hardware_value_aggregation.py",
    ]
    dataset_digest = hashlib.sha256()
    for path in sorted(item for item in args.data_root.rglob("*") if item.is_file()):
        relative = path.relative_to(args.data_root).as_posix()
        dataset_digest.update(f"{relative}\0{path.stat().st_size}\n".encode("utf-8"))
    split_payload = json.dumps({"seed": 20260711, "valid_size": args.valid_size}, sort_keys=True)
    return {
        "dataset": args.dataset_provenance,
        "dataset_train_manifest_names_sizes_sha256": dataset_digest.hexdigest(),
        "split_spec_sha256": hashlib.sha256(split_payload.encode("utf-8")).hexdigest(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "files": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in files},
    }


def canonical_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    values.pop("resume")
    values["data_root"] = str(values["data_root"].resolve())
    values["out_dir"] = str(values["out_dir"].resolve())
    return values


def protocol_hash(args: argparse.Namespace, source_provenance: dict) -> str:
    payload = json.dumps({"args": canonical_args(args), "provenance": source_provenance},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_datasets(args: argparse.Namespace):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    eval_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_full = CIFAR10(args.data_root, train=True, download=False, transform=train_transform)
    eval_full = CIFAR10(args.data_root, train=True, download=False, transform=eval_transform)
    split_generator = torch.Generator().manual_seed(20260711)
    order = torch.randperm(len(train_full), generator=split_generator).tolist()
    valid_indices = order[: args.valid_size]
    train_indices = order[args.valid_size :]
    if not train_indices or not valid_indices:
        raise ValueError("valid-size must leave non-empty train and validation subsets")
    return Subset(train_full, train_indices), Subset(eval_full, valid_indices)


def make_model(args: argparse.Namespace, variant: str) -> torch.nn.Module:
    return vit_tiny(
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        mlp_ratio=4.0, drop_path_rate=0.0, attention_only=False,
        attention_k=args.attention_k, use_thermometer_encoding=True,
        n_thresholds=3, v_n_thresholds=3, decode_output=True,
        thermometer_decode_use_weights=False, topk_impl="winner-tree",
        topk_forward_mode="topk", topk_surrogate_mode="kth",
        topk_surrogate_proxy_source="encoded", majority_train_temperature=0.125,
        majority_surrogate_mode="count", value_aggregation=VARIANTS[variant],
        value_gap_shift=1, value_max_gap_bucket=3, value_global_tail_weight=0,
        value_global_tail_exclude_cls=True, value_output_bits=0,
        value_final_channel_bits=1,
    )


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = count = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        correct += int((model(images).argmax(-1) == targets).sum())
        count += targets.numel()
    model.train()
    return correct / count


def lr_factor(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    floor = args.min_learning_rate / args.learning_rate
    return floor + (1.0 - floor) * cosine


def rng_state() -> dict:
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_one(args, variant, seed, train_set, valid_set, base_state, device,
            source_provenance, run_protocol_hash):
    run_dir = args.out_dir / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("protocol_hash") != run_protocol_hash:
            raise RuntimeError(f"existing result belongs to a different protocol: {result_path}")
        return result

    seed_everything(seed)
    model = make_model(args, variant).to(device)
    model.load_state_dict(base_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, args))
    sampler = StatefulBatchSampler(len(train_set), args.batch_size, seed + 10_000)
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=0,
                        pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(seed + 20_000))
    valid_loader = DataLoader(valid_set, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    start_step, history = 0, []
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved["provenance"] != source_provenance
                or saved["protocol_hash"] != run_protocol_hash):
            raise RuntimeError(f"refusing incompatible checkpoint: {checkpoint_path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        sampler.load_state_dict(saved["sampler"])
        start_step, history = saved["step"], saved["history"]

    iterator = iter(loader)
    if start_step:
        restore_rng(saved["rng"])
    model.train()
    started = time.perf_counter()
    for step in range(start_step, args.steps):
        images, targets = next(iterator)
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.steps:
            accuracy = evaluate(model, valid_loader, device)
            record = {"step": completed, "validation_accuracy": accuracy,
                      "train_loss": float(loss.detach()),
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(record)
            print(json.dumps({"variant": variant, "seed": seed, **record}), flush=True)
        if completed % args.checkpoint_every == 0 or completed == args.steps:
            checkpoint = {
                "step": completed, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "sampler": sampler.state_dict(), "rng": rng_state(), "history": history,
                "args": canonical_args(args), "provenance": source_provenance,
                "protocol_hash": run_protocol_hash,
            }
            atomic_torch_save(checkpoint_path, checkpoint)

    result = {
        "variant": variant, "seed": seed, "steps": args.steps,
        "best_validation_accuracy": max(row["validation_accuracy"] for row in history),
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "history": history, "elapsed_this_invocation_seconds": time.perf_counter() - started,
        "provenance": source_provenance, "protocol_hash": run_protocol_hash,
        "args": canonical_args(args),
    }
    if provenance(args) != source_provenance:
        raise RuntimeError("source or runtime provenance changed during the run")
    atomic_json(result_path, result)
    return result


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.checkpoint_every < 1 or args.eval_every < 1:
        raise ValueError("steps/checkpoint-every/eval-every must be positive")
    if not (0 <= args.min_learning_rate <= args.learning_rate):
        raise ValueError("min-learning-rate must be in [0, learning-rate]")
    if not (0 <= args.warmup_steps < args.steps):
        raise ValueError("warmup-steps must be in [0, steps)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    source_provenance = provenance(args)
    run_protocol_hash = protocol_hash(args, source_provenance)
    atomic_json(args.out_dir / "protocol.json",
                {"args": canonical_args(args), "provenance": source_provenance,
                 "protocol_hash": run_protocol_hash,
                 "test_policy": "CIFAR-10 test split is not loaded"})
    train_set, valid_set = make_datasets(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for seed in args.seeds:
        seed_everything(seed)
        base = make_model(args, "uniform-final1")
        base_state = copy.deepcopy(base.state_dict())
        del base
        for variant in args.variants:
            results.append(run_one(args, variant, seed, train_set, valid_set,
                                   base_state, device, source_provenance, run_protocol_hash))
    atomic_json(args.out_dir / "summary.json",
                {"results": results, "provenance": source_provenance,
                 "protocol_hash": run_protocol_hash})


if __name__ == "__main__":
    main()
