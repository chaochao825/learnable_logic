from __future__ import annotations

import argparse
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
    parser.add_argument("--learned-gap", action="store_true")
    parser.add_argument("--group-lut-groups", type=int, default=0)
    parser.add_argument("--local-layers", type=int, default=0)
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


def protocol(args: argparse.Namespace) -> dict:
    source_root = Path(__file__).resolve().parent
    canonical_args = {
        **vars(args),
        "data_root": str(args.data_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
    }
    canonical_args.pop("resume")
    sources = {
        name: sha256(source_root / name)
        for name in (
            "model.py", "enhanced_model.py", "enhancements_lut.py",
            "enhancements_spatial.py", "enhancements_expert.py",
            "shiftadd.py", "train_cifar.py",
        )
    }
    archive = args.data_root / "cifar-10-python.tar.gz"
    payload = {
        "args": canonical_args,
        "sources": sources,
        "dataset_archive_sha256": sha256(archive) if archive.exists() else "missing",
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
    if hasattr(model, "reset_state_statistics"):
        model.reset_state_statistics()
    correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += int((model(images).argmax(-1) == labels).sum())
        count += labels.numel()
    model.train()
    result = {"validation_accuracy": correct / count}
    if hasattr(model, "state_statistics"):
        statistics = model.state_statistics()
        if statistics:
            result["state_histogram_per_block"] = statistics
    return result


def atomic_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
    protocol_path.write_text(json.dumps(run_protocol, indent=2, sort_keys=True), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedFullDiscreteViT(
        dim=args.dim, depth=args.depth, heads=args.heads, topk=args.topk,
        mlp_ratio=args.mlp_ratio, weight_bits=args.weight_magnitude_bits,
        activation_bits=args.activation_bits, qk_lanes=args.qk_lanes,
        learned_gap=args.learned_gap,
        group_lut_groups=args.group_lut_groups,
        local_layers=args.local_layers,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, args))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    checkpoint_path = args.out_dir / "checkpoint.pt"
    history, start_step = [], 0
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("protocol_sha256") != run_protocol["sha256"]:
            raise RuntimeError("checkpoint protocol/source/data hash does not match this run")
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
    started = time.perf_counter()
    for step in range(start_step, args.steps):
        images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.steps:
            evaluation = evaluate(model, valid_loader, device)
            row = {"step": completed, **evaluation,
                   "train_loss": float(loss.detach()),
                   "next_learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_every == 0 or completed == args.steps:
            atomic_save(checkpoint_path, {
                "step": completed, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "history": history,
                "protocol_sha256": run_protocol["sha256"],
                "args": {**vars(args), "data_root": str(args.data_root), "out_dir": str(args.out_dir)},
            })

    source_root = Path(__file__).resolve().parent
    result = {
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "history": history,
        "deployment_contract": model.deployment_contract(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": {
            name: sha256(source_root / name)
            for name in (
                "model.py", "enhanced_model.py", "enhancements_lut.py",
                "enhancements_spatial.py", "enhancements_expert.py",
                "shiftadd.py", "train_cifar.py",
            )
        },
        "protocol_sha256": run_protocol["sha256"],
    }
    temporary = args.out_dir / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.out_dir / "result.json")


if __name__ == "__main__":
    main()
