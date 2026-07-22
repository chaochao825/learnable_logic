from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .model import BitStateConfig, BitStateViT
from .regularization import (
    collapse_regularization,
    gate_distribution_metrics,
    gate_entropy_target_penalty,
)
from .teacher import ATTENTION_CLEAN_PROFILE, load_attention_clean_teacher


RESULT_COLUMNS = (
    "method",
    "dataset",
    "soft_acc",
    "discrete_acc",
    "acc_gap",
    "soft_loss",
    "discrete_loss",
    "loss_gap",
    "train_time",
    "epochs_to_target",
    "unused_gate_ratio",
    "gate_count",
    "predicate_count",
    "depth",
    "fanout_max",
    "state_entropy",
    "state_constant_ratio",
    "state_duplicate_ratio",
    "state_flip_rate",
    "gate_entropy",
    "gate_confidence",
    "hard_carrier_acc",
    "hard_bit_acc",
    "hard_path_acc_gap",
    "hard_carrier_loss",
    "hard_bit_loss",
    "hard_path_loss_gap",
    "bit_exact_verified",
)


class SyntheticPatternDataset(Dataset[tuple[Tensor, Tensor]]):
    """Class prototypes with independent bit-flip noise and no download."""

    def __init__(
        self,
        *,
        samples: int,
        image_size: int,
        channels: int,
        classes: int,
        sample_seed: int,
        prototype_seed: int = 771,
        noise_probability: float = 0.08,
    ) -> None:
        if samples < 1 or not 0.0 <= noise_probability < 0.5:
            raise ValueError((samples, noise_probability))
        prototype_generator = torch.Generator().manual_seed(prototype_seed)
        prototypes = torch.randint(
            0,
            2,
            (classes, channels, image_size, image_size),
            generator=prototype_generator,
            dtype=torch.int64,
        ).bool()
        generator = torch.Generator().manual_seed(sample_seed)
        labels = torch.arange(samples, dtype=torch.long) % classes
        permutation = torch.randperm(samples, generator=generator)
        labels = labels[permutation]
        noise = torch.rand(
            samples,
            channels,
            image_size,
            image_size,
            generator=generator,
        ) < noise_probability
        self.images = torch.logical_xor(prototypes[labels], noise).to(torch.float32)
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.images[index], self.labels[index]


@dataclass
class CageController:
    tau_max: float = 3.0
    tau_min: float = 0.5
    beta: float = 0.99
    choices: int = 16
    ema_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.tau_min <= 0.0 or self.tau_max < self.tau_min:
            raise ValueError((self.tau_min, self.tau_max))
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(self.beta)
        if self.choices < 2:
            raise ValueError(self.choices)

    def update(self, model: BitStateViT) -> tuple[float, float]:
        with torch.no_grad():
            confidence = float(
                torch.stack([layer.confidence() for layer in model.gate_layers()]).mean()
            )
        random_confidence = 1.0 / self.choices
        confidence = min(1.0, max(random_confidence, confidence))
        if self.ema_confidence is None:
            self.ema_confidence = random_confidence
        self.ema_confidence = (
            self.beta * self.ema_confidence + (1.0 - self.beta) * confidence
        )
        progress = (self.ema_confidence - random_confidence) / (1.0 - random_confidence)
        progress = min(1.0, max(0.0, progress))
        tau = self.tau_max + (self.tau_min - self.tau_max) * progress
        return float(tau), confidence


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(dataset: Dataset, limit: int, seed: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=generator)[:limit].tolist())


def split_train_validation(
    train_source: Dataset,
    validation_source: Dataset,
    validation_size: int,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    if validation_size <= 0:
        return train_source, None
    if validation_size >= len(train_source):
        raise ValueError((validation_size, len(train_source)))
    if len(train_source) != len(validation_source):
        raise ValueError((len(train_source), len(validation_source)))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_source), generator=generator)
    validation_indices = permutation[:validation_size].tolist()
    train_indices = permutation[validation_size:].tolist()
    return (
        Subset(train_source, train_indices),
        Subset(validation_source, validation_indices),
    )


def make_datasets(
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, Dataset, int, int, int]:
    if args.dataset == "synthetic_patterns":
        train_source = SyntheticPatternDataset(
            samples=args.train_limit if args.train_limit > 0 else 4096,
            image_size=args.image_size,
            channels=args.in_channels,
            classes=args.num_classes,
            sample_seed=args.seed + 10,
        )
        test = SyntheticPatternDataset(
            samples=args.eval_limit if args.eval_limit > 0 else 1024,
            image_size=args.image_size,
            channels=args.in_channels,
            classes=args.num_classes,
            sample_seed=args.seed + 20,
        )
        train, validation = split_train_validation(
            train_source,
            train_source,
            args.validation_size,
            args.seed + 25,
        )
        return (
            train,
            test if validation is None else validation,
            test,
            args.image_size,
            args.in_channels,
            args.num_classes,
        )

    if args.dataset == "sklearn_digits":
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        images = torch.from_numpy(digits.images).unsqueeze(1).float() / 16.0
        labels = torch.from_numpy(digits.target).long()
        indices = torch.arange(labels.numel()).numpy()
        train_ids, test_ids = train_test_split(
            indices,
            test_size=0.25,
            random_state=args.seed,
            stratify=labels.numpy(),
        )
        train_source = TensorDataset(images[train_ids], labels[train_ids])
        test = TensorDataset(images[test_ids], labels[test_ids])
        train, validation = split_train_validation(
            train_source,
            train_source,
            args.validation_size,
            args.seed + 25,
        )
        return (
            subset(train, args.train_limit, args.seed + 30),
            subset(
                test if validation is None else validation,
                args.eval_limit,
                args.seed + 35,
            ),
            subset(test, args.eval_limit, args.seed + 40),
            8,
            1,
            10,
        )

    from torchvision import datasets, transforms

    evaluation_transform = transforms.ToTensor()
    root = Path(args.data_root)
    if args.dataset == "mnist":
        train_transform = evaluation_transform
        train_source = datasets.MNIST(
            root,
            train=True,
            transform=train_transform,
            download=args.download,
        )
        validation_source = train_source
        test = datasets.MNIST(root, train=False, transform=evaluation_transform, download=args.download)
        image_size, channels, classes = 28, 1, 10
    elif args.dataset == "cifar10":
        train_transform = (
            transforms.Compose(
                (
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                )
            )
            if args.augment
            else evaluation_transform
        )
        train_source = datasets.CIFAR10(
            root,
            train=True,
            transform=train_transform,
            download=args.download,
        )
        validation_source = datasets.CIFAR10(
            root,
            train=True,
            transform=evaluation_transform,
            download=args.download,
        )
        test = datasets.CIFAR10(root, train=False, transform=evaluation_transform, download=args.download)
        image_size, channels, classes = 32, 3, 10
    else:
        raise ValueError(args.dataset)
    train, validation = split_train_validation(
        train_source,
        validation_source,
        args.validation_size,
        args.seed + 25,
    )
    return (
        subset(train, args.train_limit, args.seed + 30),
        subset(
            test if validation is None else validation,
            args.eval_limit,
            args.seed + 35,
        ),
        subset(test, args.eval_limit, args.seed + 40),
        image_size,
        channels,
        classes,
    )


def make_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, int, int]]:
    train, validation, test, image_size, channels, classes = make_datasets(args)
    generator = torch.Generator().manual_seed(args.seed + 50)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation, shuffle=False, **common)
    test_loader = DataLoader(test, shuffle=False, **common)
    return train_loader, validation_loader, test_loader, (image_size, channels, classes)


def geometric_temperature(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    progress = epoch / (epochs - 1)
    return start * (end / start) ** progress


def linear_schedule(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    return start + (end - start) * (epoch / (epochs - 1))


def epoch_learning_rate(epoch: int, args: argparse.Namespace) -> float:
    if args.lr_schedule == "constant":
        return args.learning_rate
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return args.learning_rate * (epoch + 1) / warmup_epochs
    decay_epochs = max(1, args.epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / decay_epochs))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def training_mode(method: str, *, epoch: int = 0, soft_warmup_epochs: int = 0) -> str:
    if method == "progressive_hard_st":
        return "soft" if epoch < soft_warmup_epochs else "hard_st"
    return {
        "soft": "soft",
        "anneal": "soft",
        "gumbel_st": "gumbel_st",
        "hard_st": "hard_st",
        "hard_st_cage": "hard_st",
    }[method]


def supervised_distillation_loss(
    student_logits: Tensor,
    labels: Tensor,
    *,
    label_smoothing: float,
    teacher_logits: Tensor | None,
    teacher_alpha: float,
    teacher_temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    supervised = F.cross_entropy(
        student_logits,
        labels,
        label_smoothing=label_smoothing,
    )
    if teacher_logits is None or teacher_alpha == 0.0:
        return supervised, supervised, supervised.new_zeros(())
    temperature = teacher_temperature
    distillation = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    combined = (1 - teacher_alpha) * supervised + teacher_alpha * distillation
    return combined, supervised, distillation


@torch.no_grad()
def evaluate(
    model: BitStateViT,
    loader: DataLoader,
    device: torch.device,
    *,
    discrete: bool,
    tau: float,
    group_sum_temperature: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if discrete:
            raw_logits = model.forward_bits(images).to(torch.float32)
        else:
            raw_logits = model(images, mode="soft", tau=tau)
        logits = raw_logits / group_sum_temperature
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == labels).sum())
        count += labels.numel()
    return total_loss / count, correct / count


@torch.no_grad()
def evaluate_hard_carrier(
    model: BitStateViT,
    loader: DataLoader,
    device: torch.device,
    group_sum_temperature: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images, mode="hard") / group_sum_temperature
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == labels).sum())
        count += labels.numel()
    return total_loss / count, correct / count


def measure_inactive(
    model: BitStateViT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    def batches() -> Iterable[Tensor]:
        for batch_id, (images, _labels) in enumerate(loader):
            if batch_id >= max_batches:
                break
            yield images.to(device, non_blocking=True)

    model.eval()
    return model.inactive_gate_ratio(batches())


@torch.no_grad()
def measure_state_diagnostics(
    model: BitStateViT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    model.eval()
    for batch_id, (images, _labels) in enumerate(loader):
        if batch_id >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        _logits, trace = model(images, mode="hard", return_trace=True)
        _penalty, metrics = collapse_regularization(
            trace,
            balance_weight=0.0,
            diversity_weight=0.0,
            flip_weight=0.0,
        )
        batch_size = images.shape[0]
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError("at least one diagnostic batch is required")
    return {key: value / count for key, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, validation_loader, test_loader, dataset_shape = make_loaders(args)
    image_size, channels, classes = dataset_shape
    if image_size % args.patch_size:
        raise ValueError(f"patch size {args.patch_size} does not divide image size {image_size}")
    config = BitStateConfig(
        image_size=image_size,
        patch_size=args.patch_size,
        in_channels=channels,
        threshold_levels=args.threshold_levels,
        state_width=args.state_width,
        encoder_kind=args.encoder_kind,
        predicate_fanin=args.predicate_fanin,
        encoder_identity_width=args.encoder_identity_width,
        predicate_temperature=args.predicate_temperature,
        predicate_chunk_size=args.predicate_chunk_size,
        local_depth=args.local_depth,
        global_depth=args.global_depth,
        heads=args.heads,
        qk_bits=args.qk_bits,
        topk=args.topk,
        num_classes=classes,
        votes_per_class=args.votes_per_class,
        update_fraction=args.update_fraction,
        attention_temperature=args.attention_temperature,
        exclude_self=args.exclude_self,
        seed=args.seed,
    )
    model = BitStateViT(config).to(device)
    teacher = None
    teacher_metadata: dict[str, object] | None = None
    if args.teacher_alpha > 0.0:
        if args.dataset != "cifar10":
            raise ValueError("the attention-clean teacher requires cifar10")
        teacher, teacher_metadata = load_attention_clean_teacher(
            source_dir=args.teacher_source_dir,
            checkpoint_path=args.teacher_checkpoint,
            device=device,
            profile=args.teacher_profile,
        )
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    cage = CageController(args.cage_tau_max, args.cage_tau_min, args.cage_beta)
    tau = args.tau
    epochs_to_target: int | None = None
    history: list[dict[str, object]] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_discrete_acc = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    regularization_active = any(
        weight > 0.0
        for weight in (
            args.state_balance_weight,
            args.state_diversity_weight,
            args.state_flip_weight,
        )
    )
    autocast_enabled = args.amp_bfloat16 and device.type == "cuda"
    started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        learning_rate = epoch_learning_rate(epoch, args)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        if args.method == "anneal":
            tau = geometric_temperature(epoch, args.epochs, args.tau_start, args.tau_end)
        elif args.method == "progressive_hard_st" and epoch >= args.soft_warmup_epochs:
            hard_epoch = epoch - args.soft_warmup_epochs
            hard_epochs = max(1, args.epochs - args.soft_warmup_epochs)
            tau = geometric_temperature(
                hard_epoch,
                hard_epochs,
                args.tau_start,
                args.tau_end,
            )
        mode = training_mode(
            args.method,
            epoch=epoch,
            soft_warmup_epochs=args.soft_warmup_epochs,
        )
        entropy_target = linear_schedule(
            epoch,
            args.epochs,
            args.gate_entropy_target_start,
            args.gate_entropy_target_end,
        )
        running_loss = 0.0
        running_task_loss = 0.0
        running_metrics: dict[str, float] = {}
        count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            teacher_logits = None
            if teacher is not None:
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.teacher_amp_bfloat16 and device.type == "cuda",
                ):
                    teacher_logits = teacher(images).to(torch.float32)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                if regularization_active:
                    raw_logits, trace = model(
                        images,
                        mode=mode,
                        tau=tau,
                        return_trace=True,
                    )
                    collapse_loss, collapse_metrics = collapse_regularization(
                        trace,
                        balance_weight=args.state_balance_weight,
                        diversity_weight=args.state_diversity_weight,
                        flip_weight=args.state_flip_weight,
                        diversity_pairs=args.state_diversity_pairs,
                        maximum_similarity=args.state_maximum_similarity,
                        minimum_flip=args.state_minimum_flip,
                        maximum_flip=args.state_maximum_flip,
                    )
                else:
                    raw_logits = model(images, mode=mode, tau=tau)
                    collapse_loss = raw_logits.new_zeros(())
                    collapse_metrics = {}
                logits = raw_logits / args.group_sum_temperature
                task_loss, supervised_loss, distillation_loss = supervised_distillation_loss(
                    logits,
                    labels,
                    label_smoothing=args.label_smoothing,
                    teacher_logits=teacher_logits,
                    teacher_alpha=args.teacher_alpha,
                    teacher_temperature=args.teacher_temperature,
                )
                gate_penalty, gate_entropy, gate_confidence = gate_entropy_target_penalty(
                    model.gate_layers(),
                    entropy_target,
                )
                loss = (
                    task_loss
                    + collapse_loss
                    + args.gate_entropy_weight * gate_penalty
                )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args.method == "hard_st_cage":
                tau, _confidence = cage.update(model)
            running_loss += float(loss.detach()) * labels.numel()
            running_task_loss += float(task_loss.detach()) * labels.numel()
            batch_metrics = {
                **collapse_metrics,
                "gate_entropy": gate_entropy,
                "gate_confidence": gate_confidence,
                "gate_entropy_penalty": gate_penalty,
                "supervised_loss": supervised_loss,
                "distillation_loss": distillation_loss,
            }
            for key, value in batch_metrics.items():
                running_metrics[key] = (
                    running_metrics.get(key, 0.0)
                    + float(value.detach()) * labels.numel()
                )
            count += labels.numel()

        soft_loss, soft_acc = evaluate(
            model,
            validation_loader,
            device,
            discrete=False,
            tau=args.eval_tau,
            group_sum_temperature=args.group_sum_temperature,
        )
        discrete_loss, discrete_acc = evaluate(
            model,
            validation_loader,
            device,
            discrete=True,
            tau=tau,
            group_sum_temperature=args.group_sum_temperature,
        )
        if epochs_to_target is None and discrete_acc >= args.target_accuracy:
            epochs_to_target = epoch + 1
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / count,
            "train_task_loss": running_task_loss / count,
            "soft_loss": soft_loss,
            "soft_acc": soft_acc,
            "discrete_loss": discrete_loss,
            "discrete_acc": discrete_acc,
            "acc_gap": abs(soft_acc - discrete_acc),
            "tau": tau,
            "mode": mode,
            "learning_rate": learning_rate,
            "gate_entropy_target": entropy_target,
            "elapsed": time.perf_counter() - started,
        }
        epoch_row.update({key: value / count for key, value in running_metrics.items()})
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        if discrete_acc > best_discrete_acc:
            best_discrete_acc = discrete_acc
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if args.save_best_checkpoint:
                torch.save(
                    {
                        "model": best_state,
                        "config": asdict(config),
                        "epoch": best_epoch,
                        "discrete_acc": best_discrete_acc,
                    },
                    output_dir / "checkpoint_best.pt",
                )

    train_time = time.perf_counter() - started
    if args.restore_best and best_state is not None:
        model.load_state_dict(best_state)
    soft_loss, soft_acc = evaluate(
        model,
        test_loader,
        device,
        discrete=False,
        tau=args.eval_tau,
        group_sum_temperature=args.group_sum_temperature,
    )
    discrete_loss, discrete_acc = evaluate(
        model,
        test_loader,
        device,
        discrete=True,
        tau=args.eval_tau,
        group_sum_temperature=args.group_sum_temperature,
    )
    hard_carrier_loss, hard_carrier_acc = evaluate_hard_carrier(
        model,
        test_loader,
        device,
        args.group_sum_temperature,
    )
    first_images, _ = next(iter(test_loader))
    model.assert_bit_exact(first_images.to(device))
    unused_ratio = measure_inactive(model, train_loader, device, args.inactive_batches)
    state_diagnostics = measure_state_diagnostics(
        model,
        test_loader,
        device,
        args.inactive_batches,
    )
    with torch.no_grad():
        final_gate_entropy, final_gate_confidence = gate_distribution_metrics(
            model.gate_layers()
        )
    method_name = {
        "soft": "bitstate_soft_argmax",
        "anneal": "bitstate_anneal_argmax",
        "gumbel_st": "bitstate_gumbel_st",
        "hard_st": "bitstate_hard_st",
        "hard_st_cage": "bitstate_hard_st_cage",
        "progressive_hard_st": "bitstate_progressive_hard_st",
    }[args.method]
    row: dict[str, object] = {
        "method": method_name,
        "dataset": args.dataset,
        "soft_acc": soft_acc,
        "discrete_acc": discrete_acc,
        "acc_gap": abs(soft_acc - discrete_acc),
        "soft_loss": soft_loss,
        "discrete_loss": discrete_loss,
        "loss_gap": abs(soft_loss - discrete_loss),
        "train_time": train_time,
        "epochs_to_target": epochs_to_target if epochs_to_target is not None else "",
        "unused_gate_ratio": unused_ratio,
        "gate_count": model.gate_count(),
        "predicate_count": model.predicate_count(),
        "depth": model.logic_depth(),
        "fanout_max": model.fanout_max(),
        "state_entropy": state_diagnostics["state_entropy"],
        "state_constant_ratio": state_diagnostics["state_constant_ratio"],
        "state_duplicate_ratio": state_diagnostics["state_duplicate_ratio"],
        "state_flip_rate": state_diagnostics["state_flip_rate"],
        "gate_entropy": float(final_gate_entropy),
        "gate_confidence": float(final_gate_confidence),
        "hard_carrier_acc": hard_carrier_acc,
        "hard_bit_acc": discrete_acc,
        "hard_path_acc_gap": abs(hard_carrier_acc - discrete_acc),
        "hard_carrier_loss": hard_carrier_loss,
        "hard_bit_loss": discrete_loss,
        "hard_path_loss_gap": abs(hard_carrier_loss - discrete_loss),
        "seed": args.seed,
        "final_tau": tau,
        "soft_eval_tau": args.eval_tau,
        "group_sum_temperature": args.group_sum_temperature,
        "best_discrete_acc": best_discrete_acc,
        "best_epoch": best_epoch,
        "selection_split": "validation" if args.validation_size > 0 else "test",
        "restored_best": bool(args.restore_best),
        "bit_exact_verified": True,
        "history": history,
        "model_config": asdict(config),
        "training_args": vars(args),
        "teacher": teacher_metadata,
        "state_diagnostics": state_diagnostics,
    }

    (output_dir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    with (output_dir / "result.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    if args.save_checkpoint:
        torch.save(
            {"model": model.state_dict(), "config": asdict(config), "result": row},
            output_dir / "checkpoint.pt",
        )
    print(json.dumps({key: row[key] for key in RESULT_COLUMNS}, indent=2))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic_patterns", "sklearn_digits", "mnist", "cifar10"), default="synthetic_patterns")
    parser.add_argument(
        "--method",
        choices=(
            "soft",
            "anneal",
            "gumbel_st",
            "hard_st",
            "hard_st_cage",
            "progressive_hard_st",
        ),
        default="hard_st_cage",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="runs/bitstate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--amp-bfloat16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-alpha", type=float, default=0.0)
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-source-dir", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--teacher-profile", default=ATTENTION_CLEAN_PROFILE)
    parser.add_argument(
        "--teacher-amp-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--target-accuracy", type=float, default=0.8)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--inactive-batches", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--threshold-levels", type=int, default=2)
    parser.add_argument("--state-width", type=int, default=64)
    parser.add_argument(
        "--encoder-kind",
        choices=("thermometer", "redundant_predicate"),
        default="thermometer",
    )
    parser.add_argument("--predicate-fanin", type=int, default=9)
    parser.add_argument("--encoder-identity-width", type=int, default=0)
    parser.add_argument("--predicate-temperature", type=float, default=1.0)
    parser.add_argument("--predicate-chunk-size", type=int, default=1024)
    parser.add_argument("--local-depth", type=int, default=2)
    parser.add_argument("--global-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--qk-bits", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--votes-per-class", type=int, default=16)
    parser.add_argument("--group-sum-temperature", type=float, default=1.0)
    parser.add_argument("--update-fraction", type=float, default=0.5)
    parser.add_argument("--attention-temperature", type=float, default=0.25)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--tau-start", type=float, default=3.0)
    parser.add_argument("--tau-end", type=float, default=0.25)
    parser.add_argument("--eval-tau", type=float, default=1.0)
    parser.add_argument("--soft-warmup-epochs", type=int, default=2)
    parser.add_argument("--cage-tau-max", type=float, default=3.0)
    parser.add_argument("--cage-tau-min", type=float, default=0.5)
    parser.add_argument("--cage-beta", type=float, default=0.99)
    parser.add_argument("--state-balance-weight", type=float, default=0.0)
    parser.add_argument("--state-diversity-weight", type=float, default=0.0)
    parser.add_argument("--state-flip-weight", type=float, default=0.0)
    parser.add_argument("--state-diversity-pairs", type=int, default=128)
    parser.add_argument("--state-maximum-similarity", type=float, default=0.9)
    parser.add_argument("--state-minimum-flip", type=float, default=0.02)
    parser.add_argument("--state-maximum-flip", type=float, default=0.5)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.0)
    parser.add_argument("--gate-entropy-target-start", type=float, default=0.8)
    parser.add_argument("--gate-entropy-target-end", type=float, default=0.1)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--restore-best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.inactive_batches < 1:
        parser.error("epochs, batch-size, and inactive-batches must be positive")
    if not 0.0 <= args.target_accuracy <= 1.0:
        parser.error("target-accuracy must be in [0, 1]")
    if args.warmup_epochs < 0 or args.soft_warmup_epochs < 0:
        parser.error("warmup epochs must be non-negative")
    if args.validation_size < 0:
        parser.error("validation-size must be non-negative")
    if args.predicate_fanin < 1 or args.predicate_chunk_size < 1:
        parser.error("predicate fanin and chunk size must be positive")
    if args.encoder_identity_width < 0 or args.predicate_temperature <= 0.0:
        parser.error("invalid predicate encoder settings")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("label-smoothing must be in [0, 1)")
    if args.group_sum_temperature <= 0.0:
        parser.error("group-sum-temperature must be positive")
    if not 0.0 <= args.teacher_alpha <= 1.0 or args.teacher_temperature <= 0.0:
        parser.error("invalid teacher distillation settings")
    if args.teacher_alpha > 0.0 and (
        not args.teacher_source_dir or not args.teacher_checkpoint
    ):
        parser.error("teacher source and checkpoint are required when teacher-alpha > 0")
    if min(
        args.state_balance_weight,
        args.state_diversity_weight,
        args.state_flip_weight,
        args.gate_entropy_weight,
    ) < 0.0:
        parser.error("regularization weights must be non-negative")
    if not 0.5 <= args.state_maximum_similarity < 1.0:
        parser.error("state-maximum-similarity must be in [0.5, 1)")
    if not 0.0 <= args.state_minimum_flip <= args.state_maximum_flip <= 1.0:
        parser.error("invalid state flip interval")
    if not 0.0 <= args.gate_entropy_target_start <= 1.0:
        parser.error("gate entropy targets must be in [0, 1]")
    if not 0.0 <= args.gate_entropy_target_end <= 1.0:
        parser.error("gate entropy targets must be in [0, 1]")
    if args.tau <= 0 or args.tau_start <= 0 or args.tau_end <= 0 or args.eval_tau <= 0:
        parser.error("temperatures must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
