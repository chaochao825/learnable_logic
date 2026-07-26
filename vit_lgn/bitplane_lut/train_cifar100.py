from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision.datasets import CIFAR100

from vit_lgn.bitplane_lut.executor import (
    BooleanRuntimeAudit,
    StrictBitPlaneLUTExecutor,
    validate_hard_payload,
)
from vit_lgn.bitplane_lut.model import (
    BitPlaneLUTClassifier,
    boolean_state_diagnostics,
    trainable_parameters,
)
from vit_lgn.bitplane_lut.train_digits import (
    atomic_json,
    atomic_torch_save,
    clone_state,
    evaluate,
    file_sha256,
    json_sha256,
    make_loader,
    resolve_device,
    seed_all,
    set_temperatures,
    write_csv,
)


SOURCE_FILES = (
    "__init__.py",
    "executor.py",
    "layers.py",
    "model.py",
    "run_cifar100_scale.sh",
    "train_cifar100.py",
    "train_digits.py",
)
IMAGE_SHAPE = (32, 32, 3)
INPUT_SYMBOLS = math.prod(IMAGE_SHAPE)
NUM_CLASSES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block-wise A8 bit-plane Hard-LGN on CIFAR-100"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260724)
    parser.add_argument("--votes-per-class", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--layers-per-block", type=int, default=1)
    parser.add_argument("--arity", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument(
        "--candidate-policy",
        choices=["mixed", "image_spatial"],
        default="image_spatial",
    )
    parser.add_argument(
        "--refit-mode",
        choices=["argmax", "truth_refit", "wiring_truth_refit"],
        default="argmax",
    )
    parser.add_argument("--wiring-passes", type=int, default=1)
    parser.add_argument("--calibration-samples", type=int, default=5000)
    parser.add_argument("--epochs-per-block", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--group-temperature", type=float, default=0.0)
    parser.add_argument("--code-loss-weight", type=float, default=0.25)
    parser.add_argument("--wiring-constraint-weight", type=float, default=0.001)
    parser.add_argument("--table-cost-weight", type=float, default=0.01)
    parser.add_argument("--fanout-cap", type=float, default=8.0)
    parser.add_argument("--temperature-start", type=float, default=1.5)
    parser.add_argument("--temperature-end", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.set_defaults(augment=True)
    parser.add_argument("--diagnostic-rows", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_splits(
    data_root: Path,
    *,
    download: bool,
    split_seed: int,
    max_train_samples: int,
    calibration_samples: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    dict[str, object],
]:
    train_data = CIFAR100(root=data_root, train=True, download=download)
    test_data = CIFAR100(root=data_root, train=False, download=download)
    train_images = np.asarray(train_data.data, dtype=np.uint8).reshape(-1, INPUT_SYMBOLS)
    train_labels = np.asarray(train_data.targets, dtype=np.int64)
    test_images = np.asarray(test_data.data, dtype=np.uint8).reshape(-1, INPUT_SYMBOLS)
    test_labels = np.asarray(test_data.targets, dtype=np.int64)
    all_indices = np.arange(train_labels.shape[0], dtype=np.int64)
    train_indices, valid_indices = train_test_split(
        all_indices,
        test_size=5000,
        random_state=split_seed,
        stratify=train_labels,
    )
    if max_train_samples:
        if max_train_samples < NUM_CLASSES:
            raise ValueError("max_train_samples must include every class")
        train_indices, _ = train_test_split(
            train_indices,
            train_size=min(max_train_samples, train_indices.shape[0]),
            random_state=split_seed + 1,
            stratify=train_labels[train_indices],
        )
    calibration_size = min(calibration_samples, train_indices.shape[0])
    if calibration_size == train_indices.shape[0]:
        calibration_indices = train_indices.copy()
    else:
        calibration_indices, _ = train_test_split(
            train_indices,
            train_size=calibration_size,
            random_state=split_seed + 2,
            stratify=train_labels[train_indices],
        )

    def convert(images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        return (
            torch.from_numpy(images[indices].copy()).to(torch.uint8),
            torch.from_numpy(labels[indices].copy()).to(torch.int64),
        )

    test_indices = np.arange(test_labels.shape[0], dtype=np.int64)
    archive = data_root / "cifar-100-python.tar.gz"
    metadata: dict[str, object] = {
        "train_size": int(train_indices.shape[0]),
        "validation_size": int(valid_indices.shape[0]),
        "test_size": int(test_indices.shape[0]),
        "calibration_size": int(calibration_indices.shape[0]),
        "train_index_sha256": _array_sha256(train_indices),
        "validation_index_sha256": _array_sha256(valid_indices),
        "calibration_index_sha256": _array_sha256(calibration_indices),
        "official_train_image_sha256": _array_sha256(train_images),
        "official_train_label_sha256": _array_sha256(train_labels),
        "official_test_image_sha256": _array_sha256(test_images),
        "official_test_label_sha256": _array_sha256(test_labels),
        "archive_sha256": file_sha256(archive) if archive.is_file() else None,
        "pixel_min": int(min(train_images.min(), test_images.min())),
        "pixel_max": int(max(train_images.max(), test_images.max())),
    }
    return (
        convert(train_images, train_labels, train_indices),
        convert(train_images, train_labels, valid_indices),
        convert(test_images, test_labels, test_indices),
        convert(train_images, train_labels, calibration_indices),
        metadata,
    )


def augment_symbols(
    symbols: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    images = symbols.reshape(-1, *IMAGE_SHAPE)
    channels_first = images.permute(0, 3, 1, 2)
    padded = F.pad(channels_first, (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
    batch = images.shape[0]
    offsets_y = torch.randint(0, 9, (batch,), generator=generator)
    offsets_x = torch.randint(0, 9, (batch,), generator=generator)
    rows = offsets_y[:, None] + torch.arange(32)[None, :]
    columns = offsets_x[:, None] + torch.arange(32)[None, :]
    batch_index = torch.arange(batch)[:, None, None]
    cropped = padded[
        batch_index,
        rows[:, :, None],
        columns[:, None, :],
    ]
    flip = torch.rand(batch, generator=generator) < 0.5
    cropped[flip] = cropped[flip].flip(dims=(2,))
    return cropped.contiguous().reshape(batch, INPUT_SYMBOLS)


@torch.no_grad()
def collect_states(
    model: BitPlaneLUTClassifier,
    symbols: torch.Tensor,
    device: torch.device,
    *,
    block_count: int,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    for start in range(0, symbols.shape[0], batch_size):
        rows.append(
            model.state_after(
                symbols[start : start + batch_size].to(device),
                block_count=block_count,
                mode="hard",
            ).cpu()
        )
    return torch.cat(rows, dim=0)


@torch.no_grad()
def strict_evaluate(
    model: BitPlaneLUTClassifier,
    executor: StrictBitPlaneLUTExecutor,
    split: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    batch_size: int,
    group_temperature: float,
) -> tuple[dict[str, float], int]:
    symbols, labels = split
    audit = BooleanRuntimeAudit()
    logits_rows = []
    expected_rows = []
    for start in range(0, symbols.shape[0], batch_size):
        batch = symbols[start : start + batch_size]
        expected_rows.append(model.hard_logits(batch.to(device)).cpu())
        with audit:
            logits_rows.append(executor.logits(batch))
    logits = torch.cat(logits_rows)
    expected = torch.cat(expected_rows)
    if not torch.equal(logits, expected):
        raise AssertionError("strict CIFAR-100 logits differ from hardened carrier")
    accuracy = float((logits.argmax(dim=-1) == labels).float().mean().item())
    loss = float(
        F.cross_entropy(logits.float() / group_temperature, labels).item()
    )
    return {"hard_acc": accuracy, "hard_loss": loss}, audit.operations


def main() -> None:
    args = parse_args()
    if args.votes_per_class < 2 or args.votes_per_class % 2:
        raise ValueError("votes_per_class must be a positive even integer")
    if not 1 <= args.minimum_epochs <= args.epochs_per_block:
        raise ValueError("minimum_epochs must be inside the epoch budget")
    if args.patience < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("patience and batch sizes must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("out_dir is not empty; use a new run directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = resolve_device(args.device)
    train_split, valid_split, test_split, calibration_split, split_metadata = load_splits(
        args.data_root,
        download=args.download,
        split_seed=args.split_seed,
        max_train_samples=args.max_train_samples,
        calibration_samples=args.calibration_samples,
    )
    train_loader = make_loader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed + 10_000,
    )
    valid_loader = make_loader(
        valid_split,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed + 20_000,
    )
    test_loader = make_loader(
        test_split,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed + 30_000,
    )
    vote_bits = args.votes_per_class * NUM_CLASSES
    state_bits = INPUT_SYMBOLS * 8 + vote_bits
    model = BitPlaneLUTClassifier(
        input_symbols=INPUT_SYMBOLS,
        state_bits=state_bits,
        num_classes=NUM_CLASSES,
        blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        arity=args.arity,
        candidate_count=args.candidate_count,
        seed=args.seed,
        candidate_policy=args.candidate_policy,
        input_shape=IMAGE_SHAPE,
    ).to(device)
    group_temperature = args.group_temperature or math.sqrt(args.votes_per_class)
    train_class_count = torch.bincount(
        train_split[1], minlength=NUM_CLASSES
    ).to(device=device, dtype=torch.float32)
    positive_class_weight = (
        (train_split[1].numel() - train_class_count) / train_class_count
    )
    vote_positive_weight = positive_class_weight[model.readout_group]
    source_root = Path(__file__).resolve().parent
    manifest = {
        "args": {
            **vars(args),
            "out_dir": str(args.out_dir.resolve()),
            "data_root": str(args.data_root.resolve()),
            "protocol_path": str(args.protocol_path.resolve()),
            "group_temperature_effective": group_temperature,
            "state_bits": state_bits,
            "vote_bits": vote_bits,
        },
        "split": split_metadata,
        "protocol_sha256": file_sha256(args.protocol_path),
        "source_sha256": {
            name: file_sha256(source_root / name) for name in SOURCE_FILES
        },
        "python": os.sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    manifest["sha256"] = json_sha256(manifest)
    atomic_json(args.out_dir / "run_manifest.json", manifest)

    epoch_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    gradient_finite = True
    final_pre_hard_test: dict[str, float] | None = None
    for block_index, block in enumerate(model.blocks):
        model.set_trainable_block(block_index)
        parameters = trainable_parameters([block])
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_per_block,
            eta_min=args.learning_rate * 0.05,
        )
        best_accuracy = -1.0
        best_loss = math.inf
        best_epoch = 0
        stale_epochs = 0
        best_state: dict[str, torch.Tensor] | None = None
        block_history: list[dict[str, object]] = []
        block_started = time.perf_counter()
        for epoch in range(args.epochs_per_block):
            model.train()
            progress = epoch / max(1, args.epochs_per_block - 1)
            temperature = set_temperatures(
                block, args.temperature_start, args.temperature_end, progress
            )
            augmentation_generator = torch.Generator().manual_seed(
                args.seed * 1_000_003 + block_index * 10_007 + epoch
            )
            loss_sum = task_sum = code_sum = constraint_sum = 0.0
            samples = 0
            maximum_gradient = 0.0
            for symbols, labels in train_loader:
                if args.augment:
                    symbols = augment_symbols(symbols, augmentation_generator)
                symbols = symbols.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                state = model.state_after(
                    symbols, block_count=block_index + 1, mode="hard_st"
                )
                logits = model.group_sum(state) / group_temperature
                target = model.target_state(labels).to(state.dtype)
                vote_state = state[:, model.preserved_bits :]
                code_weight = torch.where(
                    target.bool(),
                    vote_positive_weight.view(1, -1),
                    torch.ones_like(target),
                )
                code_weight = code_weight / code_weight.mean().detach()
                task_loss = F.cross_entropy(logits, labels)
                code_loss = ((vote_state - target).square() * code_weight).mean()
                constraint = block.constraint_loss(
                    fanout_cap=args.fanout_cap,
                    table_cost_weight=args.table_cost_weight,
                )
                loss = (
                    task_loss
                    + args.code_loss_weight * code_loss
                    + args.wiring_constraint_weight * constraint
                )
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    parameters, args.gradient_clip
                )
                finite = bool(torch.isfinite(gradient).item()) and bool(
                    torch.isfinite(loss).item()
                )
                gradient_finite &= finite
                if not finite:
                    raise FloatingPointError("non-finite CIFAR-100 training state")
                maximum_gradient = max(maximum_gradient, float(gradient.item()))
                optimizer.step()
                count = labels.numel()
                samples += count
                loss_sum += float(loss.detach().item()) * count
                task_sum += float(task_loss.detach().item()) * count
                code_sum += float(code_loss.detach().item()) * count
                constraint_sum += float(constraint.detach().item()) * count
            scheduler.step()
            validation = evaluate(
                model,
                valid_loader,
                device,
                block_count=block_index + 1,
                group_temperature=group_temperature,
            )
            row = {
                "block": block_index,
                "epoch": epoch + 1,
                "train_loss": loss_sum / samples,
                "train_task_loss": task_sum / samples,
                "train_code_loss": code_sum / samples,
                "train_constraint_loss": constraint_sum / samples,
                "temperature": temperature,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "unclipped_gradient_l2_max": maximum_gradient,
                **{f"validation_{key}": value for key, value in validation.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            epoch_rows.append(row)
            block_history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            accuracy = validation["hard_acc"]
            hard_loss = validation["hard_loss"]
            improved = accuracy > best_accuracy or (
                accuracy == best_accuracy and hard_loss < best_loss
            )
            if improved:
                best_accuracy = accuracy
                best_loss = hard_loss
                best_epoch = epoch + 1
                best_state = clone_state(block)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch + 1 >= args.minimum_epochs and stale_epochs >= args.patience:
                break
        if best_state is None:
            raise RuntimeError("block training did not produce a checkpoint")
        block.load_state_dict(best_state, strict=True)
        pre_refit = evaluate(
            model,
            valid_loader,
            device,
            block_count=block_index + 1,
            group_temperature=group_temperature,
        )
        if block_index == len(model.blocks) - 1:
            final_pre_hard_test = evaluate(
                model,
                test_loader,
                device,
                block_count=block_index + 1,
                group_temperature=group_temperature,
            )
        calibration_symbols, calibration_labels = calibration_split
        calibration_input = collect_states(
            model,
            calibration_symbols,
            device,
            block_count=block_index,
            batch_size=args.eval_batch_size,
        ).to(device)
        calibration_labels_device = calibration_labels.to(device)
        calibration_target = model.target_state(calibration_labels_device)
        calibration_weight = None
        if args.refit_mode != "argmax":
            calibration_weight = model.target_weight(calibration_labels_device)
        refit = block.refit(
            calibration_input,
            calibration_target,
            args.refit_mode,
            wiring_passes=args.wiring_passes,
            target_weight=calibration_weight,
        )
        block.freeze_hard()
        post_refit = evaluate(
            model,
            valid_loader,
            device,
            block_count=block_index + 1,
            group_temperature=group_temperature,
        )
        diagnostic_symbols = valid_split[0][: args.diagnostic_rows]
        diagnostic_input = collect_states(
            model,
            diagnostic_symbols,
            device,
            block_count=block_index,
            batch_size=args.eval_batch_size,
        )
        diagnostic_output = collect_states(
            model,
            diagnostic_symbols,
            device,
            block_count=block_index + 1,
            batch_size=args.eval_batch_size,
        )
        threshold = 0.90 * best_accuracy
        epochs_to_target = next(
            int(row["epoch"])
            for row in block_history
            if float(row["validation_hard_acc"]) >= threshold
        )
        block_row = {
            "block": block_index,
            "best_epoch": best_epoch,
            "epochs_ran": len(block_history),
            "epochs_to_90pct_best": epochs_to_target,
            **{f"pre_refit_{key}": value for key, value in pre_refit.items()},
            **{f"post_refit_{key}": value for key, value in post_refit.items()},
            "refit_bit_error": refit.bit_error,
            "refit_address_coverage": refit.address_coverage,
            "changed_truth_ratio": refit.changed_truth_ratio,
            "changed_wiring_ratio": refit.changed_wiring_ratio,
            "block_train_seconds": time.perf_counter() - block_started,
            "state_diagnostics": boolean_state_diagnostics(
                diagnostic_output, previous=diagnostic_input
            ),
            "vote_state_diagnostics": boolean_state_diagnostics(
                diagnostic_output[:, model.preserved_bits :],
                previous=diagnostic_input[:, model.preserved_bits :],
            ),
        }
        block_rows.append(block_row)
        print(json.dumps(block_row, sort_keys=True), flush=True)
        del calibration_input, calibration_target, diagnostic_input, diagnostic_output
        if calibration_weight is not None:
            del calibration_weight
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_pre_hard_test is None:
        raise RuntimeError("missing final pre-hard test metrics")
    training_seconds = time.perf_counter() - started
    payload = model.hard_payload()
    validate_hard_payload(payload)
    payload_path = args.out_dir / "hard_payload.pt"
    atomic_torch_save(payload_path, payload)
    executor = StrictBitPlaneLUTExecutor(payload)
    strict_validation, validation_audit_ops = strict_evaluate(
        model,
        executor,
        valid_split,
        device,
        batch_size=args.eval_batch_size,
        group_temperature=group_temperature,
    )
    strict_test, test_audit_ops = strict_evaluate(
        model,
        executor,
        test_split,
        device,
        batch_size=args.eval_batch_size,
        group_temperature=group_temperature,
    )
    last_pre_refit = block_rows[-1]
    structure = model.structural_diagnostics()
    result = {
        "method_id": {
            "argmax": "bitplane_lut_argmax",
            "truth_refit": "bitplane_lut_refit",
            "wiring_truth_refit": "bitplane_lut_wiring_refit",
        }[args.refit_mode],
        "dataset": "cifar100",
        "seed": args.seed,
        "state_bits": state_bits,
        "votes_per_class": args.votes_per_class,
        "vote_bits": vote_bits,
        "candidate_policy": args.candidate_policy,
        "refit_mode": args.refit_mode,
        "soft_acc": last_pre_refit["pre_refit_soft_acc"],
        "hard_acc": strict_validation["hard_acc"],
        "acc_gap": abs(
            float(last_pre_refit["pre_refit_soft_acc"])
            - strict_validation["hard_acc"]
        ),
        "soft_loss": last_pre_refit["pre_refit_soft_loss"],
        "hard_loss": strict_validation["hard_loss"],
        "loss_gap": abs(
            float(last_pre_refit["pre_refit_soft_loss"])
            - strict_validation["hard_loss"]
        ),
        "test_soft_acc": final_pre_hard_test["soft_acc"],
        "test_hard_acc": strict_test["hard_acc"],
        "test_acc_gap": abs(
            final_pre_hard_test["soft_acc"] - strict_test["hard_acc"]
        ),
        "test_hard_loss": strict_test["hard_loss"],
        "train_time_s": training_seconds,
        "epochs_to_target": sum(
            int(row["epochs_to_90pct_best"]) for row in block_rows
        ),
        "strict_runtime": {
            "compliance": "operator_audited_bool_int",
            "float_tensor_count": 0,
            "audit_operations": validation_audit_ops + test_audit_ops,
            "validation_rows": int(valid_split[0].shape[0]),
            "test_rows": int(test_split[0].shape[0]),
            "exact_carrier_logit_match": True,
        },
        "hard_payload_sha256": file_sha256(payload_path),
        "hard_payload_bytes": payload_path.stat().st_size,
        "training_health": {
            "finite_gradients": gradient_finite,
            "late_validation_drop": max(
                0.0,
                max(float(row["validation_hard_acc"]) for row in epoch_rows)
                - strict_validation["hard_acc"],
            ),
        },
        "structure": structure,
        "block_results": block_rows,
        "run_manifest_sha256": manifest["sha256"],
    }
    atomic_json(args.out_dir / "result.json", result)
    write_csv(args.out_dir / "per_epoch.csv", epoch_rows)
    write_csv(
        args.out_dir / "block_results.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"state_diagnostics", "vote_state_diagnostics"}
            }
            for row in block_rows
        ],
    )
    print(
        json.dumps(
            {
                "result": str((args.out_dir / "result.json").resolve()),
                "validation_hard_acc": strict_validation["hard_acc"],
                "test_hard_acc": strict_test["hard_acc"],
                "audit_operations": validation_audit_ops + test_audit_ops,
                "payload_sha256": result["hard_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
