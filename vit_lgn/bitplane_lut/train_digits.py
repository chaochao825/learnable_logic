from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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


SOURCE_FILES = (
    "__init__.py",
    "executor.py",
    "layers.py",
    "model.py",
    "train_digits.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block-wise A8 bit-plane Hard-LGN on sklearn digits"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260724)
    parser.add_argument("--state-bits", type=int, choices=[672, 832], default=672)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--layers-per-block", type=int, default=1)
    parser.add_argument("--arity", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument(
        "--refit-mode",
        choices=["argmax", "truth_refit", "wiring_truth_refit"],
        default="truth_refit",
    )
    parser.add_argument("--wiring-passes", type=int, default=1)
    parser.add_argument("--epochs-per-block", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_split(
    split_seed: int, max_train_samples: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | tuple:
    dataset = load_digits()
    symbols = dataset.images.reshape(-1, 64).astype(np.uint8)
    labels = dataset.target.astype(np.int64)
    indices = np.arange(labels.shape[0])
    train_index, held_index = train_test_split(
        indices,
        test_size=0.30,
        random_state=split_seed,
        stratify=labels,
    )
    valid_index, test_index = train_test_split(
        held_index,
        test_size=0.50,
        random_state=split_seed + 1,
        stratify=labels[held_index],
    )
    if max_train_samples:
        if max_train_samples < 10:
            raise ValueError("max_train_samples must be zero or at least ten")
        selected, _ = train_test_split(
            train_index,
            train_size=min(max_train_samples, train_index.shape[0]),
            random_state=split_seed + 2,
            stratify=labels[train_index],
        )
        train_index = selected

    def convert(index: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(symbols[index].copy()).to(torch.uint8),
            torch.from_numpy(labels[index].copy()).to(torch.int64),
        )

    split_metadata = {
        "train_size": int(train_index.shape[0]),
        "validation_size": int(valid_index.shape[0]),
        "test_size": int(test_index.shape[0]),
        "train_index_sha256": hashlib.sha256(train_index.tobytes()).hexdigest(),
        "validation_index_sha256": hashlib.sha256(valid_index.tobytes()).hexdigest(),
        "test_index_sha256": hashlib.sha256(test_index.tobytes()).hexdigest(),
        "pixel_min": int(symbols.min()),
        "pixel_max": int(symbols.max()),
    }
    return convert(train_index), convert(valid_index), convert(test_index), split_metadata


def make_loader(
    split: tuple[torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(*split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate(
    model: BitPlaneLUTClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    block_count: int,
    group_temperature: float,
) -> dict[str, float]:
    model.eval()
    count = hard_correct = soft_correct = 0
    hard_loss_sum = soft_loss_sum = 0.0
    for symbols, labels in loader:
        symbols = symbols.to(device)
        labels = labels.to(device)
        hard_integer = model.hard_logits(symbols, block_count=block_count)
        soft_logits = model(
            symbols, block_count=block_count, mode="soft"
        )
        hard_logits = hard_integer.to(torch.float32) / group_temperature
        normalized_soft = soft_logits / group_temperature
        hard_loss_sum += float(
            F.cross_entropy(hard_logits, labels, reduction="sum").item()
        )
        soft_loss_sum += float(
            F.cross_entropy(normalized_soft, labels, reduction="sum").item()
        )
        hard_correct += int((hard_integer.argmax(dim=-1) == labels).sum().item())
        soft_correct += int((soft_logits.argmax(dim=-1) == labels).sum().item())
        count += labels.numel()
    model.train()
    hard_acc = hard_correct / count
    soft_acc = soft_correct / count
    hard_loss = hard_loss_sum / count
    soft_loss = soft_loss_sum / count
    return {
        "soft_acc": soft_acc,
        "hard_acc": hard_acc,
        "acc_gap": abs(soft_acc - hard_acc),
        "soft_loss": soft_loss,
        "hard_loss": hard_loss,
        "loss_gap": abs(soft_loss - hard_loss),
    }


@torch.no_grad()
def collect_state(
    model: BitPlaneLUTClassifier,
    split: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    block_count: int,
) -> torch.Tensor:
    symbols, _ = split
    rows = []
    for start in range(0, symbols.shape[0], 256):
        rows.append(
            model.state_after(
                symbols[start:start + 256].to(device),
                block_count=block_count,
                mode="hard",
            ).cpu()
        )
    return torch.cat(rows, dim=0)


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def set_temperatures(
    block: torch.nn.Module, start: float, end: float, progress: float
) -> float:
    temperature = start * ((end / start) ** progress)
    for layer in block.layers:
        layer.wiring_temperature = temperature
        layer.truth_temperature = temperature
    return temperature


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.epochs_per_block < 1 or args.batch_size < 1:
        raise ValueError("training budget must be positive")
    if args.temperature_start <= 0 or args.temperature_end <= 0:
        raise ValueError("temperatures must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("out_dir is not empty; use a new reproducible run directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = resolve_device(args.device)
    train_split, valid_split, test_split, split_metadata = make_split(
        args.split_seed, args.max_train_samples
    )
    train_loader = make_loader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed + 10_000,
    )
    valid_loader = make_loader(
        valid_split,
        batch_size=512,
        shuffle=False,
        seed=args.seed + 20_000,
    )
    test_loader = make_loader(
        test_split,
        batch_size=512,
        shuffle=False,
        seed=args.seed + 30_000,
    )
    model = BitPlaneLUTClassifier(
        input_symbols=64,
        state_bits=args.state_bits,
        num_classes=10,
        blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        arity=args.arity,
        candidate_count=args.candidate_count,
        seed=args.seed,
    ).to(device)
    group_temperature = args.group_temperature or math.sqrt(model.vote_bits / 10)
    train_class_count = torch.bincount(
        train_split[1], minlength=model.num_classes
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
            "group_temperature_effective": group_temperature,
        },
        "split": split_metadata,
        "source_sha256": {
            name: file_sha256(source_root / name) for name in SOURCE_FILES
        },
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sklearn": sklearn.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    manifest["sha256"] = json_sha256(manifest)
    atomic_json(args.out_dir / "run_manifest.json", manifest)

    epoch_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    gradient_finite = True
    for block_index, block in enumerate(model.blocks):
        model.set_trainable_block(block_index)
        parameters = trainable_parameters([block])
        if not parameters:
            raise RuntimeError("current block has no trainable LUT/wiring shadows")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_block, eta_min=args.learning_rate * 0.05
        )
        best_accuracy = -1.0
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        block_history: list[dict[str, object]] = []
        block_started = time.perf_counter()
        for epoch in range(args.epochs_per_block):
            model.train()
            progress = epoch / max(1, args.epochs_per_block - 1)
            temperature = set_temperatures(
                block,
                args.temperature_start,
                args.temperature_end,
                progress,
            )
            loss_sum = task_sum = code_sum = constraint_sum = 0.0
            samples = 0
            maximum_gradient = 0.0
            for symbols, labels in train_loader:
                symbols = symbols.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                state = model.state_after(
                    symbols, block_count=block_index + 1, mode="hard_st"
                )
                logits = model.group_sum(state) / group_temperature
                target = model.target_state(labels).to(state.dtype)
                vote_state = state[:, model.preserved_bits:]
                code_weight = torch.where(
                    target.bool(),
                    vote_positive_weight.view(1, -1),
                    torch.ones_like(target),
                )
                code_weight = code_weight / code_weight.mean().detach()
                task_loss = F.cross_entropy(logits, labels)
                code_loss = (
                    (vote_state - target).square() * code_weight
                ).mean()
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
                    raise FloatingPointError("non-finite block-wise training state")
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
            accuracy = validation["hard_acc"]
            hard_loss = validation["hard_loss"]
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and hard_loss < best_loss
            ):
                best_accuracy = accuracy
                best_loss = hard_loss
                best_epoch = epoch + 1
                best_state = clone_state(block)
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
        train_symbols, train_labels = train_split
        calibration_input, calibration_target, calibration_weight = model.calibration_states(
            train_symbols.to(device),
            train_labels.to(device),
            block_index,
        )
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
        valid_input_state = collect_state(
            model,
            valid_split,
            device,
            block_count=block_index,
        )
        valid_output_state = collect_state(
            model,
            valid_split,
            device,
            block_count=block_index + 1,
        )
        threshold = 0.90 * best_accuracy
        epochs_to_target = next(
            int(row["epoch"])
            for row in block_history
            if float(row["validation_hard_acc"]) >= threshold
        )
        block_rows.append({
            "block": block_index,
            "best_epoch": best_epoch,
            "epochs_to_90pct_best": epochs_to_target,
            **{f"pre_refit_{key}": value for key, value in pre_refit.items()},
            **{f"post_refit_{key}": value for key, value in post_refit.items()},
            "refit_bit_error": refit.bit_error,
            "refit_address_coverage": refit.address_coverage,
            "changed_truth_ratio": refit.changed_truth_ratio,
            "changed_wiring_ratio": refit.changed_wiring_ratio,
            "block_train_seconds": time.perf_counter() - block_started,
            "state_diagnostics": boolean_state_diagnostics(
                valid_output_state, previous=valid_input_state
            ),
            "vote_state_diagnostics": boolean_state_diagnostics(
                valid_output_state[:, model.preserved_bits:],
                previous=valid_input_state[:, model.preserved_bits:],
            ),
        })
        print(json.dumps(block_rows[-1], sort_keys=True), flush=True)

    training_seconds = time.perf_counter() - started
    final_carrier_validation = evaluate(
        model,
        valid_loader,
        device,
        block_count=len(model.blocks),
        group_temperature=group_temperature,
    )
    final_carrier_test = evaluate(
        model,
        test_loader,
        device,
        block_count=len(model.blocks),
        group_temperature=group_temperature,
    )
    expected_validation_logits = model.hard_logits(
        valid_split[0].to(device)
    ).cpu()
    expected_test_logits = model.hard_logits(test_split[0].to(device)).cpu()
    payload = model.hard_payload()
    validate_hard_payload(payload)
    payload_path = args.out_dir / "hard_payload.pt"
    atomic_torch_save(payload_path, payload)
    payload_hash = file_sha256(payload_path)
    executor = StrictBitPlaneLUTExecutor(payload)
    audit = BooleanRuntimeAudit()
    with audit:
        strict_validation_logits = executor.logits(valid_split[0])
        strict_test_logits = executor.logits(test_split[0])
        strict_trace = executor.state_trace(valid_split[0])
    if not torch.equal(strict_validation_logits, expected_validation_logits):
        raise AssertionError("strict validation logits differ from hardened model")
    if not torch.equal(strict_test_logits, expected_test_logits):
        raise AssertionError("strict test logits differ from hardened model")

    strict_validation_accuracy = float(
        (strict_validation_logits.argmax(dim=-1) == valid_split[1]).float().mean().item()
    )
    strict_test_accuracy = float(
        (strict_test_logits.argmax(dim=-1) == test_split[1]).float().mean().item()
    )
    strict_validation_loss = float(
        F.cross_entropy(
            strict_validation_logits.float() / group_temperature,
            valid_split[1],
        ).item()
    )
    strict_test_loss = float(
        F.cross_entropy(
            strict_test_logits.float() / group_temperature,
            test_split[1],
        ).item()
    )
    trace_diagnostics = []
    for index, state in enumerate(strict_trace):
        previous = strict_trace[index - 1] if index else None
        trace_diagnostics.append({
            "state_index": index,
            **boolean_state_diagnostics(state, previous=previous),
            "vote_state": boolean_state_diagnostics(
                state[:, model.preserved_bits:],
                previous=(
                    previous[:, model.preserved_bits:]
                    if previous is not None else None
                ),
            ),
        })
    structure = model.structural_diagnostics()
    last_pre_refit = block_rows[-1]
    result = {
        "method_id": {
            "argmax": "bitplane_lut_argmax",
            "truth_refit": "bitplane_lut_refit",
            "wiring_truth_refit": "bitplane_lut_wiring_refit",
        }[args.refit_mode],
        "dataset": "sklearn_digits",
        "seed": args.seed,
        "state_bits": args.state_bits,
        "refit_mode": args.refit_mode,
        "soft_acc": last_pre_refit["pre_refit_soft_acc"],
        "hard_acc": strict_validation_accuracy,
        "acc_gap": abs(
            float(last_pre_refit["pre_refit_soft_acc"])
            - strict_validation_accuracy
        ),
        "soft_loss": last_pre_refit["pre_refit_soft_loss"],
        "hard_loss": strict_validation_loss,
        "loss_gap": abs(
            float(last_pre_refit["pre_refit_soft_loss"])
            - strict_validation_loss
        ),
        "test_soft_acc": final_carrier_test["soft_acc"],
        "test_hard_acc": strict_test_accuracy,
        "test_acc_gap": abs(final_carrier_test["soft_acc"] - strict_test_accuracy),
        "test_hard_loss": strict_test_loss,
        "train_time_s": training_seconds,
        "epochs_to_target": sum(
            int(row["epochs_to_90pct_best"]) for row in block_rows
        ),
        "strict_runtime": {
            "compliance": "operator_audited_bool_int",
            "float_tensor_count": 0,
            "audit_operations": audit.operations,
            "validation_rows": int(valid_split[0].shape[0]),
            "test_rows": int(test_split[0].shape[0]),
            "exact_carrier_logit_match": True,
        },
        "hard_payload_sha256": payload_hash,
        "hard_payload_bytes": payload_path.stat().st_size,
        "training_health": {
            "finite_gradients": gradient_finite,
            "late_validation_drop": max(
                0.0,
                max(float(row["validation_hard_acc"]) for row in epoch_rows)
                - strict_validation_accuracy,
            ),
        },
        "structure": structure,
        "block_results": block_rows,
        "state_trace_diagnostics": trace_diagnostics,
        "final_carrier_validation": final_carrier_validation,
        "final_carrier_test": final_carrier_test,
        "run_manifest_sha256": manifest["sha256"],
    }
    atomic_json(args.out_dir / "result.json", result)
    write_csv(args.out_dir / "per_epoch.csv", epoch_rows)
    flat_block_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"state_diagnostics", "vote_state_diagnostics"}
        }
        for row in block_rows
    ]
    write_csv(args.out_dir / "block_results.csv", flat_block_rows)
    print(json.dumps({
        "result": str((args.out_dir / "result.json").resolve()),
        "validation_hard_acc": strict_validation_accuracy,
        "test_hard_acc": strict_test_accuracy,
        "audit_operations": audit.operations,
        "payload_sha256": payload_hash,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
