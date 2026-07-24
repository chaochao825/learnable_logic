from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

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


EXPECTED_CORPUS_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
)
SOURCE_FILES = (
    "__init__.py",
    "executor.py",
    "layers.py",
    "model.py",
    "run_shakespeare_scale.sh",
    "train_digits.py",
    "train_shakespeare.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict bit-plane LUT next-character model"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--prefix-run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260724)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--votes-per-class", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--layers-per-block", type=int, default=1)
    parser.add_argument("--arity", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument(
        "--candidate-policy",
        choices=["mixed", "sequence_causal"],
        default="sequence_causal",
    )
    parser.add_argument("--epochs-per-block", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
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
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--validation-samples", type=int, default=20_000)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=5_000)
    parser.add_argument("--diagnostic-rows", type=int, default=512)
    parser.add_argument("--generation-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def sampled_windows(
    encoded: torch.Tensor,
    *,
    context_length: int,
    samples: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if encoded.dtype != torch.uint8 or encoded.ndim != 1:
        raise TypeError("encoded corpus segment must be one-dimensional uint8")
    windows = encoded.unfold(0, context_length + 1, 1)
    available = windows.shape[0]
    if available < 1:
        raise ValueError("corpus segment is shorter than one context window")
    if samples < 0:
        raise ValueError("sample count cannot be negative")
    if samples and samples < available:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(available, generator=generator)[:samples]
        indices = indices.sort().values
    else:
        indices = torch.arange(available, dtype=torch.int64)
    selected = windows[indices]
    return (
        selected[:, :-1].contiguous(),
        selected[:, -1].to(torch.int64).contiguous(),
    ), indices


def load_corpus_splits(
    corpus_path: Path,
    *,
    context_length: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    calibration_samples: int,
    split_seed: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    list[str],
    dict[str, object],
]:
    corpus_sha256 = file_sha256(corpus_path)
    if corpus_sha256 != EXPECTED_CORPUS_SHA256:
        raise RuntimeError("Tiny Shakespeare corpus hash mismatch")
    text = corpus_path.read_text(encoding="utf-8")
    characters = sorted(set(text))
    if len(characters) != 65:
        raise RuntimeError("unexpected Tiny Shakespeare vocabulary")
    stoi = {character: index for index, character in enumerate(characters)}
    encoded = torch.tensor([stoi[character] for character in text], dtype=torch.uint8)
    train_end = int(encoded.numel() * 0.90)
    validation_end = int(encoded.numel() * 0.95)
    train_stream = encoded[:train_end]
    validation_stream = encoded[train_end:validation_end]
    test_stream = encoded[validation_end:]
    train_split, train_indices = sampled_windows(
        train_stream,
        context_length=context_length,
        samples=train_samples,
        seed=split_seed,
    )
    validation_split, validation_indices = sampled_windows(
        validation_stream,
        context_length=context_length,
        samples=validation_samples,
        seed=split_seed + 1,
    )
    test_split, test_indices = sampled_windows(
        test_stream,
        context_length=context_length,
        samples=test_samples,
        seed=split_seed + 2,
    )
    calibration_size = min(calibration_samples, train_split[0].shape[0])
    calibration_order = torch.randperm(
        train_split[0].shape[0],
        generator=torch.Generator().manual_seed(split_seed + 3),
    )[:calibration_size]
    calibration_split = (
        train_split[0][calibration_order].contiguous(),
        train_split[1][calibration_order].contiguous(),
    )

    def missing_characters(labels: torch.Tensor) -> list[str]:
        counts = torch.bincount(labels, minlength=len(characters))
        return [characters[index] for index in torch.nonzero(counts == 0).flatten()]

    metadata = {
        "corpus_sha256": corpus_sha256,
        "corpus_characters": len(text),
        "vocab_size": len(characters),
        "vocab_sha256": hashlib.sha256("".join(characters).encode()).hexdigest(),
        "train_stream_characters": int(train_stream.numel()),
        "validation_stream_characters": int(validation_stream.numel()),
        "test_stream_characters": int(test_stream.numel()),
        "train_rows": int(train_split[0].shape[0]),
        "validation_rows": int(validation_split[0].shape[0]),
        "test_rows": int(test_split[0].shape[0]),
        "calibration_rows": int(calibration_split[0].shape[0]),
        "train_index_sha256": tensor_sha256(train_indices),
        "validation_index_sha256": tensor_sha256(validation_indices),
        "test_index_sha256": tensor_sha256(test_indices),
        "train_stream_sha256": tensor_sha256(train_stream),
        "validation_stream_sha256": tensor_sha256(validation_stream),
        "test_stream_sha256": tensor_sha256(test_stream),
        "train_missing_characters": missing_characters(train_split[1]),
        "validation_missing_characters": missing_characters(validation_split[1]),
        "test_missing_characters": missing_characters(test_split[1]),
    }
    return (
        train_split,
        validation_split,
        test_split,
        calibration_split,
        train_stream,
        characters,
        metadata,
    )


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


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    choices = logits.topk(k=min(k, logits.shape[1]), dim=-1).indices
    return int((choices == labels[:, None]).any(dim=1).sum().item())


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
    hard_correct = top5_correct = 0
    loss_sum = 0.0
    for start in range(0, symbols.shape[0], batch_size):
        batch = symbols[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        expected = model.hard_logits(batch.to(device)).cpu()
        with audit:
            logits = executor.logits(batch)
        if not torch.equal(logits, expected):
            raise AssertionError("strict sequence logits differ from hardened carrier")
        hard_correct += int((logits.argmax(dim=-1) == batch_labels).sum().item())
        top5_correct += topk_correct(logits, batch_labels, 5)
        loss_sum += float(
            F.cross_entropy(
                logits.float() / group_temperature,
                batch_labels,
                reduction="sum",
            ).item()
        )
    rows = labels.numel()
    loss = loss_sum / rows
    return {
        "hard_acc": hard_correct / rows,
        "hard_top5_acc": top5_correct / rows,
        "hard_loss": loss,
        "hard_perplexity": math.exp(min(loss, 30.0)),
        "hard_bits_per_character": loss / math.log(2.0),
    }, audit.operations


def ngram_reference(
    train_stream: torch.Tensor,
    split: tuple[torch.Tensor, torch.Tensor],
    *,
    vocab_size: int,
    order: int,
) -> dict[str, float]:
    if order not in {0, 1, 2}:
        raise ValueError("only unigram, bigram, and trigram references are supported")
    unigram = torch.bincount(train_stream.to(torch.int64), minlength=vocab_size)
    if order == 0:
        table = unigram.view(1, -1)
    else:
        windows = train_stream.unfold(0, order + 1, 1).to(torch.int64)
        prefix = torch.zeros(windows.shape[0], dtype=torch.int64)
        for slot in range(order):
            prefix = prefix * vocab_size + windows[:, slot]
        flat = prefix * vocab_size + windows[:, -1]
        table = torch.bincount(
            flat, minlength=(vocab_size**order) * vocab_size
        ).reshape(vocab_size**order, vocab_size)
    symbols, labels = split
    if order == 0:
        rows = table.expand(symbols.shape[0], -1)
    else:
        prefix = torch.zeros(symbols.shape[0], dtype=torch.int64)
        for slot in range(symbols.shape[1] - order, symbols.shape[1]):
            prefix = prefix * vocab_size + symbols[:, slot].to(torch.int64)
        rows = table[prefix].clone()
        unseen = rows.sum(dim=1) == 0
        rows[unseen] = unigram
    labels = labels.to(torch.int64)
    return {
        "hard_acc": float((rows.argmax(dim=-1) == labels).float().mean().item()),
        "hard_top5_acc": topk_correct(rows, labels, 5) / labels.numel(),
    }


def greedy_generate(
    executor: StrictBitPlaneLUTExecutor,
    prompt: torch.Tensor,
    characters: list[str],
    length: int,
) -> tuple[str, int]:
    context = prompt.clone().to(torch.uint8)
    generated = context.tolist()
    audit = BooleanRuntimeAudit()
    for _ in range(length):
        with audit:
            next_token = int(executor.predict(context.view(1, -1)).item())
        generated.append(next_token)
        context = torch.cat(
            (context[1:], torch.tensor([next_token], dtype=torch.uint8))
        )
    return "".join(characters[token] for token in generated), audit.operations


@torch.no_grad()
def load_hard_prefix(
    model: BitPlaneLUTClassifier,
    payload: dict[str, object],
) -> int:
    """Load an exact strict payload as frozen leading blocks."""

    validate_hard_payload(payload)
    expected_scalars = {
        "input_symbols": model.input_symbols,
        "input_bits": model.input_bits,
        "state_bits": model.state_bits,
        "preserved_bits": model.preserved_bits,
        "vote_bits": model.vote_bits,
        "num_classes": model.num_classes,
    }
    for key, expected in expected_scalars.items():
        if int(payload[key]) != expected:
            raise ValueError(f"prefix payload mismatch: {key}")
    if payload["candidate_policy"] != model.candidate_policy:
        raise ValueError("prefix candidate policy mismatch")
    if list(payload.get("input_shape") or []) != list(model.input_shape or ()):
        raise ValueError("prefix input shape mismatch")
    if not torch.equal(
        payload["encoder_route"].to(torch.int64),
        model.encoder_route.detach().cpu().to(torch.int64),
    ):
        raise ValueError("prefix encoder route mismatch")
    if not torch.equal(
        payload["readout"]["group"].to(torch.int64),
        model.readout_group.detach().cpu().to(torch.int64),
    ):
        raise ValueError("prefix readout mismatch")

    prefix_blocks = payload["blocks"]
    if not isinstance(prefix_blocks, (list, tuple)) or not prefix_blocks:
        raise ValueError("prefix payload has no blocks")
    if len(prefix_blocks) >= len(model.blocks):
        raise ValueError("prefix must be shallower than the target model")
    for block_index, block_payload in enumerate(prefix_blocks):
        block = model.blocks[block_index]
        layer_payloads = block_payload["layers"]
        if len(layer_payloads) != len(block.layers):
            raise ValueError("prefix layer count mismatch")
        for layer, layer_payload in zip(block.layers, layer_payloads):
            sources = layer_payload["source_indices"].to(
                device=layer.frozen_sources.device, dtype=torch.int64
            )
            truth = layer_payload["truth_table"].to(
                device=layer.frozen_truth.device, dtype=torch.bool
            )
            if sources.shape != layer.frozen_sources.shape:
                raise ValueError("prefix source shape mismatch")
            if truth.shape != layer.frozen_truth.shape:
                raise ValueError("prefix truth shape mismatch")
            candidates = layer.candidate_indices.to(sources.device)
            matches = sources.unsqueeze(-1).eq(candidates)
            if not bool(matches.any(dim=-1).all()):
                raise ValueError("prefix source is outside the candidate pool")
            layer.frozen_sources.copy_(sources)
            layer.frozen_truth.copy_(truth)
            layer.frozen_flag.fill_(True)
            layer.wiring_logits.requires_grad_(False)
            layer.truth_logits.requires_grad_(False)
    return len(prefix_blocks)


def main() -> None:
    args = parse_args()
    if args.context_length < 2 or args.context_length % 8:
        raise ValueError("context_length must be a positive multiple of eight")
    if args.votes_per_class < 8 or args.votes_per_class % 8:
        raise ValueError("votes_per_class must be a positive multiple of eight")
    if not 1 <= args.minimum_epochs <= args.epochs_per_block:
        raise ValueError("minimum_epochs must be inside the epoch budget")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("out_dir is not empty; use a new run directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = resolve_device(args.device)
    (
        train_split,
        validation_split,
        test_split,
        calibration_split,
        train_stream,
        characters,
        split_metadata,
    ) = load_corpus_splits(
        args.corpus_path,
        context_length=args.context_length,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        calibration_samples=args.calibration_samples,
        split_seed=args.split_seed,
    )
    vocab_size = len(characters)
    train_loader = make_loader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed + 10_000,
    )
    validation_loader = make_loader(
        validation_split,
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
    vote_bits = args.votes_per_class * vocab_size
    state_bits = args.context_length * 8 + vote_bits
    model = BitPlaneLUTClassifier(
        input_symbols=args.context_length,
        state_bits=state_bits,
        num_classes=vocab_size,
        blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        arity=args.arity,
        candidate_count=args.candidate_count,
        seed=args.seed,
        candidate_policy=args.candidate_policy,
        input_shape=(args.context_length,),
    ).to(device)
    prefix_blocks = 0
    prefix_block_rows: list[dict[str, object]] = []
    prefix_training_seconds = 0.0
    prefix_gradient_finite = True
    prefix_metadata: dict[str, object] | None = None
    if args.prefix_run_dir is not None:
        prefix_dir = args.prefix_run_dir.resolve()
        prefix_result_path = prefix_dir / "result.json"
        prefix_manifest_path = prefix_dir / "run_manifest.json"
        prefix_payload_path = prefix_dir / "hard_payload.pt"
        prefix_result = json.loads(prefix_result_path.read_text(encoding="utf-8"))
        prefix_manifest = json.loads(prefix_manifest_path.read_text(encoding="utf-8"))
        prefix_payload_sha256 = file_sha256(prefix_payload_path)
        if prefix_payload_sha256 != prefix_result["hard_payload_sha256"]:
            raise RuntimeError("prefix payload hash mismatch")
        if prefix_manifest["protocol_sha256"] != file_sha256(args.protocol_path):
            raise RuntimeError("prefix protocol hash mismatch")
        prefix_args = prefix_manifest["args"]
        for key in (
            "seed",
            "split_seed",
            "context_length",
            "votes_per_class",
            "layers_per_block",
            "arity",
            "candidate_count",
            "candidate_policy",
            "train_samples",
            "validation_samples",
            "test_samples",
            "calibration_samples",
        ):
            if prefix_args[key] != getattr(args, key):
                raise RuntimeError(f"prefix training argument mismatch: {key}")
        for key, value in split_metadata.items():
            if prefix_manifest["split"].get(key) != value:
                raise RuntimeError(f"prefix split provenance mismatch: {key}")
        prefix_payload = torch.load(
            prefix_payload_path, map_location="cpu", weights_only=True
        )
        prefix_blocks = load_hard_prefix(model, prefix_payload)
        if prefix_blocks != int(prefix_args["blocks"]):
            raise RuntimeError("prefix block count differs from its manifest")
        probe = validation_split[0][: min(256, validation_split[0].shape[0])]
        expected_prefix_logits = StrictBitPlaneLUTExecutor(prefix_payload).logits(probe)
        actual_prefix_logits = model.hard_logits(
            probe.to(device), block_count=prefix_blocks
        ).cpu()
        if not torch.equal(actual_prefix_logits, expected_prefix_logits):
            raise RuntimeError("loaded prefix differs from its strict executor")
        prefix_block_rows = list(prefix_result["block_results"])
        if len(prefix_block_rows) != prefix_blocks:
            raise RuntimeError("prefix result has incomplete block metrics")
        prefix_training_seconds = float(prefix_result["train_time_s"])
        prefix_gradient_finite = bool(
            prefix_result["training_health"]["finite_gradients"]
        )
        prefix_metadata = {
            "run_dir": str(prefix_dir),
            "blocks": prefix_blocks,
            "payload_sha256": prefix_payload_sha256,
            "result_sha256": file_sha256(prefix_result_path),
            "manifest_file_sha256": file_sha256(prefix_manifest_path),
            "manifest_sha256": prefix_manifest["sha256"],
            "exact_strict_logit_match": True,
        }
    group_temperature = args.group_temperature or math.sqrt(args.votes_per_class)
    train_class_count = torch.bincount(
        train_split[1], minlength=vocab_size
    ).to(device=device, dtype=torch.float32)
    safe_train_class_count = train_class_count.clamp_min(1.0)
    positive_class_weight = (
        (train_split[1].numel() - safe_train_class_count)
        / safe_train_class_count
    )
    vote_positive_weight = positive_class_weight[model.readout_group]
    references = {
        name: {
            "validation": ngram_reference(
                train_stream, validation_split, vocab_size=vocab_size, order=order
            ),
            "test": ngram_reference(
                train_stream, test_split, vocab_size=vocab_size, order=order
            ),
        }
        for name, order in (("unigram", 0), ("bigram", 1), ("trigram", 2))
    }
    source_root = Path(__file__).resolve().parent
    manifest = {
        "args": {
            **vars(args),
            "out_dir": str(args.out_dir.resolve()),
            "corpus_path": str(args.corpus_path.resolve()),
            "protocol_path": str(args.protocol_path.resolve()),
            "prefix_run_dir": (
                str(args.prefix_run_dir.resolve())
                if args.prefix_run_dir is not None
                else None
            ),
            "group_temperature_effective": group_temperature,
            "vocab_size": vocab_size,
            "state_bits": state_bits,
            "vote_bits": vote_bits,
        },
        "split": split_metadata,
        "vocabulary": characters,
        "prefix": prefix_metadata,
        "protocol_sha256": file_sha256(args.protocol_path),
        "source_sha256": {
            name: file_sha256(source_root / name) for name in SOURCE_FILES
        },
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    manifest["sha256"] = json_sha256(manifest)
    atomic_json(args.out_dir / "run_manifest.json", manifest)

    epoch_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = list(prefix_block_rows)
    started = time.perf_counter()
    gradient_finite = prefix_gradient_finite
    final_pre_hard_test: dict[str, float] | None = None
    for block_index in range(prefix_blocks, len(model.blocks)):
        block = model.blocks[block_index]
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
            loss_sum = task_sum = code_sum = constraint_sum = 0.0
            hard_correct = samples = 0
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
                    raise FloatingPointError("non-finite sequence training state")
                maximum_gradient = max(maximum_gradient, float(gradient.item()))
                optimizer.step()
                count = labels.numel()
                samples += count
                hard_correct += int((logits.argmax(dim=-1) == labels).sum().item())
                loss_sum += float(loss.detach().item()) * count
                task_sum += float(task_loss.detach().item()) * count
                code_sum += float(code_loss.detach().item()) * count
                constraint_sum += float(constraint.detach().item()) * count
            scheduler.step()
            validation = evaluate(
                model,
                validation_loader,
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
                "train_hard_acc": hard_correct / samples,
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
            validation_loader,
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
        calibration_target = model.target_state(calibration_labels.to(device))
        refit = block.refit(
            calibration_input,
            calibration_target,
            "argmax",
            wiring_passes=1,
            target_weight=None,
        )
        block.freeze_hard()
        post_refit = evaluate(
            model,
            validation_loader,
            device,
            block_count=block_index + 1,
            group_temperature=group_temperature,
        )
        diagnostic_symbols = validation_split[0][: args.diagnostic_rows]
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
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if final_pre_hard_test is None:
        raise RuntimeError("missing final pre-hard test metrics")
    continuation_training_seconds = time.perf_counter() - started
    training_seconds = prefix_training_seconds + continuation_training_seconds
    payload = model.hard_payload()
    validate_hard_payload(payload)
    payload_path = args.out_dir / "hard_payload.pt"
    atomic_torch_save(payload_path, payload)
    executor = StrictBitPlaneLUTExecutor(payload)
    strict_train, train_audit_ops = strict_evaluate(
        model,
        executor,
        train_split,
        device,
        batch_size=args.eval_batch_size,
        group_temperature=group_temperature,
    )
    strict_validation, validation_audit_ops = strict_evaluate(
        model,
        executor,
        validation_split,
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
    generated_text, generation_audit_ops = greedy_generate(
        executor,
        test_split[0][0],
        characters,
        args.generation_length,
    )
    structure = model.structural_diagnostics()
    last_pre_refit = block_rows[-1]
    audit_operations = (
        train_audit_ops
        + validation_audit_ops
        + test_audit_ops
        + generation_audit_ops
    )
    method_id = {
        "mixed": "bitplane_lut_sequence_mixed_argmax",
        "sequence_causal": "bitplane_lut_sequence_causal_argmax",
    }[args.candidate_policy]
    result = {
        "method_id": method_id,
        "dataset": "tiny_shakespeare_char",
        "seed": args.seed,
        "context_length": args.context_length,
        "vocab_size": vocab_size,
        "state_bits": state_bits,
        "votes_per_class": args.votes_per_class,
        "vote_bits": vote_bits,
        "candidate_policy": args.candidate_policy,
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
        "train_hard_metrics": strict_train,
        "validation_hard_metrics": strict_validation,
        "test_soft_acc": final_pre_hard_test["soft_acc"],
        "test_hard_acc": strict_test["hard_acc"],
        "test_acc_gap": abs(
            final_pre_hard_test["soft_acc"] - strict_test["hard_acc"]
        ),
        "test_hard_metrics": strict_test,
        "integer_ngram_references": references,
        "train_time_s": training_seconds,
        "continuation_train_time_s": continuation_training_seconds,
        "prefix_train_time_s": prefix_training_seconds,
        "prefix": prefix_metadata,
        "epochs_to_target": sum(
            int(row["epochs_to_90pct_best"]) for row in block_rows
        ),
        "strict_runtime": {
            "compliance": "operator_audited_bool_int",
            "float_tensor_count": 0,
            "audit_operations": audit_operations,
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
        "greedy_generation": generated_text,
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
                "train_hard_acc": strict_train["hard_acc"],
                "audit_operations": audit_operations,
                "payload_sha256": result["hard_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
