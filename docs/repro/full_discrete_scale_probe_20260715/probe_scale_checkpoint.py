"""Bounded frozen-checkpoint probe for the e192/e384 scale ablation.

Run this script against a checkout of the exact scale-ablation source commit.
It does not train or modify a checkpoint.  It evaluates the first attention
block on a deterministic prefix of the validation split and prints one JSON
object containing the probe protocol and two run summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED_COMMIT = "47e0250cdbeb327925891ec6cf6bd436eca35dbb"
DEFAULT_RUNS = (
    "fdscale_d6e192_control_seed42",
    "fdscale_d6e384_seed42",
)
EXPECTED_RUNS = {
    "fdscale_d6e192_control_seed42": {
        "checkpoint_sha256": "a16ddb2505d1d394653a372bc0f28895c1402cd5d24c51582974836af16095c2",
        "protocol_sha256": "b0389ebb920bea1125ccebfa87179eeea18b7d1c7b0fbc3433d5b82f8c1b22a9",
    },
    "fdscale_d6e384_seed42": {
        "checkpoint_sha256": "ca0120b860d510e1c2d81772e9b7add44f67ae594822852589533bf9bf048a8e",
        "protocol_sha256": "c5ac3b723126f00396c95909d96b9b20f7847e2991b02310ba9bb23fb89816ef",
    },
}
CIFAR10_PAYLOAD_SHA256 = {
    "batches.meta": "f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb",
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "test_batch": "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_batch_hash(images, labels) -> str:
    digest = hashlib.sha256()
    for name, tensor in (("images", images), ("labels", labels)):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", nargs=2, default=list(DEFAULT_RUNS))
    parser.add_argument("--probe-images", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def require_frozen_source(repo_root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"scale probe requires source commit {EXPECTED_COMMIT}, got {commit}"
        )
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    )
    if status.strip():
        raise RuntimeError("scale probe requires a completely clean source checkout")
    return commit


def verify_dataset(data_root: Path) -> dict[str, object]:
    payload_root = data_root / "cifar-10-batches-py"
    observed = {
        name: sha256(payload_root / name) for name in CIFAR10_PAYLOAD_SHA256
    }
    if observed != CIFAR10_PAYLOAD_SHA256:
        raise RuntimeError("CIFAR-10 extracted payload hashes do not match the frozen set")
    archive = data_root / "cifar-10-python.tar.gz"
    return {
        "archive_sha256": sha256(archive),
        "extracted_files_sha256": observed,
    }


def verify_protocol(
    *, run: str, protocol: dict, repo_root: Path, dataset: dict[str, object]
) -> None:
    if run not in EXPECTED_RUNS:
        raise RuntimeError(f"no frozen manifest exists for run {run}")
    expected = EXPECTED_RUNS[run]
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    if canonical_hash(payload) != protocol.get("sha256"):
        raise RuntimeError(f"{run} protocol digest does not match its payload")
    if protocol.get("sha256") != expected["protocol_sha256"]:
        raise RuntimeError(f"{run} protocol is not the frozen protocol")
    if protocol.get("dataset_archive_sha256") != dataset["archive_sha256"]:
        raise RuntimeError(f"{run} dataset archive does not match its protocol")
    source_root = repo_root / "vit_lgn" / "full_discrete"
    for name, expected_hash in protocol.get("sources", {}).items():
        if sha256(source_root / name) != expected_hash:
            raise RuntimeError(f"{run} source hash mismatch: {name}")


def build_model(model_class, protocol: dict):
    args = protocol["args"]
    return model_class(
        dim=args["dim"],
        depth=args["depth"],
        heads=args["heads"],
        topk=args["topk"],
        mlp_ratio=args["mlp_ratio"],
        weight_bits=args["weight_magnitude_bits"],
        activation_bits=args["activation_bits"],
        qk_lanes=args["qk_lanes"],
        learned_gap=args.get("learned_gap", False),
        group_lut_groups=args.get("group_lut_groups", 0),
        local_layers=args.get("local_layers", 0),
        logic_expert_width=args.get("logic_expert_width", 0),
        logic_expert_count=args.get("logic_expert_count", 1),
        state_control=args.get("state_control", "none"),
        state_expert_width=args.get("state_expert_width", 0),
    )


def weight_statistics(model, shift_add_linear) -> dict[str, float | int]:
    count = zero = saturated = 0
    error_square = reference_square = 0.0
    exponents: list[float] = []
    for module in model.modules():
        if not isinstance(module, shift_add_linear):
            continue
        code, scale = module.integer_weight_and_scale()
        quantized = code * scale
        count += code.numel()
        zero += int((code == 0).sum())
        saturated += int((code.abs() == module.qmax).sum())
        error_square += float((quantized - module.weight).square().sum())
        reference_square += float(module.weight.square().sum())
        exponents.extend(scale.log2().round().reshape(-1).tolist())
    return {
        "trainable_weight_scalars": count,
        "weight_zero_pct": 100.0 * zero / count,
        "weight_saturation_pct": 100.0 * saturated / count,
        "weight_relative_rmse": (error_square / reference_square) ** 0.5,
        "row_scale_exponent_mean": sum(exponents) / len(exponents),
        "row_scale_exponent_min": min(exponents),
        "row_scale_exponent_max": max(exponents),
    }


def probe_run(
    *,
    run: str,
    images,
    runs_root: Path,
    repo_root: Path,
    dataset: dict[str, object],
    device,
    torch,
    model_class,
    shift_add_linear,
) -> dict[str, object]:
    run_root = runs_root / run
    protocol_path = run_root / "protocol.json"
    checkpoint_path = run_root / "checkpoint.pt"
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    verify_protocol(
        run=run, protocol=protocol, repo_root=repo_root, dataset=dataset
    )
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_RUNS[run]["checkpoint_sha256"]:
        raise RuntimeError(f"{run} checkpoint hash is not the frozen checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["step"]) != 50_000:
        raise RuntimeError(f"{run} checkpoint is at step {checkpoint['step']}, not 50000")
    if checkpoint.get("protocol_sha256") != protocol["sha256"]:
        raise RuntimeError(f"{run} checkpoint and protocol do not match")

    model = build_model(model_class, protocol)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    images = images.to(device)

    with torch.inference_mode():
        tokens = model.patch_embed(images)
        cls = model.parameter_quantizer(model.cls_token).expand(
            images.shape[0], -1, -1
        )
        position = model.parameter_quantizer(model.position)
        tokens = model.token_quantizer(
            torch.cat((cls, tokens), dim=1) + position
        )
        attention = model.blocks[0].attn
        attention_input = model.blocks[0].norm1(tokens)
        input_code, _ = attention.qkv.input_quantizer.integer_code_and_scale(
            attention_input
        )

        qkv = attention.qkv(attention_input).reshape(
            images.shape[0],
            attention_input.shape[1],
            3,
            attention.heads,
            attention.head_dim,
        )
        query, key, _ = qkv.permute(2, 0, 3, 1, 4)
        query_bits, _ = attention._threshold_bits(query)
        key_bits, _ = attention._threshold_bits(key)
        scores = attention.xnor_popcount(query_bits, key_bits)

        # Commit 47e0250 used framework topk without the later stable tie ABI.
        indices = scores.topk(
            min(attention.topk, attention_input.shape[1]), dim=-1
        ).indices
        selected = torch.gather(scores.to(torch.int64), -1, indices)
        best = selected.max(dim=-1, keepdim=True).values
        gaps = best - selected
        buckets = torch.bitwise_right_shift(
            gaps, attention.gap_shift
        ).clamp(0, attention.max_gap_bucket)
        histogram = torch.bincount(buckets.reshape(-1), minlength=4)

        sorted_scores = torch.sort(scores, dim=-1, descending=True).values
        kth = sorted_scores[..., attention.topk - 1]
        boundary_tie_count = (scores == kth.unsqueeze(-1)).sum(-1)

        result: dict[str, object] = {
            "run": run,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_sha256": checkpoint_hash,
            "protocol_sha256": protocol["sha256"],
            "dim": protocol["args"]["dim"],
            "heads": protocol["args"]["heads"],
            "head_dim": attention.head_dim,
            "xnor_width": attention.head_dim * attention.qk_lanes,
            "qkv_input_saturation_pct": 100.0
            * float((input_code.abs() == 127).float().mean()),
            "q_bit_one_pct": 100.0 * float(query_bits.float().mean()),
            "k_bit_one_pct": 100.0 * float(key_bits.float().mean()),
            "selected_gap_mean": float(gaps.float().mean()),
            "selected_gap_median": float(gaps.float().median()),
            "selected_gap_max": int(gaps.max()),
            "weight_8_4_2_1_pct": [
                100.0 * int(value) / int(histogram.sum()) for value in histogram
            ],
            "topk_boundary_tie_count_mean": float(
                boundary_tie_count.float().mean()
            ),
            "topk_boundary_tie_gt1_pct": 100.0
            * float((boundary_tie_count > 1).float().mean()),
        }
        result.update(weight_statistics(model, shift_add_linear))
        return result


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.runs_root = args.runs_root.resolve()
    args.data_root = args.data_root.resolve()
    if args.probe_images < 1 or args.probe_images > 5000:
        raise ValueError("probe-images must be in [1,5000]")
    source_commit = require_frozen_source(args.repo_root)

    sys.path.insert(0, str(args.repo_root))
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import v2
    from vit_lgn.full_discrete.enhanced_model import EnhancedFullDiscreteViT
    from vit_lgn.full_discrete.shiftadd import ShiftAddLinear

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dataset = verify_dataset(args.data_root)
    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    full = CIFAR10(
        args.data_root, train=True, download=False, transform=transform
    )
    order = torch.randperm(
        len(full), generator=torch.Generator().manual_seed(args.split_seed)
    ).tolist()
    sample_indices = order[: args.probe_images]
    images, labels = next(
        iter(
            DataLoader(
                Subset(full, sample_indices),
                batch_size=args.probe_images,
                num_workers=0,
            )
        )
    )

    payload = {
        "probe_script_sha256": sha256(Path(__file__).resolve()),
        "source_commit": source_commit,
        "split_seed": args.split_seed,
        "sample_indices": sample_indices,
        "sample_indices_sha256": canonical_hash(sample_indices),
        "input_batch_sha256": tensor_batch_hash(images, labels),
        "dataset": dataset,
        "probe_images": args.probe_images,
        "attention_block": 0,
        "device": str(device),
        "runs": [
            probe_run(
                run=run,
                images=images,
                runs_root=args.runs_root,
                repo_root=args.repo_root,
                dataset=dataset,
                device=device,
                torch=torch,
                model_class=EnhancedFullDiscreteViT,
                shift_add_linear=ShiftAddLinear,
            )
            for run in args.runs
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
