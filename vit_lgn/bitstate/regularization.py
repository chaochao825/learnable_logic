from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .gates import HardSTGateLayer


MIND_GAP_ENTROPY_THRESHOLD = 1.8843154907226562
NONTRIVIAL_GATE_IDS = (1, 2, 4, 6, 7, 8, 9, 11, 13, 14)


def gate_distribution_metrics(
    layers: Iterable[HardSTGateLayer],
    *,
    tau: float = 1.0,
) -> tuple[Tensor, Tensor]:
    layers = list(layers)
    if not layers:
        raise ValueError("at least one gate layer is required")
    probabilities = [F.softmax(layer.logits / tau, dim=-1) for layer in layers]
    entropy = torch.cat(
        [-(p * p.clamp_min(1e-8).log()).sum(dim=-1) / math.log(16) for p in probabilities]
    ).mean()
    confidence = torch.cat([p.amax(dim=-1) for p in probabilities]).mean()
    return entropy, confidence


@torch.no_grad()
def entropy_unused_gate_ratio(
    layers: Iterable[HardSTGateLayer],
    *,
    threshold: float = MIND_GAP_ENTROPY_THRESHOLD,
) -> float:
    """Return the Mind-the-Gap high-entropy unused-gate fraction."""

    if threshold <= 0.0:
        raise ValueError(threshold)
    entropy_values = []
    for layer in layers:
        probabilities = F.softmax(layer.logits.detach(), dim=-1)
        entropy_values.append(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        )
    if not entropy_values:
        raise ValueError("at least one gate layer is required")
    entropy = torch.cat(entropy_values)
    return float((entropy > threshold).to(torch.float32).mean())


@torch.no_grad()
def selected_gate_function_metrics(
    layers: Iterable[HardSTGateLayer],
) -> dict[str, float | int | list[int]]:
    """Summarize the deterministic functions selected by gate argmax.

    Entropy-unused measures commitment, not whether a committed gate performs a
    two-input operation. Literal functions are reported separately so identity
    highways cannot be mistaken for learned Boolean computation.
    """

    operations = [layer.hard_ops().cpu() for layer in layers]
    if not operations:
        raise ValueError("at least one gate layer is required")
    operations_tensor = torch.cat(operations)
    histogram = torch.bincount(operations_tensor, minlength=16)
    total = operations_tensor.numel()

    def ratio(indices: tuple[int, ...]) -> float:
        return float(histogram[list(indices)].sum()) / total

    constant_ratio = ratio((0, 15))
    wire_ratio = ratio((3, 5))
    inverted_literal_ratio = ratio((10, 12))
    literal_ratio = wire_ratio + inverted_literal_ratio
    nontrivial_ratio = ratio(NONTRIVIAL_GATE_IDS)
    return {
        "constant_gate_ratio": constant_ratio,
        "wire_gate_ratio": wire_ratio,
        "inverted_literal_gate_ratio": inverted_literal_ratio,
        "literal_gate_ratio": literal_ratio,
        "nontrivial_gate_ratio": nontrivial_ratio,
        "selected_function_count": int(torch.count_nonzero(histogram)),
        "selected_function_coverage": float(torch.count_nonzero(histogram)) / 16,
        "selected_function_histogram": histogram.tolist(),
    }


def gate_entropy_target_penalty(
    layers: Iterable[HardSTGateLayer],
    target: float,
    *,
    tau: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    if not 0.0 <= target <= 1.0:
        raise ValueError(target)
    entropy, confidence = gate_distribution_metrics(layers, tau=tau)
    return (entropy - target).square(), entropy, confidence


def gate_nontrivial_target_penalty(
    layers: Iterable[HardSTGateLayer],
    target: float,
    *,
    tau: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Penalize gates with too little probability on true two-input logic."""

    if not 0.0 <= target <= 1.0:
        raise ValueError(target)
    if tau <= 0.0:
        raise ValueError(tau)
    masses = []
    for layer in layers:
        probabilities = F.softmax(layer.logits / tau, dim=-1)
        masses.append(probabilities[..., list(NONTRIVIAL_GATE_IDS)].sum(dim=-1))
    if not masses:
        raise ValueError("at least one gate layer is required")
    nontrivial_mass = torch.cat(masses)
    penalty = F.relu(target - nontrivial_mass).square().mean()
    return penalty, nontrivial_mass.mean()


def _activity_metrics(state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flattened = state.reshape(-1, state.shape[-1])
    occupancy = flattened.mean(dim=0)
    balance = ((occupancy - 0.5) * 2).square().mean()
    probability = occupancy.clamp(1e-6, 1 - 1e-6)
    entropy = (
        -(probability * probability.log() + (1 - probability) * (1 - probability).log())
        / math.log(2)
    ).mean()
    constant_ratio = ((occupancy <= 1e-6) | (occupancy >= 1 - 1e-6)).to(state.dtype).mean()
    return balance, entropy, constant_ratio


def _diversity_metrics(
    state: Tensor,
    *,
    max_pairs: int,
    maximum_similarity: float,
) -> tuple[Tensor, Tensor, Tensor]:
    width = state.shape[-1]
    pair_count = min(width, max_pairs)
    left = torch.div(
        torch.arange(pair_count, device=state.device) * width,
        pair_count,
        rounding_mode="floor",
    )
    offset = max(1, width // 2 + 1)
    right = (left + offset) % width
    if width > 1:
        right = torch.where(right == left, (right + 1) % width, right)
    x = state[..., left].reshape(-1, pair_count)
    y = state[..., right].reshape(-1, pair_count)
    equal_probability = (x * y + (1 - x) * (1 - y)).mean(dim=0)
    similarity = 0.5 + (equal_probability - 0.5).abs()
    scale = max(1e-6, 1.0 - maximum_similarity)
    penalty = (F.relu(similarity - maximum_similarity) / scale).square().mean()
    duplicate_ratio = (similarity >= 1 - 1e-6).to(state.dtype).mean()
    return penalty, similarity.mean(), duplicate_ratio


def _flip_metrics(
    previous: Tensor,
    current: Tensor,
    *,
    minimum: float,
    maximum: float,
) -> tuple[Tensor, Tensor]:
    flip_rate = (previous + current - 2 * previous * current).mean()
    low_scale = max(minimum, 1e-6)
    high_scale = max(1.0 - maximum, 1e-6)
    penalty = (F.relu(minimum - flip_rate) / low_scale).square()
    penalty = penalty + (F.relu(flip_rate - maximum) / high_scale).square()
    return penalty, flip_rate


def collapse_regularization(
    trace: Sequence[Tensor],
    *,
    balance_weight: float,
    diversity_weight: float,
    flip_weight: float,
    diversity_pairs: int = 128,
    maximum_similarity: float = 0.9,
    minimum_flip: float = 0.02,
    maximum_flip: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    if not trace:
        raise ValueError("trace must not be empty")
    if min(balance_weight, diversity_weight, flip_weight) < 0.0:
        raise ValueError((balance_weight, diversity_weight, flip_weight))
    if diversity_pairs < 1 or not 0.5 <= maximum_similarity < 1.0:
        raise ValueError((diversity_pairs, maximum_similarity))
    if not 0.0 <= minimum_flip <= maximum_flip <= 1.0:
        raise ValueError((minimum_flip, maximum_flip))

    states = [state for state in trace if state.ndim >= 2]
    balance_values = []
    entropy_values = []
    constant_values = []
    diversity_values = []
    similarity_values = []
    duplicate_values = []
    for state in states:
        balance, entropy, constant = _activity_metrics(state)
        diversity, similarity, duplicate = _diversity_metrics(
            state,
            max_pairs=diversity_pairs,
            maximum_similarity=maximum_similarity,
        )
        balance_values.append(balance)
        entropy_values.append(entropy)
        constant_values.append(constant)
        diversity_values.append(diversity)
        similarity_values.append(similarity)
        duplicate_values.append(duplicate)

    flip_values = []
    flip_penalties = []
    for previous, current in zip(trace, trace[1:]):
        if previous.shape != current.shape:
            continue
        penalty, rate = _flip_metrics(
            previous,
            current,
            minimum=minimum_flip,
            maximum=maximum_flip,
        )
        flip_penalties.append(penalty)
        flip_values.append(rate)

    reference = states[0]
    zero = reference.new_zeros(())
    balance_penalty = torch.stack(balance_values).mean() if balance_values else zero
    diversity_penalty = torch.stack(diversity_values).mean() if diversity_values else zero
    flip_penalty = torch.stack(flip_penalties).mean() if flip_penalties else zero
    total = (
        balance_weight * balance_penalty
        + diversity_weight * diversity_penalty
        + flip_weight * flip_penalty
    )
    metrics = {
        "state_balance_penalty": balance_penalty,
        "state_diversity_penalty": diversity_penalty,
        "state_flip_penalty": flip_penalty,
        "state_entropy": torch.stack(entropy_values).mean() if entropy_values else zero,
        "state_constant_ratio": torch.stack(constant_values).mean() if constant_values else zero,
        "state_pair_similarity": torch.stack(similarity_values).mean() if similarity_values else zero,
        "state_duplicate_ratio": torch.stack(duplicate_values).mean() if duplicate_values else zero,
        "state_flip_rate": torch.stack(flip_values).mean() if flip_values else zero,
    }
    return total, metrics
