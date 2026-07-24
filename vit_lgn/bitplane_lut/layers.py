from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    return soft + (hard - soft).detach()


def uint8_to_bitplanes(symbols: torch.Tensor) -> torch.Tensor:
    """Expand unsigned A8 symbols into little-endian Boolean bit-planes."""

    if symbols.dtype != torch.uint8:
        raise TypeError("A8 symbols must use torch.uint8")
    shifts = torch.arange(8, dtype=torch.uint8, device=symbols.device)
    return torch.bitwise_and(
        torch.bitwise_right_shift(symbols.unsqueeze(-1), shifts), 1
    ).bool().flatten(-2)


def bitplanes_to_uint8(bits: torch.Tensor) -> torch.Tensor:
    """Pack little-endian Boolean bit-planes into unsigned A8 symbols."""

    if bits.dtype != torch.bool:
        raise TypeError("bit-planes must use torch.bool")
    if bits.shape[-1] % 8:
        raise ValueError("bit-plane width must be divisible by eight")
    shifts = torch.arange(8, dtype=torch.int16, device=bits.device)
    weights = torch.bitwise_left_shift(torch.ones_like(shifts), shifts)
    values = (
        bits.reshape(*bits.shape[:-1], -1, 8).to(torch.int16)
        * weights
    ).sum(dim=-1, dtype=torch.int16)
    return values.to(torch.uint8)


def _patterns(arity: int, device: torch.device | None = None) -> torch.Tensor:
    addresses = torch.arange(1 << arity, dtype=torch.int64, device=device)
    shifts = torch.arange(arity, dtype=torch.int64, device=device)
    return torch.bitwise_and(
        torch.bitwise_right_shift(addresses.unsqueeze(-1), shifts), 1
    ).bool()


def deterministic_candidate_indices(
    input_bits: int,
    output_bits: int,
    arity: int,
    candidate_count: int,
    seed: int,
    identity_offset: int = 0,
) -> torch.Tensor:
    """Build local, same-plane, and globally hashed wiring candidates.

    Slot zero starts on the identity route.  Other slots start on distinct
    nearby routes while retaining identity in their pool.  The remaining
    candidates mix nearby bit-planes, nearby A8 symbols, and deterministic
    global sources.
    Candidate construction is fixed support logic and is not exported after a
    source has been selected.
    """

    if input_bits < 1 or output_bits < 1:
        raise ValueError("input_bits and output_bits must be positive")
    if arity not in {2, 3, 4}:
        raise ValueError("arity must be one of 2, 3, or 4")
    if not 1 <= candidate_count <= input_bits:
        raise ValueError("candidate_count must be in [1,input_bits]")

    result = torch.empty(
        output_bits, arity, candidate_count, dtype=torch.int64
    )
    symbol_count = max(1, input_bits // 8)
    for output in range(output_bits):
        identity = (int(identity_offset) + output) % input_bits
        symbol = (identity // 8) % symbol_count
        plane = identity % 8
        for slot in range(arity):
            local_plane = symbol * 8 + ((plane + slot + 1) % 8)
            next_symbol = ((symbol + slot + 1) % symbol_count) * 8 + plane
            previous_symbol = ((symbol - slot - 1) % symbol_count) * 8 + plane
            ordered = (
                [identity, local_plane, next_symbol, previous_symbol]
                if slot == 0
                else [local_plane, next_symbol, previous_symbol, identity]
            )
            unique: list[int] = []
            for value in ordered:
                value %= input_bits
                if value not in unique:
                    unique.append(value)
            probe = 0
            while len(unique) < candidate_count:
                # Integer-only deterministic hash; no process-randomized hash().
                value = (
                    (output + 1) * 1_315_423_911
                    + (slot + 1) * 2_654_435_761
                    + (probe + 1) * 97_531
                    + int(seed) * 433_494_437
                ) % input_bits
                if value not in unique:
                    unique.append(value)
                probe += 1
            result[output, slot] = torch.tensor(
                unique[:candidate_count], dtype=torch.int64
            )
    return result


def spatial_candidate_indices(
    state_bits: int,
    preserved_bits: int,
    output_bits: int,
    arity: int,
    candidate_count: int,
    num_classes: int,
    image_shape: tuple[int, int, int],
    seed: int,
) -> torch.Tensor:
    """Build fixed image-local, class-state, and global source candidates."""

    height, width, channels = (int(value) for value in image_shape)
    if min(height, width, channels) < 1:
        raise ValueError("image dimensions must be positive")
    if preserved_bits != height * width * channels * 8:
        raise ValueError("preserved bit width does not match the image shape")
    if state_bits != preserved_bits + output_bits:
        raise ValueError("spatial routing expects raw planes plus vote planes")
    if output_bits % num_classes:
        raise ValueError("output bits must divide evenly across classes")
    if arity not in {2, 3, 4}:
        raise ValueError("arity must be one of 2, 3, or 4")
    if not 1 <= candidate_count <= state_bits:
        raise ValueError("invalid candidate_count")

    result = torch.empty(
        output_bits, arity, candidate_count, dtype=torch.int64
    )
    pixel_count = height * width
    votes_per_class = output_bits // num_classes
    offsets = (
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    for output in range(output_bits):
        class_index = output % num_classes
        vote_index = output // num_classes
        identity = preserved_bits + output
        anchor = (
            vote_index * 1_315_423_911
            + class_index * 2_654_435_761
            + int(seed) * 433_494_437
        ) % pixel_count
        anchor_y, anchor_x = divmod(anchor, width)
        for slot in range(arity):
            ordered = [identity] if slot == 0 else []

            # Previous vote planes from the same class let deeper blocks grow
            # Boolean support without introducing a numeric mixing matrix.
            for delta in (slot + 1, -(slot + 1), slot + 3):
                neighbor_vote = (vote_index + delta) % votes_per_class
                ordered.append(
                    preserved_bits + neighbor_vote * num_classes + class_index
                )

            # Tile deterministic 3x3 pixel neighborhoods, channels, and bit
            # planes across class votes. Every entry is a raw Boolean source.
            for probe in range(max(candidate_count // 2, len(offsets))):
                dy, dx = offsets[(probe + slot) % len(offsets)]
                y = (anchor_y + dy) % height
                x = (anchor_x + dx) % width
                channel = (class_index + vote_index + slot + probe) % channels
                plane = (vote_index + 2 * slot + probe) % 8
                symbol = (y * width + x) * channels + channel
                ordered.append(symbol * 8 + plane)

            ordered.append(identity)
            unique: list[int] = []
            for value in ordered:
                value %= state_bits
                if value not in unique:
                    unique.append(value)
                if len(unique) == candidate_count:
                    break
            probe = 0
            while len(unique) < candidate_count:
                value = (
                    (output + 1) * 2_246_822_519
                    + (slot + 1) * 3_266_489_917
                    + (probe + 1) * 668_265_263
                    + int(seed) * 374_761_393
                ) % state_bits
                if value not in unique:
                    unique.append(value)
                probe += 1
            result[output, slot] = torch.tensor(unique, dtype=torch.int64)
    return result


@dataclass(frozen=True)
class RefitMetrics:
    bit_error: float
    address_coverage: float
    changed_truth_ratio: float
    changed_wiring_ratio: float


class LearnableLUTLayer(nn.Module):
    """Hard-forward k-input LUT bank with learned discrete candidate wiring."""

    def __init__(
        self,
        input_bits: int,
        output_bits: int,
        *,
        arity: int = 4,
        candidate_count: int = 16,
        seed: int = 0,
        identity_offset: int = 0,
        wiring_temperature: float = 1.0,
        truth_temperature: float = 1.0,
        candidate_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if arity not in {2, 3, 4}:
            raise ValueError("arity must be one of 2, 3, or 4")
        if input_bits < 1 or output_bits < 1:
            raise ValueError("layer widths must be positive")
        if not 1 <= candidate_count <= input_bits:
            raise ValueError("invalid candidate_count")
        if wiring_temperature <= 0 or truth_temperature <= 0:
            raise ValueError("temperatures must be positive")
        self.input_bits = int(input_bits)
        self.output_bits = int(output_bits)
        self.arity = int(arity)
        self.candidate_count = int(candidate_count)
        self.table_size = 1 << self.arity
        self.wiring_temperature = float(wiring_temperature)
        self.truth_temperature = float(truth_temperature)

        if candidate_indices is None:
            candidates = deterministic_candidate_indices(
                self.input_bits,
                self.output_bits,
                self.arity,
                self.candidate_count,
                seed,
                identity_offset,
            )
        else:
            candidates = candidate_indices.detach().cpu().to(torch.int64).clone()
            expected = (self.output_bits, self.arity, self.candidate_count)
            if tuple(candidates.shape) != expected:
                raise ValueError("fixed candidate tensor shape mismatch")
            if candidates.numel() and (
                int(candidates.min().item()) < 0
                or int(candidates.max().item()) >= self.input_bits
            ):
                raise ValueError("fixed candidate source is out of range")
            ordered, _ = candidates.sort(dim=-1)
            if self.candidate_count > 1 and bool(
                (ordered[..., 1:] == ordered[..., :-1]).any()
            ):
                raise ValueError("fixed candidate pools must contain unique sources")
        self.register_buffer("candidate_indices", candidates, persistent=True)
        self.register_buffer("truth_patterns", _patterns(self.arity), persistent=False)
        self.register_buffer(
            "frozen_sources",
            torch.zeros(self.output_bits, self.arity, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "frozen_truth",
            torch.zeros(self.output_bits, self.table_size, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "frozen_flag", torch.tensor(False, dtype=torch.bool), persistent=True
        )
        self.wiring_logits = nn.Parameter(
            torch.empty(self.output_bits, self.arity, self.candidate_count)
        )
        self.truth_logits = nn.Parameter(
            torch.empty(self.output_bits, self.table_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.wiring_logits, mean=0.0, std=0.02)
        with torch.no_grad():
            self.wiring_logits[..., 0].add_(0.5)
            projection = self.truth_patterns[:, 0].to(self.truth_logits.dtype)
            # Identity is the exact initial hard function, but the small margin
            # lets task gradients change individual truth entries quickly.
            self.truth_logits.copy_(
                projection.mul(2.0).sub(1.0).mul(0.25).expand_as(
                    self.truth_logits
                )
            )
            self.frozen_flag.fill_(False)

    @property
    def is_frozen(self) -> bool:
        return bool(self.frozen_flag.item())

    def selected_sources(self) -> torch.Tensor:
        if self.is_frozen:
            return self.frozen_sources
        choice = self.wiring_logits.argmax(dim=-1, keepdim=True)
        return self.candidate_indices.gather(-1, choice).squeeze(-1)

    def hard_truth_table(self) -> torch.Tensor:
        if self.is_frozen:
            return self.frozen_truth
        return self.truth_logits.ge(0)

    def _selected_inputs(
        self, bits: torch.Tensor, mode: str
    ) -> torch.Tensor:
        candidates = bits[:, self.candidate_indices].to(self.wiring_logits.dtype)
        soft = torch.softmax(
            self.wiring_logits / self.wiring_temperature, dim=-1
        )
        if mode == "soft":
            selector = soft
        else:
            hard = F.one_hot(
                self.wiring_logits.argmax(dim=-1),
                num_classes=self.candidate_count,
            ).to(soft.dtype)
            selector = _ste(hard, soft)
        return (candidates * selector.unsqueeze(0)).sum(dim=-1)

    def _relaxed_forward(self, bits: torch.Tensor, mode: str) -> torch.Tensor:
        selected = self._selected_inputs(bits, mode)
        probability = torch.sigmoid(
            self.truth_logits / self.truth_temperature
        )
        if mode == "soft":
            table = probability
        else:
            table = _ste(self.truth_logits.ge(0).to(probability.dtype), probability)
        pattern = self.truth_patterns.to(selected.device)
        factors = torch.where(
            pattern.view(1, 1, self.table_size, self.arity),
            selected.unsqueeze(-2),
            1.0 - selected.unsqueeze(-2),
        )
        address_weight = factors.prod(dim=-1)
        return (address_weight * table.unsqueeze(0)).sum(dim=-1)

    def hard_forward(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[-1] != self.input_bits:
            raise ValueError("hard LUT input shape mismatch")
        if bits.dtype != torch.bool:
            if not bits.dtype.is_floating_point:
                raise TypeError("hard LUT input must be Boolean or a binary carrier")
            if bool(((bits != 0) & (bits != 1)).any()):
                raise ValueError("hard LUT carrier contains non-binary values")
            hard_bits = bits.bool()
        else:
            hard_bits = bits
        selected = hard_bits[:, self.selected_sources()]
        address = torch.zeros(
            selected.shape[:-1], dtype=torch.int64, device=selected.device
        )
        for slot in range(self.arity):
            address = torch.bitwise_or(
                address,
                torch.bitwise_left_shift(
                    selected[..., slot].to(torch.int64), slot
                ),
            )
        table = self.hard_truth_table()
        gate = torch.arange(self.output_bits, device=bits.device).view(1, -1)
        output = table[gate, address]
        return output if bits.dtype == torch.bool else output.to(bits.dtype)

    def forward(self, bits: torch.Tensor, mode: str = "hard_st") -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[-1] != self.input_bits:
            raise ValueError(
                f"expected [batch,{self.input_bits}] input, got {tuple(bits.shape)}"
            )
        if mode not in {"hard", "hard_st", "soft"}:
            raise ValueError("mode must be hard, hard_st, or soft")
        if mode == "hard" or self.is_frozen:
            return self.hard_forward(bits)
        if not bits.dtype.is_floating_point:
            bits = bits.to(self.wiring_logits.dtype)
        return self._relaxed_forward(bits, mode)

    @staticmethod
    def _fit_table_for_addresses(
        address: torch.Tensor,
        target: torch.Tensor,
        table_size: int,
        fallback: torch.Tensor,
        target_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if address.shape != target.shape:
            raise ValueError("address and target shapes must match")
        # Layout [output,sample] makes scatter over local LUT addresses direct.
        index = address.transpose(0, 1).to(torch.int64)
        target_os = target.transpose(0, 1).to(torch.int64)
        if target_weight is None:
            weight_os = torch.ones_like(target_os)
        else:
            if target_weight.shape != target.shape:
                raise ValueError("target_weight shape must match target")
            weight_os = target_weight.transpose(0, 1).to(torch.int64)
            if weight_os.numel() and int(weight_os.min().item()) < 1:
                raise ValueError("target weights must be positive integers")
        count = torch.zeros(
            target_os.shape[0], table_size,
            dtype=torch.int64, device=target.device,
        )
        ones = torch.zeros_like(count)
        visits = torch.zeros_like(count)
        count.scatter_add_(1, index, weight_os)
        ones.scatter_add_(1, index, target_os * weight_os)
        visits.scatter_add_(1, index, torch.ones_like(target_os))
        fitted = ones * 2 >= count
        fitted = torch.where(count > 0, fitted, fallback.to(fitted.device))
        errors = torch.minimum(ones, count - ones).sum(dim=1)
        coverage = (visits > 0).sum(dim=1)
        return fitted, errors, coverage

    @torch.no_grad()
    def refit_truth_tables(
        self,
        input_bits: torch.Tensor,
        target_bits: torch.Tensor,
        target_weight: torch.Tensor | None = None,
    ) -> RefitMetrics:
        input_bool = input_bits.bool()
        target_bool = target_bits.bool()
        if input_bool.ndim != 2 or input_bool.shape[-1] != self.input_bits:
            raise ValueError("refit input shape mismatch")
        if target_bool.shape != (input_bool.shape[0], self.output_bits):
            raise ValueError("refit target shape mismatch")
        old_truth = self.hard_truth_table().detach().clone()
        sources = self.selected_sources()
        selected = input_bool[:, sources]
        address = torch.zeros(
            selected.shape[:-1], dtype=torch.int64, device=selected.device
        )
        for slot in range(self.arity):
            address |= selected[..., slot].to(torch.int64) << slot
        fitted, errors, coverage = self._fit_table_for_addresses(
            address,
            target_bool,
            self.table_size,
            old_truth,
            target_weight,
        )
        self.truth_logits.copy_(
            fitted.to(self.truth_logits.dtype).mul(2.0).sub(1.0).mul(4.0)
        )
        prediction = self.hard_forward(input_bool)
        return RefitMetrics(
            bit_error=float((prediction != target_bool).float().mean().item()),
            address_coverage=float(
                coverage.sum().item() / (self.output_bits * self.table_size)
            ),
            changed_truth_ratio=float((fitted != old_truth).float().mean().item()),
            changed_wiring_ratio=0.0,
        )

    @torch.no_grad()
    def greedy_refit_wiring_and_truth(
        self,
        input_bits: torch.Tensor,
        target_bits: torch.Tensor,
        *,
        passes: int = 1,
        target_weight: torch.Tensor | None = None,
    ) -> RefitMetrics:
        """Coordinate-search candidate wires, fitting the optimal local table.

        Each coordinate compares all registered candidate sources under the
        empirical bit target.  It never introduces a numeric weight or a source
        outside the pre-registered candidate pool.
        """

        if passes < 1:
            raise ValueError("passes must be positive")
        input_bool = input_bits.bool()
        target_bool = target_bits.bool()
        if input_bool.ndim != 2 or input_bool.shape[-1] != self.input_bits:
            raise ValueError("refit input shape mismatch")
        if target_bool.shape != (input_bool.shape[0], self.output_bits):
            raise ValueError("refit target shape mismatch")
        original_sources = self.selected_sources().detach().clone()
        original_truth = self.hard_truth_table().detach().clone()
        chosen = self.wiring_logits.argmax(dim=-1)
        sources = self.candidate_indices.gather(
            -1, chosen.unsqueeze(-1)
        ).squeeze(-1)

        for _ in range(passes):
            selected = input_bool[:, sources]
            for slot in range(self.arity):
                best_choice = chosen[:, slot].clone()
                best_error = torch.full(
                    (self.output_bits,),
                    input_bool.shape[0] + 1,
                    dtype=torch.int64,
                    device=input_bool.device,
                )
                for candidate in range(self.candidate_count):
                    proposal = selected.clone()
                    candidate_source = self.candidate_indices[:, slot, candidate]
                    proposal[..., slot] = input_bool[:, candidate_source]
                    address = torch.zeros(
                        proposal.shape[:-1],
                        dtype=torch.int64,
                        device=input_bool.device,
                    )
                    for bit in range(self.arity):
                        address |= proposal[..., bit].to(torch.int64) << bit
                    _, error, _ = self._fit_table_for_addresses(
                        address,
                        target_bool,
                        self.table_size,
                        original_truth,
                        target_weight,
                    )
                    improve = error < best_error
                    best_error = torch.where(improve, error, best_error)
                    best_choice = torch.where(
                        improve,
                        torch.full_like(best_choice, candidate),
                        best_choice,
                    )
                chosen[:, slot] = best_choice
                sources[:, slot] = self.candidate_indices[
                    torch.arange(self.output_bits, device=input_bool.device),
                    slot,
                    best_choice,
                ]
                selected[..., slot] = input_bool[:, sources[:, slot]]

        self.wiring_logits.fill_(-4.0)
        self.wiring_logits.scatter_(-1, chosen.unsqueeze(-1), 4.0)
        truth_metrics = self.refit_truth_tables(
            input_bool, target_bool, target_weight
        )
        return RefitMetrics(
            bit_error=truth_metrics.bit_error,
            address_coverage=truth_metrics.address_coverage,
            changed_truth_ratio=float(
                (self.hard_truth_table() != original_truth).float().mean().item()
            ),
            changed_wiring_ratio=float((sources != original_sources).float().mean().item()),
        )

    @torch.no_grad()
    def freeze_hard(self) -> None:
        self.frozen_sources.copy_(self.selected_sources())
        self.frozen_truth.copy_(self.hard_truth_table())
        self.frozen_flag.fill_(True)
        self.wiring_logits.requires_grad_(False)
        self.truth_logits.requires_grad_(False)

    def soft_wiring_usage(self) -> torch.Tensor:
        probability = torch.softmax(
            self.wiring_logits / self.wiring_temperature, dim=-1
        )
        usage = probability.new_zeros(self.input_bits)
        usage.scatter_add_(
            0,
            self.candidate_indices.flatten(),
            probability.flatten(),
        )
        return usage

    def wiring_constraint_loss(self, fanout_cap: float = 8.0) -> torch.Tensor:
        usage = self.soft_wiring_usage()
        fanout = torch.relu(usage - float(fanout_cap)).square().mean()
        unused = torch.exp(-usage).mean()
        return fanout + unused

    def truth_complexity_loss(self) -> torch.Tensor:
        probability = torch.sigmoid(
            self.truth_logits / self.truth_temperature
        )
        transitions = []
        address = torch.arange(self.table_size, device=probability.device)
        for slot in range(self.arity):
            transitions.append(
                (probability - probability[:, address ^ (1 << slot)]).abs().mean()
            )
        return torch.stack(transitions).mean()

    def hard_diagnostics(self) -> dict[str, float | int]:
        table = self.hard_truth_table().detach().cpu()
        patterns = _patterns(self.arity)
        constant = (table == table[:, :1]).all(dim=1)
        literal = torch.zeros(self.output_bits, dtype=torch.bool)
        for slot in range(self.arity):
            projection = patterns[:, slot]
            literal |= (table == projection).all(dim=1)
            literal |= (table == ~projection).all(dim=1)
        literal &= ~constant
        sources = self.selected_sources().detach().cpu().flatten()
        fanout = torch.bincount(sources, minlength=self.input_bits)
        source_bits = max(1, math.ceil(math.log2(self.input_bits)))
        return {
            "gate_count": self.output_bits,
            "constant_gate_ratio": float(constant.float().mean().item()),
            "literal_gate_ratio": float(literal.float().mean().item()),
            "nontrivial_gate_ratio": float((~(constant | literal)).float().mean().item()),
            "fanout_max": int(fanout.max().item()),
            "selected_source_ratio": float((fanout > 0).float().mean().item()),
            "logical_payload_bits": int(
                self.output_bits * (self.arity * source_bits + self.table_size)
            ),
        }

    @torch.no_grad()
    def hard_payload(self) -> dict[str, object]:
        return {
            "kind": "boolean_lut_layer",
            "input_bits": self.input_bits,
            "output_bits": self.output_bits,
            "arity": self.arity,
            "source_indices": self.selected_sources().detach().cpu().to(torch.int32),
            "truth_table": self.hard_truth_table().detach().cpu().to(torch.uint8),
        }


__all__ = [
    "LearnableLUTLayer",
    "RefitMetrics",
    "bitplanes_to_uint8",
    "deterministic_candidate_indices",
    "spatial_candidate_indices",
    "uint8_to_bitplanes",
]
