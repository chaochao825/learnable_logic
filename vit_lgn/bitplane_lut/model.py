from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn

from .layers import (
    LearnableLUTLayer,
    RefitMetrics,
    bitplanes_to_uint8,
    uint8_to_bitplanes,
)


def _entropy_from_probability(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-7, 1.0 - 1e-7)
    return -probability * torch.log2(probability) - (
        1.0 - probability
    ) * torch.log2(1.0 - probability)


def boolean_state_diagnostics(
    state: torch.Tensor,
    *,
    previous: torch.Tensor | None = None,
) -> dict[str, object]:
    """Measure marginal and joint state use; entropy alone is not a verdict."""

    if state.dtype != torch.bool or state.ndim != 2:
        raise TypeError("state diagnostics require a [sample,bits] Boolean tensor")
    if state.shape[1] % 8:
        raise ValueError("state width must contain complete A8 symbols")
    state_cpu = state.detach().cpu()
    probability = state_cpu.float().mean(dim=0)
    entropy = _entropy_from_probability(probability)
    inactive = (probability == 0) | (probability == 1)
    plane_entropy = entropy.reshape(-1, 8).mean(dim=0)
    plane_active = (~inactive).reshape(-1, 8).float().mean(dim=0)
    packed = bitplanes_to_uint8(state_cpu)
    symbol_entropies = []
    symbol_unique = []
    for column in range(packed.shape[1]):
        counts = torch.bincount(packed[:, column].to(torch.int64), minlength=256)
        observed = counts[counts > 0].float()
        distribution = observed / observed.sum().clamp_min(1)
        symbol_entropies.append(float((-(distribution * distribution.log2()).sum()).item()))
        symbol_unique.append(int(observed.numel()))
    unique_states = int(torch.unique(state_cpu.to(torch.uint8), dim=0).shape[0])
    result: dict[str, object] = {
        "bit_entropy_mean": float(entropy.mean().item()),
        "bit_entropy_min": float(entropy.min().item()),
        "inactive_bit_ratio": float(inactive.float().mean().item()),
        "per_plane_entropy": [float(value) for value in plane_entropy.tolist()],
        "per_plane_active_ratio": [float(value) for value in plane_active.tolist()],
        "a8_symbol_entropy_mean": float(sum(symbol_entropies) / len(symbol_entropies)),
        "a8_symbol_unique_mean": float(sum(symbol_unique) / len(symbol_unique)),
        "joint_unique_states": unique_states,
        "joint_unique_state_ratio": float(unique_states / max(1, state.shape[0])),
    }
    if previous is not None:
        if previous.dtype != torch.bool or previous.shape != state.shape:
            raise ValueError("previous state must be a matching Boolean tensor")
        result["input_output_bit_flip_ratio"] = float(
            (state_cpu != previous.detach().cpu()).float().mean().item()
        )
    return result


class BitPlaneLUTBlock(nn.Module):
    """Preserve raw A8 planes and update only the learned vote planes."""

    def __init__(
        self,
        state_bits: int,
        preserved_bits: int,
        *,
        layers: int = 1,
        arity: int = 4,
        candidate_count: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if state_bits < 8 or state_bits % 8:
            raise ValueError("state_bits must be a positive multiple of eight")
        if not 0 < preserved_bits < state_bits:
            raise ValueError("preserved_bits must be inside the state width")
        if preserved_bits % 8 or (state_bits - preserved_bits) % 8:
            raise ValueError("preserved and learned state regions must be A8 aligned")
        if layers < 1:
            raise ValueError("layers must be positive")
        self.state_bits = int(state_bits)
        self.preserved_bits = int(preserved_bits)
        self.vote_bits = self.state_bits - self.preserved_bits
        self.layers = nn.ModuleList(
            [
                LearnableLUTLayer(
                    self.state_bits,
                    self.vote_bits,
                    arity=arity,
                    candidate_count=candidate_count,
                    seed=seed + index * 10_007,
                    identity_offset=self.preserved_bits,
                )
                for index in range(layers)
            ]
        )

    @property
    def is_frozen(self) -> bool:
        return all(layer.is_frozen for layer in self.layers)

    def forward(self, state: torch.Tensor, mode: str = "hard_st") -> torch.Tensor:
        for layer in self.layers:
            update = layer(state, mode=mode)
            state = torch.cat((state[:, :self.preserved_bits], update), dim=-1)
        return state

    def hard_forward(self, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            update = layer.hard_forward(state)
            state = torch.cat((state[:, :self.preserved_bits], update), dim=-1)
        return state

    @torch.no_grad()
    def refit(
        self,
        input_state: torch.Tensor,
        target_state: torch.Tensor,
        mode: str,
        *,
        wiring_passes: int = 1,
        target_weight: torch.Tensor | None = None,
    ) -> RefitMetrics:
        if mode not in {"argmax", "truth_refit", "wiring_truth_refit"}:
            raise ValueError("invalid hardening mode")
        current = input_state.bool()
        for layer in self.layers[:-1]:
            update = layer.hard_forward(current)
            current = torch.cat(
                (current[:, :self.preserved_bits], update), dim=-1
            )
        final = self.layers[-1]
        if mode == "argmax":
            prediction = final.hard_forward(current)
            return RefitMetrics(
                bit_error=float((prediction != target_state.bool()).float().mean().item()),
                address_coverage=0.0,
                changed_truth_ratio=0.0,
                changed_wiring_ratio=0.0,
            )
        if mode == "truth_refit":
            return final.refit_truth_tables(
                current, target_state, target_weight
            )
        return final.greedy_refit_wiring_and_truth(
            current,
            target_state,
            passes=wiring_passes,
            target_weight=target_weight,
        )

    @torch.no_grad()
    def freeze_hard(self) -> None:
        for layer in self.layers:
            layer.freeze_hard()

    def constraint_loss(
        self,
        *,
        fanout_cap: float,
        table_cost_weight: float,
    ) -> torch.Tensor:
        losses = [
            layer.wiring_constraint_loss(fanout_cap)
            + float(table_cost_weight) * layer.truth_complexity_loss()
            for layer in self.layers
        ]
        return torch.stack(losses).mean()


class BitPlaneLUTClassifier(nn.Module):
    """Logic-native classifier with A8 bit-plane state and fixed GroupSum."""

    def __init__(
        self,
        *,
        input_symbols: int,
        state_bits: int,
        num_classes: int,
        blocks: int = 2,
        layers_per_block: int = 1,
        arity: int = 4,
        candidate_count: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_symbols < 1:
            raise ValueError("input_symbols must be positive")
        if state_bits <= input_symbols * 8:
            raise ValueError("state needs learned vote planes after preserving every input plane")
        if state_bits % 8:
            raise ValueError("state_bits must be divisible by eight")
        if blocks < 1:
            raise ValueError("blocks must be positive")
        self.input_symbols = int(input_symbols)
        self.input_bits = self.input_symbols * 8
        self.state_bits = int(state_bits)
        self.preserved_bits = self.input_bits
        self.vote_bits = self.state_bits - self.preserved_bits
        self.num_classes = int(num_classes)
        if self.vote_bits % 8 or self.vote_bits % self.num_classes:
            raise ValueError("learned vote planes must align to A8 and GroupSum")
        self.arity = int(arity)
        self.candidate_count = int(candidate_count)

        # Every source bit appears once before deterministic repetition.  This
        # is fixed wiring, not learned projection capacity.
        route = torch.arange(self.state_bits, dtype=torch.int64) % self.input_bits
        self.register_buffer("encoder_route", route, persistent=True)
        group = torch.arange(self.vote_bits, dtype=torch.int64) % self.num_classes
        self.register_buffer("readout_group", group, persistent=True)
        self.blocks = nn.ModuleList(
            [
                BitPlaneLUTBlock(
                    self.state_bits,
                    self.preserved_bits,
                    layers=layers_per_block,
                    arity=self.arity,
                    candidate_count=self.candidate_count,
                    seed=seed + index * 100_003,
                )
                for index in range(blocks)
            ]
        )

    def encode(self, symbols_or_bits: torch.Tensor, *, carrier: bool) -> torch.Tensor:
        if symbols_or_bits.dtype == torch.uint8:
            if symbols_or_bits.ndim != 2 or symbols_or_bits.shape[1] != self.input_symbols:
                raise ValueError("A8 input shape mismatch")
            bits = uint8_to_bitplanes(symbols_or_bits)
        elif symbols_or_bits.dtype == torch.bool:
            if symbols_or_bits.ndim != 2 or symbols_or_bits.shape[1] != self.input_bits:
                raise ValueError("Boolean input shape mismatch")
            bits = symbols_or_bits
        else:
            raise TypeError("input must be uint8 symbols or Boolean planes")
        state = bits[:, self.encoder_route]
        return state.float() if carrier else state

    def state_after(
        self,
        symbols_or_bits: torch.Tensor,
        *,
        block_count: int | None = None,
        mode: str = "hard_st",
    ) -> torch.Tensor:
        count = len(self.blocks) if block_count is None else int(block_count)
        if not 0 <= count <= len(self.blocks):
            raise ValueError("invalid block_count")
        carrier = mode != "hard"
        state = self.encode(symbols_or_bits, carrier=carrier)
        for block in self.blocks[:count]:
            state = block(state, mode=mode)
        return state

    def group_sum(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_bits:
            raise ValueError("readout state shape mismatch")
        # readout_group is interleaved by construction, so reshape/sum is the
        # deterministic GroupSum implementation used during optimization.
        votes = state[:, self.preserved_bits:]
        return votes.reshape(votes.shape[0], -1, self.num_classes).sum(dim=1)

    def forward(
        self,
        symbols_or_bits: torch.Tensor,
        *,
        block_count: int | None = None,
        mode: str = "hard_st",
    ) -> torch.Tensor:
        state = self.state_after(
            symbols_or_bits, block_count=block_count, mode=mode
        )
        if state.dtype == torch.bool:
            state = state.float()
        return self.group_sum(state)

    def hard_logits(
        self, symbols_or_bits: torch.Tensor, *, block_count: int | None = None
    ) -> torch.Tensor:
        state = self.state_after(
            symbols_or_bits, block_count=block_count, mode="hard"
        )
        votes = state[:, self.preserved_bits:]
        return votes.reshape(
            votes.shape[0], -1, self.num_classes
        ).sum(dim=1, dtype=torch.int32)

    def target_state(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        return labels.view(-1, 1) == self.readout_group.view(1, -1)

    def target_weight(self, labels: torch.Tensor) -> torch.Tensor:
        """Integer class balancing for empirical truth-table fitting."""

        target = self.target_state(labels)
        counts = torch.bincount(labels, minlength=self.num_classes).to(torch.int64)
        if bool((counts == 0).any()):
            raise ValueError("every class must appear in the refit split")
        class_count = counts[self.readout_group]
        positive_weight = labels.numel() - class_count
        negative_weight = class_count
        return torch.where(
            target,
            positive_weight.view(1, -1),
            negative_weight.view(1, -1),
        )

    def set_trainable_block(self, index: int) -> None:
        if not 0 <= index < len(self.blocks):
            raise ValueError("invalid block index")
        for block_index, block in enumerate(self.blocks):
            for parameter in block.parameters():
                parameter.requires_grad_(block_index == index and not block.is_frozen)

    @torch.no_grad()
    def calibration_states(
        self,
        symbols: torch.Tensor,
        labels: torch.Tensor,
        block_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_state = self.state_after(
            symbols, block_count=block_index, mode="hard"
        )
        return input_state, self.target_state(labels), self.target_weight(labels)

    def structural_diagnostics(self) -> dict[str, object]:
        layers = [layer for block in self.blocks for layer in block.layers]
        if not layers:
            raise RuntimeError("classifier has no learned layers")
        live = torch.zeros(self.state_bits, dtype=torch.bool)
        live[self.preserved_bits:] = True
        unused = 0
        layer_rows = []
        fanout_max = 0
        for reverse_index, layer in enumerate(reversed(layers)):
            diagnostics = layer.hard_diagnostics()
            gate_live = live[self.preserved_bits:]
            unused += int((~gate_live).sum().item())
            sources = layer.selected_sources().detach().cpu()
            source_live = torch.zeros(layer.input_bits, dtype=torch.bool)
            selected_for_live_gates = sources[gate_live]
            if selected_for_live_gates.numel():
                source_live[selected_for_live_gates.flatten()] = True
            # Raw A8 planes are fixed wires across every block/layer.  They are
            # live only when a later learned gate actually consumes them.
            source_live[:self.preserved_bits] |= live[:self.preserved_bits]
            live = source_live
            fanout_max = max(fanout_max, int(diagnostics["fanout_max"]))
            layer_rows.append({
                "reverse_layer_index": reverse_index,
                **diagnostics,
            })
        layer_rows.reverse()
        gate_count = sum(layer.output_bits for layer in layers)
        encoder_index_bits = max(1, math.ceil(math.log2(self.input_bits)))
        readout_index_bits = max(1, math.ceil(math.log2(self.num_classes)))
        learned_payload_bits = sum(
            int(layer.hard_diagnostics()["logical_payload_bits"])
            for layer in layers
        )
        return {
            "state_symbols": self.state_bits // 8,
            "state_bits": self.state_bits,
            "preserved_input_bits": self.preserved_bits,
            "learned_vote_bits": self.vote_bits,
            "gate_count": gate_count,
            "learned_logic_depth": len(layers),
            "fanout_max": fanout_max,
            "unused_gate_ratio": float(unused / max(1, gate_count)),
            "learned_payload_bits": learned_payload_bits,
            "total_logical_payload_bits": int(
                learned_payload_bits
                + self.state_bits * encoder_index_bits
                + self.vote_bits * readout_index_bits
            ),
            "learned_dense_integer_matrix_count": 0,
            "learned_numeric_weight_count": 0,
            "layers": layer_rows,
        }

    @torch.no_grad()
    def hard_payload(self) -> dict[str, object]:
        return {
            "schema": "a8_bitplane_hard_lgn_v1",
            "kind": "bitplane_lut_classifier",
            "input_symbols": self.input_symbols,
            "input_bits": self.input_bits,
            "state_bits": self.state_bits,
            "state_symbols": self.state_bits // 8,
            "preserved_bits": self.preserved_bits,
            "vote_bits": self.vote_bits,
            "num_classes": self.num_classes,
            "encoder_route": self.encoder_route.detach().cpu().to(torch.int32),
            "blocks": [
                {
                    "kind": "hard_lut_block",
                    "preserved_bits": self.preserved_bits,
                    "layers": [layer.hard_payload() for layer in block.layers],
                }
                for block in self.blocks
            ],
            "readout": {
                "kind": "integer_group_sum",
                "group": self.readout_group.detach().cpu().to(torch.int16),
            },
            "learned_dense_integer_matrix_count": 0,
            "learned_numeric_weight_count": 0,
        }


def trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


__all__ = [
    "BitPlaneLUTBlock",
    "BitPlaneLUTClassifier",
    "boolean_state_diagnostics",
    "trainable_parameters",
]
