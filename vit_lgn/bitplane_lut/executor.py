"""Strict Boolean/integer executor for A8 bit-plane Hard-LGN payloads."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from hard_lgn_gap_proto.boolean_executor import BooleanRuntimeAudit

from .layers import uint8_to_bitplanes


_INTEGER_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _tensor_leaves(value: object):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensor_leaves(child)


def validate_hard_payload(value: object) -> None:
    """Fail closed if a deployment artifact contains a real-valued value."""

    if isinstance(value, float):
        raise TypeError("real-valued payload scalar")
    for tensor in _tensor_leaves(value):
        if tensor.device.type != "cpu":
            raise ValueError("hard payload tensors must be on CPU")
        if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            raise TypeError(f"real-valued payload tensor: {tensor.dtype}")
        if tensor.dtype != torch.bool and tensor.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"unsupported hard payload dtype: {tensor.dtype}")
    if isinstance(value, Mapping):
        for child in value.values():
            validate_hard_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_hard_payload(child)


def _integer_index(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must be an integer tensor")
    return value.to(torch.int64)


def execute_lut_layer(
    state: torch.Tensor, payload: Mapping[str, object]
) -> torch.Tensor:
    if state.dtype != torch.bool or state.device.type != "cpu" or state.ndim != 2:
        raise TypeError("strict LUT state must be a two-dimensional CPU Boolean tensor")
    if payload.get("kind") != "boolean_lut_layer":
        raise ValueError("unexpected LUT layer kind")
    input_bits = int(payload["input_bits"])
    output_bits = int(payload["output_bits"])
    arity = int(payload["arity"])
    if arity not in {2, 3, 4}:
        raise ValueError("unsupported LUT arity")
    if state.shape[1] != input_bits:
        raise ValueError("strict LUT input width mismatch")
    sources = _integer_index(payload["source_indices"], "source_indices")
    truth = payload["truth_table"]
    if sources.shape != (output_bits, arity):
        raise ValueError("source index shape mismatch")
    if not isinstance(truth, torch.Tensor) or truth.dtype not in {
        torch.bool,
        torch.uint8,
    }:
        raise TypeError("truth table must be bool or uint8")
    if truth.shape != (output_bits, 1 << arity):
        raise ValueError("truth table shape mismatch")
    if sources.numel() and (
        int(sources.min().item()) < 0 or int(sources.max().item()) >= input_bits
    ):
        raise ValueError("source index out of range")
    if truth.dtype == torch.uint8 and truth.numel() and int(truth.max().item()) > 1:
        raise ValueError("truth table contains a non-Boolean code")

    selected = state[:, sources]
    address = torch.zeros(
        selected.shape[:-1], dtype=torch.int64, device=state.device
    )
    for slot in range(arity):
        address = torch.bitwise_or(
            address,
            torch.bitwise_left_shift(selected[..., slot].to(torch.int64), slot),
        )
    gate = torch.arange(output_bits, dtype=torch.int64).view(1, -1)
    return truth[gate, address].bool()


class StrictBitPlaneLUTExecutor:
    """Run exported LUT/wiring and fixed GroupSum with no real tensor."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        validate_hard_payload(payload)
        if payload.get("schema") != "a8_bitplane_hard_lgn_v1":
            raise ValueError("unexpected payload schema")
        if payload.get("kind") != "bitplane_lut_classifier":
            raise ValueError("unexpected payload kind")
        if int(payload.get("learned_dense_integer_matrix_count", -1)) != 0:
            raise ValueError("payload contains learned dense integer matrices")
        if int(payload.get("learned_numeric_weight_count", -1)) != 0:
            raise ValueError("payload contains learned numeric weights")
        self.payload = payload
        self.input_symbols = int(payload["input_symbols"])
        self.input_bits = int(payload["input_bits"])
        self.state_bits = int(payload["state_bits"])
        self.preserved_bits = int(payload["preserved_bits"])
        self.vote_bits = int(payload["vote_bits"])
        self.num_classes = int(payload["num_classes"])
        if self.input_bits != self.input_symbols * 8:
            raise ValueError("input A8 symbol/plane count mismatch")
        if self.preserved_bits != self.input_bits:
            raise ValueError("v1 payload must preserve every raw input plane")
        if self.preserved_bits + self.vote_bits != self.state_bits:
            raise ValueError("preserved/vote state partition mismatch")
        if self.state_bits % 8 or self.vote_bits % self.num_classes:
            raise ValueError("state vote width is incompatible with A8/GroupSum")
        route = _integer_index(payload["encoder_route"], "encoder_route")
        if route.shape != (self.state_bits,):
            raise ValueError("encoder route shape mismatch")
        if route.numel() and (
            int(route.min().item()) < 0 or int(route.max().item()) >= self.input_bits
        ):
            raise ValueError("encoder route out of range")
        readout = payload["readout"]
        if not isinstance(readout, Mapping) or readout.get("kind") != "integer_group_sum":
            raise ValueError("strict executor requires integer GroupSum")
        group = _integer_index(readout["group"], "readout group")
        if group.shape != (self.vote_bits,):
            raise ValueError("readout group shape mismatch")
        if group.numel() and (
            int(group.min().item()) < 0 or int(group.max().item()) >= self.num_classes
        ):
            raise ValueError("readout group out of range")

    def encode(self, symbols_or_bits: torch.Tensor) -> torch.Tensor:
        if symbols_or_bits.device.type != "cpu":
            raise ValueError("strict input must be on CPU")
        if symbols_or_bits.dtype == torch.uint8:
            if symbols_or_bits.ndim != 2 or symbols_or_bits.shape[1] != self.input_symbols:
                raise ValueError("strict A8 input shape mismatch")
            bits = uint8_to_bitplanes(symbols_or_bits)
        elif symbols_or_bits.dtype == torch.bool:
            if symbols_or_bits.ndim != 2 or symbols_or_bits.shape[1] != self.input_bits:
                raise ValueError("strict Boolean input shape mismatch")
            bits = symbols_or_bits
        else:
            raise TypeError("strict input must use torch.uint8 or torch.bool")
        route = _integer_index(self.payload["encoder_route"], "encoder_route")
        return bits[:, route]

    def state_trace(self, symbols_or_bits: torch.Tensor) -> list[torch.Tensor]:
        state = self.encode(symbols_or_bits)
        trace = [state]
        blocks = self.payload["blocks"]
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise ValueError("payload must contain hard LUT blocks")
        for block in blocks:
            if not isinstance(block, Mapping) or block.get("kind") != "hard_lut_block":
                raise ValueError("unexpected hard block payload")
            layers = block["layers"]
            preserved_bits = int(block.get("preserved_bits", -1))
            if preserved_bits != self.preserved_bits:
                raise ValueError("block preserved-state ABI mismatch")
            if not isinstance(layers, (list, tuple)) or not layers:
                raise ValueError("hard block must contain LUT layers")
            for layer in layers:
                update = execute_lut_layer(state, layer)
                if update.shape[1] != self.vote_bits:
                    raise ValueError("LUT block vote width mismatch")
                state = torch.cat((state[:, :self.preserved_bits], update), dim=-1)
                trace.append(state)
        return trace

    def logits(self, symbols_or_bits: torch.Tensor) -> torch.Tensor:
        state = self.state_trace(symbols_or_bits)[-1]
        group = _integer_index(self.payload["readout"]["group"], "readout group")
        canonical = torch.arange(self.vote_bits, dtype=torch.int64).remainder(
            self.num_classes
        )
        if not torch.equal(group, canonical):
            raise ValueError("v1 executor requires canonical interleaved GroupSum")
        votes = state[:, self.preserved_bits:]
        return votes.reshape(
            votes.shape[0], -1, self.num_classes
        ).sum(dim=1, dtype=torch.int32)

    def predict(self, symbols_or_bits: torch.Tensor) -> torch.Tensor:
        return self.logits(symbols_or_bits).argmax(dim=-1).to(torch.int64)


__all__ = [
    "BooleanRuntimeAudit",
    "StrictBitPlaneLUTExecutor",
    "execute_lut_layer",
    "validate_hard_payload",
]
