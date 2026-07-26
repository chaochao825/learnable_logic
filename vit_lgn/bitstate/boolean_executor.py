"""Strict Boolean/integer executor for exported persistent BitState models."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from hard_lgn_gap_proto.boolean_executor import BOOLEAN_GATE_TRUTH


def _tensor_leaves(value: object):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _tensor_leaves(child)


def _validate_payload(value: object) -> None:
    for tensor in _tensor_leaves(value):
        if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            raise TypeError(f"real-valued payload tensor: {tensor.dtype}")
        if tensor.device.type != "cpu":
            raise ValueError("strict payload tensors must be on CPU")
    if isinstance(value, float):
        raise TypeError("real-valued payload scalar")
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_payload(child)


def _index(value: Tensor) -> Tensor:
    return value.to(torch.int64)


def _gate_layer(state: Tensor, payload: Mapping[str, object]) -> Tensor:
    if state.dtype != torch.bool:
        raise TypeError("gate state must be Boolean")
    left_source = _index(payload["indices_0"])
    right_source = _index(payload["indices_1"])
    operation = _index(payload["op_ids"])
    left = state[..., left_source]
    right = state[..., right_source]
    address = torch.bitwise_or(
        torch.bitwise_left_shift(left.to(torch.uint8), 1),
        right.to(torch.uint8),
    ).to(torch.int64)
    operation_shape = (1,) * (address.ndim - 1) + (operation.numel(),)
    return BOOLEAN_GATE_TRUTH[operation.reshape(operation_shape), address]


def _raw_patch_bits(images: Tensor, encoder: Mapping[str, object]) -> Tensor:
    if images.device.type != "cpu" or images.dtype != torch.uint8:
        raise TypeError("strict BitState input must be a CPU uint8 tensor")
    image_size = int(encoder["image_size"])
    patch_size = int(encoder["patch_size"])
    channels = int(encoder["in_channels"])
    levels = int(encoder["threshold_levels"])
    if images.ndim != 4 or tuple(images.shape[1:]) != (
        channels,
        image_size,
        image_size,
    ):
        raise ValueError("strict BitState input shape mismatch")
    thresholds = encoder["thresholds"]
    bits = images.unsqueeze(-1) >= thresholds.view(1, 1, 1, 1, levels)
    batch = images.shape[0]
    grid = image_size // patch_size
    raw_width = channels * patch_size * patch_size * levels
    return bits.reshape(
        batch,
        channels,
        grid,
        patch_size,
        grid,
        patch_size,
        levels,
    ).permute(0, 2, 4, 1, 3, 5, 6).reshape(
        batch,
        grid * grid,
        raw_width,
    )


def _append_global(
    state: Tensor,
    thresholds: Tensor,
    append_global: bool,
) -> Tensor:
    if not append_global:
        return state
    global_token = state.to(torch.int32).sum(dim=1) >= thresholds
    return torch.cat((global_token.unsqueeze(1), state), dim=1)


def _encode(images: Tensor, payload: Mapping[str, object]) -> Tensor:
    patches = _raw_patch_bits(images, payload)
    kind = payload["kind"]
    append_global = bool(payload["append_global_token"])
    if kind == "uint8_thermometer_patch_encoder":
        state = patches[..., _index(payload["state_route"])]
        patch_tokens = state.shape[1]
        thresholds = torch.full(
            (state.shape[-1],),
            (patch_tokens + 1) // 2,
            dtype=torch.int32,
        )
        return _append_global(state, thresholds, append_global)
    if kind != "uint8_thermometer_sparse_popcount_predicates":
        raise ValueError(f"unsupported encoder kind: {kind}")
    identity = patches[..., _index(payload["identity_route"])]
    route = _index(payload["predicate_route"])
    predicate_thresholds = payload["predicate_thresholds"].to(torch.int32)
    predicate_polarity = payload["predicate_polarity"]
    if route.numel():
        counts = patches[..., route].sum(dim=-1, dtype=torch.int32)
        base = counts >= predicate_thresholds
        predicates = torch.where(predicate_polarity, base, ~base)
        state = torch.cat((identity, predicates), dim=-1)
    else:
        state = identity
    return _append_global(
        state,
        payload["global_token_thresholds"].to(torch.int32),
        append_global,
    )


def _local_block(state: Tensor, payload: Mapping[str, object]) -> Tensor:
    token_route = _index(payload["token_route"])
    channel_route = _index(payload["channel_route"])
    routed = state[:, token_route, channel_route]
    updates = _gate_layer(routed, payload["logic"])
    output = state.clone()
    output[..., _index(payload["update_positions"])] = updates
    return output


def _reshape_qk(value: Tensor, heads: int, qk_bits: int) -> Tensor:
    batch, tokens, _ = value.shape
    return value.reshape(batch, tokens, heads, qk_bits).permute(0, 2, 1, 3)


def _topk_indices(
    query: Tensor,
    key: Tensor,
    topk: int,
    exclude_self: bool,
) -> Tensor:
    scores = (query.unsqueeze(3) == key.unsqueeze(2)).to(torch.int32).sum(dim=-1)
    tokens = scores.shape[-1]
    priority = torch.arange(tokens - 1, -1, -1, dtype=torch.int32)
    ranking = scores * (tokens + 1) + priority.view(1, 1, 1, -1)
    if exclude_self:
        diagonal = torch.arange(tokens)
        ranking[:, :, diagonal, diagonal] = -1
    return ranking.topk(topk, dim=-1, largest=True, sorted=True).indices


def _selected_values(state: Tensor, indices: Tensor, heads: int) -> Tensor:
    batch, tokens, width = state.shape
    head_width = width // heads
    values = state.reshape(batch, tokens, heads, head_width).permute(0, 2, 1, 3)
    batch_index = torch.arange(batch).view(batch, 1, 1, 1)
    head_index = torch.arange(heads).view(1, heads, 1, 1)
    return values[batch_index, head_index, indices]


def _message_from_counts(
    counts: Tensor,
    payload: Mapping[str, object],
    topk: int,
) -> Tensor:
    majority = counts * 2 >= topk
    mode = payload["message_mode"]
    if mode == "majority":
        return majority
    if mode != "count_threshold_hybrid":
        raise ValueError(f"unsupported message mode: {mode}")
    source = _index(payload["message_count_source_indices"])
    thresholds = payload["message_count_thresholds"].to(torch.int32)
    count_output_width = int(payload["count_output_width"])
    encoded = (counts[..., source].unsqueeze(-1) >= thresholds).flatten(-2)
    return torch.cat((encoded, majority[..., count_output_width:]), dim=-1)


def _global_block(state: Tensor, payload: Mapping[str, object]) -> Tensor:
    heads = int(payload["heads"])
    qk_bits = int(payload["qk_bits"])
    topk = int(payload["topk"])
    query = _reshape_qk(_gate_layer(state, payload["query"]), heads, qk_bits)
    key = _reshape_qk(_gate_layer(state, payload["key"]), heads, qk_bits)
    indices = _topk_indices(query, key, topk, bool(payload["exclude_self"]))
    selected = _selected_values(state, indices, heads)
    counts = selected.to(torch.int32).sum(dim=3)
    message = _message_from_counts(counts, payload, topk)
    message = message.permute(0, 2, 1, 3).reshape_as(state)
    return _gate_layer(torch.cat((state, message), dim=-1), payload["merge"])


class StrictBitStateExecutor:
    """Execute an exported BitState payload without model shadow parameters."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        _validate_payload(payload)
        if payload.get("kind") != "persistent_boolean_state_vit":
            raise ValueError("unexpected BitState payload kind")
        if payload.get("persistent_state_dtype") != "bool":
            raise ValueError("persistent state must be Boolean")
        if payload.get("classifier_dtype") != "int32":
            raise ValueError("classifier must be int32")
        self.payload = payload

    def logits(
        self,
        images: Tensor,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        state = _encode(images, self.payload["encoder"])
        trace = [state] if return_trace else []
        for block in self.payload["local_blocks"]:
            state = _local_block(state, block)
            if return_trace:
                trace.append(state)
        for block in self.payload["global_blocks"]:
            state = _global_block(state, block)
            if return_trace:
                trace.append(state)
        votes = _gate_layer(state[:, 0], self.payload["head"])
        group = self.payload["group_sum"]
        logits = votes.reshape(
            votes.shape[0],
            int(group["num_classes"]),
            int(group["votes_per_class"]),
        ).sum(dim=-1, dtype=torch.int32)
        if return_trace:
            trace.append(votes)
            return logits, trace
        return logits

    def predict(self, images: Tensor) -> Tensor:
        return self.logits(images).argmax(dim=-1).to(torch.int64)


__all__ = ["StrictBitStateExecutor"]
