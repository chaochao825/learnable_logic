from __future__ import annotations

import torch
from torch import Tensor, nn

from .gates import HardSTGateLayer, hard_forward_soft_backward, random_connections


class LocalBitLogicBlock(nn.Module):
    """Spatially shared Boolean update with an exact identity highway."""

    OFFSETS = (
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

    def __init__(
        self,
        *,
        state_width: int,
        grid_size: int,
        update_fraction: float,
        seed: int,
        gate_init_strength: float = 1.5,
    ) -> None:
        super().__init__()
        if state_width < 2 or grid_size < 1:
            raise ValueError((state_width, grid_size))
        if not 0.0 < update_fraction <= 1.0:
            raise ValueError(update_fraction)
        if gate_init_strength <= 0.0:
            raise ValueError(gate_init_strength)
        self.state_width = int(state_width)
        self.grid_size = int(grid_size)
        self.num_tokens = 1 + grid_size * grid_size
        self.update_width = max(1, min(state_width, round(state_width * update_fraction)))
        generator = torch.Generator().manual_seed(seed)

        token_route = torch.zeros(self.num_tokens, state_width, dtype=torch.long)
        channel_route = torch.arange(state_width, dtype=torch.long).repeat(self.num_tokens, 1)
        channel_permutation = torch.randperm(state_width, generator=generator)
        channel_route[:] = channel_permutation
        channel_route[0] = torch.arange(state_width)
        for patch_id in range(grid_size * grid_size):
            row, col = divmod(patch_id, grid_size)
            token_id = patch_id + 1
            for channel in range(state_width):
                dr, dc = self.OFFSETS[channel % len(self.OFFSETS)]
                source_row = min(grid_size - 1, max(0, row + dr))
                source_col = min(grid_size - 1, max(0, col + dc))
                token_route[token_id, channel] = 1 + source_row * grid_size + source_col
        self.register_buffer("token_route", token_route)
        self.register_buffer("channel_route", channel_route)

        idx0, idx1 = random_connections(state_width, self.update_width, generator)
        init_ops = torch.full((self.update_width,), 3, dtype=torch.long)
        self.logic = HardSTGateLayer(
            state_width,
            self.update_width,
            idx0,
            idx1,
            init_ops=init_ops,
            init_strength=gate_init_strength,
            surrogate_inputs=True,
        )
        positions = torch.randperm(state_width, generator=generator)[: self.update_width]
        self.register_buffer("update_positions", positions.sort().values)

    def _route(self, state: Tensor) -> Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.num_tokens, self.state_width):
            raise ValueError((state.shape, self.num_tokens, self.state_width))
        return state[:, self.token_route, self.channel_route]

    def forward(self, state: Tensor, *, mode: str = "hard_st", tau: float = 1.0) -> Tensor:
        updates = self.logic(self._route(state), mode=mode, tau=tau)
        output = state.clone()
        output[..., self.update_positions] = updates
        return output

    def forward_bits(self, state: Tensor) -> Tensor:
        if state.dtype != torch.bool:
            raise TypeError("forward_bits requires torch.bool")
        updates = self.logic.forward_bits(self._route(state))
        output = state.clone()
        output[..., self.update_positions] = updates
        return output

    def deployment_payload(self) -> dict[str, object]:
        return {
            "kind": "local_bit_logic_block",
            "state_width": self.state_width,
            "grid_size": self.grid_size,
            "token_route": self.token_route.detach().cpu().to(torch.int32),
            "channel_route": self.channel_route.detach().cpu().to(torch.int32),
            "update_positions": self.update_positions.detach().cpu().to(torch.int32),
            "logic": self.logic.deployment_payload(),
        }


class BinaryTopKBlock(nn.Module):
    """XNOR-popcount routing, discrete value messages, and Boolean merge."""

    def __init__(
        self,
        *,
        state_width: int,
        num_tokens: int,
        heads: int,
        qk_bits: int,
        topk: int,
        seed: int,
        attention_temperature: float = 0.25,
        exclude_self: bool = False,
        gate_init_strength: float = 2.0,
        message_mode: str = "majority",
        message_count_thresholds: tuple[int, ...] = (1, 3, 5, 7),
        message_count_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if state_width % heads:
            raise ValueError((state_width, heads))
        if not 1 <= topk <= num_tokens - int(exclude_self):
            raise ValueError((topk, num_tokens, exclude_self))
        if qk_bits < 1 or attention_temperature <= 0.0:
            raise ValueError((qk_bits, attention_temperature))
        if gate_init_strength <= 0.0:
            raise ValueError(gate_init_strength)
        if message_mode not in {"majority", "count_threshold_hybrid"}:
            raise ValueError(message_mode)
        if not 0.0 < message_count_fraction <= 1.0:
            raise ValueError(message_count_fraction)
        self.state_width = int(state_width)
        self.num_tokens = int(num_tokens)
        self.heads = int(heads)
        self.qk_bits = int(qk_bits)
        self.topk = int(topk)
        self.head_width = state_width // heads
        self.attention_temperature = float(attention_temperature)
        self.exclude_self = bool(exclude_self)
        self.message_mode = str(message_mode)
        thresholds = tuple(int(value) for value in message_count_thresholds)
        if self.message_mode == "count_threshold_hybrid":
            if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
                raise ValueError("message count thresholds must be strictly increasing")
            if thresholds[0] < 1 or thresholds[-1] > self.topk:
                raise ValueError((thresholds, self.topk))
            count_output_width = int(self.head_width * message_count_fraction)
            count_output_width -= count_output_width % len(thresholds)
            if count_output_width < len(thresholds):
                raise ValueError("head width is too small for count thresholds")
            count_sources = count_output_width // len(thresholds)
            source_indices = torch.div(
                torch.arange(count_sources, dtype=torch.long) * self.head_width,
                count_sources,
                rounding_mode="floor",
            )
        else:
            count_output_width = 0
            source_indices = torch.empty(0, dtype=torch.long)
        self.count_output_width = int(count_output_width)
        self.register_buffer(
            "message_count_thresholds",
            torch.tensor(thresholds, dtype=torch.int32),
        )
        self.register_buffer("message_count_source_indices", source_indices)
        generator = torch.Generator().manual_seed(seed)

        qk_width = heads * qk_bits
        q0, q1 = random_connections(state_width, qk_width, generator)
        k0, k1 = random_connections(state_width, qk_width, generator)
        qk_init = torch.full((qk_width,), 3, dtype=torch.long)
        self.query = HardSTGateLayer(
            state_width,
            qk_width,
            q0,
            q1,
            init_ops=qk_init,
            init_strength=gate_init_strength,
            surrogate_inputs=True,
        )
        self.key = HardSTGateLayer(
            state_width,
            qk_width,
            k0,
            k1,
            init_ops=qk_init,
            init_strength=gate_init_strength,
            surrogate_inputs=True,
        )
        merge_left = torch.arange(state_width)
        merge_right = torch.arange(state_width, 2 * state_width)
        merge_init = torch.full((state_width,), 3, dtype=torch.long)
        self.merge = HardSTGateLayer(
            2 * state_width,
            state_width,
            merge_left,
            merge_right,
            init_ops=merge_init,
            init_strength=gate_init_strength,
            surrogate_inputs=True,
        )
        # Break the soft-LUT symmetry around the hard identity operation so
        # query/key routing receives gradient from the first optimizer step.
        with torch.no_grad():
            noise_scale = min(0.01, gate_init_strength * 0.1)
            self.merge.logits.add_(
                torch.randn(
                    self.merge.logits.shape,
                    generator=generator,
                    device=self.merge.logits.device,
                )
                .clamp(-1, 1)
                * noise_scale
            )

    def _reshape_qk(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.heads, self.qk_bits).permute(0, 2, 1, 3)

    def _topk_indices(self, query_bits: Tensor, key_bits: Tensor) -> Tensor:
        scores = (query_bits.unsqueeze(3) == key_bits.unsqueeze(2)).to(torch.int32).sum(dim=-1)
        key_priority = torch.arange(
            self.num_tokens - 1,
            -1,
            -1,
            device=scores.device,
            dtype=torch.int32,
        )
        ranking = scores * (self.num_tokens + 1) + key_priority.view(1, 1, 1, -1)
        if self.exclude_self:
            diagonal = torch.arange(self.num_tokens, device=scores.device)
            ranking[:, :, diagonal, diagonal] = -1
        return ranking.topk(self.topk, dim=-1, largest=True, sorted=True).indices

    def _gather_values(self, state: Tensor, indices: Tensor) -> Tensor:
        batch = state.shape[0]
        values = state.reshape(batch, self.num_tokens, self.heads, self.head_width)
        values = values.permute(0, 2, 1, 3)
        batch_index = torch.arange(batch, device=state.device).view(batch, 1, 1, 1)
        head_index = torch.arange(self.heads, device=state.device).view(1, self.heads, 1, 1)
        selected = values[batch_index, head_index, indices]
        return selected

    def _encode_hard_counts(self, counts: Tensor) -> Tensor:
        majority = counts * 2 >= self.topk
        if self.message_mode == "majority":
            return majority
        selected_counts = counts[..., self.message_count_source_indices]
        threshold_bits = (
            selected_counts.unsqueeze(-1)
            >= self.message_count_thresholds.view(1, 1, 1, 1, -1)
        ).flatten(-2)
        return torch.cat((threshold_bits, majority[..., self.count_output_width :]), dim=-1)

    def _encode_soft_counts(self, average: Tensor, tau: float) -> Tensor:
        if self.message_mode == "majority":
            return average
        selected_average = average[..., self.message_count_source_indices]
        thresholds = self.message_count_thresholds.to(average.dtype) - 0.5
        soft_count = selected_average.unsqueeze(-1) * self.topk
        threshold_bits = torch.sigmoid(
            (soft_count - thresholds.view(1, 1, 1, 1, -1))
            / (self.attention_temperature * tau)
        ).flatten(-2)
        return torch.cat(
            (threshold_bits, average[..., self.count_output_width :]),
            dim=-1,
        )

    def _hard_message(
        self,
        state_bits: Tensor,
        query_bits: Tensor,
        key_bits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        indices = self._topk_indices(query_bits, key_bits)
        selected = self._gather_values(state_bits, indices)
        counts = selected.to(torch.int32).sum(dim=3)
        message = self._encode_hard_counts(counts)
        message = message.permute(0, 2, 1, 3).reshape(
            state_bits.shape[0],
            self.num_tokens,
            self.state_width,
        )
        return message, indices

    def forward(self, state: Tensor, *, mode: str = "hard_st", tau: float = 1.0) -> Tensor:
        if state.shape[1:] != (self.num_tokens, self.state_width):
            raise ValueError((state.shape, self.num_tokens, self.state_width))
        query = self._reshape_qk(self.query(state, mode=mode, tau=tau))
        key = self._reshape_qk(self.key(state, mode=mode, tau=tau))
        hard_message, indices = self._hard_message(
            state >= 0.5,
            query.detach() >= 0.5,
            key.detach() >= 0.5,
        )

        soft_scores = (
            query.unsqueeze(3) * key.unsqueeze(2)
            + (1 - query.unsqueeze(3)) * (1 - key.unsqueeze(2))
        ).mean(dim=-1)
        if self.exclude_self:
            diagonal = torch.arange(self.num_tokens, device=state.device)
            soft_scores[:, :, diagonal, diagonal] = torch.finfo(soft_scores.dtype).min
        weights = torch.softmax(soft_scores / self.attention_temperature, dim=-1)
        values = state.reshape(
            state.shape[0], self.num_tokens, self.heads, self.head_width
        ).permute(0, 2, 1, 3)
        soft_message = torch.einsum("bhqk,bhkd->bhqd", weights, values)
        soft_message = self._encode_soft_counts(soft_message, tau)
        soft_message = soft_message.permute(0, 2, 1, 3).reshape_as(state)
        message = hard_forward_soft_backward(
            hard_message.to(state.dtype),
            soft_message,
        )
        merged_input = torch.cat((state, message), dim=-1)
        output = self.merge(merged_input, mode=mode, tau=tau)
        self._last_indices = indices.detach()
        return output

    def forward_bits(self, state: Tensor, *, return_indices: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if state.dtype != torch.bool:
            raise TypeError("forward_bits requires torch.bool")
        query = self._reshape_qk(self.query.forward_bits(state))
        key = self._reshape_qk(self.key.forward_bits(state))
        message, indices = self._hard_message(state, query, key)
        output = self.merge.forward_bits(torch.cat((state, message), dim=-1))
        if return_indices:
            return output, indices
        return output

    def deployment_payload(self) -> dict[str, object]:
        return {
            "kind": "binary_xnor_popcount_topk",
            "state_width": self.state_width,
            "num_tokens": self.num_tokens,
            "heads": self.heads,
            "qk_bits": self.qk_bits,
            "topk": self.topk,
            "exclude_self": self.exclude_self,
            "message_mode": self.message_mode,
            "message_count_thresholds": self.message_count_thresholds.detach()
            .cpu()
            .to(torch.int32),
            "message_count_source_indices": self.message_count_source_indices.detach()
            .cpu()
            .to(torch.int32),
            "count_output_width": self.count_output_width,
            "query": self.query.deployment_payload(),
            "key": self.key.deployment_payload(),
            "merge": self.merge.deployment_payload(),
        }
