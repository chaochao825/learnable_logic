from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .blocks import BinaryTopKBlock, LocalBitLogicBlock
from .encoder import RedundantPredicatePatchEncoder, ThermometerPatchEncoder
from .gates import HardSTGateLayer, random_connections, total_gate_count


@dataclass(frozen=True)
class BitStateConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    threshold_levels: int = 4
    state_width: int = 192
    encoder_kind: str = "thermometer"
    predicate_fanin: int = 9
    encoder_identity_width: int = 0
    predicate_temperature: float = 1.0
    predicate_chunk_size: int = 1024
    local_depth: int = 2
    global_depth: int = 2
    heads: int = 6
    qk_bits: int = 16
    topk: int = 8
    num_classes: int = 10
    votes_per_class: int = 32
    update_fraction: float = 0.5
    attention_temperature: float = 0.25
    exclude_self: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        positive = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "threshold_levels": self.threshold_levels,
            "state_width": self.state_width,
            "predicate_fanin": self.predicate_fanin,
            "predicate_chunk_size": self.predicate_chunk_size,
            "heads": self.heads,
            "qk_bits": self.qk_bits,
            "topk": self.topk,
            "num_classes": self.num_classes,
            "votes_per_class": self.votes_per_class,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(positive)
        if self.image_size % self.patch_size:
            raise ValueError((self.image_size, self.patch_size))
        if self.encoder_kind not in {"thermometer", "redundant_predicate"}:
            raise ValueError(self.encoder_kind)
        if self.encoder_identity_width < 0 or self.predicate_temperature <= 0.0:
            raise ValueError((self.encoder_identity_width, self.predicate_temperature))
        if self.state_width % self.heads:
            raise ValueError((self.state_width, self.heads))
        if self.local_depth < 0 or self.global_depth < 0:
            raise ValueError((self.local_depth, self.global_depth))
        if not 0.0 < self.update_fraction <= 1.0:
            raise ValueError(self.update_fraction)
        if self.attention_temperature <= 0.0:
            raise ValueError(self.attention_temperature)
        if self.topk > self.num_tokens - int(self.exclude_self):
            raise ValueError((self.topk, self.num_tokens, self.exclude_self))

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return 1 + self.grid_size * self.grid_size


class BitStateViT(nn.Module):
    """Persistent Boolean-state vision model with a GroupSum vote head.

    ``forward`` is the training carrier. In ``hard`` and ``hard_st`` modes its
    values are exactly Boolean at every hidden boundary. ``forward_bits`` is
    the deployment reference: it accepts uint8 (or quantized float) images,
    carries torch.bool state only, and returns integer class vote counts.
    """

    def __init__(self, config: BitStateConfig) -> None:
        super().__init__()
        self.config = config
        encoder_kwargs = {
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "in_channels": config.in_channels,
            "threshold_levels": config.threshold_levels,
            "state_width": config.state_width,
            "append_global_token": True,
        }
        if config.encoder_kind == "redundant_predicate":
            self.encoder = RedundantPredicatePatchEncoder(
                **encoder_kwargs,
                predicate_fanin=config.predicate_fanin,
                identity_width=config.encoder_identity_width,
                predicate_temperature=config.predicate_temperature,
                predicate_chunk_size=config.predicate_chunk_size,
                seed=config.seed + 500,
            )
        else:
            self.encoder = ThermometerPatchEncoder(**encoder_kwargs)
        self.local_blocks = nn.ModuleList(
            LocalBitLogicBlock(
                state_width=config.state_width,
                grid_size=config.grid_size,
                update_fraction=config.update_fraction,
                seed=config.seed + 1000 + block_id,
            )
            for block_id in range(config.local_depth)
        )
        self.global_blocks = nn.ModuleList(
            BinaryTopKBlock(
                state_width=config.state_width,
                num_tokens=config.num_tokens,
                heads=config.heads,
                qk_bits=config.qk_bits,
                topk=config.topk,
                seed=config.seed + 2000 + block_id,
                attention_temperature=config.attention_temperature,
                exclude_self=config.exclude_self,
            )
            for block_id in range(config.global_depth)
        )

        generator = torch.Generator().manual_seed(config.seed + 3000)
        vote_width = config.num_classes * config.votes_per_class
        head_0, head_1 = random_connections(config.state_width, vote_width, generator)
        # Avoid constant truth tables at initialization while retaining all 16
        # functions as trainable candidates.
        initial_ops = torch.randint(1, 15, (vote_width,), generator=generator)
        self.head = HardSTGateLayer(
            config.state_width,
            vote_width,
            head_0,
            head_1,
            init_ops=initial_ops,
            init_strength=1.0,
            surrogate_inputs=True,
        )

    def gate_layers(self) -> list[HardSTGateLayer]:
        return [module for module in self.modules() if isinstance(module, HardSTGateLayer)]

    def _group_sum(self, votes: Tensor) -> Tensor:
        return votes.reshape(
            votes.shape[0],
            self.config.num_classes,
            self.config.votes_per_class,
        ).sum(dim=-1, dtype=votes.dtype)

    def forward(
        self,
        images: Tensor,
        *,
        mode: str = "hard_st",
        tau: float = 1.0,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        state = self.encoder(images, mode=mode, tau=tau)
        trace = [state] if return_trace else []
        for block in self.local_blocks:
            state = block(state, mode=mode, tau=tau)
            if return_trace:
                trace.append(state)
        for block in self.global_blocks:
            state = block(state, mode=mode, tau=tau)
            if return_trace:
                trace.append(state)
        votes = self.head(state[:, 0], mode=mode, tau=tau)
        logits = self._group_sum(votes)
        if return_trace:
            trace.append(votes)
            return logits, trace
        return logits

    def forward_bits(
        self,
        images: Tensor,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        state = self.encoder.forward_bits(images)
        trace = [state] if return_trace else []
        for block in self.local_blocks:
            state = block.forward_bits(state)
            if return_trace:
                trace.append(state)
        for block in self.global_blocks:
            state = block.forward_bits(state)
            if return_trace:
                trace.append(state)
        votes = self.head.forward_bits(state[:, 0])
        logits = self._group_sum(votes.to(torch.int32))
        if return_trace:
            trace.append(votes)
            return logits, trace
        return logits

    @torch.no_grad()
    def assert_bit_exact(self, images: Tensor) -> None:
        hard_logits, hard_trace = self(images, mode="hard", return_trace=True)
        bit_logits, bit_trace = self.forward_bits(images, return_trace=True)
        if len(hard_trace) != len(bit_trace):
            raise AssertionError((len(hard_trace), len(bit_trace)))
        for layer_id, (carrier, bits) in enumerate(zip(hard_trace, bit_trace)):
            if not torch.equal(carrier >= 0.5, bits):
                mismatch = int(torch.count_nonzero((carrier >= 0.5) != bits))
                raise AssertionError(f"layer {layer_id} differs at {mismatch} bits")
        if not torch.equal(hard_logits.to(torch.int32), bit_logits):
            raise AssertionError("GroupSum logits differ from integer execution")

    @torch.no_grad()
    def inactive_gate_ratio(self, image_batches: Iterable[Tensor]) -> float:
        """Measure gates whose hard outputs are constant on the supplied data."""

        layers = self.gate_layers()
        minima = [torch.ones(layer.out_dim, dtype=torch.bool) for layer in layers]
        maxima = [torch.zeros(layer.out_dim, dtype=torch.bool) for layer in layers]
        seen = False

        def hook(layer_id: int):
            def collect(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
                bits = output.detach() >= 0.5
                flattened = bits.reshape(-1, bits.shape[-1]).cpu()
                minima[layer_id].logical_and_(flattened.all(dim=0))
                maxima[layer_id].logical_or_(flattened.any(dim=0))

            return collect

        handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
        try:
            for images in image_batches:
                seen = True
                self(images, mode="hard")
        finally:
            for handle in handles:
                handle.remove()
        if not seen:
            raise ValueError("at least one image batch is required")
        inactive = sum(int(torch.count_nonzero(low == high)) for low, high in zip(minima, maxima))
        return inactive / max(self.gate_count(), 1)

    def gate_count(self) -> int:
        return total_gate_count(self.gate_layers())

    def logic_depth(self) -> int:
        # Encoder stages, one gate stage per local block, query/key plus merge
        # per global block, and one vote-head stage.
        return self.encoder.logic_depth() + self.config.local_depth + 2 * self.config.global_depth + 1

    def fanout_max(self) -> int:
        gate_fanout = max((layer.fanout_max() for layer in self.gate_layers()), default=0)
        return max(gate_fanout, self.encoder.fanout_max())

    def predicate_count(self) -> int:
        return int(getattr(self.encoder, "predicate_width", 0))

    def deployment_payload(self) -> dict[str, object]:
        discrete_config = {
            key: value
            for key, value in asdict(self.config).items()
            if isinstance(value, (bool, int))
        }
        return {
            "kind": "persistent_boolean_state_vit",
            "config": discrete_config,
            "encoder": self.encoder.deployment_payload(),
            "local_blocks": [block.deployment_payload() for block in self.local_blocks],
            "global_blocks": [block.deployment_payload() for block in self.global_blocks],
            "head": self.head.deployment_payload(),
            "group_sum": {
                "num_classes": self.config.num_classes,
                "votes_per_class": self.config.votes_per_class,
            },
            "gate_count": self.gate_count(),
            "predicate_count": self.predicate_count(),
            "logic_depth": self.logic_depth(),
            "fanout_max": self.fanout_max(),
            "general_matrix_multipliers": 0,
            "persistent_state_dtype": "bool",
            "classifier_dtype": "int32",
        }


def bitstate_vit(**kwargs: object) -> BitStateViT:
    return BitStateViT(BitStateConfig(**kwargs))
