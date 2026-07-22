from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .gates import hard_forward_soft_backward


class ThermometerPatchEncoder(nn.Module):
    """Convert uint8 images directly to a persistent Boolean token state."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        threshold_levels: int,
        state_width: int,
        append_global_token: bool = True,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError((image_size, patch_size))
        if in_channels <= 0 or threshold_levels <= 0 or state_width <= 0:
            raise ValueError((in_channels, threshold_levels, state_width))
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.threshold_levels = int(threshold_levels)
        self.state_width = int(state_width)
        self.append_global_token = bool(append_global_token)
        self.grid_size = self.image_size // self.patch_size
        self.patch_tokens = self.grid_size * self.grid_size
        self.num_tokens = self.patch_tokens + int(self.append_global_token)

        thresholds = torch.arange(1, threshold_levels + 1, dtype=torch.int32)
        thresholds = torch.div(
            thresholds * 255,
            threshold_levels + 1,
            rounding_mode="floor",
        ).clamp(1, 254)
        self.register_buffer("thresholds", thresholds.to(torch.uint8))

        raw_width = in_channels * patch_size * patch_size * threshold_levels
        route = torch.div(
            torch.arange(state_width, dtype=torch.long) * raw_width,
            state_width,
            rounding_mode="floor",
        ).clamp_max(raw_width - 1)
        self.raw_patch_width = int(raw_width)
        self.register_buffer("state_route", route)

    def _validate(self, images: Tensor) -> None:
        expected = (self.in_channels, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError((images.shape, expected))

    def to_uint8(self, images: Tensor) -> Tensor:
        self._validate(images)
        if images.dtype == torch.uint8:
            return images
        if not images.is_floating_point():
            raise TypeError(f"unsupported image dtype: {images.dtype}")
        return images.detach().clamp(0, 1).mul(255).round().to(torch.uint8)

    def _raw_patch_bits(self, images: Tensor) -> Tensor:
        images_u8 = self.to_uint8(images)
        threshold_shape = (1, 1, 1, 1, self.threshold_levels)
        bits = images_u8.unsqueeze(-1) >= self.thresholds.view(threshold_shape)
        batch = images_u8.shape[0]
        patch = self.patch_size
        grid = self.grid_size
        patches = bits.reshape(
            batch,
            self.in_channels,
            grid,
            patch,
            grid,
            patch,
            self.threshold_levels,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5, 6).reshape(
            batch,
            self.patch_tokens,
            self.raw_patch_width,
        )
        return patches

    def _append_global_bits(self, state: Tensor) -> Tensor:
        if not self.append_global_token:
            return state
        global_token = state.to(torch.int32).sum(dim=1) * 2 >= self.patch_tokens
        return torch.cat((global_token.unsqueeze(1), state), dim=1)

    def forward_bits(self, images: Tensor) -> Tensor:
        patches = self._raw_patch_bits(images)
        state = patches[..., self.state_route]
        return self._append_global_bits(state)

    def forward(
        self,
        images: Tensor,
        *,
        mode: str = "hard_st",
        tau: float = 1.0,
    ) -> Tensor:
        del mode, tau
        return self.forward_bits(images).to(torch.float32)

    def logic_depth(self) -> int:
        return 1

    def fanout_max(self) -> int:
        counts = torch.bincount(self.state_route.detach().cpu(), minlength=self.raw_patch_width)
        return int(counts.max().item())

    def deployment_payload(self) -> dict[str, Tensor | int | str | bool]:
        return {
            "kind": "uint8_thermometer_patch_encoder",
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "threshold_levels": self.threshold_levels,
            "state_width": self.state_width,
            "append_global_token": self.append_global_token,
            "thresholds": self.thresholds.detach().cpu(),
            "state_route": self.state_route.detach().cpu().to(torch.int32),
        }


class RedundantPredicatePatchEncoder(ThermometerPatchEncoder):
    """Preserve raw bits and add sparse deployable popcount predicates.

    Every learned feature compares the popcount of a fixed sparse subset of
    thermometer bits with an integer threshold, followed by an optional
    inversion. Training uses a hard forward / sigmoid backward estimator;
    deployment stores only routes, integer thresholds, and polarity bits.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        threshold_levels: int,
        state_width: int,
        predicate_fanin: int = 9,
        identity_width: int = 0,
        predicate_temperature: float = 1.0,
        predicate_chunk_size: int = 1024,
        global_token_mode: str = "majority",
        seed: int = 0,
        append_global_token: bool = True,
    ) -> None:
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            threshold_levels=threshold_levels,
            state_width=state_width,
            append_global_token=append_global_token,
        )
        if predicate_fanin < 1 or predicate_temperature <= 0.0:
            raise ValueError((predicate_fanin, predicate_temperature))
        if predicate_chunk_size < 1:
            raise ValueError(predicate_chunk_size)
        if identity_width < 0:
            raise ValueError(identity_width)
        if global_token_mode not in {"majority", "learned_count"}:
            raise ValueError(global_token_mode)

        automatic_identity = min(self.raw_patch_width, self.state_width)
        self.identity_width = min(
            automatic_identity if identity_width == 0 else identity_width,
            self.raw_patch_width,
            self.state_width,
        )
        self.predicate_width = self.state_width - self.identity_width
        self.predicate_fanin = min(int(predicate_fanin), self.raw_patch_width)
        self.predicate_temperature = float(predicate_temperature)
        self.predicate_chunk_size = int(predicate_chunk_size)
        self.global_token_mode = global_token_mode

        identity_route = torch.div(
            torch.arange(self.identity_width, dtype=torch.long) * self.raw_patch_width,
            max(self.identity_width, 1),
            rounding_mode="floor",
        ).clamp_max(self.raw_patch_width - 1)
        self.register_buffer("identity_route", identity_route)

        generator = torch.Generator().manual_seed(seed)
        if self.predicate_width:
            random_scores = torch.rand(
                self.predicate_width,
                self.raw_patch_width,
                generator=generator,
            )
            predicate_route = random_scores.topk(
                self.predicate_fanin,
                dim=-1,
                largest=False,
                sorted=False,
            ).indices
            low = max(1, self.predicate_fanin // 3)
            high = max(low, min(self.predicate_fanin, math.ceil(2 * self.predicate_fanin / 3)))
            initial_thresholds = torch.randint(
                low,
                high + 1,
                (self.predicate_width,),
                generator=generator,
            )
            if self.predicate_fanin == 1:
                threshold_logits = torch.zeros(self.predicate_width)
            else:
                target = initial_thresholds.to(torch.float32).clamp(
                    1.25,
                    self.predicate_fanin - 0.25,
                )
                fraction = (target - 1.0) / (self.predicate_fanin - 1.0)
                threshold_logits = torch.logit(fraction.clamp(1e-4, 1 - 1e-4))
            polarity = torch.randint(
                0,
                2,
                (self.predicate_width,),
                generator=generator,
            ).to(torch.float32)
            polarity_logits = (polarity * 2 - 1) * 0.5
        else:
            predicate_route = torch.empty(0, self.predicate_fanin, dtype=torch.long)
            threshold_logits = torch.empty(0)
            polarity_logits = torch.empty(0)
        self.register_buffer("predicate_route", predicate_route)
        self.threshold_logits = nn.Parameter(threshold_logits)
        self.polarity_logits = nn.Parameter(polarity_logits)

        if self.global_token_mode == "learned_count":
            initial_global_thresholds = 1 + torch.div(
                torch.arange(self.state_width, dtype=torch.long) * self.patch_tokens,
                self.state_width,
                rounding_mode="floor",
            )
            initial_global_thresholds = initial_global_thresholds[
                torch.randperm(self.state_width, generator=generator)
            ]
            if self.patch_tokens == 1:
                global_threshold_logits = torch.zeros(self.state_width)
            else:
                target = initial_global_thresholds.to(torch.float32).clamp(
                    1.25,
                    self.patch_tokens - 0.25,
                )
                fraction = (target - 1.0) / (self.patch_tokens - 1.0)
                global_threshold_logits = torch.logit(fraction.clamp(1e-4, 1 - 1e-4))
            self.global_threshold_logits = nn.Parameter(global_threshold_logits)
        else:
            self.register_parameter("global_threshold_logits", None)

    def _threshold_real(self) -> Tensor:
        if self.predicate_fanin == 1:
            return torch.ones_like(self.threshold_logits)
        return 1.0 + (self.predicate_fanin - 1.0) * torch.sigmoid(self.threshold_logits)

    def hard_thresholds(self) -> Tensor:
        return self._threshold_real().detach().round().clamp(1, self.predicate_fanin).long()

    def _global_threshold_real(self) -> Tensor:
        if self.global_token_mode != "learned_count":
            raise RuntimeError("global count thresholds are disabled")
        if self.patch_tokens == 1:
            return torch.ones_like(self.global_threshold_logits)
        return 1.0 + (self.patch_tokens - 1.0) * torch.sigmoid(
            self.global_threshold_logits
        )

    def hard_global_thresholds(self) -> Tensor:
        if self.global_token_mode == "majority":
            return torch.full(
                (self.state_width,),
                math.ceil(self.patch_tokens / 2),
                device=self.threshold_logits.device,
                dtype=torch.long,
            )
        return (
            self._global_threshold_real()
            .detach()
            .round()
            .clamp(1, self.patch_tokens)
            .long()
        )

    def _predicate_chunk(
        self,
        patches: Tensor,
        start: int,
        end: int,
        *,
        mode: str,
        tau: float,
    ) -> tuple[Tensor, Tensor]:
        route = self.predicate_route[start:end]
        counts = patches[..., route].sum(dim=-1, dtype=torch.int32)
        hard_base = counts >= self.hard_thresholds()[start:end]
        hard_polarity = self.polarity_logits[start:end].detach() >= 0
        hard = torch.where(hard_polarity, hard_base, ~hard_base)
        if mode == "hard":
            return hard.to(torch.float32), hard

        temperature = self.predicate_temperature * tau
        threshold = self._threshold_real()[start:end]
        soft_base = torch.sigmoid(
            (counts.to(threshold.dtype) - threshold + 0.5) / temperature
        )
        polarity = torch.sigmoid(self.polarity_logits[start:end] / tau)
        soft = polarity * soft_base + (1 - polarity) * (1 - soft_base)
        if mode == "soft":
            return soft, hard
        if mode not in {"hard_st", "gumbel_st"}:
            raise ValueError(mode)
        return hard_forward_soft_backward(hard.to(soft.dtype), soft), hard

    def _patch_state(
        self,
        patches: Tensor,
        *,
        mode: str,
        tau: float,
    ) -> tuple[Tensor, Tensor]:
        if tau <= 0.0:
            raise ValueError(tau)
        identity_bits = patches[..., self.identity_route]
        carriers = [identity_bits.to(torch.float32)]
        hard_parts = [identity_bits]
        for start in range(0, self.predicate_width, self.predicate_chunk_size):
            end = min(start + self.predicate_chunk_size, self.predicate_width)
            carrier, hard = self._predicate_chunk(
                patches,
                start,
                end,
                mode=mode,
                tau=tau,
            )
            carriers.append(carrier)
            hard_parts.append(hard)
        return torch.cat(carriers, dim=-1), torch.cat(hard_parts, dim=-1)

    def _append_global_carrier(
        self,
        state: Tensor,
        hard_state: Tensor,
        *,
        mode: str,
        tau: float,
    ) -> Tensor:
        if not self.append_global_token:
            return state
        hard_global = (
            hard_state.to(torch.int32).sum(dim=1) >= self.hard_global_thresholds()
        )
        if mode == "hard":
            global_token = hard_global.to(state.dtype)
        else:
            if self.global_token_mode == "majority":
                threshold = state.new_full(
                    (self.state_width,),
                    math.ceil(self.patch_tokens / 2),
                )
            else:
                threshold = self._global_threshold_real()
            soft_global = torch.sigmoid(
                (state.sum(dim=1) - threshold + 0.5)
                / (self.predicate_temperature * tau)
            )
            if mode == "soft":
                global_token = soft_global
            else:
                global_token = hard_forward_soft_backward(
                    hard_global.to(soft_global.dtype),
                    soft_global,
                )
        return torch.cat((global_token.unsqueeze(1), state), dim=1)

    def forward_bits(self, images: Tensor) -> Tensor:
        patches = self._raw_patch_bits(images)
        identity = patches[..., self.identity_route]
        predicates = []
        hard_thresholds = self.hard_thresholds()
        hard_polarity = self.polarity_logits.detach() >= 0
        for start in range(0, self.predicate_width, self.predicate_chunk_size):
            end = min(start + self.predicate_chunk_size, self.predicate_width)
            counts = patches[..., self.predicate_route[start:end]].sum(
                dim=-1,
                dtype=torch.int32,
            )
            base = counts >= hard_thresholds[start:end]
            predicates.append(torch.where(hard_polarity[start:end], base, ~base))
        state = torch.cat((identity, *predicates), dim=-1)
        if not self.append_global_token:
            return state
        global_token = (
            state.to(torch.int32).sum(dim=1) >= self.hard_global_thresholds()
        )
        return torch.cat((global_token.unsqueeze(1), state), dim=1)

    def forward(
        self,
        images: Tensor,
        *,
        mode: str = "hard_st",
        tau: float = 1.0,
    ) -> Tensor:
        patches = self._raw_patch_bits(images)
        state, hard_state = self._patch_state(patches, mode=mode, tau=tau)
        return self._append_global_carrier(
            state,
            hard_state,
            mode=mode,
            tau=tau,
        )

    def logic_depth(self) -> int:
        return 2 if self.predicate_width else 1

    def fanout_max(self) -> int:
        routes = torch.cat(
            (self.identity_route.reshape(-1), self.predicate_route.reshape(-1)),
        ).detach().cpu()
        counts = torch.bincount(routes, minlength=self.raw_patch_width)
        return int(counts.max().item())

    def deployment_payload(self) -> dict[str, Tensor | int | str | bool]:
        return {
            "kind": "uint8_thermometer_sparse_popcount_predicates",
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "threshold_levels": self.threshold_levels,
            "state_width": self.state_width,
            "append_global_token": self.append_global_token,
            "thresholds": self.thresholds.detach().cpu(),
            "identity_route": self.identity_route.detach().cpu().to(torch.int32),
            "predicate_route": self.predicate_route.detach().cpu().to(torch.int32),
            "predicate_thresholds": self.hard_thresholds().cpu().to(torch.uint8),
            "predicate_polarity": (self.polarity_logits.detach() >= 0).cpu(),
            "predicate_fanin": self.predicate_fanin,
            "predicate_count": self.predicate_width,
            "global_token_mode": self.global_token_mode,
            "global_token_thresholds": (
                self.hard_global_thresholds().cpu().to(torch.int32)
            ),
        }
