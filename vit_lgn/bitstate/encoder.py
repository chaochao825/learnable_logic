from __future__ import annotations

import torch
from torch import Tensor, nn


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

    def forward_bits(self, images: Tensor) -> Tensor:
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
        state = patches[..., self.state_route]
        if not self.append_global_token:
            return state
        global_token = state.to(torch.int32).sum(dim=1) * 2 >= self.patch_tokens
        return torch.cat((global_token.unsqueeze(1), state), dim=1)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_bits(images).to(torch.float32)

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
