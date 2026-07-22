"""
Vision Transformer Tiny 模型定义。
仅替换 TransformerBlock 中的 MultiHeadAttention 为 AttentionLogic。
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from vit_tiny_baseline import DropPath, PatchEmbedding, MLP
from src.attention_logic import AttentionLogic

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from vit_lgn.spatial_logic_pyramid.convlogic_pyramid import ConvLogicInspiredPyramidMixer


class LogicTransformerBlock(nn.Module):
    """仅将注意力层替换为 AttentionLogic 的 Transformer block。"""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attention_only: bool = False,
        attention_k: int = 15,
        validate_input: bool = False,
        use_thermometer_encoding: bool = False,
        n_thresholds: int = 8,
        v_n_thresholds: int | None = None,
        encoding_scale: float = 10.0,
        apply_sigmoid_before_encoding: bool = True,
        decode_output: bool = False,
        boundary_surrogate_temperature: float = 1.0,
        topk_impl: str = "winner-tree",
        topk_forward_mode: str = "topk",
        topk_surrogate_mode: str = "kth",
        topk_surrogate_proxy_source: str = "encoded",
        topk_kth_detach_value: bool = True,
        topk_kth_use_midpoint_theta: bool = False,
        topk_kth_normalize_soft_mask: bool = True,
        majority_train_temperature: float = 0.125,
        fast_vote: bool = False,
        majority_surrogate_mode: str = "count",
        majority_k_root_degree: int = 2,
        thermometer_decode_use_weights: bool = True,
        value_aggregation: str = "majority",
        value_gap_shift: int = 1,
        value_max_gap_bucket: int = 3,
        value_global_tail_weight: int = 0,
        value_global_tail_exclude_cls: bool = True,
        value_output_bits: int = 0,
        value_final_channel_bits: int = 0,
        value_score_gap_source: str = "raw",
        token_grid_size: int | tuple[int, int] | None = None,
        irpe_mode: str = "none",
        irpe_clip_size: int | tuple[int, int] | None = None,
        irpe_manhattan_clip: int | None = None,
        irpe_bias_bits: int = 6,
        irpe_lut_value_bits: int = 6,
        irpe_scale_bits: int = 4,
        irpe_scale_frac_bits: int = 2,
        irpe_initial_head_scale: float = 1.0,
        irpe_learnable_head_scale: bool = True,
        irpe_learnable_cls_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AttentionLogic(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k=attention_k,
            validate_input=validate_input,
            use_thermometer_encoding=use_thermometer_encoding,
            n_thresholds=n_thresholds,
            v_n_thresholds=v_n_thresholds,
            encoding_scale=encoding_scale,
            apply_sigmoid_before_encoding=apply_sigmoid_before_encoding,
            decode_output=decode_output,
            boundary_surrogate_temperature=boundary_surrogate_temperature,
            topk_impl=topk_impl,
            topk_forward_mode=topk_forward_mode,
            topk_surrogate_mode=topk_surrogate_mode,
            topk_surrogate_proxy_source=topk_surrogate_proxy_source,
            topk_kth_detach_value=topk_kth_detach_value,
            topk_kth_use_midpoint_theta=topk_kth_use_midpoint_theta,
            topk_kth_normalize_soft_mask=topk_kth_normalize_soft_mask,
            majority_train_temperature=majority_train_temperature,
            fast_vote=fast_vote,
            majority_surrogate_mode=majority_surrogate_mode,
            majority_k_root_degree=majority_k_root_degree,
            thermometer_decode_use_weights=thermometer_decode_use_weights,
            value_aggregation=value_aggregation,
            value_gap_shift=value_gap_shift,
            value_max_gap_bucket=value_max_gap_bucket,
            value_global_tail_weight=value_global_tail_weight,
            value_global_tail_exclude_cls=value_global_tail_exclude_cls,
            value_output_bits=value_output_bits,
            value_final_channel_bits=value_final_channel_bits,
            value_score_gap_source=value_score_gap_source,
            token_grid_size=token_grid_size,
            irpe_mode=irpe_mode,
            irpe_clip_size=irpe_clip_size,
            irpe_manhattan_clip=irpe_manhattan_clip,
            irpe_bias_bits=irpe_bias_bits,
            irpe_lut_value_bits=irpe_lut_value_bits,
            irpe_scale_bits=irpe_scale_bits,
            irpe_scale_frac_bits=irpe_scale_frac_bits,
            irpe_initial_head_scale=irpe_initial_head_scale,
            irpe_learnable_head_scale=irpe_learnable_head_scale,
            irpe_learnable_cls_bias=irpe_learnable_cls_bias,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Identity() if attention_only else MLP(embed_dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTTiny(nn.Module):
    """ViT-Tiny 模型。"""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        attention_only: bool = False,
        attention_k: int = 15,
        validate_input: bool = False,
        use_thermometer_encoding: bool = False,
        n_thresholds: int = 8,
        v_n_thresholds: int | None = None,
        encoding_scale: float = 10.0,
        apply_sigmoid_before_encoding: bool = True,
        decode_output: bool = False,
        boundary_surrogate_temperature: float = 1.0,
        topk_impl: str = "winner-tree",
        topk_forward_mode: str = "topk",
        topk_surrogate_mode: str = "kth",
        topk_surrogate_proxy_source: str = "encoded",
        topk_kth_detach_value: bool = True,
        topk_kth_use_midpoint_theta: bool = False,
        topk_kth_normalize_soft_mask: bool = True,
        majority_train_temperature: float = 0.125,
        fast_vote: bool = False,
        majority_surrogate_mode: str = "count",
        majority_k_root_degree: int = 2,
        thermometer_decode_use_weights: bool = True,
        value_aggregation: str = "majority",
        value_gap_shift: int = 1,
        value_max_gap_bucket: int = 3,
        value_global_tail_weight: int = 0,
        value_global_tail_exclude_cls: bool = True,
        value_output_bits: int = 0,
        value_final_channel_bits: int = 0,
        value_score_gap_source: str = "raw",
        irpe_mode: str = "none",
        irpe_clip_size: int | tuple[int, int] | None = None,
        irpe_manhattan_clip: int | None = None,
        irpe_bias_bits: int = 6,
        irpe_lut_value_bits: int = 6,
        irpe_scale_bits: int = 4,
        irpe_scale_frac_bits: int = 2,
        irpe_initial_head_scale: float = 1.0,
        irpe_learnable_head_scale: bool = True,
        irpe_learnable_cls_bias: bool = True,
        spatial_pyramid_mode: str = "none",
        spatial_pyramid_bits: int = 3,
        spatial_pyramid_clip: float = 1.0,
        spatial_pyramid_channel_wiring: str = "fixed_roll",
        spatial_pyramid_residual_shift: int = 1,
        spatial_pyramid_permute_patches: bool = False,
        spatial_pyramid_permutation_seed: int = 20260712,
        late_cls_blocks: int = 0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        patches_per_axis = img_size // patch_size
        token_grid_size = (patches_per_axis, patches_per_axis)
        if irpe_mode != "none" and token_grid_size[0] * token_grid_size[1] != num_patches:
            raise ValueError("integer iRPE requires a regular square patch grid")
        if spatial_pyramid_mode not in {
            "none",
            "flat-cls-mean",
            "cls-only",
            "topdown-broadcast",
        }:
            raise ValueError(
                "spatial_pyramid_mode must be 'none', 'flat-cls-mean', "
                "'cls-only', or 'topdown-broadcast'"
            )
        if spatial_pyramid_residual_shift < 0 or spatial_pyramid_residual_shift > 8:
            raise ValueError("spatial_pyramid_residual_shift must be in [0, 8]")
        if late_cls_blocks < 0 or late_cls_blocks > depth:
            raise ValueError("late_cls_blocks must be in [0, depth]")
        if late_cls_blocks and irpe_mode != "none":
            raise ValueError(
                "late CLS currently requires irpe_mode='none'; early patch-only blocks "
                "need a no-CLS relative-position table before the two mechanisms can be combined"
            )
        self.spatial_pyramid_mode = spatial_pyramid_mode
        self.spatial_pyramid_residual_shift = int(spatial_pyramid_residual_shift)
        self.late_cls_blocks = int(late_cls_blocks)
        if spatial_pyramid_permute_patches and spatial_pyramid_mode != "cls-only":
            raise ValueError(
                "spatial_pyramid_permute_patches is currently a cls-only hierarchy control"
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        self.spatial_pyramid = None
        if spatial_pyramid_mode in {"cls-only", "topdown-broadcast"}:
            self.spatial_pyramid = ConvLogicInspiredPyramidMixer(
                channels=embed_dim,
                grid_size=patches_per_axis,
                bits=spatial_pyramid_bits,
                clip=spatial_pyramid_clip,
                injection_mode=(
                    "cls_only"
                    if spatial_pyramid_mode == "cls-only"
                    else "topdown_broadcast"
                ),
                output_mode="delta",
                channel_wiring=spatial_pyramid_channel_wiring,
            )
        if spatial_pyramid_permute_patches:
            generator = torch.Generator().manual_seed(int(spatial_pyramid_permutation_seed))
            permutation = torch.randperm(num_patches, generator=generator)
            self.register_buffer(
                "_spatial_pyramid_patch_permutation",
                permutation,
                persistent=True,
            )
        else:
            self._spatial_pyramid_patch_permutation = None

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                LogicTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    attention_only=attention_only,
                    attention_k=attention_k,
                    validate_input=validate_input,
                    use_thermometer_encoding=use_thermometer_encoding,
                    n_thresholds=n_thresholds,
                    v_n_thresholds=v_n_thresholds,
                    encoding_scale=encoding_scale,
                    apply_sigmoid_before_encoding=apply_sigmoid_before_encoding,
                    decode_output=decode_output,
                    boundary_surrogate_temperature=boundary_surrogate_temperature,
                    topk_impl=topk_impl,
                    topk_forward_mode=topk_forward_mode,
                    topk_surrogate_mode=topk_surrogate_mode,
                    topk_surrogate_proxy_source=topk_surrogate_proxy_source,
                    topk_kth_detach_value=topk_kth_detach_value,
                    topk_kth_use_midpoint_theta=topk_kth_use_midpoint_theta,
                    topk_kth_normalize_soft_mask=topk_kth_normalize_soft_mask,
                    majority_train_temperature=majority_train_temperature,
                    fast_vote=fast_vote,
                    majority_surrogate_mode=majority_surrogate_mode,
                    majority_k_root_degree=majority_k_root_degree,
                    thermometer_decode_use_weights=thermometer_decode_use_weights,
                    value_aggregation=value_aggregation,
                    value_gap_shift=value_gap_shift,
                    value_max_gap_bucket=value_max_gap_bucket,
                    value_global_tail_weight=value_global_tail_weight,
                    value_global_tail_exclude_cls=value_global_tail_exclude_cls,
                    value_output_bits=value_output_bits,
                    value_final_channel_bits=value_final_channel_bits,
                    value_score_gap_source=value_score_gap_source,
                    token_grid_size=token_grid_size,
                    irpe_mode=irpe_mode,
                    irpe_clip_size=irpe_clip_size,
                    irpe_manhattan_clip=irpe_manhattan_clip,
                    irpe_bias_bits=irpe_bias_bits,
                    irpe_lut_value_bits=irpe_lut_value_bits,
                    irpe_scale_bits=irpe_scale_bits,
                    irpe_scale_frac_bits=irpe_scale_frac_bits,
                    irpe_initial_head_scale=irpe_initial_head_scale,
                    irpe_learnable_head_scale=irpe_learnable_head_scale,
                    irpe_learnable_cls_bias=irpe_learnable_cls_bias,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_fn)

    @staticmethod
    def _init_weights_fn(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            if isinstance(m.weight, UninitializedParameter):
                return
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _spatial_context_delta(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.spatial_pyramid_mode == "none":
            return None
        if self.spatial_pyramid_mode == "flat-cls-mean":
            delta = torch.zeros_like(x)
            delta[:, 0] = x[:, 1:].mean(dim=1)
            return delta
        if self.spatial_pyramid is None:
            raise RuntimeError("spatial pyramid mode has no mixer")
        pyramid_input = x
        if self._spatial_pyramid_patch_permutation is not None:
            patches = x[:, 1:].index_select(1, self._spatial_pyramid_patch_permutation)
            pyramid_input = torch.cat((x[:, :1], patches), dim=1)
        return self.spatial_pyramid(pyramid_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        batch_size = x.shape[0]

        if self.late_cls_blocks == 0:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            x = self.pos_drop(x + self.pos_embed)
            pyramid_delta = self._spatial_context_delta(x)
            if pyramid_delta is not None:
                x = x + pyramid_delta / float(1 << self.spatial_pyramid_residual_shift)
            for block in self.blocks:
                x = block(x)
        else:
            trunk_block_count = len(self.blocks) - self.late_cls_blocks
            x = self.pos_drop(x + self.pos_embed[:, 1:])
            for block in self.blocks[:trunk_block_count]:
                x = block(x)
            cls_tokens = self.pos_drop(
                self.cls_token.expand(batch_size, -1, -1) + self.pos_embed[:, :1]
            )
            x = torch.cat([cls_tokens, x], dim=1)
            pyramid_delta = self._spatial_context_delta(x)
            if pyramid_delta is not None:
                x = x + pyramid_delta / float(1 << self.spatial_pyramid_residual_shift)
            for block in self.blocks[trunk_block_count:]:
                x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x


def vit_tiny(**kwargs) -> ViTTiny:
    model = ViTTiny(
        img_size=kwargs.get("img_size", 32),
        patch_size=kwargs.get("patch_size", 4),
        in_channels=kwargs.get("in_channels", 3),
        num_classes=kwargs.get("num_classes", 10),
        embed_dim=kwargs.get("embed_dim", 192),
        depth=kwargs.get("depth", 12),
        num_heads=kwargs.get("num_heads", 3),
        mlp_ratio=kwargs.get("mlp_ratio", 4.0),
        drop_path_rate=kwargs.get("drop_path_rate", 0.0),
        attention_only=kwargs.get("attention_only", False),
        attention_k=kwargs.get("attention_k", 15),
        validate_input=kwargs.get("validate_input", False),
        use_thermometer_encoding=kwargs.get("use_thermometer_encoding", False),
        n_thresholds=kwargs.get("n_thresholds", 8),
        v_n_thresholds=kwargs.get("v_n_thresholds", None),
        encoding_scale=kwargs.get("encoding_scale", 10.0),
        apply_sigmoid_before_encoding=kwargs.get("apply_sigmoid_before_encoding", True),
        decode_output=kwargs.get("decode_output", False),
        boundary_surrogate_temperature=kwargs.get("boundary_surrogate_temperature", 1.0),
        topk_impl=kwargs.get("topk_impl", "winner-tree"),
        topk_forward_mode=kwargs.get("topk_forward_mode", "topk"),
        topk_surrogate_mode=kwargs.get("topk_surrogate_mode", "kth"),
        topk_surrogate_proxy_source=kwargs.get("topk_surrogate_proxy_source", "encoded"),
        topk_kth_detach_value=kwargs.get("topk_kth_detach_value", True),
        topk_kth_use_midpoint_theta=kwargs.get("topk_kth_use_midpoint_theta", False),
        topk_kth_normalize_soft_mask=kwargs.get("topk_kth_normalize_soft_mask", True),
        majority_train_temperature=kwargs.get("majority_train_temperature", 0.125),
        fast_vote=kwargs.get("fast_vote", False),
        majority_surrogate_mode=kwargs.get("majority_surrogate_mode", "count"),
        majority_k_root_degree=kwargs.get("majority_k_root_degree", 2),
        thermometer_decode_use_weights=kwargs.get("thermometer_decode_use_weights", True),
        value_aggregation=kwargs.get("value_aggregation", "majority"),
        value_gap_shift=kwargs.get("value_gap_shift", 1),
        value_max_gap_bucket=kwargs.get("value_max_gap_bucket", 3),
        value_global_tail_weight=kwargs.get("value_global_tail_weight", 0),
        value_global_tail_exclude_cls=kwargs.get("value_global_tail_exclude_cls", True),
        value_output_bits=kwargs.get("value_output_bits", 0),
        value_final_channel_bits=kwargs.get("value_final_channel_bits", 0),
        value_score_gap_source=kwargs.get("value_score_gap_source", "raw"),
        irpe_mode=kwargs.get("irpe_mode", "none"),
        irpe_clip_size=kwargs.get("irpe_clip_size", None),
        irpe_manhattan_clip=kwargs.get("irpe_manhattan_clip", None),
        irpe_bias_bits=kwargs.get("irpe_bias_bits", 6),
        irpe_lut_value_bits=kwargs.get("irpe_lut_value_bits", 6),
        irpe_scale_bits=kwargs.get("irpe_scale_bits", 4),
        irpe_scale_frac_bits=kwargs.get("irpe_scale_frac_bits", 2),
        irpe_initial_head_scale=kwargs.get("irpe_initial_head_scale", 1.0),
        irpe_learnable_head_scale=kwargs.get("irpe_learnable_head_scale", True),
        irpe_learnable_cls_bias=kwargs.get("irpe_learnable_cls_bias", True),
        spatial_pyramid_mode=kwargs.get("spatial_pyramid_mode", "none"),
        spatial_pyramid_bits=kwargs.get("spatial_pyramid_bits", 3),
        spatial_pyramid_clip=kwargs.get("spatial_pyramid_clip", 1.0),
        spatial_pyramid_channel_wiring=kwargs.get(
            "spatial_pyramid_channel_wiring", "fixed_roll"
        ),
        spatial_pyramid_residual_shift=kwargs.get("spatial_pyramid_residual_shift", 1),
        spatial_pyramid_permute_patches=kwargs.get(
            "spatial_pyramid_permute_patches", False
        ),
        spatial_pyramid_permutation_seed=kwargs.get(
            "spatial_pyramid_permutation_seed", 20260712
        ),
        late_cls_blocks=kwargs.get("late_cls_blocks", 0),
    )
    return model


if __name__ == "__main__":
    model = vit_tiny()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
