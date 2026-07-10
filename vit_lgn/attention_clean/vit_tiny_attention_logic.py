"""
Vision Transformer Tiny 模型定义。
仅替换 TransformerBlock 中的 MultiHeadAttention 为 AttentionLogic。
"""

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from vit_tiny_baseline import DropPath, PatchEmbedding, MLP
from src.attention_logic import AttentionLogic


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
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
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
    )
    return model


if __name__ == "__main__":
    model = vit_tiny()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
