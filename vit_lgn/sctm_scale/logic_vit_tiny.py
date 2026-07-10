"""
Logic-ViT Tiny
--------------

基于 `vit_tiny_baseline.py` 中的 ViT-Tiny 结构，
将每个 Transformer Block 的 MLP 替换为 3 层等维的 LogicLayer 逻辑 FFN（纯 PyTorch 版本）。

本文件只负责定义“逻辑版 ViT-Tiny” 模型结构，不包含训练逻辑。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from vit_tiny_baseline import DropPath, PatchEmbedding, MultiHeadAttention
from logic_iwp import LogicFFN_IWP, RandomForestLogicFFN_IWP


class NonNegativeLinear(nn.Module):
    """Linear head with nonnegative feature weights and a free class bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.trunc_normal_(self.raw_weight, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class SignedSparseLinear(nn.Linear):
    """Dense trainable linear head that can be converted to signed per-class top-k."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        topk: int = 32,
        use_topk_mask: bool = False,
        counter_mode: bool = False,
        scaled_counter: bool = False,
    ):
        super().__init__(in_features, out_features)
        self.topk = int(topk)
        self.register_buffer("topk_mask", torch.ones(out_features, in_features))
        self.register_buffer("use_topk_mask_flag", torch.tensor(bool(use_topk_mask), dtype=torch.bool))
        self.register_buffer("counter_mode_flag", torch.tensor(bool(counter_mode), dtype=torch.bool))
        self.register_buffer("scaled_counter_flag", torch.tensor(bool(scaled_counter), dtype=torch.bool))
        self.register_buffer("counter_scale", torch.ones(out_features))
        if use_topk_mask:
            self.apply_topk_mask()

    def apply_topk_mask(self) -> None:
        if self.topk <= 0:
            self.topk_mask.zero_()
            self.use_topk_mask_flag.fill_(True)
            self.counter_scale.fill_(1.0)
            return
        with torch.no_grad():
            if self.topk >= self.weight.shape[1]:
                mask = torch.ones_like(self.weight)
            else:
                mask = torch.zeros_like(self.weight)
                idx = self.weight.abs().topk(self.topk, dim=1).indices
                mask.scatter_(1, idx, 1.0)
            self.topk_mask.copy_(mask)
            selected_count = mask.sum(dim=1).clamp_min(1.0)
            scale = (self.weight.abs() * mask).sum(dim=1) / selected_count
            self.counter_scale.copy_(scale.clamp_min(1e-6))
            self.use_topk_mask_flag.fill_(True)

    def effective_weight(self) -> torch.Tensor:
        if not bool(self.use_topk_mask_flag.item()):
            return self.weight
        if bool(self.counter_mode_flag.item()):
            weight = self.weight.sign() * self.topk_mask
            if bool(self.scaled_counter_flag.item()):
                weight = weight * self.counter_scale.unsqueeze(1)
            return weight
        return self.weight * self.topk_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class FixedGroupSumHead(nn.Module):
    """Fixed DLGN-style GroupSum readout over contiguous feature groups."""

    def __init__(self, in_features: int, out_features: int, use_bias: bool = False, tau: float = 1.0):
        super().__init__()
        group_size = in_features // out_features
        if group_size < 1:
            raise ValueError(
                f"group_sum head needs in_features >= out_features, got {in_features} and {out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.used_features = int(group_size * out_features)
        self.tau = float(tau)
        weight = torch.zeros(out_features, in_features)
        for cls_idx in range(out_features):
            start = cls_idx * group_size
            weight[cls_idx, start : start + group_size] = 1.0 / max(self.tau, 1e-6)
        self.register_buffer("weight", weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def build_classifier_head(
    in_features: int,
    out_features: int,
    head_type: str = "linear",
    signed_head_topk: int = 32,
    signed_head_use_topk_mask: bool = False,
) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(in_features, out_features)
    if head_type == "group_sum":
        return FixedGroupSumHead(in_features, out_features, use_bias=False)
    if head_type == "group_sum_bias":
        return FixedGroupSumHead(in_features, out_features, use_bias=True)
    if head_type == "nonnegative_linear":
        return NonNegativeLinear(in_features, out_features)
    if head_type in {"signed_sparse_linear", "signed_counter_linear", "scaled_signed_counter_linear"}:
        return SignedSparseLinear(
            in_features,
            out_features,
            topk=signed_head_topk,
            use_topk_mask=signed_head_use_topk_mask,
            counter_mode=head_type in {"signed_counter_linear", "scaled_signed_counter_linear"},
            scaled_counter=head_type == "scaled_signed_counter_linear",
        )
    raise ValueError(f"Unknown head_type: {head_type}")


class LogicTransformerBlock(nn.Module):
    """
    使用 LogicFFN_IWP 作为 FFN 的 Transformer Block。

    包含：
    - LayerNorm + MultiHeadAttention（自注意力）
    - LayerNorm + LogicFFN_IWP（逻辑前馈网络）
    - DropPath 正则化
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        drop_path: float = 0.0,
        ffn_layers: int = 3,
        logic_hidden_multiplier: float = 1.0,
        logic_grad_factor: float = 1.0,
        logic_resconnection_init: bool = True,
        logic_weight_init: str = "ri",
        logic_weight_init_sigma: float = 0.5,
        logic_res_connect_fraction: Optional[float] = None,
        logic_shift_init: bool = True,
        logic_shift_init_type: str = "ri",
        logic_shift_init_shift: float = 1.2,
        logic_shift_init_direction: str = "0101",
        logic_soft_eval: bool = True,
        logic_connections: str = "unique",
        logic_n_thresholds: int = 31,
        logic_use_thermometer: bool = True,
        logic_encoding_temperature: float = 10.0,
        logic_act_fn: str = "SIN01",
        logic_connectivity: str = "fixed",
        learnable_conn_k: int = 64,
        learnable_conn_use_skip_bias: bool = True,
    ):
        """
        初始化逻辑 Transformer 块。

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            drop_path: DropPath 比率
            ffn_layers: 逻辑 FFN 层数
            logic_grad_factor: 逻辑层梯度因子
            logic_resconnection_init: 是否使用残差连接初始化（改 indices + 可选 freeze）
            logic_weight_init: 权重初始化方式（"ri" 软直通 A / "gauss" 高斯随机）
            logic_weight_init_sigma: weight_init 的尺度（ri 时 ±sigma 送入激活；gauss 时标准差）
            logic_res_connect_fraction: 若设置，覆盖残差连接比例（0 即不使用残差连接）
            logic_shift_init: 是否执行重尾初始化（shift_init）
            logic_shift_init_type: 重尾类型 ri / and-or / and-or-ri / uniform
            logic_shift_init_shift: 重尾偏移量
            logic_shift_init_direction: 4 位方向串（仅 type=ri 时）
            logic_soft_eval: 是否使用软评估模式
            logic_connections: 连接模式（"random" 或 "unique"）
            logic_n_thresholds: 温度计编码阈值数量
            logic_use_thermometer: 是否使用温度计编码
            logic_encoding_temperature: 编码温度参数
            logic_act_fn: 逻辑层激活函数（"SIN01" / "sigmoid" / "sigmoid-st" / "sin-st" / "linear"）
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = LogicFFN_IWP(
            embed_dim=embed_dim,
            num_layers=ffn_layers,
            hidden_multiplier=logic_hidden_multiplier,
            n_thresholds=logic_n_thresholds,
            use_thermometer=logic_use_thermometer,
            encoding_temperature=logic_encoding_temperature,
            grad_factor=logic_grad_factor,
            residual_init=logic_resconnection_init,
            weight_init_choice=logic_weight_init,
            weight_init_sigma=logic_weight_init_sigma,
            residual_connect_fraction=logic_res_connect_fraction,
            shift_init_enable=logic_shift_init,
            shift_init_type=logic_shift_init_type,
            init_shift=logic_shift_init_shift,
            init_shift_direction=logic_shift_init_direction,
            soft_eval=logic_soft_eval,
            connections=logic_connections,
            act_fn=logic_act_fn,
            connectivity=logic_connectivity,
            learnable_conn_k=learnable_conn_k,
            learnable_conn_use_skip_bias=learnable_conn_use_skip_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 [batch_size, num_patches, embed_dim]

        Returns:
            torch.Tensor: 输出张量 [batch_size, num_patches, embed_dim]
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class TreeLogicTransformerBlock(nn.Module):
    """
    使用 RandomForestLogicFFN_IWP（向量化严格森林）作为 FFN 的 Transformer Block。
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        drop_path: float = 0.0,
        logic_grad_factor: float = 1.0,
        logic_resconnection_init: bool = True,
        logic_weight_init: str = "ri",
        logic_weight_init_sigma: float = 0.5,
        logic_res_connect_fraction: Optional[float] = None,
        logic_shift_init: bool = True,
        logic_shift_init_type: str = "ri",
        logic_shift_init_shift: float = 1.2,
        logic_shift_init_direction: str = "0101",
        logic_soft_eval: bool = True,
        logic_connections: str = "unique",
        logic_n_thresholds: int = 31,
        logic_use_thermometer: bool = True,
        logic_encoding_temperature: float = 10.0,
        num_forest_layers: int = 2,
        logic_act_fn: str = "SIN01",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = RandomForestLogicFFN_IWP(
            embed_dim=embed_dim,
            n_thresholds=logic_n_thresholds,
            use_thermometer=logic_use_thermometer,
            encoding_temperature=logic_encoding_temperature,
            residual_connect_fraction=logic_res_connect_fraction,
            grad_factor=logic_grad_factor,
            residual_init=logic_resconnection_init,
            weight_init_choice=logic_weight_init,
            weight_init_sigma=logic_weight_init_sigma,
            shift_init_enable=logic_shift_init,
            shift_init_type=logic_shift_init_type,
            init_shift=logic_shift_init_shift,
            init_shift_direction=logic_shift_init_direction,
            soft_eval=logic_soft_eval,
            num_forest_layers=num_forest_layers,
            act_fn=logic_act_fn,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 [batch_size, num_patches, embed_dim]

        Returns:
            torch.Tensor: 输出张量 [batch_size, num_patches, embed_dim]
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class LogicViTTiny(nn.Module):
    """
    逻辑版 ViT-Tiny：
    - PatchEmbedding 与原始 ViT 相同
    - Attention 与原始 ViT 相同
    - FFN 完全由 LogicFFN_IWP 替换
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        drop_path_rate: float = 0.0,
        logic_ffn_layers: int = 3,
        logic_hidden_multiplier: float = 1.0,
        logic_grad_factor: float = 1.0,
        logic_resconnection_init: bool = True,
        logic_weight_init: str = "ri",
        logic_weight_init_sigma: float = 0.5,
        logic_res_connect_fraction: Optional[float] = None,
        logic_shift_init: bool = True,
        logic_shift_init_type: str = "ri",
        logic_shift_init_shift: float = 1.2,
        logic_shift_init_direction: str = "0101",
        logic_soft_eval: bool = True,
        logic_connections: str = "unique",
        logic_n_thresholds: int = 31,
        logic_use_thermometer: bool = True,
        logic_encoding_temperature: float = 10.0,
        ffn_type: str = "logic",
        num_forest_layers: int = 2,
        logic_act_fn: str = "SIN01",
        logic_connectivity: str = "fixed",
        learnable_conn_k: int = 64,
        learnable_conn_use_skip_bias: bool = True,
        head_type: str = "linear",
        signed_head_topk: int = 32,
        signed_head_use_topk_mask: bool = False,
    ):
        """
        初始化逻辑版 ViT-Tiny 模型。

        Args:
            img_size: 输入图像大小
            patch_size: 补丁大小
            in_channels: 输入通道数
            num_classes: 分类类别数
            embed_dim: 嵌入维度
            depth: Transformer 层数
            num_heads: 注意力头数
            drop_path_rate: DropPath 比率
            logic_ffn_layers: 逻辑 FFN 层数
            logic_grad_factor: 逻辑层梯度因子
            logic_resconnection_init: 是否使用残差连接初始化（改 indices + 可选 freeze）
            logic_weight_init: 权重初始化方式（"ri" 软直通 A / "gauss" 高斯随机）
            logic_weight_init_sigma: weight_init 的尺度（ri 时 ±sigma 送入激活；gauss 时标准差）
            logic_res_connect_fraction: 若设置，覆盖残差连接比例（0-1）；0 即不使用残差连接；未设置时由 logic_resconnection_init 控制
            logic_shift_init: 是否执行重尾初始化（shift_init）
            logic_shift_init_type: 重尾类型 ri / and-or / and-or-ri / uniform
            logic_shift_init_shift: 重尾偏移量
            logic_shift_init_direction: 4 位方向串（仅 type=ri 时）
            logic_soft_eval: 是否使用软评估模式
            logic_connections: 连接模式（"random" 或 "unique"）
            logic_n_thresholds: 温度计编码阈值数量
            logic_use_thermometer: 是否使用温度计编码
            logic_encoding_temperature: 编码温度参数
            logic_act_fn: 逻辑层激活函数（"SIN01" / "sigmoid" / "sigmoid-st" / "sin-st" / "linear"）
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # class token & 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        # Transformer 编码器（可选择使用 Logic 或 TreeLogic 版本的 FFN）
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        if ffn_type not in {"logic", "tree-logic"}:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        block_cls = LogicTransformerBlock if ffn_type == "logic" else TreeLogicTransformerBlock
        blocks = []
        for i in range(depth):
            kwargs = dict(
                embed_dim=embed_dim,
                num_heads=num_heads,
                drop_path=dpr[i],
                logic_grad_factor=logic_grad_factor,
                logic_resconnection_init=logic_resconnection_init,
                logic_weight_init=logic_weight_init,
                logic_weight_init_sigma=logic_weight_init_sigma,
                logic_res_connect_fraction=logic_res_connect_fraction,
                logic_shift_init=logic_shift_init,
                logic_shift_init_type=logic_shift_init_type,
                logic_shift_init_shift=logic_shift_init_shift,
                logic_shift_init_direction=logic_shift_init_direction,
                logic_soft_eval=logic_soft_eval,
                logic_connections=logic_connections,
                logic_n_thresholds=logic_n_thresholds,
                logic_use_thermometer=logic_use_thermometer,
                logic_encoding_temperature=logic_encoding_temperature,
                logic_act_fn=logic_act_fn,
            )
            if block_cls is TreeLogicTransformerBlock:
                kwargs["num_forest_layers"] = num_forest_layers
            if block_cls is LogicTransformerBlock:
                kwargs["ffn_layers"] = logic_ffn_layers
                kwargs["logic_hidden_multiplier"] = logic_hidden_multiplier
                kwargs["logic_connectivity"] = logic_connectivity
                kwargs["learnable_conn_k"] = learnable_conn_k
                kwargs["learnable_conn_use_skip_bias"] = learnable_conn_use_skip_bias
            blocks.append(block_cls(**kwargs))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = build_classifier_head(
            embed_dim,
            num_classes,
            head_type=head_type,
            signed_head_topk=signed_head_topk,
            signed_head_use_topk_mask=signed_head_use_topk_mask,
        )

        self._init_weights()
        if signed_head_use_topk_mask and hasattr(self.head, "apply_topk_mask"):
            self.head.apply_topk_mask()

    def _init_weights(self):
        """
        初始化模型权重。

        使用截断正态分布初始化位置编码和类别 token，
        并对所有线性层和 LayerNorm 层进行初始化。
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights_fn)

    def _init_weights_fn(self, m):
        """
        权重初始化辅助函数。

        Args:
            m: 模块对象
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入图像张量 [batch_size, in_channels, img_size, img_size]

        Returns:
            torch.Tensor: 分类 logits [batch_size, num_classes]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        B = x.shape[0]

        # 添加 cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # 位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 编码器
        for blk in self.blocks:
            x = blk(x)

        # 分类头
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x


def logic_vit_tiny(**kwargs) -> LogicViTTiny:
    """
    工厂函数：创建 LogicViTTiny 模型。

    Args:
        **kwargs: 模型参数（与 LogicViTTiny.__init__ 的参数相同）

    Returns:
        LogicViTTiny: 创建好的模型实例
    """
    model = LogicViTTiny(
        img_size=kwargs.get("img_size", 32),
        patch_size=kwargs.get("patch_size", 4),
        in_channels=kwargs.get("in_channels", 3),
        num_classes=kwargs.get("num_classes", 10),
        embed_dim=kwargs.get("embed_dim", 192),
        depth=kwargs.get("depth", 6),
        num_heads=kwargs.get("num_heads", 3),
        drop_path_rate=kwargs.get("drop_path_rate", 0.05),
        logic_ffn_layers=kwargs.get("logic_ffn_layers", 3),
        logic_hidden_multiplier=kwargs.get("logic_hidden_multiplier", 1.0),
        logic_grad_factor=kwargs.get("logic_grad_factor", 1.0),
        logic_resconnection_init=kwargs.get("logic_resconnection_init", True),
        logic_weight_init=kwargs.get("logic_weight_init", "ri"),
        logic_weight_init_sigma=kwargs.get("logic_weight_init_sigma", 0.5),
        logic_res_connect_fraction=kwargs.get("logic_res_connect_fraction"),
        logic_shift_init=kwargs.get("logic_shift_init", True),
        logic_shift_init_type=kwargs.get("logic_shift_init_type", "ri"),
        logic_shift_init_shift=kwargs.get("logic_shift_init_shift", 1.2),
        logic_shift_init_direction=kwargs.get("logic_shift_init_direction", "0101"),
        logic_soft_eval=kwargs.get("logic_soft_eval", True),
        logic_connections=kwargs.get("logic_connections", "unique"),
        logic_n_thresholds=kwargs.get("logic_n_thresholds", 31),
        logic_use_thermometer=kwargs.get("logic_use_thermometer", True),
        logic_encoding_temperature=kwargs.get("logic_encoding_temperature", 10.0),
        ffn_type=kwargs.get("ffn_type", "logic"),
        num_forest_layers=kwargs.get("num_forest_layers", 2),
        logic_act_fn=kwargs.get("logic_act_fn", "SIN01"),
        logic_connectivity=kwargs.get("logic_connectivity", "fixed"),
        learnable_conn_k=kwargs.get("learnable_conn_k", 64),
        learnable_conn_use_skip_bias=kwargs.get("learnable_conn_use_skip_bias", True),
        head_type=kwargs.get("head_type", "linear"),
        signed_head_topk=kwargs.get("signed_head_topk", 32),
        signed_head_use_topk_mask=kwargs.get("signed_head_use_topk_mask", False),
    )
    return model


if __name__ == "__main__":
    # 简单自测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = logic_vit_tiny().to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print(f"Params : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
