"""
Vision Transformer Tiny 模型定义
适用于 CIFAR-10 数据集
"""
import torch
import torch.nn as nn
import math


class DropPath(nn.Module):
    """
    随机深度 / DropPath 正则化。

    在训练时随机丢弃路径，有助于正则化和提高泛化能力。
    """
    def __init__(self, drop_prob=0.):
        """
        初始化 DropPath 模块。

        Args:
            drop_prob: 丢弃概率（0.0 到 1.0 之间）
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入张量

        Returns:
            torch.Tensor: 输出张量（训练时可能被随机丢弃）
        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x / keep_prob * random_tensor
        return output


class PatchEmbedding(nn.Module):
    """
    图像补丁嵌入层。

    将图像分割成补丁并投影到嵌入空间。
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        """
        初始化补丁嵌入层。

        Args:
            img_size: 输入图像大小
            patch_size: 补丁大小
            in_channels: 输入通道数
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入图像张量 [batch_size, in_channels, img_size, img_size]

        Returns:
            torch.Tensor: 补丁嵌入 [batch_size, n_patches, embed_dim]
        """
        # x: [B, 3, 32, 32]
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制。

    实现标准的缩放点积注意力，支持多个注意力头。
    """
    def __init__(self, embed_dim=192, num_heads=3):
        """
        初始化多头注意力层。

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数（必须能整除 embed_dim）
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入张量 [batch_size, num_patches, embed_dim]

        Returns:
            torch.Tensor: 输出张量 [batch_size, num_patches, embed_dim]
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """
    前馈神经网络。

    标准的 MLP，包含两个线性层和 GELU 激活函数。
    """
    def __init__(self, embed_dim=192, mlp_ratio=4.0):
        """
        初始化 MLP。

        Args:
            embed_dim: 嵌入维度
            mlp_ratio: MLP 扩展比例（隐藏层维度 = embed_dim * mlp_ratio）
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入张量 [batch_size, num_patches, embed_dim]

        Returns:
            torch.Tensor: 输出张量 [batch_size, num_patches, embed_dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer 编码器块。

    包含自注意力层和前馈网络，使用残差连接和 LayerNorm。
    可以通过 attention_only 开关关闭 FFN，仅保留 Attention（用于对比实验）。
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attention_only: bool = False,
    ):
        """
        初始化 Transformer 块。

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            mlp_ratio: MLP 扩展比例
            drop_path: DropPath 比率
            attention_only: 若为 True，则不使用 FFN，仅保留 Attention + 残差
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        # attention_only 时将 FFN 替换为恒等映射，方便做 attention-only 对照实验
        self.mlp = nn.Identity() if attention_only else MLP(embed_dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入张量 [batch_size, num_patches, embed_dim]

        Returns:
            torch.Tensor: 输出张量 [batch_size, num_patches, embed_dim]
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTTiny(nn.Module):
    """ViT-Tiny 模型

    Args:
        img_size: 输入图像大小（默认 32,用于 CIFAR-10)
        patch_size: 补丁大小（默认 4)
        in_channels: 输入通道数（默认 3,RGB 图像)
        num_classes: 分类数量（默认 10,用于 CIFAR-10)
        embed_dim: 嵌入维度（默认 192)
        depth: Transformer 层数（默认 12)
        num_heads: 注意力头数（默认 3)
        mlp_ratio: MLP 扩展比例（默认 4.0)
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        attention_only: bool = False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # 类别 token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        # Transformer 编码器
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                attention_only=attention_only,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
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

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入图像张量 [batch_size, in_channels, img_size, img_size]

        Returns:
            torch.Tensor: 分类 logits [batch_size, num_classes]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]
        B = x.shape[0]

        # 添加类别 token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, embed_dim]

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer 编码器
        for blk in self.blocks:
            x = blk(x)

        # 分类头
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 使用类别 token 进行分类
        x = self.head(cls_token_final)

        return x


def vit_tiny(**kwargs):
    """
    工厂函数：创建 ViT-Tiny 模型。

    Args:
        **kwargs: 模型参数（与 ViTTiny.__init__ 的参数相同）

    Returns:
        ViTTiny: 创建好的模型实例
    """
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
    )
    return model


if __name__ == "__main__":
    # 测试模型
    model = vit_tiny()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
