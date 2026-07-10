"""
功能函数模块：提供逻辑运算、连接生成和权重初始化等工具函数
"""
import torch
import numpy as np

BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


# 16种二元逻辑运算的真值表：
# | id | 运算符                  | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|-------------------------|-------|-------|-------|-------|
# | 0  | 0（常量0）              | 0     | 0     | 0     | 0     |
# | 1  | A and B（与）           | 0     | 0     | 0     | 1     |
# | 2  | not(A implies B)       | 0     | 0     | 1     | 0     |
# | 3  | A                      | 0     | 0     | 1     | 1     |
# | 4  | not(B implies A)       | 0     | 1     | 0     | 0     |
# | 5  | B                      | 0     | 1     | 0     | 1     |
# | 6  | A xor B（异或）         | 0     | 1     | 1     | 0     |
# | 7  | A or B（或）            | 0     | 1     | 1     | 1     |
# | 8  | not(A or B)（或非）     | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)（同或）    | 1     | 0     | 0     | 1     |
# | 10 | not(B)                 | 1     | 0     | 1     | 0     |
# | 11 | B implies A            | 1     | 0     | 1     | 1     |
# | 12 | not(A)                 | 1     | 1     | 0     | 0     |
# | 13 | A implies B            | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)（与非）    | 1     | 1     | 1     | 0     |
# | 15 | 1（常量1）              | 1     | 1     | 1     | 1     |

def bin_op(a, b, i):
    """
    执行第i种二元逻辑运算

    参数:
        a: 第一个输入张量
        b: 第二个输入张量
        i: 逻辑运算的索引（0-15）

    返回:
        运算结果张量
    """
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)

    if i == 0:
        return torch.zeros_like(a)
    elif i == 1:
        return a * b
    elif i == 2:
        return a - a * b
    elif i == 3:
        return a
    elif i == 4:
        return b - a * b
    elif i == 5:
        return b
    elif i == 6:
        return a + b - 2 * a * b
    elif i == 7:
        return a + b - a * b
    elif i == 8:
        return 1 - (a + b - a * b)
    elif i == 9:
        return 1 - (a + b - 2 * a * b)
    elif i == 10:
        return 1 - b
    elif i == 11:
        return 1 - b + a * b
    elif i == 12:
        return 1 - a
    elif i == 13:
        return 1 - a + a * b
    elif i == 14:
        return 1 - a * b
    elif i == 15:
        return torch.ones_like(a)


def bin_op_s(a, b, i_s):
    """
    加权组合16种二元逻辑运算

    参数:
        a: 第一个输入张量
        b: 第二个输入张量
        i_s: 16种运算的权重（概率分布），形状为(..., 16)

    返回:
        加权组合后的结果张量
    """
    r = torch.zeros_like(a)
    for i in range(16):
        u = bin_op(a, b, i)
        r = r + i_s[..., i] * u
    return r

def bin_gate(p, q, w):
    """
    IWP二元逻辑门函数：计算输入p和q通过权重w的逻辑门输出

    参数:
        p: 第一个输入概率
        q: 第二个输入概率
        w: 4种输入组合(00, 01, 10, 11)的权重，形状为(..., 4)

    返回:
        逻辑门输出概率
    """
    prob = torch.stack([
        (1 - p) * (1 - q),  # 00
        (1 - p) * q,        # 01
        p * (1 - q),        # 10
        p * q               # 11
    ], dim=-1)
    res = prob * w
    return torch.sum(res, dim=-1)

########################################################################################################################


def get_unique_connections(in_dim, out_dim, device='cuda'):
    """
    生成唯一的输入到输出连接对

    确保每个输入至少被使用一次，生成唯一（非随机）的连接模式。
    首先取对(0,1), (2,3), (4,5), ...，如果不够则增加偏移量。

    参数:
        in_dim: 输入维度
        out_dim: 输出维度（神经元数量）
        device: 计算设备

    返回:
        (a, b): 两个输入索引张量，形状为(out_dim,)
    """
    assert out_dim * 2 >= in_dim, '神经元数量 ({}) 必须不小于输入数量 ({}) 的一半，否则无法使用或考虑所有输入。'.format(
        out_dim, in_dim
    )

    x = torch.arange(in_dim).long().unsqueeze(0)

    # 取对 (0, 1), (2, 3), (4, 5), ...
    a, b = x[..., ::2], x[..., 1::2]
    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]

    # 如果还不够，取对 (1, 2), (3, 4), (5, 6), ...
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]

    # 如果还不够，使用偏移量 >= 2 的对：
    offset = 2
    while out_dim > a.shape[-1] > offset:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])

    if a.shape[-1] >= out_dim:
        a = a[..., :out_dim]
        b = b[..., :out_dim]
    else:
        assert False, (a.shape[-1], offset, out_dim)

    perm = torch.randperm(out_dim)

    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)

    a, b = a.to(torch.int64), b.to(torch.int64)
    a, b = a.to(device), b.to(device)
    a, b = a.contiguous(), b.contiguous()
    return a, b


########################################################################################################################


class GradFactor(torch.autograd.Function):
    """
    梯度缩放函数：在前向传播中不改变输入，但在反向传播中缩放梯度

    用于深层网络中避免梯度消失，通过增加梯度因子来提高梯度稳定性。
    """
    @staticmethod
    def forward(ctx, x, f):
        """前向传播：直接返回输入"""
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        """反向传播：返回缩放后的梯度"""
        return grad_y * ctx.f, None

########################################################################################################################


class SignGrad(torch.autograd.Function):
    """
    符号梯度函数：前向传播不改变输入，反向传播使用梯度的符号

    将梯度量化为+1或-1，用于梯度估计和优化。
    """
    @staticmethod
    def forward(ctx, x):
        """前向传播：直接返回输入"""
        return x

    @staticmethod
    def backward(ctx, grad_y):
        """反向传播：返回梯度的符号"""
        return torch.sign(grad_y)


########################################################################################################################


class SinSkipGrad(torch.autograd.Function):
    """
    正弦跳过梯度函数：前向传播使用正弦函数，反向传播使用符号梯度

    前向：y = 0.5 + 0.5*sin(x)，将输入映射到[0,1]区间
    反向：使用符号梯度而非真实梯度，假设局部梯度范数为1
    """
    @staticmethod
    def forward(ctx, x):
        """前向传播：应用正弦激活函数"""
        ctx.save_for_backward(x)
        return 0.5 + 0.5*torch.sin(x)

    @staticmethod
    def backward(ctx, grad_y):
        """反向传播：使用符号梯度"""
        # 假设局部梯度范数为1
        x, = ctx.saved_tensors
        local_grad = 0.5*torch.cos(x)
        return torch.sign(local_grad) * grad_y

########################################################################################################################


class SigmoidStraightThrough(torch.autograd.Function):
    """
    Sigmoid直通梯度函数：前向传播使用sigmoid，反向传播直通梯度

    这是一种直通估计器（straight-through estimator），
    在前向传播中应用sigmoid激活，但在反向传播中直接传递梯度而不计算sigmoid的导数。
    """
    @staticmethod
    def forward(ctx, x):
        """前向传播：应用sigmoid激活函数"""
        return torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_y):
        """反向传播：直通梯度（不应用sigmoid导数）"""
        return grad_y

########################################################################################################################


class LinearStraightThrough(torch.autograd.Function):
    """
    线性直通梯度函数：前向传播裁剪到[0,1]区间，反向传播直通梯度

    在前向传播中将输入裁剪到[0,1]范围，但在反向传播中直接传递梯度。
    """
    @staticmethod
    def forward(ctx, x):
        """前向传播：将输入裁剪到[0,1]区间"""
        return torch.clamp(x, 0, 1)

    @staticmethod
    def backward(ctx, grad_y):
        """反向传播：直通梯度"""
        return grad_y

########################################################################################################################

def weight_init(shape, choice, sigma, device='cuda'):
    """
    权重初始化函数

    参数:
        shape: 权重张量的形状
        choice: 初始化方法（'gauss' 或 'ri'）
            - 'gauss': 高斯随机初始化
            - 'ri': 残差初始化（Residual Initialization）
                - 16 维：将索引 3（对应输入 A）设为 sigma，其余为 0
                - 4 维（IWP）：直通 A 模式 [0,0,1,1]，即 w[0]=w[1]=-sigma，w[2]=w[3]=sigma
        sigma: 初始化标准差或值
        device: 计算设备

    返回:
        初始化后的权重张量
    """
    if choice == 'gauss':
        # 高斯随机初始化
        return torch.randn(*shape, device=device) * sigma
    elif choice == 'ri':
        if shape[-1] == 16:
            # 16 维：只激活索引 3（对应输入 A 的逻辑门）
            w = torch.zeros(*shape, device=device)
            w[..., 3] = sigma
            return w
        elif shape[-1] == 4:
            # 4 维 IWP：直通 A（00/01/10/11 -> 0/0/1/1），sigmoid/SIN01 下 ±sigma 可得 0/1
            w = torch.zeros(*shape, device=device)
            w[..., 0] = -sigma
            w[..., 1] = -sigma
            w[..., 2] = sigma
            w[..., 3] = sigma
            return w
        else:
            raise ValueError(f"weight_init(choice='ri') expects shape[-1] in (4, 16), got {shape[-1]}")
    else:
        raise NotImplementedError(choice)
